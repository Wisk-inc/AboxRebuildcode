"""Measure a real 1B-parameter training step on this machine with Bantam's memory stack.

    python experiments/bench_1b.py --steps 3 --threads 2

Builds a 1.0B-parameter ZoomGPT (d=2048, 17 layers, 32k vocab), stores weights
in bf16 (stochastic-rounding updates) with 8-bit Muon/AdamW state, runs full
forward/backward/optimizer steps on real TinyStories tokens and reports
peak memory, seconds per step, tokens/s and the time those rates imply for a
Chinchilla-optimal 20B-token run. The standard recipe (fp32 weights + fp32
AdamW = 16 B/param = 16 GB before activations) is reported by calculation
only: it cannot fit in this machine's RAM.
"""
import torch

torch.set_flush_denormal(True)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from zoomtrain.lowmem import state_bytes  # noqa: E402
from zoomtrain.models import ZoomGPT  # noqa: E402
from zoomtrain.optim import build_optimizer  # noqa: E402


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--d", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=17)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--chinchilla_tokens", type=float, default=2e10)
    ap.add_argument("--data", default="/home/user/data/tinystories/tok4k/train.bin")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bench_1b.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    t0 = time.time()
    model = ZoomGPT(args.vocab, args.d, args.layers, args.heads).to(torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    rss_model = rss_gb()
    opt = build_optimizer(model, "muon", 1e-3, 0.1, state_bits=8)
    toks = np.memmap(args.data, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    times, losses = [], []
    for step in range(args.steps + 1):           # step 0 is warm-up (allocates optimizer state)
        starts = rng.integers(0, len(toks) - args.seq - 2, args.batch)
        chunk = torch.from_numpy(np.stack([toks[s:s + args.seq + 1] for s in starts]).astype(np.int64))
        s = time.perf_counter()
        loss = model(chunk[:, :-1], chunk[:, 1:], k=1)
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.perf_counter() - s
        losses.append(loss.item())
        if step > 0:
            times.append(dt)
        print(f"step {step}: {dt:.1f}s loss {loss.item():.3f} rss {rss_gb():.2f} GB", flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    sec = float(np.median(times))
    tok_s = args.batch * args.seq / sec
    flops = model.flops_per_step(args.batch, args.seq, 1)
    out = {
        "params": n, "config": vars(args),
        "weights_gb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9,
        "optimizer_state_gb": state_bytes(opt) / 1e9,
        "rss_after_model_gb": rss_model, "peak_rss_gb": peak,
        "machine_ram_gb": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9,
        "standard_recipe_static_gb": n * 16 / 1e9,
        "sec_per_step": sec, "tokens_per_s": tok_s, "achieved_gflops": flops / sec / 1e9,
        "chinchilla_tokens": args.chinchilla_tokens,
        "chinchilla_days_at_measured_rate": args.chinchilla_tokens / tok_s / 86400,
        "losses": losses, "build_seconds": t0 and (time.time() - t0),
    }
    out["chinchilla_years_at_measured_rate"] = out["chinchilla_days_at_measured_rate"] / 365.25
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("config", "losses")}, indent=1))


if __name__ == "__main__":
    main()
