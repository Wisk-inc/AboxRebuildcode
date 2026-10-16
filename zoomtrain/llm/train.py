"""Compute-matched LLM training: Zoom Training x {AdamW, Muon} on CPU or GPU.

Every run gets the same FLOP budget and reads the same pre-tokenized chunks in
the same order. Baseline: --zoom 1. Zoom Training: e.g.
--zoom 4:0.35,2:0.25,1:0.4 (zoom level : fraction of the FLOP budget).
Validation is always plain next-token loss at k=1 (the deployable model).

    python llm/train.py --data DIR --budget 6.8e13 --zoom 4:0.35,2:0.25,1:0.4 \
        --optim muon --name my_run [--device cuda --precision bf16 --compile]
"""
import torch

torch.set_flush_denormal(True)  # before any parallel op (see zoomtrain.device)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from zoomtrain.data.tokpack import TokPack  # noqa: E402
from zoomtrain.device import (DutyCycleLimiter, describe, maybe_compile,  # noqa: E402
                              setup_runtime)
from zoomtrain.models import ZoomGPT  # noqa: E402
from zoomtrain.lowmem import state_bytes  # noqa: E402
from zoomtrain.optim import build_optimizer  # noqa: E402
from zoomtrain.zoom import ZoomSchedule, lr_at  # noqa: E402

KMAX = 4


def open_tokens(data_dir, split):
    pack = os.path.join(data_dir, f"{split}.tokpack")
    if os.path.exists(pack):
        return TokPack(pack)
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")


class ChunkStream:
    """Shuffled, non-overlapping windows of the token stream, consumed in order."""

    def __init__(self, tokens, seq, seed):
        self.data = tokens
        self.seq = seq
        n = (len(self.data) - seq - KMAX) // seq
        self.starts = np.random.default_rng(seed).permutation(n) * seq
        self.pos = 0
        self.epochs = 0

    def next(self, batch):
        if self.pos + batch > len(self.starts):
            self.pos, self.epochs = 0, self.epochs + 1
        s = self.starts[self.pos:self.pos + batch]
        self.pos += batch
        chunk = np.stack([self.data[i:i + self.seq + KMAX] for i in s]).astype(np.int64)
        return torch.from_numpy(chunk)


@torch.no_grad()
def evaluate(model, val, args, rt, n_seqs):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, n_seqs, 64):
        rows = [val[s * args.seq:s * args.seq + args.seq + 1]
                for s in range(i, min(i + 64, n_seqs))]
        c = rt.to(torch.from_numpy(np.stack(rows).astype(np.int64)))
        with rt.autocast():
            tot += model(c[:, :-1], c[:, 1:], k=1).float().item() * c.shape[0]
        n += c.shape[0]
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train/val .bin or .tokpack")
    ap.add_argument("--zoom", default="1")
    ap.add_argument("--budget", type=float, required=True, help="training FLOPs")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--state_bits", type=int, default=32, choices=[32, 8],
                    help="8: blockwise 8-bit optimizer state")
    ap.add_argument("--param_dtype", default="fp32", choices=["fp32", "bf16"],
                    help="bf16: weights in bf16, updates written with stochastic rounding")
    ap.add_argument("--grad_ckpt", type=int, default=0, help="activation checkpointing per block")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batch_scale_k", type=int, default=0,
                    help="use batch*k sequences at zoom k (constant positions per step)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--lr_restart", type=int, default=0,
                    help="restart warmup+cosine at every zoom stage switch")
    ap.add_argument("--mix_fine", type=float, default=0.0,
                    help="in coarse stages, run a k=1 step with this probability")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--gated", type=int, default=1)
    ap.add_argument("--merge_mode", default="", help="gate|mean|linear (overrides --gated)")
    ap.add_argument("--out_mode", default="slot", choices=["slot", "bag"])
    ap.add_argument("--rope_pos", default="token", choices=["token", "patch"])
    ap.add_argument("--qk_norm", type=int, default=0)
    ap.add_argument("--evals", type=int, default=10)
    ap.add_argument("--eval_seqs", type=int, default=256)
    ap.add_argument("--final_eval_seqs", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="auto", help="auto|fp32|bf16|fp16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--duty", type=float, default=1.0, help="cap average device load")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "llm"))
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    rt = setup_runtime(args.device, args.precision, args.compile, args.threads, args.seed)
    if rt.device.type == "cpu":
        rt.amp_dtype = None  # fp32 on CPU (no bf16 matmul units on this class of CPU)
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    sched = ZoomSchedule.parse(args.zoom)
    model = ZoomGPT(meta["vocab_size"], args.d, args.layers, args.heads, zoom_ks=sched.ks,
                    gated=bool(args.gated), out_mode=args.out_mode, rope_pos=args.rope_pos,
                    qk_norm=bool(args.qk_norm), merge_mode=args.merge_mode or None,
                    grad_ckpt=bool(args.grad_ckpt)).to(rt.device)
    if args.param_dtype == "bf16":
        model = model.to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    opt = build_optimizer(model, args.optim, args.lr, args.wd, state_bits=args.state_bits)
    fwd = maybe_compile(model, rt)
    stream = ChunkStream(open_tokens(args.data, "train"), args.seq, args.seed)
    val = open_tokens(args.data, "val")
    limiter = DutyCycleLimiter(args.duty, rt.device)

    log = {"args": vars(args), "schedule": repr(sched), "params": n_params, "evals": [],
           "train": [], "runtime": describe(rt)}
    print(f"[{args.name}] params={n_params/1e6:.2f}M zoom={sched} optim={args.optim} "
          f"budget={args.budget:.3e} {describe(rt)}", flush=True)

    flops, tokens, step, train_time = 0.0, 0, 0, 0.0
    next_eval = 0.0
    mix_rng = np.random.default_rng(args.seed + 1)
    while True:
        frac = flops / args.budget
        if frac >= next_eval - 1e-12:
            vl = evaluate(model, val, args, rt, args.eval_seqs)
            log["evals"].append({"frac": frac, "flops": flops, "tokens": tokens, "step": step,
                                 "train_time": train_time, "val_loss": vl})
            print(f"  frac={frac:.3f} step={step} tok={tokens/1e6:.2f}M "
                  f"t={train_time:.0f}s val={vl:.4f}", flush=True)
            next_eval += 1.0 / args.evals
        if frac >= 1.0:
            break
        k, s0, s1 = sched.stage_at(frac)
        if k > 1 and args.mix_fine > 0 and mix_rng.random() < args.mix_fine:
            k = 1
        batch = args.batch * (k if args.batch_scale_k else 1)
        for g in opt.param_groups:
            g["lr"] = lr_at(frac, args.lr, warmup=args.warmup,
                            stage=(s0, s1) if args.lr_restart else None)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            chunk = rt.to(stream.next(batch))
            x, y = chunk[:, :args.seq], chunk[:, k:args.seq + k]
            with rt.autocast():
                loss = fwd(x, y, k=k) / args.accum
            rt.backward(loss, opt, model.parameters(), step=False)
        rt.step(opt, model.parameters())
        loss = loss * args.accum
        if rt.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        train_time += dt
        limiter(dt)
        flops += model.flops_per_step(batch, args.seq, k) * args.accum
        tokens += batch * args.seq * args.accum
        step += 1
        if step % 25 == 0:
            lv = loss.item()
            log["train"].append({"step": step, "frac": frac, "k": k, "loss": lv,
                                 "train_time": train_time})
            if not math.isfinite(lv):
                print("non-finite loss, abort", flush=True)
                break

    log["final_val_loss"] = evaluate(model, val, args, rt, args.final_eval_seqs)
    log["optimizer_state_bytes"] = state_bytes(opt)
    log["param_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
    log["train_time"] = train_time
    log["tokens"] = tokens
    log["steps"] = step
    log["data_epochs"] = stream.epochs + stream.pos / len(stream.starts)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.name}.json"), "w") as f:
        json.dump(log, f, indent=1)
    if args.save:
        torch.save(model.state_dict(), args.save)
    print(f"[{args.name}] final val={log['final_val_loss']:.4f} tokens={tokens/1e6:.1f}M "
          f"steps={step} train_time={train_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
