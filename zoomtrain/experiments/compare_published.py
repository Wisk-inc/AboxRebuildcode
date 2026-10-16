"""Score our models and published checkpoints on the same held-out stories.

    python experiments/compare_published.py --stories 1000 \
        --ours results/llm_lowmem/save_muon_base.json:results/llm_lowmem/save_muon_base.pt \
        --hf roneneldan/TinyStories-1M roneneldan/TinyStories-8M ...

Metric: bits per byte (BPB) = total negative log-likelihood of every story's
tokens given a start token, divided by ln 2 and by the story's UTF-8 bytes.
It is tokenizer independent, so a 4k-BPE model, a 50k GPT-Neo model and a
32k Llama model are compared on the same footing. Each model scores the
first N stories of the TinyStories validation split, which none of them
trained on. Stories longer than a model's context are scored in consecutive
windows (each window conditions on its first token).

Writes results/published/compare.json.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

torch.set_flush_denormal(True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from zoomtrain.models import ZoomGPT  # noqa: E402
from zoomtrain.zoom import ZoomSchedule  # noqa: E402


def load_stories(path, n):
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    out = [t.strip() for t in texts if t and t.strip()]
    return out[:n]


def windows(ids, bos, ctx):
    """[bos] + ids split so every id is predicted exactly once with <= ctx inputs."""
    seq = [bos] + ids
    out, s = [], 0
    while s < len(seq) - 1:
        chunk = seq[s:s + ctx + 1]          # ctx inputs, ctx targets
        out.append(chunk)
        s += ctx
    return out


@torch.no_grad()
def score(forward_logits, chunks, batch=32, pad_id=0):
    """Sum of NLL (nats) over all target tokens; chunks are lists of ids (inputs+1)."""
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]))
    total, n_tok = 0.0, 0
    for i in range(0, len(order), batch):
        group = [chunks[j] for j in order[i:i + batch]]
        L = max(len(c) for c in group)
        x = torch.full((len(group), L - 1), pad_id, dtype=torch.long)
        y = torch.full((len(group), L - 1), -100, dtype=torch.long)
        mask = torch.zeros((len(group), L - 1), dtype=torch.long)
        for r, c in enumerate(group):
            t = torch.tensor(c)
            x[r, :len(c) - 1] = t[:-1]
            y[r, :len(c) - 1] = t[1:]
            mask[r, :len(c) - 1] = 1
        logits = forward_logits(x, mask).float()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              ignore_index=-100, reduction="sum")
        total += nll.item()
        n_tok += int((y != -100).sum())
    return total, n_tok


def eval_ours(spec, stories, threads):
    run_json, weights = spec.split(":")
    log = json.load(open(run_json))
    a = argparse.Namespace(**log["args"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.data, "tokenizer.json"))
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    m = ZoomGPT(meta["vocab_size"], a.d, a.layers, a.heads,
                zoom_ks=ZoomSchedule.parse(a.zoom).ks, gated=bool(a.gated),
                out_mode=a.out_mode, rope_pos=a.rope_pos, qk_norm=bool(a.qk_norm))
    sd = torch.load(weights, map_location="cpu")
    m.load_state_dict({k: v.float() for k, v in sd.items()})
    m.eval()
    eos = meta["eos_id"]
    chunks = [c for s in stories for c in windows(tok.encode(s).ids, eos, a.seq)]

    def fwd(x, mask):
        return m.hidden(x, 1) @ m.emb.weight.t()

    nll, ntok = score(fwd, chunks)
    params = sum(p.numel() for p in m.parameters())
    return {"name": os.path.basename(weights).replace(".pt", ""), "family": "ours",
            "params": params, "params_non_embedding": params - m.emb.weight.numel(),
            "vocab": meta["vocab_size"], "context": a.seq, "nll_nats": nll, "tokens": ntok,
            "train_flops": log["args"]["budget"], "train_tokens": log["tokens"],
            "train_seconds_cpu": log["train_time"], "run": run_json}


def eval_hf(repo, stories, ctx_cap, tokenizer_repo=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_repo or repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)
    m.eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ctx = min(getattr(m.config, "max_position_embeddings", 2048) or 2048, ctx_cap)
    chunks = [c for s in stories
              for c in windows(tok(s, add_special_tokens=False)["input_ids"], bos, ctx)]

    def fwd(x, mask):
        return m(input_ids=x, attention_mask=mask).logits

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    nll, ntok = score(fwd, chunks, batch=16, pad_id=pad)
    params = sum(p.numel() for p in m.parameters())
    emb = m.get_input_embeddings().weight.numel()
    out_emb = m.get_output_embeddings()
    tied = out_emb is not None and out_emb.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()
    non_emb = params - emb - (0 if tied or out_emb is None else out_emb.weight.numel())
    return {"name": repo, "family": "published", "params": params,
            "params_non_embedding": non_emb, "vocab": m.config.vocab_size, "context": ctx,
            "nll_nats": nll, "tokens": ntok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_parquet", default="/home/user/data/tinystories/val.parquet")
    ap.add_argument("--stories", type=int, default=1000)
    ap.add_argument("--ours", nargs="*", default=[], help="run.json:weights.pt")
    ap.add_argument("--hf", nargs="*", default=[], help="HF repo ids (optionally repo@tokenizer_repo)")
    ap.add_argument("--ctx_cap", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "published", "compare.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    stories = load_stories(args.val_parquet, args.stories)
    n_bytes = sum(len(s.encode("utf-8")) for s in stories)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    results["_eval"] = {"stories": len(stories), "bytes": n_bytes, "source": args.val_parquet}
    for spec in args.ours:
        t0 = time.time()
        r = eval_ours(spec, stories, args.threads)
        r["eval_seconds"] = time.time() - t0
        results[r["name"]] = r
        print(r["name"], flush=True)
    for spec in args.hf:
        repo, _, tok_repo = spec.partition("@")
        t0 = time.time()
        try:
            r = eval_hf(repo, stories, args.ctx_cap, tok_repo or None)
        except Exception as e:  # keep going; report the failure
            print(f"{repo}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        r["eval_seconds"] = time.time() - t0
        results[repo] = r
        print(repo, flush=True)
    for k, r in results.items():
        if k.startswith("_"):
            continue
        r["bpb"] = r["nll_nats"] / math.log(2) / n_bytes
        r["nats_per_token"] = r["nll_nats"] / r["tokens"]
        r["bytes_per_token"] = n_bytes / r["tokens"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    rows = sorted((r for k, r in results.items() if not k.startswith("_")), key=lambda r: r["bpb"])
    print(f"\n{'model':40s} {'params':>9s} {'non-emb':>9s} {'vocab':>6s} {'BPB':>7s} {'nats/tok':>8s}")
    for r in rows:
        print(f"{r['name']:40s} {r['params']/1e6:8.2f}M {r['params_non_embedding']/1e6:8.2f}M "
              f"{r['vocab']:6d} {r['bpb']:7.4f} {r['nats_per_token']:8.4f}")


if __name__ == "__main__":
    main()
