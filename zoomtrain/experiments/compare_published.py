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
sys.path.insert(0, os.path.join(ROOT, "experiments"))


def load_stories(path, n, clean_json=None):
    texts = [t.strip() for t in pq.read_table(path, columns=["text"]).column("text").to_pylist()]
    if clean_json:
        idx = json.load(open(clean_json))["clean_indices"]
        return [texts[i] for i in idx][:n]
    return [t for t in texts if t][:n]


def l2c_text(s):
    """llama2.c trained on TinyStories_all_data, where paragraphs are split by a single newline."""
    return s.replace("\n\n", "\n")


def windows(ids, bos, ctx, end=None):
    """[bos] + ids (+ end) split so every target is predicted exactly once with <= ctx inputs."""
    seq = [bos] + ids + ([end] if end is not None else [])
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


def eval_ours(spec, stories, threads, end_token=True):
    run_json, weights = spec.split(":")
    log = json.load(open(run_json))
    a = argparse.Namespace(**log["args"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.data, "tokenizer.json"))
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    m = build_model(a, meta["vocab_size"])
    sd = torch.load(weights, map_location="cpu")
    m.load_state_dict({k: v.float() for k, v in sd.items()})
    m.eval()
    eos = meta["eos_id"]
    chunks = [c for s in stories
              for c in windows(tok.encode(s).ids, eos, a.seq, eos if end_token else None)]

    def fwd(x, mask):
        return m.logits(m.hidden(x, 1))

    nll, ntok = score(fwd, chunks)
    params = sum(p.numel() for p in m.parameters())
    # intermediate checkpoints (--save_every_eval) are charged what they had seen at that point
    tr_tokens, tr_flops, tr_secs = log.get("tokens"), log["args"]["budget"], log.get("train_time")
    for e in log.get("evals", []):
        if e.get("checkpoint") and os.path.abspath(e["checkpoint"]) == os.path.abspath(weights):
            tr_tokens, tr_flops, tr_secs = e["tokens"], e["flops"], e["train_time"]
    return {"name": os.path.basename(weights).replace(".pt", ""), "family": "ours",
            "params": params, "params_non_embedding": params - m.emb.weight.numel(),
            "vocab": meta["vocab_size"], "context": a.seq, "nll_nats": nll, "tokens": ntok,
            "train_flops": tr_flops, "train_tokens": tr_tokens,
            "train_seconds_cpu": tr_secs, "run": run_json}


def eval_hf(repo, stories, ctx_cap, tokenizer_repo=None, end_token=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_repo or repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)
    m.eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ctx = min(getattr(m.config, "max_position_embeddings", 2048) or 2048, ctx_cap)
    end = tok.eos_token_id if end_token else None
    chunks = [c for s in stories
              for c in windows(tok(s, add_special_tokens=False)["input_ids"], bos, ctx, end)]

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


def eval_llama2c(pt_path, spm_path, stories, end_token=True):
    import sentencepiece as spm
    from llama2c import load
    m, ck = load(pt_path)
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    bos = sp.bos_id()
    ctx = m.max_seq_len
    # llama2.c packs [BOS] story [BOS] story ..., so the end of a story is predicting the next BOS
    chunks = [c for s in stories
              for c in windows(sp.encode(l2c_text(s)), bos, ctx, bos if end_token else None)]

    def fwd(x, mask):
        return m(x)

    nll, ntok = score(fwd, chunks, batch=16, pad_id=0)
    params = sum(p.numel() for p in m.parameters())
    cfg = ck.get("config", {})
    return {"name": os.path.basename(pt_path).replace(".pt", ""), "family": "published-llama2c",
            "params": params, "params_non_embedding": params - m.tok_embeddings.weight.numel(),
            "vocab": m.tok_embeddings.weight.shape[0], "context": ctx, "nll_nats": nll,
            "tokens": ntok, "iter_num": ck.get("iter_num"),
            "train_tokens": (ck.get("iter_num") or 0) * cfg.get("batch_size", 0)
            * cfg.get("max_seq_len", 0) * cfg.get("gradient_accumulation_steps", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_parquet", default="/home/user/data/tinystories/val.parquet")
    ap.add_argument("--stories", type=int, default=1000)
    ap.add_argument("--ours", nargs="*", default=[], help="run.json:weights.pt")
    ap.add_argument("--hf", nargs="*", default=[], help="HF repo ids (optionally repo@tokenizer_repo)")
    ap.add_argument("--l2c", nargs="*", default=[], help="llama2.c checkpoint.pt:tokenizer.model")
    ap.add_argument("--clean", default="", help="JSON with clean_indices (stories no model saw)")
    ap.add_argument("--no_end", action="store_true", help="do not score the end-of-story token")
    ap.add_argument("--ctx_cap", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "published", "compare.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    stories = load_stories(args.val_parquet, args.stories, args.clean or None)
    end = not args.no_end
    n_bytes = sum(len(s.encode("utf-8")) for s in stories)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    prev = results.get("_eval")
    results["_eval"] = {"stories": len(stories), "bytes": n_bytes, "source": args.val_parquet,
                        "clean_subset": bool(args.clean), "end_of_story_scored": end,
                        "denominator": "UTF-8 bytes of the original V1 validation text"}
    if prev and (prev["stories"], prev["bytes"]) != (len(stories), n_bytes):
        results = {"_eval": results["_eval"]}         # different story set: start fresh
    for spec in args.ours:
        t0 = time.time()
        r = eval_ours(spec, stories, args.threads, end)
        r["eval_seconds"] = time.time() - t0
        results[r["name"]] = r
        print(r["name"], flush=True)
    for spec in args.hf:
        repo, _, tok_repo = spec.partition("@")
        t0 = time.time()
        try:
            r = eval_hf(repo, stories, args.ctx_cap, tok_repo or None, end)
        except Exception as e:  # keep going; report the failure
            print(f"{repo}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        r["eval_seconds"] = time.time() - t0
        results[repo] = r
        print(repo, flush=True)
    for spec in args.l2c:
        pt, _, spm_path = spec.partition(":")
        t0 = time.time()
        r = eval_llama2c(pt, spm_path, stories, end)
        r["eval_seconds"] = time.time() - t0
        results[r["name"]] = r
        print(r["name"], flush=True)
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
