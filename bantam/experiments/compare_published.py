"""Score our models and published checkpoints on the same held-out stories.

    python experiments/compare_published.py --stories 1000 \
        --ours results/llm_lowmem/save_muon_base.json:results/llm_lowmem/save_muon_base.pt \
        --hf roneneldan/TinyStories-1M roneneldan/TinyStories-8M ... \
        --l2c stories15M.pt:tokenizer.model --out results/published/compare_clean1000.json

Metric: bits per byte (BPB) = total negative log-likelihood of every story's
tokens plus its end-of-story event, divided by ln 2 and by the story's UTF-8
bytes. It is tokenizer independent, so a 4k-BPE model, a 50k GPT-Neo model and a
32k Llama model are compared on the same footing.

Stories: by default the first N of the 4,194 TinyStories V1 validation stories
found in neither V1 train nor llama2.c's training shards
(experiments/clean_val_indices.json; the split is pinned by revision and text
hash and downloaded with huggingface_hub when --val_parquet is not given).
--all_val scores the first N validation stories instead: 30-69 % of those were in
the published models' training data.

Story boundaries: each family is scored in the format it was trained on, so no
model pays for tokens the others get for free. Boundary tokens before the story
are context only; the end-of-story event is scored (unless --no_end):
  ours       [eos] story [eos]
  llama2.c   [BOS] story [BOS]
  roneneldan "<|endoftext|>\\n" story "\\n<|endoftext|>" (the V1 .txt layout); other
             HF models [eos] story [eos]. Override with --hf repo#eos_nl|eos|none.
Stories longer than a model's context are scored in consecutive windows (each
window conditions on its first token). Every row also gives the NLL of the
end-of-story tokens alone (nll_end_nats) and bpb_no_end, the BPB without them.
--hf repo=local_dir scores a local copy of a hub repo under the repo's name (the
row records the weight files' sha256, comparable with the hub's LFS ids).

Every row records the scoring signature (story set hash, end-of-story policy,
context cap, boundary policy and --hf format overrides); a results file never
mixes conventions: re-scoring into an --out written with another convention
needs --overwrite.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

torch.set_flush_denormal(True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from bantamtrain.cli.pretrain import build_model  # noqa: E402

CLEAN = os.path.join(ROOT, "experiments", "clean_val_indices.json")
FORMATS = ("eos_nl", "eos", "none")
SIGNATURE_VERSION = 3                  # 3: --hf #format overrides are part of the convention


def text_sha256(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def fetch_val_parquet(clean_json=CLEAN, cache=None):
    """The pinned V1 validation parquet (huggingface_hub cache shared with prepare-text)."""
    src = json.load(open(clean_json))["source"]
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(src["repo"], src["filename"], repo_type=src["repo_type"],
                               revision=src["revision"], cache_dir=cache)
    except Exception as e:  # noqa: BLE001 - offline, missing package, ...
        raise SystemExit(f"could not fetch {src['repo']}/{src['filename']} at {src['revision'][:10]} "
                         f"({type(e).__name__}: {e}); pass --val_parquet")


def load_stories(path, n, clean_json=None):
    """First n clean stories (clean_json indexes the V1 validation split, whose text hash
    is checked), or the first n non-empty validation stories when clean_json is None."""
    import pyarrow.parquet as pq
    texts = [t.strip() for t in pq.read_table(path, columns=["text"]).column("text").to_pylist()]
    if clean_json:
        meta = json.load(open(clean_json))
        src = meta.get("source", {})
        if "text_sha256" in src and (len(texts) != src["n_texts"] or text_sha256(texts) != src["text_sha256"]):
            raise ValueError(f"{path} is not the validation split {os.path.basename(clean_json)} "
                             f"indexes ({src.get('repo')}@{str(src.get('revision'))[:10]}): its "
                             f"positional indices would select other stories")
        return [texts[i] for i in meta["clean_indices"]][:n]
    return [t for t in texts if t][:n]


def l2c_text(s):
    """llama2.c trained on TinyStories_all_data, where paragraphs are split by a single newline.
    Not invertible: single and double newlines merge (l2c_lossy counts the stories affected)."""
    return s.replace("\n\n", "\n")


def l2c_lossy(stories):
    return sum(1 for s in stories if l2c_text(s).replace("\n", "\n\n") != s)


def boundary_windows(ids, prefix, ctx, end=()):
    """prefix + ids + end split into chunks of <= ctx + 1 tokens (ctx inputs, ctx targets) so
    every target is predicted exactly once. Returns (chunk, n_skip) pairs: the first n_skip
    targets of a chunk are prefix tokens, which condition the story but are not scored."""
    seq = list(prefix) + list(ids) + list(end)
    out, s = [], 0
    while s < len(seq) - 1:
        out.append((seq[s:s + ctx + 1], max(0, len(prefix) - 1 - s)))
        s += ctx
    return out


def story_windows(ids, prefix, ctx, end=()):
    """boundary_windows as (chunk, n_skip, n_end) triples: the last n_end targets of a chunk are
    end-of-story tokens (a multi-token end can straddle two chunks), so their NLL can be
    reported apart from the story's."""
    first_end = len(prefix) + len(ids)                 # seq index of the first end token
    out = []
    for i, (c, skip) in enumerate(boundary_windows(ids, prefix, ctx, end)):
        s = i * ctx                                    # c's targets are seq[s+1 : s+len(c)]
        out.append((c, skip, max(0, s + len(c) - max(first_end, s + 1))))
    return out


def windows(ids, bos, ctx, end=None):
    """[bos] + ids (+ end) split so every target is predicted exactly once with <= ctx inputs."""
    return [c for c, _ in boundary_windows(ids, [bos], ctx, [] if end is None else [end])]


@torch.no_grad()
def score_split(forward_logits, chunks, batch=32, pad_id=0):
    """(nll, n_tok, nll_end, n_end): NLL (nats) summed over all scored targets, and the part
    of it on end-of-story targets. chunks are id lists (inputs+1), (ids, n_skip) pairs from
    boundary_windows or (ids, n_skip, n_end) triples from story_windows."""
    chunks = [(c, 0, 0) if not isinstance(c, tuple) else tuple(c) + (0,) * (3 - len(c))
              for c in chunks]
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i][0]))
    total, n_tok, end_total, n_end = 0.0, 0, 0.0, 0
    for i in range(0, len(order), batch):
        group = [chunks[j] for j in order[i:i + batch]]
        L = max(len(c) for c, _, _ in group)
        x = torch.full((len(group), L - 1), pad_id, dtype=torch.long)
        y = torch.full((len(group), L - 1), -100, dtype=torch.long)
        mask = torch.zeros((len(group), L - 1), dtype=torch.long)
        is_end = torch.zeros((len(group), L - 1), dtype=torch.bool)
        for r, (c, skip, ne) in enumerate(group):
            t = torch.tensor(c)
            x[r, :len(c) - 1] = t[:-1]
            y[r, skip:len(c) - 1] = t[1 + skip:]
            mask[r, :len(c) - 1] = 1
            if ne:
                is_end[r, len(c) - 1 - ne:len(c) - 1] = True
        logits = forward_logits(x, mask).float()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              ignore_index=-100, reduction="sum")
        total += nll.item()
        n_tok += int((y != -100).sum())
        if is_end.any():
            end_total += F.cross_entropy(logits[is_end], y[is_end], reduction="sum").item()
            n_end += int(is_end.sum())
    return total, n_tok, end_total, n_end


def score(forward_logits, chunks, batch=32, pad_id=0):
    """Sum of NLL (nats) over all scored targets and their count (see score_split)."""
    return score_split(forward_logits, chunks, batch, pad_id)[:2]


def _nll_fields(parts):
    nll, ntok, nll_end, n_end = parts
    return {"nll_nats": nll, "tokens": ntok, "nll_end_nats": nll_end, "end_tokens": n_end}


def _resolve(path, run_json):
    """A path a run stored relative to the cwd it trained in (normally ROOT)."""
    if os.path.isabs(path):
        return path
    bases = [ROOT, os.getcwd()]
    d = os.path.dirname(os.path.abspath(run_json))
    while d != os.path.dirname(d):
        bases.append(d)
        d = os.path.dirname(d)
    for b in bases:
        if os.path.exists(os.path.join(b, path)):
            return os.path.join(b, path)
    return os.path.join(ROOT, path)


def _same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


def train_cost(log, weights, run_json):
    """(tokens, FLOPs, seconds) a checkpoint had trained for: the matching --save_every_eval
    eval for intermediate checkpoints, else the whole run. An in-progress run whose evals
    do not name these weights raises instead of charging an unknown amount."""
    for e in log.get("evals", []):
        if e.get("checkpoint") and _same_file(_resolve(e["checkpoint"], run_json), weights):
            return e["tokens"], e["flops"], e["train_time"]
    if log.get("in_progress"):
        raise ValueError(f"{weights}: no eval of the in-progress run {run_json} names this checkpoint")
    return log.get("tokens"), log["args"]["budget"], log.get("train_time")


def _state_dict(weights):
    path = os.path.join(weights, "model.safetensors") if os.path.isdir(weights) else weights
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def eval_ours(spec, stories, threads, end_token=True):
    run_json, weights = spec.split(":")
    log = json.load(open(run_json))
    a = argparse.Namespace(**log["args"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.data, "tokenizer.json"))
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    m = build_model(a, meta["vocab_size"])
    sd = _state_dict(weights)
    m.load_state_dict({k: v.float() for k, v in sd.items()})
    m.eval()
    eos = meta["eos_id"]
    chunks = [c for s in stories
              for c in story_windows(tok.encode(s).ids, [eos], a.seq, [eos] if end_token else [])]

    def fwd(x, mask):
        return m.logits(m.hidden(x, 1))

    parts = score_split(fwd, chunks)
    params = sum(p.numel() for p in m.parameters())
    tr_tokens, tr_flops, tr_secs = train_cost(log, weights, run_json)
    return {"name": os.path.basename(weights.rstrip("/")).replace(".pt", ""), "family": "ours",
            "params": params, "params_non_embedding": params - m.emb.weight.numel(),
            "vocab": meta["vocab_size"], "context": a.seq, **_nll_fields(parts),
            "boundary_format": "eos", "train_flops": tr_flops, "train_tokens": tr_tokens,
            "train_seconds_cpu": tr_secs, "run": run_json}


def default_format(repo):
    return "eos_nl" if repo.startswith("roneneldan/") else "eos"


def parse_hf_spec(spec):
    """repo[@tokenizer_repo][#format][=local_dir] -> (repo, tokenizer_repo or None, format);
    split_local() removes the =local_dir part."""
    spec = split_local(spec)[0]
    spec, _, fmt = spec.partition("#")
    repo, _, tok_repo = spec.partition("@")
    fmt = fmt or default_format(repo)
    if fmt not in FORMATS:
        raise SystemExit(f"--hf {repo}: unknown boundary format {fmt!r} (use {'|'.join(FORMATS)})")
    return repo, tok_repo or None, fmt


def split_local(spec):
    """'spec=local_dir' -> (spec, local_dir): score a local copy of a hub model (same files)
    under its repo name, so published_compute.json and the family defaults still apply."""
    spec, _, local = spec.partition("=")
    return spec, local or None


def _weights_sha256(d):
    """{file: sha256} of the weight files in a local model dir (checkable against the hub's LFS ids)."""
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith((".bin", ".safetensors")):
            h = hashlib.sha256()
            with open(os.path.join(d, f), "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            out[f] = h.hexdigest()
    return out


def hf_boundary(fmt, tok, end_token=True):
    """(prefix, end) token ids of a boundary format."""
    eos = tok.eos_token_id
    start = tok.bos_token_id if tok.bos_token_id is not None else eos
    if fmt == "eos_nl":
        nl = tok("\n", add_special_tokens=False)["input_ids"]
        prefix, end = [eos] + nl, nl + [eos]
    elif fmt == "eos":
        prefix, end = [start], [eos]
    else:                                   # "none": the model never learned a separator
        prefix, end = [start], []
    return prefix, (end if end_token else [])


def eval_hf(repo, stories, ctx_cap, tokenizer_repo=None, end_token=True, fmt=None, local_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    fmt = fmt or parse_hf_spec(repo)[2]
    src = local_dir or repo
    tok = AutoTokenizer.from_pretrained(tokenizer_repo or src)
    m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)
    m.eval()
    ctx = min(getattr(m.config, "max_position_embeddings", 2048) or 2048, ctx_cap)
    prefix, end = hf_boundary(fmt, tok, end_token)
    chunks = [c for s in stories
              for c in story_windows(tok(s, add_special_tokens=False)["input_ids"], prefix, ctx, end)]

    def fwd(x, mask):
        return m(input_ids=x, attention_mask=mask).logits

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    parts = score_split(fwd, chunks, batch=16, pad_id=pad)
    params = sum(p.numel() for p in m.parameters())
    emb = m.get_input_embeddings().weight.numel()
    out_emb = m.get_output_embeddings()
    tied = out_emb is not None and out_emb.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()
    non_emb = params - emb - (0 if tied or out_emb is None else out_emb.weight.numel())
    row = {"name": repo, "family": "published", "params": params,
           "params_non_embedding": non_emb, "vocab": m.config.vocab_size, "context": ctx,
           **_nll_fields(parts), "boundary_format": fmt}
    if local_dir:
        row.update(weights=os.path.abspath(local_dir), weights_sha256=_weights_sha256(local_dir))
    return row


def eval_llama2c(pt_path, spm_path, stories, end_token=True):
    import sentencepiece as spm
    from llama2c import load
    m, ck = load(pt_path)
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    bos = sp.bos_id()
    ctx = m.max_seq_len
    # llama2.c packs [BOS] story [BOS] story ..., so the end of a story is predicting the next BOS
    chunks = [c for s in stories
              for c in story_windows(sp.encode(l2c_text(s)), [bos], ctx, [bos] if end_token else [])]

    def fwd(x, mask):
        return m(x)

    parts = score_split(fwd, chunks, batch=16, pad_id=0)
    params = sum(p.numel() for p in m.parameters())
    cfg = ck.get("config", {})
    return {"name": os.path.basename(pt_path).replace(".pt", ""), "family": "published-llama2c",
            "params": params, "params_non_embedding": params - m.tok_embeddings.weight.numel(),
            "vocab": m.tok_embeddings.weight.shape[0], "context": ctx, **_nll_fields(parts),
            "boundary_format": "bos", "iter_num": ck.get("iter_num"),
            "train_tokens": (ck.get("iter_num") or 0) * cfg.get("batch_size", 0)
            * cfg.get("max_seq_len", 0) * cfg.get("gradient_accumulation_steps", 0)}


def boundary_overrides(hf_specs):
    """'repo#format' for every --hf spec scored in another than its family's native format."""
    out = set()
    for spec in hf_specs:
        repo, _, fmt = parse_hf_spec(spec)
        if fmt != default_format(repo):
            out.add(f"{repo}#{fmt}")
    return sorted(out)


def signature(stories, n_bytes, clean, end, ctx_cap, overrides=()):
    """Everything that changes what a BPB number means; stored in _eval, its id in every row.
    A run with --hf repo#format overrides is a different convention from a native run."""
    return {"version": SIGNATURE_VERSION, "stories": len(stories), "bytes": n_bytes,
            "text_sha256": text_sha256(stories), "clean_subset": bool(clean),
            "end_of_story_scored": end, "ctx_cap": ctx_cap,
            "boundary_policy": "native: each family in its training format, prefix unscored",
            "boundary_overrides": list(overrides)}


def sig_id(sig):
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_parquet", default=None,
                    help="V1 validation parquet (default: download the pinned revision)")
    ap.add_argument("--hf_cache", default=os.path.expanduser("~/.cache/bantamtrain_hf"))
    ap.add_argument("--stories", type=int, default=1000)
    ap.add_argument("--ours", nargs="*", default=[], help="run.json:weights.pt")
    ap.add_argument("--hf", nargs="*", default=[],
                    help="HF repo ids: repo[@tokenizer_repo][#eos_nl|eos|none][=local_dir] "
                         "(=local_dir: a local copy of the repo, scored under the repo's name)")
    ap.add_argument("--l2c", nargs="*", default=[], help="llama2.c checkpoint.pt:tokenizer.model")
    ap.add_argument("--clean", default=CLEAN, help="JSON with clean_indices (stories no model saw)")
    ap.add_argument("--all_val", action="store_true",
                    help="score the first N validation stories (contaminated) instead of the clean set")
    ap.add_argument("--no_end", action="store_true", help="do not score the end-of-story event")
    ap.add_argument("--ctx_cap", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true",
                    help="discard an --out file scored with another convention")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "published", "compare.json"))
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    clean = None if args.all_val else (args.clean or CLEAN)
    parquet = args.val_parquet or fetch_val_parquet(args.clean or CLEAN, args.hf_cache)
    stories = load_stories(parquet, args.stories, clean)
    end = not args.no_end
    n_bytes = sum(len(s.encode("utf-8")) for s in stories)
    sig = signature(stories, n_bytes, clean, end, args.ctx_cap, boundary_overrides(args.hf))
    sid = sig_id(sig)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    rows = [k for k in results if not k.startswith("_")]
    if rows and results.get("_eval", {}).get("signature") != sig:
        if not args.overwrite:
            raise SystemExit(f"{args.out} holds {len(rows)} rows scored with another (or an "
                             f"unrecorded) convention; use --overwrite or another --out")
        results = {}
    results["_eval"] = {"stories": len(stories), "bytes": n_bytes, "source": parquet,
                        "clean_subset": bool(clean), "end_of_story_scored": end,
                        "denominator": "UTF-8 bytes of the original V1 validation text",
                        "l2c_lossy_stories": l2c_lossy(stories),
                        "signature": sig, "signature_id": sid}

    def add(r, t0):
        r["eval_seconds"] = time.time() - t0
        r["signature_id"] = sid
        results[r["name"]] = r
        print(r["name"], flush=True)

    for spec in args.ours:
        t0 = time.time()
        add(eval_ours(spec, stories, args.threads, end), t0)
    for spec in args.hf:
        repo, tok_repo, fmt = parse_hf_spec(spec)
        t0 = time.time()
        try:
            r = eval_hf(repo, stories, args.ctx_cap, tok_repo, end, fmt, split_local(spec)[1])
        except Exception as e:  # keep going; report the failure
            print(f"{repo}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        add(r, t0)
    for spec in args.l2c:
        pt, _, spm_path = spec.partition(":")
        t0 = time.time()
        add(eval_llama2c(pt, spm_path, stories, end), t0)
    for k, r in results.items():
        if k.startswith("_"):
            continue
        r["bpb"] = r["nll_nats"] / math.log(2) / n_bytes
        r["nats_per_token"] = r["nll_nats"] / r["tokens"]
        r["bytes_per_token"] = n_bytes / r["tokens"]
        if "nll_end_nats" in r:           # the same scores with the end-of-story event left out
            r["bpb_no_end"] = (r["nll_nats"] - r["nll_end_nats"]) / math.log(2) / n_bytes
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    rows = sorted((r for k, r in results.items() if not k.startswith("_")), key=lambda r: r["bpb"])
    print(f"\n{'model':40s} {'params':>9s} {'non-emb':>9s} {'vocab':>6s} {'BPB':>7s} {'no end':>7s} "
          f"{'nats/tok':>8s}")
    for r in rows:
        no_end = f"{r['bpb_no_end']:7.4f}" if "bpb_no_end" in r else f"{'':7s}"
        print(f"{r['name']:40s} {r['params']/1e6:8.2f}M {r['params_non_embedding']/1e6:8.2f}M "
              f"{r['vocab']:6d} {r['bpb']:7.4f} {no_end} {r['nats_per_token']:8.4f}")
    return 0


if __name__ == "__main__":
    main()
