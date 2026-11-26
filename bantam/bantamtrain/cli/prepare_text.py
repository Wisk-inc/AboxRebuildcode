"""Turn a text dataset into a tokenizer + pre-tokenized token files.

Storing the corpus once as packed token ids (uint16, 2 bytes/token; uint32 when
the vocabulary exceeds 65536) means the training loop never re-tokenizes and
every batch is a zero-copy slice. Preparation streams: parquet record batches
are tokenized --batch documents at a time straight into the .bin files, so
memory stays O(batch) whatever the corpus size, and the BPE tokenizer is
trained on the first --tok_docs training documents (or --tokenizer reuses an
existing Hugging Face tokenizer). Training documents identical to a validation
document are dropped (meta.json: train_docs_dropped_exact_val) so the held-out
loss is honest; pack-text also removes near-duplicates.

Usage:
    bantam prepare-text --hf roneneldan/TinyStories --out DIR [--max_shards 1]
    bantam prepare-text --train train0.parquet [train1.parquet ...] --val val.parquet --out DIR
    bantam prepare-text ... --tokenizer gpt2     (or a tokenizer.json / model directory)
--hf downloads a dataset repo's parquet shards through huggingface_hub: one
config directory (--config, else the first with train shards), up to
--max_shards training shards and its validation split (validation/valid/dev,
else test). A repo whose shards are not named by split uses all shards of the
config for training with a --val_frac hold-out; a repo without parquet files
falls back to the Hub's refs/convert/parquet branch. --column names the text
column (default "text"). Documents are encoded without BOS, each followed by EOS.

Writes DIR/tokenizer.json, train.bin, val.bin and meta.json (vocab_size, eos_id,
dtype, token and document counts, and a manifest: prep_id plus the size and
blake2b of every file). Re-preparing a directory deletes the .tokpack files and
pack_stats.json of an earlier pack-text run, whose data no longer matches.
"""
import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import uuid
from collections import deque

import numpy as np

EOS = "<|eos|>"
EOS_CANDIDATES = (EOS, "<|endoftext|>", "</s>", "<|end_of_text|>", "<eos>", "<|im_end|>")
STALE = ("train.tokpack", "val.tokpack", "pack_stats.json")


def _pq():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("prepare-text needs pyarrow: pip install 'bantamtrain[data]'")
    return pq


def iter_texts(paths, column, limit=None, batch_rows=1024):
    """Stripped, non-empty texts of `column`, streamed one parquet record batch at a time
    (no pre-buffering or threaded read-ahead, so memory stays at about one row group)."""
    pq = _pq()
    n = 0
    for path in ([paths] if isinstance(paths, (str, os.PathLike)) else paths):
        try:   # pre_buffer would read every row group's column up front
            pf = pq.ParquetFile(path, pre_buffer=False)
        except TypeError:
            pf = pq.ParquetFile(path)
        if column not in pf.schema_arrow.names:
            raise SystemExit(f"{path} has no column {column!r}; columns: {pf.schema_arrow.names}")
        for rb in pf.iter_batches(batch_size=batch_rows, columns=[column], use_threads=False):
            for t in rb.column(0).to_pylist():
                if t and t.strip():
                    yield t.strip()
                    n += 1
                    if limit and n >= limit:
                        return


def read_texts(paths, column, limit=None):
    """All texts in a list (kept for callers that want them in RAM)."""
    return list(iter_texts(paths, column, limit))


def _n_rows(paths):
    pq = _pq()
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)


def _digest(text):
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()


_SPLIT = re.compile(r"(train|validation|valid|dev|test)(?=[-_.\d]|$)")


def _split_of(f):
    """(config_dir, split or None) of a repo file: split from the file name prefix, or
    from its parent directory (the refs/convert/parquet layout CONFIG/SPLIT/0000.parquet)."""
    d, base = os.path.split(f)
    m = _SPLIT.match(base.lower())
    if m:
        return d, m.group(1)
    parent = os.path.basename(d).lower()
    parent = parent[len("partial-"):] if parent.startswith("partial-") else parent
    if parent in ("train", "validation", "valid", "dev", "test"):
        return os.path.dirname(d), parent
    return d, None


def pick_shards(files, max_shards, config=None):
    """(train_files, val_files) of one config of a repo file listing (see the module doc)."""
    files = sorted(f for f in files if f.endswith(".parquet"))
    by_cfg = {}
    for f in files:
        cfg, split = _split_of(f)
        by_cfg.setdefault(cfg, []).append((f, split))
    if config is not None:
        hits = [c for c in by_cfg if c == config or c.split("/")[-1] == config
                or c.startswith(config + "/")]
        if not hits:
            raise SystemExit(f"no parquet files for config {config!r}; configs: {sorted(by_cfg)}")
        cfg = hits[0]
    else:
        with_train = [c for c in by_cfg if any(s == "train" for _, s in by_cfg[c])]
        cfg = (with_train or sorted(by_cfg))[0]
    entries = by_cfg[cfg]
    train = [f for f, s in entries if s == "train"]
    val = []
    for want in (("validation", "valid", "dev"), ("test",)):
        val = [f for f, s in entries if s in want]
        if val:
            break
    if not train:   # shards not named by split: all of them are training data
        train, val = [f for f, s in entries if s is None], []
    return train[:max_shards], val[:1]


def hf_shards(repo, max_shards, cache, config=None):
    """Download a dataset repo's parquet shards; returns (train_paths, val_paths)."""
    from huggingface_hub import HfApi, hf_hub_download
    api, revision = HfApi(), None
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    if not files:   # the Hub keeps a parquet conversion of most datasets on this branch
        revision = "refs/convert/parquet"
        try:
            files = [f for f in api.list_repo_files(repo, repo_type="dataset", revision=revision)
                     if f.endswith(".parquet")]
        except Exception as e:  # noqa: BLE001 - any hub error means "no conversion"
            print(f"{repo}: no {revision} branch ({type(e).__name__})", file=sys.stderr)
    if not files:
        raise SystemExit(f"{repo} has no parquet files (nor a {revision} conversion); download "
                         f"it yourself and pass --train/--val parquet files")
    train, val = pick_shards(files, max_shards, config)
    if not train:
        raise SystemExit(f"no train parquet shards in {repo}: {files[:10]}")
    kw = {"revision": revision} if revision else {}
    paths = [[hf_hub_download(repo, f, repo_type="dataset", cache_dir=cache, **kw) for f in group]
             for group in (train, val)]
    return paths[0], paths[1]


def train_tokenizer(texts, vocab_size):
    """Byte-level BPE with the EOS special token; `texts` may be any iterator."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def _guess_eos(tok, folder):
    """EOS named by a prepared directory's meta.json or a tokenizer_config.json next to the
    tokenizer, else the first known end-of-text token in its vocabulary."""
    for name, key in (("meta.json", "eos_token"), ("tokenizer_config.json", "eos_token")):
        cfg = os.path.join(folder, name) if folder else ""
        if cfg and os.path.exists(cfg):
            e = json.load(open(cfg)).get(key)
            e = e.get("content") if isinstance(e, dict) else e
            if e and tok.token_to_id(e) is not None:
                return e
    return next((c for c in EOS_CANDIDATES if tok.token_to_id(c) is not None), None)


def load_tokenizer(name_or_path, eos=None):
    """(tokenizers.Tokenizer, eos_token) from a tokenizer.json file, or a directory / Hub id
    through transformers.AutoTokenizer (fast tokenizers only) when installed."""
    from tokenizers import Tokenizer
    p, tok, eos_tok = name_or_path, None, None
    if os.path.isfile(p):
        tok, folder = Tokenizer.from_file(p), os.path.dirname(p)
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            AutoTokenizer = None
        if AutoTokenizer is not None:
            hf = AutoTokenizer.from_pretrained(p, use_fast=True)
            if not getattr(hf, "is_fast", False):
                raise SystemExit(f"{p} has no fast (tokenizers) version")
            tok, eos_tok, folder = hf.backend_tokenizer, hf.eos_token, None
        elif os.path.exists(os.path.join(p, "tokenizer.json")):
            tok, folder = Tokenizer.from_file(os.path.join(p, "tokenizer.json")), p
        else:
            tok, folder = Tokenizer.from_pretrained(p), None
    tok.no_truncation()
    tok.no_padding()
    eos_tok = eos or eos_tok or _guess_eos(tok, folder)
    if eos_tok is None or tok.token_to_id(eos_tok) is None:
        raise SystemExit(f"no end-of-text token found in {name_or_path}; pass --eos TOKEN")
    return tok, eos_tok


def vocab_size_of(tok):
    """Embedding rows needed for every id the tokenizer can emit."""
    return max(tok.get_vocab_size(), max(tok.get_vocab().values(), default=-1) + 1)


def _batches(texts, batch, max_chars=1 << 18):
    """Lists of at most `batch` texts and about `max_chars` characters (tokenizers keeps
    ~150 B per token of Encoding objects, so a batch is bounded by size, not only count)."""
    out, chars = [], 0
    for t in texts:
        out.append(t)
        chars += len(t)
        if len(out) >= batch or chars >= max_chars:
            yield out
            out, chars = [], 0
    if out:
        yield out


def _encode(tok, texts, path, batch, dtype, eos):
    """Tokenize `texts` (any iterable) batch by batch, appending ids + EOS to `path`.
    Returns (n_tokens, n_docs, blake2b hex of the file)."""
    from ..data.tokpack import token_dtype
    if dtype is None:
        dtype = token_dtype(tok.get_vocab_size()) if hasattr(tok, "get_vocab_size") else np.uint16
    dtype = np.dtype(dtype)
    hi = np.iinfo(dtype).max
    enc = getattr(tok, "encode_batch_fast", None) or tok.encode_batch
    special = getattr(tok, "encode_special_tokens", None)
    if special is not None:   # a literal "<|eos|>" inside a document is text, not the separator
        tok.encode_special_tokens = True
    h = hashlib.blake2b()
    n = docs = 0
    try:
        with open(path, "wb") as f:
            for chunk in _batches(texts, batch):
                try:
                    out = enc(chunk, add_special_tokens=False)
                except TypeError:   # duck-typed tokenizers without the keyword
                    out = enc(chunk)
                ids = [e.ids for e in out]
                flat = itertools.chain.from_iterable(itertools.chain(x, (eos,)) for x in ids)
                a = np.fromiter(flat, dtype=np.int64, count=sum(len(x) + 1 for x in ids))
                if a.size and (a.min() < 0 or a.max() > hi):
                    raise ValueError(f"token id {int(a.max())} does not fit {dtype.name}; "
                                     f"use dtype=np.uint32")
                b = a.astype(dtype).tobytes()
                f.write(b)
                h.update(b)
                n += a.size
                docs += len(chunk)
    finally:
        if special is not None:
            tok.encode_special_tokens = special
    return n, docs, h.hexdigest()


def encode_to_memmap(tok, texts, path, batch=4096, dtype=None, eos_id=None):
    """Write the token ids (+ EOS after each text) to `path`; returns the token count.
    dtype defaults to uint16, or uint32 when the tokenizer's vocab exceeds 65536."""
    eos = tok.token_to_id(EOS) if eos_id is None else eos_id
    return _encode(tok, texts, path, batch, dtype, eos)[0]


def _file_entry(path):
    h = hashlib.blake2b()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return {"bytes": os.path.getsize(path), "blake2b": h.hexdigest()}


def clear_stale(out):
    """Delete files that described an earlier preparation of `out`; returns their names."""
    gone = []
    for name in sorted(os.listdir(out)):
        if name in STALE + ("meta.json",) or name.endswith((".payload.tmp", ".docs.tmp")) \
                or (name.endswith(".tmp") and ".tokpack" in name):
            os.remove(os.path.join(out, name))
            gone.append(name)
    return gone


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bantam prepare-text", description=__doc__.split("\n")[0])
    ap.add_argument("--hf", default="",
                    help="HuggingFace dataset repo id, e.g. roneneldan/TinyStories")
    ap.add_argument("--config", default=None, help="dataset config (subdirectory) to use with --hf")
    ap.add_argument("--max_shards", type=int, default=1)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/bantamtrain_hf"))
    ap.add_argument("--train", nargs="+", default=[], help="training parquet file(s)")
    ap.add_argument("--val", default="")
    ap.add_argument("--val_frac", type=float, default=0.01,
                    help="held-out fraction when the dataset has no validation split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--column", default="text")
    ap.add_argument("--vocab", type=int, default=4096, help="BPE vocabulary to train")
    ap.add_argument("--tokenizer", default="",
                    help="reuse this tokenizer (tokenizer.json, model dir or Hub id) instead of "
                         "training one")
    ap.add_argument("--eos", default=None,
                    help="end-of-text token of --tokenizer (default: guessed)")
    ap.add_argument("--tok_docs", type=int, default=200_000)
    ap.add_argument("--max_train_docs", type=int, default=None)
    ap.add_argument("--drop_val_overlap", type=int, default=1,
                    help="drop training documents identical to a validation document (1) "
                         "or keep them (0)")
    ap.add_argument("--batch", type=int, default=4096, help="documents per tokenizer batch")
    args = ap.parse_args(argv)
    if not args.hf and not args.train:
        ap.error("give --hf REPO or --train FILE.parquet")

    if args.hf:
        train_paths, val_paths = hf_shards(args.hf, args.max_shards, args.cache, args.config)
    else:
        train_paths, val_paths = args.train, [args.val] if args.val else []
    for p in list(train_paths) + list(val_paths):   # bad file or --column: fail before touching DIR
        next(iter_texts([p], args.column), None)
    limit = args.max_train_docs
    if val_paths:
        n_train_docs = None
        vset = ({_digest(t) for t in iter_texts(val_paths, args.column)}
                if args.drop_val_overlap else set())

        def val_texts():
            return iter_texts(val_paths, args.column)
    else:  # no validation split: hold out the last val_frac of the training docs
        cap = max(1, int(min(_n_rows(train_paths), limit or float("inf")) * args.val_frac))
        tail, total = deque(maxlen=cap), 0
        for t in iter_texts(train_paths, args.column, limit):
            total += 1
            tail.append(_digest(t) if args.drop_val_overlap else None)
        n_val = max(1, int(total * args.val_frac))
        n_train_docs = total - n_val
        if n_train_docs < 1:
            raise SystemExit(f"need at least 2 documents to hold out validation, found {total}")
        vset = set(list(tail)[-n_val:]) if args.drop_val_overlap else set()

        def val_texts():
            return itertools.islice(iter_texts(train_paths, args.column, limit), n_train_docs, None)
    dropped = [0]

    def train_texts():
        it = iter_texts(train_paths, args.column, limit)
        for t in itertools.islice(it, n_train_docs):
            if vset and _digest(t) in vset:
                dropped[0] += 1
                continue
            yield t

    if args.tokenizer:
        tok, eos_tok = load_tokenizer(args.tokenizer, args.eos)
        vocab = vocab_size_of(tok)
    else:
        tok = train_tokenizer(itertools.islice(train_texts(), args.tok_docs), args.vocab)
        eos_tok = EOS
        vocab = tok.get_vocab_size()
    os.makedirs(args.out, exist_ok=True)   # only now: a run that failed on its inputs leaves
    gone = clear_stale(args.out)           # DIR as it was (or absent)
    if gone:
        print(f"removed files of an earlier preparation: {gone}")
    tok.save(os.path.join(args.out, "tokenizer.json"))
    eos_id = tok.token_to_id(eos_tok)
    from ..data.tokpack import token_dtype
    dtype = token_dtype(vocab)

    n_val_tok, val_docs, val_hash = _encode(tok, val_texts(), os.path.join(args.out, "val.bin"),
                                            args.batch, dtype, eos_id)
    dropped[0] = 0
    n_train, train_docs, train_hash = _encode(tok, train_texts(),
                                              os.path.join(args.out, "train.bin"), args.batch,
                                              dtype, eos_id)
    if not train_docs or not val_docs:
        raise SystemExit(f"empty split: {train_docs} training and {val_docs} validation documents")
    files = {"tokenizer.json": _file_entry(os.path.join(args.out, "tokenizer.json"))}
    for name, nt, hx in (("train.bin", n_train, train_hash), ("val.bin", n_val_tok, val_hash)):
        files[name] = {"bytes": nt * dtype.itemsize, "blake2b": hx}
    meta = {"vocab_size": vocab, "eos_id": eos_id,
            "train_tokens": n_train, "val_tokens": n_val_tok,
            "train_docs": train_docs, "val_docs": val_docs,
            "dtype": dtype.name, "tokenizer": args.tokenizer or "bpe", "eos_token": eos_tok,
            "train_docs_dropped_exact_val": dropped[0] if args.drop_val_overlap else None,
            "prep_id": uuid.uuid4().hex, "files": files}
    tmp = os.path.join(args.out, "meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, os.path.join(args.out, "meta.json"))
    print(json.dumps({k: v for k, v in meta.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
