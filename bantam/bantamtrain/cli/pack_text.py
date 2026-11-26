"""Deduplicate a pre-tokenized corpus and pack it into TokPack files.

    bantam pack-text --data DIR [--dedup 1] [--threshold 0.8]

Reads DIR/train.bin, DIR/val.bin, DIR/meta.json (from prepare-text), writes
DIR/train.tokpack, DIR/val.tokpack, DIR/pack_stats.json. Validation is never
deduplicated against itself; training documents that duplicate or
near-duplicate a validation document are dropped so the held-out loss is
honest (with --dedup 0, exact copies of validation documents are still dropped
unless --drop_val_overlap 0).

Everything streams from the .bin files (read by slices, never mapped): dedup keeps ~14 B per
document in RAM (MinHash signatures go to a temporary file in DIR) and the
TokPack writer holds one chunk plus the dictionary sample, so corpora much
larger than RAM can be packed. Each pack stores a document index and the
prep_id of the data it was made from, which readers check against meta.json.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from ..data import dedup, tokpack  # noqa: E402


def kept_blocks(tokens, eos, keep):
    """Blocks of the documents of `tokens` with keep[i] set (all tokens if keep is None)."""
    if keep is None:
        yield from tokpack._blocks(tokens)
        return
    d = 0
    for _, blk in dedup.doc_blocks(tokens, eos):
        starts, ends = dedup.doc_bounds(blk, eos)
        k = keep[d:d + len(starts)]
        d += len(starts)
        yield blk if k.all() else blk[np.repeat(k, ends - starts)]


def val_overlap(val, train, eos):
    """Keep mask over train documents: False where the document equals a validation document."""
    vh = np.sort(np.concatenate([np.zeros(0, np.uint64)] + [
        dedup.exact_hashes(b, *dedup.doc_bounds(b, eos)) for _, b in dedup.doc_blocks(val, eos)]))
    keep = []
    for _, blk in dedup.doc_blocks(train, eos):
        h = dedup.exact_hashes(blk, *dedup.doc_bounds(blk, eos))
        keep.append(~np.isin(h, vh))
    return np.concatenate(keep) if keep else np.ones(0, bool)


def check_manifest(data, meta, arrays):
    """SystemExit if a .bin file does not match what meta.json recorded for it."""
    for name, arr in arrays.items():
        key = name.split(".")[0] + "_tokens"
        if key in meta and len(arr) != meta[key]:
            raise SystemExit(f"{data}/{name} has {len(arr)} tokens but meta.json says {meta[key]}: "
                             f"the files come from different prepare-text runs")
        want = meta.get("files", {}).get(name, {}).get("blake2b")
        if want:
            h = hashlib.blake2b()
            for s in range(0, len(arr), 1 << 24):
                h.update(arr[s:s + (1 << 24)].tobytes())
            if h.hexdigest() != want:
                raise SystemExit(f"{data}/{name} changed since prepare-text (blake2b mismatch)")


def pack(path, blocks, vocab, eos, chunk, level, extra):
    """Two streaming passes over blocks() (frequencies, then encoding) into a TokPack."""
    counts = np.zeros(vocab, np.int64)
    for blk in blocks():
        counts += np.bincount(blk.astype(np.int64), minlength=vocab)[:vocab]
    with tokpack.TokPackWriter(path, vocab, counts, chunk=chunk, level=level, eos_id=eos,
                               extra=extra) as w:
        for blk in blocks():
            w.append(blk)
    return w.result


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bantam pack-text", description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--dedup", type=int, default=1)
    ap.add_argument("--drop_val_overlap", type=int, default=1,
                    help="with --dedup 0: still drop training documents identical to a "
                         "validation document")
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--num_perm", type=int, default=64)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--level", type=int, default=19)
    args = ap.parse_args(argv)

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    eos, vocab = meta["eos_id"], meta["vocab_size"]
    dtype = np.dtype(meta.get("dtype", "uint16"))
    train = tokpack.BinFile(os.path.join(args.data, "train.bin"), dtype)
    val = tokpack.BinFile(os.path.join(args.data, "val.bin"), dtype)
    check_manifest(args.data, meta, {"train.bin": train, "val.bin": val})
    stats = {"raw_train_bytes": int(train.nbytes), "raw_val_bytes": int(val.nbytes),
             "dtype": dtype.name, "prep_id": meta.get("prep_id")}

    keep_train = None
    if args.dedup:
        t0 = time.time()
        # dedup train together with val, then only drop from train
        keep, st = dedup.dedup_stream([val, train], eos, args.ngram, args.num_perm, args.bands,
                                      args.threshold, tmpdir=args.data)
        n_val_docs = st["docs_per_source"][0]
        keep[:n_val_docs] = True
        keep_train = keep[n_val_docs:]
        st["dedup_seconds"] = time.time() - t0
        st["train_docs_dropped"] = int((~keep_train).sum())
        stats["dedup"] = st
        print(json.dumps(st, indent=1))
    elif args.drop_val_overlap:
        keep_train = val_overlap(val, train, eos)
        stats["val_overlap"] = {"method": "exact", "train_docs": int(len(keep_train)),
                                "train_docs_dropped": int((~keep_train).sum())}
        print(json.dumps(stats["val_overlap"]))

    for name, arr, keep in (("train", train, keep_train), ("val", val, None)):
        path = os.path.join(args.data, f"{name}.tokpack")
        t0 = time.time()
        source = {"prep_id": meta.get("prep_id"), "split": name, "bin_tokens": int(len(arr)),
                  "dedup": bool(args.dedup and name == "train")}
        nbytes = pack(path, lambda: kept_blocks(arr, eos, keep), vocab, eos, args.chunk,
                      args.level, {"source": source})
        dt = time.time() - t0
        tp = tokpack.TokPack(path)
        n = len(tp)
        first = next(kept_blocks(arr, eos, keep), np.zeros(0, dtype))[:1 << 20]
        if not np.array_equal(tp[:len(first)], first):
            raise RuntimeError(f"{path}: read-back mismatch")
        t1 = time.time()
        n_read = 0
        for w in tp.stream(256, seed=0):
            n_read += len(w)
            if n_read >= 20_000_000:
                break
        read_dt = max(time.time() - t1, 1e-9)
        raw = n * dtype.itemsize                 # the .bin's dtype (uint32 above 65536 ids)
        stats[name] = {"tokens": n, "uint16_bytes": 2 * n, "tokpack_bytes": int(nbytes),
                       "bits_per_token": 8 * nbytes / n if n else 0.0,
                       "ratio_vs_uint16": 2 * n / nbytes if nbytes else 0.0,
                       "raw_bytes": raw, "ratio_vs_raw": raw / nbytes if nbytes else 0.0,
                       "write_seconds": dt, "stream_tokens_per_s": n_read / read_dt,
                       "docs": tp.h["doc_index"]["n_docs"],
                       "doc_index_bytes": tp.h["doc_index"]["bytes"]}
        if name == "train" and "dedup" in stats:
            stats["dedup"]["train_tokens_after"] = n
        print(name, json.dumps(stats[name], indent=1))
    tmp = os.path.join(args.data, "pack_stats.json.tmp")
    with open(tmp, "w") as f:
        json.dump(stats, f, indent=1)
    os.replace(tmp, os.path.join(args.data, "pack_stats.json"))


if __name__ == "__main__":
    main()
