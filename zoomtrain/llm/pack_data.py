"""Deduplicate a pre-tokenized corpus and pack it into TokPack files.

    python llm/pack_data.py --data DIR [--dedup 1] [--threshold 0.8]

Reads DIR/train.bin, DIR/val.bin, DIR/meta.json (from prepare_data.py),
writes DIR/train.tokpack, DIR/val.tokpack, DIR/pack_stats.json. Validation
is never deduplicated against itself; training documents that near-duplicate
a validation document are dropped so the held-out loss is honest.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from zoomtrain.data import dedup, tokpack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dedup", type=int, default=1)
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--num_perm", type=int, default=64)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--level", type=int, default=10)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    eos, vocab = meta["eos_id"], meta["vocab_size"]
    train = np.array(np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r"))
    val = np.array(np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r"))
    stats = {"raw_train_bytes": int(train.nbytes), "raw_val_bytes": int(val.nbytes)}

    if args.dedup:
        t0 = time.time()
        # dedup train together with val, then only drop from train
        both = np.concatenate([val, train])
        keep, st = dedup.dedup(both, eos, args.ngram, args.num_perm, args.bands,
                               args.threshold)
        n_val_docs = int((val == eos).sum())
        keep[:n_val_docs] = True
        n_leak = int((~keep[n_val_docs:]).sum())
        train = dedup.filter_tokens(both, eos, keep)[len(val):]
        st["dedup_seconds"] = time.time() - t0
        st["train_docs_dropped"] = n_leak
        st["train_tokens_after"] = int(len(train))
        stats["dedup"] = st
        print(json.dumps(st, indent=1))

    for name, arr in (("train", train), ("val", val)):
        t0 = time.time()
        nbytes = tokpack.write(os.path.join(args.data, f"{name}.tokpack"), arr, vocab,
                               chunk=args.chunk, level=args.level)
        dt = time.time() - t0
        tp = tokpack.TokPack(os.path.join(args.data, f"{name}.tokpack"))
        t1 = time.time()
        n_read = 0
        for w in tp.stream(256, seed=0):
            n_read += len(w)
            if n_read >= 20_000_000:
                break
        read_dt = time.time() - t1
        assert np.array_equal(tp[1000:5000], arr[1000:5000])
        stats[name] = {"tokens": int(len(arr)), "uint16_bytes": int(arr.nbytes),
                       "tokpack_bytes": int(nbytes),
                       "bits_per_token": 8 * nbytes / len(arr),
                       "ratio_vs_uint16": arr.nbytes / nbytes,
                       "write_seconds": dt, "stream_tokens_per_s": n_read / read_dt}
        print(name, json.dumps(stats[name], indent=1))
    with open(os.path.join(args.data, "pack_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
