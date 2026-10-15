"""Turn a HuggingFace text dataset into a compact pre-tokenized uint16 memmap.

Storing the corpus once as packed uint16 token ids (2 bytes/token) means the
training loop never re-tokenizes and every batch is a zero-copy slice. That is
the "hold the data efficiently" half of the recipe; nothing is filtered out.

Usage:
    python prepare_data.py --hf roneneldan/TinyStories --out DIR [--max_shards 1]
    python prepare_data.py --train train0.parquet --val val.parquet --out DIR
Any HF dataset with a text column works: --hf downloads its parquet shards
(train + validation/test splits, up to --max_shards training shards) through
huggingface_hub; --column names the text column (default "text").
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|eos|>"


def read_texts(paths, column, limit=None):
    texts = []
    for path in ([paths] if isinstance(paths, str) else paths):
        table = pq.read_table(path, columns=[column])
        texts.extend(t.strip() for t in table.column(column).to_pylist() if t and t.strip())
        if limit and len(texts) >= limit:
            break
    return texts[:limit] if limit else texts


def hf_shards(repo, max_shards, cache):
    """Download a dataset repo's parquet shards; returns (train_paths, val_paths)."""
    from huggingface_hub import HfApi, hf_hub_download
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if f.endswith(".parquet")]
    def pick(keys):
        return sorted(f for f in files if any(k in os.path.basename(f).lower() for k in keys))
    train = pick(["train"])[:max_shards]
    val = pick(["valid", "test"])[:1]
    if not train:
        raise SystemExit(f"no train parquet shards in {repo}: {files[:10]}")
    paths = [[hf_hub_download(repo, f, repo_type="dataset", cache_dir=cache) for f in group]
             for group in (train, val)]
    return paths[0], paths[1]


def train_tokenizer(texts, vocab_size):
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


def encode_to_memmap(tok, texts, path, batch=10_000):
    eos = tok.token_to_id(EOS)
    chunks, n = [], 0
    for i in range(0, len(texts), batch):
        for enc in tok.encode_batch(texts[i:i + batch]):
            ids = enc.ids + [eos]
            chunks.append(np.asarray(ids, dtype=np.uint16))
            n += len(ids)
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(n,))
    pos = 0
    for c in chunks:
        arr[pos:pos + len(c)] = c
        pos += len(c)
    arr.flush()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="", help="HuggingFace dataset repo id, e.g. roneneldan/TinyStories")
    ap.add_argument("--max_shards", type=int, default=1)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/zoomtrain_hf"))
    ap.add_argument("--train", default="")
    ap.add_argument("--val", default="")
    ap.add_argument("--val_frac", type=float, default=0.01,
                    help="held-out fraction when the dataset has no validation split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--column", default="text")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tok_docs", type=int, default=200_000)
    ap.add_argument("--max_train_docs", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.hf:
        train_paths, val_paths = hf_shards(args.hf, args.max_shards, args.cache)
    else:
        train_paths, val_paths = [args.train], [args.val] if args.val else []
    train = read_texts(train_paths, args.column, args.max_train_docs)
    if val_paths:
        val = read_texts(val_paths, args.column)
    else:  # no validation split: hold out the last val_frac of the training docs
        n_val = max(1, int(len(train) * args.val_frac))
        train, val = train[:-n_val], train[-n_val:]
    tok = train_tokenizer(train[:args.tok_docs], args.vocab)
    tok.save(os.path.join(args.out, "tokenizer.json"))

    n_train = encode_to_memmap(tok, train, os.path.join(args.out, "train.bin"))
    n_val = encode_to_memmap(tok, val, os.path.join(args.out, "val.bin"))
    meta = {"vocab_size": tok.get_vocab_size(), "eos_id": tok.token_to_id(EOS),
            "train_tokens": n_train, "val_tokens": n_val,
            "train_docs": len(train), "val_docs": len(val)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
