"""Turn a HuggingFace text dataset into a compact pre-tokenized uint16 memmap.

Storing the corpus once as packed uint16 token ids (2 bytes/token) means the
training loop never re-tokenizes and every batch is a zero-copy slice. That is
the "hold the data efficiently" half of the recipe; nothing is filtered out.

Usage:
    python prepare_data.py --train train0.parquet --val val.parquet --out DIR
Any HF dataset with a text column works: export it to parquet, or point
--train/--val at the parquet files under the dataset repo's data/ folder.
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|eos|>"


def read_texts(path, column, limit=None):
    table = pq.read_table(path, columns=[column])
    texts = table.column(column).to_pylist()
    texts = [t.strip() for t in texts if t and t.strip()]
    return texts[:limit] if limit else texts


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
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--column", default="text")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tok_docs", type=int, default=200_000)
    ap.add_argument("--max_train_docs", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train = read_texts(args.train, args.column, args.max_train_docs)
    val = read_texts(args.val, args.column)
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
