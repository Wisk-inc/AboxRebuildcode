"""TokPack: lossless, random-access, compressed storage for token streams.

Layout (little endian)
    b"ZTPK1\\0" | u32 header_len | header JSON
    | u64[n_chunks + 1] byte offsets (relative to data start)
    | u16[vocab] rank -> token id table (only if remapped)
    | chunk payloads

Each chunk of `chunk` tokens is encoded as
    1. frequency-rank remap: token id -> rank of its corpus frequency, so the
       common tokens become small integers and the high byte is mostly 0;
    2. byte-plane split (all low bytes, then all high bytes), which puts
       those zeros next to each other;
    3. zstd.
All three steps are invertible, so decode(encode(x)) == x exactly.

Random access decodes only the chunks that overlap the request (LRU-cached);
`stream()` walks chunks in a shuffled order with a shuffle buffer so every
chunk is decompressed once per epoch.
"""
import json
import struct
from collections import OrderedDict

import numpy as np
import zstandard as zstd

MAGIC = b"ZTPK1\0"


def _shuffle_bytes(a16):
    b = a16.view(np.uint8).reshape(-1, 2)
    return np.ascontiguousarray(b.T).tobytes()


def _unshuffle_bytes(raw, n):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(2, n)
    return np.ascontiguousarray(b.T).view(np.uint16).reshape(n)


def write(path, tokens, vocab, chunk=65536, level=10, remap=True, threads=0):
    tokens = np.asarray(tokens, dtype=np.uint16)
    n = len(tokens)
    if remap:
        counts = np.bincount(tokens, minlength=vocab)
        rank_to_tok = np.argsort(-counts, kind="stable").astype(np.uint16)
        tok_to_rank = np.empty(vocab, dtype=np.uint16)
        tok_to_rank[rank_to_tok] = np.arange(vocab, dtype=np.uint16)
    cctx = zstd.ZstdCompressor(level=level, threads=threads)
    payloads, offsets = [], [0]
    for s in range(0, n, chunk):
        c = tokens[s:s + chunk]
        if remap:
            c = tok_to_rank[c]
        p = cctx.compress(_shuffle_bytes(c))
        payloads.append(p)
        offsets.append(offsets[-1] + len(p))
    header = json.dumps({"n_tokens": n, "vocab": vocab, "chunk": chunk, "remap": remap,
                         "n_chunks": len(payloads), "codec": f"zstd-{level}",
                         "pipeline": "rank-remap|byte-shuffle|zstd"}).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        f.write(np.asarray(offsets, dtype=np.uint64).tobytes())
        if remap:
            f.write(rank_to_tok.tobytes())
        for p in payloads:
            f.write(p)
    return offsets[-1]


class TokPack:
    def __init__(self, path, cache_chunks=64):
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        assert bytes(self.mm[:6]) == MAGIC, "not a TokPack file"
        hlen = struct.unpack("<I", bytes(self.mm[6:10]))[0]
        self.h = json.loads(bytes(self.mm[10:10 + hlen]))
        pos = 10 + hlen
        nc = self.h["n_chunks"]
        self.offsets = np.frombuffer(self.mm[pos:pos + 8 * (nc + 1)].tobytes(), dtype=np.uint64)
        pos += 8 * (nc + 1)
        self.rank_to_tok = None
        if self.h["remap"]:
            v = self.h["vocab"]
            self.rank_to_tok = np.frombuffer(self.mm[pos:pos + 2 * v].tobytes(), dtype=np.uint16)
            pos += 2 * v
        self.data_start = pos
        self.n = self.h["n_tokens"]
        self.chunk = self.h["chunk"]
        self.dctx = zstd.ZstdDecompressor()
        self.cache = OrderedDict()
        self.cache_chunks = cache_chunks

    def __len__(self):
        return self.n

    @property
    def compressed_bytes(self):
        return int(self.offsets[-1])

    def _chunk(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        a = self.data_start + int(self.offsets[i])
        b = self.data_start + int(self.offsets[i + 1])
        n = min(self.chunk, self.n - i * self.chunk)
        arr = _unshuffle_bytes(self.dctx.decompress(self.mm[a:b].tobytes()), n)
        if self.rank_to_tok is not None:
            arr = self.rank_to_tok[arr]
        self.cache[i] = arr
        if len(self.cache) > self.cache_chunks:
            self.cache.popitem(last=False)
        return arr

    def read(self, start, length):
        end = start + length
        assert 0 <= start and end <= self.n
        c0, c1 = start // self.chunk, (end - 1) // self.chunk
        parts = [self._chunk(i) for i in range(c0, c1 + 1)]
        joined = parts[0] if len(parts) == 1 else np.concatenate(parts)
        off = start - c0 * self.chunk
        return joined[off:off + length]

    def __getitem__(self, sl):
        start, stop, step = sl.indices(self.n)
        assert step == 1
        return self.read(start, stop - start)

    def stream(self, seq, seed=0, buffer_chunks=32):
        """Yield length-`seq` windows in a shuffled order, one decompression per chunk."""
        rng = np.random.default_rng(seed)
        order = rng.permutation(self.h["n_chunks"])
        pool = []
        for i in order:
            c = self._chunk(int(i))
            pool.extend(c[j:j + seq] for j in range(0, len(c) - seq + 1, seq))
            if len(pool) >= buffer_chunks * (self.chunk // seq):
                rng.shuffle(pool)
                half = len(pool) // 2
                yield from pool[half:]
                pool = pool[:half]
        rng.shuffle(pool)
        yield from pool
