"""TokPack: lossless, random-access, compressed storage for token streams.

Format v2 (default, b"ZTPK2\\0"):
    1. frequency-rank remap: token id -> rank of its corpus frequency, so the
       common tokens become small integers;
    2. LEB128 varint: ranks < 128 take one byte, < 16384 two, < 2^21 three;
    3. zstd-19 per chunk of `chunk` tokens, all chunks sharing one zstd
       dictionary trained on a sample of the corpus and stored in the file.
    The dictionary is what lets small chunks (fast random access) compress
    like large ones: measured on 18M held-out TinyStories tokens (4096 vocab),
    16K-token chunks reach 5.96 bits/token (0.75 B/token) with a 1 MB
    dictionary vs 7.08 bits/token without one, and decompress in 0.05 ms.

Format v1 (b"ZTPK1\\0", read-only): rank remap -> byte-plane shuffle -> zstd.
    Kept so old files still load. The byte shuffle turned out to *hurt*
    (8.55 bits/token vs 7.86 for plain zstd on uint16), which is why v2
    replaced it.

Layout (little endian)
    magic | u32 header_len | header JSON
    | u64[n_chunks + 1] byte offsets (relative to data start)
    | u16/u32[vocab] rank -> token id table
    | (v2) zstd dictionary bytes
    | chunk payloads
All steps are invertible, so decode(encode(x)) == x exactly.
"""
import json
import struct
from collections import OrderedDict

import numpy as np
import zstandard as zstd

MAGIC_V1 = b"ZTPK1\0"
MAGIC_V2 = b"ZTPK2\0"


# ---------------------------------------------------------------- varint codec
def leb128_encode(a):
    """Non-negative ints < 2^21 -> LEB128 bytes (vectorized)."""
    a = np.asarray(a, dtype=np.uint32)
    assert a.size == 0 or int(a.max()) < (1 << 21)
    lens = 1 + (a >= 128).astype(np.int64) + (a >= 16384).astype(np.int64)
    pos = np.concatenate([[0], np.cumsum(lens)[:-1]])
    out = np.empty(int(lens.sum()), dtype=np.uint8)
    out[pos] = (a & 0x7F) | np.where(lens > 1, 0x80, 0).astype(np.uint32)
    m2 = lens >= 2
    out[pos[m2] + 1] = ((a[m2] >> 7) & 0x7F) | np.where(lens[m2] > 2, 0x80, 0).astype(np.uint32)
    m3 = lens == 3
    out[pos[m3] + 2] = a[m3] >> 14
    return out


def leb128_decode(b, n=None):
    """LEB128 bytes -> uint32 values (vectorized, values < 2^21)."""
    b = np.frombuffer(b, dtype=np.uint8) if isinstance(b, (bytes, bytearray)) else b
    cont = b >= 128
    starts = np.flatnonzero(np.concatenate([[True], ~cont[:-1]]))
    v = (b[starts] & 0x7F).astype(np.uint32)
    has2 = cont[starts]
    i2 = starts[has2] + 1
    v[has2] |= (b[i2] & 0x7F).astype(np.uint32) << 7
    has3 = np.zeros_like(has2)
    has3[has2] = cont[i2]
    v[has3] |= b[starts[has3] + 2].astype(np.uint32) << 14
    if n is not None:
        assert len(v) == n, f"decoded {len(v)} values, expected {n}"
    return v


# ---------------------------------------------------------------- v1 helpers
def _unshuffle_bytes(raw, n):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(2, n)
    return np.ascontiguousarray(b.T).view(np.uint16).reshape(n)


# ---------------------------------------------------------------- writer
def write(path, tokens, vocab, chunk=16384, level=19, dict_size=1 << 20,
          dict_sample_tokens=2_000_000, threads=0):
    """Write a v2 TokPack. Returns the total payload bytes (chunks + dictionary)."""
    tokens = np.asarray(tokens)
    assert tokens.size == 0 or int(tokens.max()) < vocab
    n = len(tokens)
    counts = np.bincount(tokens, minlength=vocab)
    rank_to_tok = np.argsort(-counts, kind="stable").astype(np.uint32)
    tok_to_rank = np.empty(vocab, dtype=np.uint32)
    tok_to_rank[rank_to_tok] = np.arange(vocab, dtype=np.uint32)
    ranks = tok_to_rank[tokens]

    dict_bytes = b""
    # the dictionary must pay for itself: cap it at ~5 % of the varint stream
    approx_varint = int(n * (1 + (counts[rank_to_tok[128:]].sum() / max(n, 1))))
    dict_size = min(dict_size, approx_varint // 20)
    if dict_size >= 4096 and n >= 8 * chunk:
        sample = ranks[:dict_sample_tokens]
        samples = [leb128_encode(sample[s:s + 4096]).tobytes()
                   for s in range(0, len(sample), 4096)]
        try:
            d = zstd.train_dictionary(dict_size, samples, level=level, threads=threads or 1)
            dict_bytes = d.as_bytes()
        except zstd.ZstdError:
            dict_bytes = b""
    zd = zstd.ZstdCompressionDict(dict_bytes) if dict_bytes else None
    cctx = zstd.ZstdCompressor(level=level, dict_data=zd, threads=threads)
    payloads, offsets = [], [0]
    for s in range(0, n, chunk):
        p = cctx.compress(leb128_encode(ranks[s:s + chunk]).tobytes())
        payloads.append(p)
        offsets.append(offsets[-1] + len(p))
    tdtype = "u16" if vocab <= 65536 else "u32"
    header = json.dumps({"version": 2, "n_tokens": n, "vocab": vocab, "chunk": chunk,
                         "n_chunks": len(payloads), "codec": f"zstd-{level}",
                         "dict_bytes": len(dict_bytes), "table_dtype": tdtype,
                         "pipeline": "rank-remap|leb128|zstd+dict"}).encode()
    with open(path, "wb") as f:
        f.write(MAGIC_V2)
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        f.write(np.asarray(offsets, dtype=np.uint64).tobytes())
        f.write(rank_to_tok.astype(np.uint16 if tdtype == "u16" else np.uint32).tobytes())
        f.write(dict_bytes)
        for p in payloads:
            f.write(p)
    return offsets[-1] + len(dict_bytes)


# ---------------------------------------------------------------- reader
class TokPack:
    def __init__(self, path, cache_chunks=64):
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        magic = bytes(self.mm[:6])
        assert magic in (MAGIC_V1, MAGIC_V2), "not a TokPack file"
        self.version = 2 if magic == MAGIC_V2 else 1
        hlen = struct.unpack("<I", bytes(self.mm[6:10]))[0]
        self.h = json.loads(bytes(self.mm[10:10 + hlen]))
        pos = 10 + hlen
        nc = self.h["n_chunks"]
        self.offsets = np.frombuffer(self.mm[pos:pos + 8 * (nc + 1)].tobytes(), dtype=np.uint64)
        pos += 8 * (nc + 1)
        v = self.h["vocab"]
        if self.version == 1:
            self.rank_to_tok = None
            if self.h["remap"]:
                self.rank_to_tok = np.frombuffer(self.mm[pos:pos + 2 * v].tobytes(), dtype=np.uint16)
                pos += 2 * v
            self.dctx = zstd.ZstdDecompressor()
            self.dict_bytes = 0
        else:
            tdt = np.uint16 if self.h["table_dtype"] == "u16" else np.uint32
            w = np.dtype(tdt).itemsize
            self.rank_to_tok = np.frombuffer(self.mm[pos:pos + w * v].tobytes(), dtype=tdt)
            pos += w * v
            nd = self.h["dict_bytes"]
            self.dict_bytes = nd
            zd = zstd.ZstdCompressionDict(self.mm[pos:pos + nd].tobytes()) if nd else None
            pos += nd
            self.dctx = zstd.ZstdDecompressor(dict_data=zd)
        self.data_start = pos
        self.n = self.h["n_tokens"]
        self.chunk = self.h["chunk"]
        self.out_dtype = np.uint16 if v <= 65536 else np.uint32
        self.cache = OrderedDict()
        self.cache_chunks = cache_chunks

    def __len__(self):
        return self.n

    @property
    def compressed_bytes(self):
        return int(self.offsets[-1]) + self.dict_bytes

    def _chunk(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        a = self.data_start + int(self.offsets[i])
        b = self.data_start + int(self.offsets[i + 1])
        n = min(self.chunk, self.n - i * self.chunk)
        raw = self.dctx.decompress(self.mm[a:b].tobytes())
        if self.version == 1:
            arr = _unshuffle_bytes(raw, n)
            if self.rank_to_tok is not None:
                arr = self.rank_to_tok[arr]
        else:
            arr = self.rank_to_tok[leb128_decode(raw, n)].astype(self.out_dtype, copy=False)
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
        per_chunk = max(1, self.chunk // seq)
        for i in order:
            c = self._chunk(int(i))
            pool.extend(c[j:j + seq] for j in range(0, len(c) - seq + 1, seq))
            if len(pool) >= buffer_chunks * per_chunk:
                rng.shuffle(pool)
                half = len(pool) // 2
                yield from pool[half:]
                pool = pool[:half]
        rng.shuffle(pool)
        yield from pool
