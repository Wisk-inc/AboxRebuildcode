"""GridPack: error-bounded, compressed storage for gridded weather fields.

Why float16 is the wrong tool: float16 has 10 mantissa bits, so at
z500 ~ 5.4e4 m^2/s^2 its step is 32 m^2/s^2 (the size of a 6 h forecast
error), and surface pressure ~ 1e5 Pa cannot be represented at all above
65504. bfloat16 keeps the range but only 7 mantissa bits.

GridPack instead stores each (channel, time) slab as linear int16 quantization
around the channel's per-step mean with a per-slab scale chosen so the
quantization error is at most `err_frac` of the channel's 6-hour change
standard deviation (an error the model cannot resolve anyway). Residuals
are then byte-shuffled and zstd compressed per chunk of `chunk` time steps.
Inside a chunk the quantized integers are stored as temporal deltas
(q[t] - q[t-1], exact in int16): consecutive 6 h states differ by ~diff_std,
i.e. ~1/(2*err_frac) quantization steps, so the deltas are small integers
with low entropy regardless of the field's absolute magnitude. Chunk
boundaries restart from an absolute slab, so random access decodes one chunk.

Every value is reconstructed with |x_hat - x| <= scale/2 <= err_frac * diff_std.

Layout
    b"ZGPK1\\0" | u32 hlen | header JSON (shape, dtype, chunk, scales, means)
    | u64[n_chunks + 1] offsets | chunk payloads (int16 residual, byte-shuffled, zstd)
"""
import json
import struct

import numpy as np
import zstandard as zstd

MAGIC = b"ZGPK1\0"


def _shuffle(a16):
    b = a16.view(np.uint8).reshape(-1, 2)
    return np.ascontiguousarray(b.T).tobytes()


def _unshuffle(raw, n):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(2, n)
    return np.ascontiguousarray(b.T).view(np.int16).reshape(n)


def quantize_slab(x, scale, mean):
    q = np.round((x - mean) / scale)
    assert np.abs(q).max() < 32767, "field range exceeds int16 at this scale"
    return q.astype(np.int16)


def write(path, fields, diff_std, err_frac=0.05, chunk=64, level=10):
    """fields: (T, C, H, W) float32; diff_std: (C,) std of one-step change per channel."""
    T, C, H, W = fields.shape
    scales = (np.asarray(diff_std, dtype=np.float64) * 2 * err_frac).astype(np.float32)
    # per-channel offset = channel mean; per-chunk int16 range check happens in quantize
    means = fields.reshape(T, C, -1).mean(axis=(0, 2)).astype(np.float32)
    cctx = zstd.ZstdCompressor(level=level)
    payloads, offsets, max_err = [], [0], np.zeros(C)
    for s in range(0, T, chunk):
        blk = fields[s:s + chunk]
        qs = []
        for c in range(C):
            q = quantize_slab(blk[:, c], scales[c], means[c])
            max_err[c] = max(max_err[c], np.abs(q.astype(np.float32) * scales[c] + means[c]
                                              - blk[:, c]).max())
            qs.append(q)
        q = np.stack(qs, axis=1)                       # (t, C, H, W) int16
        d = q.astype(np.int32)
        d[1:] -= d[:-1].copy()                         # temporal delta, first slab absolute
        assert np.abs(d).max() < 32767
        p = cctx.compress(_shuffle(d.astype(np.int16).ravel()))
        payloads.append(p)
        offsets.append(offsets[-1] + len(p))
    header = json.dumps({"shape": [T, C, H, W], "chunk": chunk, "err_frac": err_frac,
                         "scales": scales.tolist(), "means": means.tolist(),
                         "max_abs_err": max_err.tolist(), "n_chunks": len(payloads),
                         "pipeline": "int16-linear|temporal-delta|byte-shuffle|zstd"}).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        f.write(np.asarray(offsets, dtype=np.uint64).tobytes())
        for p in payloads:
            f.write(p)
    return offsets[-1], max_err


class GridPack:
    def __init__(self, path):
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        assert bytes(self.mm[:6]) == MAGIC, "not a GridPack file"
        hlen = struct.unpack("<I", bytes(self.mm[6:10]))[0]
        self.h = json.loads(bytes(self.mm[10:10 + hlen]))
        pos = 10 + hlen
        nc = self.h["n_chunks"]
        self.offsets = np.frombuffer(self.mm[pos:pos + 8 * (nc + 1)].tobytes(), dtype=np.uint64)
        self.data_start = pos + 8 * (nc + 1)
        self.T, self.C, self.H, self.W = self.h["shape"]
        self.chunk = self.h["chunk"]
        self.scales = np.asarray(self.h["scales"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.means = np.asarray(self.h["means"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.dctx = zstd.ZstdDecompressor()

    @property
    def compressed_bytes(self):
        return int(self.offsets[-1])

    def read_chunk(self, i):
        a = self.data_start + int(self.offsets[i])
        b = self.data_start + int(self.offsets[i + 1])
        t = min(self.chunk, self.T - i * self.chunk)
        n = t * self.C * self.H * self.W
        d = _unshuffle(self.dctx.decompress(self.mm[a:b].tobytes()), n)
        q = np.cumsum(d.reshape(t, self.C, self.H, self.W).astype(np.int32), axis=0)
        return q.astype(np.float32) * self.scales + self.means

    def read(self, start, length):
        c0, c1 = start // self.chunk, (start + length - 1) // self.chunk
        parts = [self.read_chunk(i) for i in range(c0, c1 + 1)]
        joined = parts[0] if len(parts) == 1 else np.concatenate(parts)
        off = start - c0 * self.chunk
        return joined[off:off + length]

    def read_all(self):
        return self.read(0, self.T)
