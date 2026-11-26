"""GridPack: error-bounded, compressed storage for gridded weather fields.

Why float16 is the wrong tool: float16 has 10 mantissa bits, so at
z500 ~ 5.4e4 m^2/s^2 its step is 32 m^2/s^2 (the size of a 6 h forecast
error), and surface pressure ~ 1e5 Pa cannot be represented at all above
65504. bfloat16 keeps the range but only 7 mantissa bits.

GridPack instead stores each channel as linear int16 quantization around the
channel's global mean (one mean and one scale per channel, applied to every
time step), with the scale chosen so the reconstruction error is at most
`err_frac` of the channel's 6-hour change standard deviation (an error the
model cannot resolve anyway). Inside a chunk of `chunk` time steps the
quantized integers are stored as temporal deltas (q[t] - q[t-1], exact in
int16): consecutive 6 h states differ by ~diff_std, i.e. ~1/(2*err_frac)
quantization steps, so the deltas are small integers with low entropy
regardless of the field's absolute magnitude. The deltas are byte-shuffled
and zstd compressed per chunk; chunk boundaries restart from an absolute
slab, so random access decodes one chunk.

Guarantee (checked by write() on every value, in float64, against the float32
reconstruction the reader computes): |x_hat - x| <= err_frac * diff_std[c].
The scale is 2*err_frac*diff_std minus a margin of a few float32 ulps of the
field's magnitude (<= 1e-3 relative for ERA5), so float32 rounding cannot push
the error past the bound. A channel with diff_std == 0 (constant in time) uses
a range-based scale instead; its bound (header "bounds") is half that scale
plus float32 rounding, and a constant channel round-trips exactly.
Non-finite values and fields whose range or 6 h change does not fit int16 at
the requested err_frac raise ValueError.

Layout
    b"ZGPK1\\0" | u32 hlen | header JSON (shape, chunk, scales, means, bounds)
    | u64[n_chunks + 1] offsets | chunk payloads (int16 deltas, byte-shuffled, zstd)
"""
import json
import os
import struct
import threading

import numpy as np
import zstandard as zstd

MAGIC = b"ZGPK1\0"
_I16 = 32767


class StalePackError(ValueError):
    """The pack was made from other data than the prepared directory now holds."""


def _shuffle(a16):
    b = a16.view(np.uint8).reshape(-1, 2)
    return np.ascontiguousarray(b.T).tobytes()


def _unshuffle(raw, n):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(2, n)
    return np.ascontiguousarray(b.T).view(np.int16).reshape(n)


def quantize_slab(x, scale, mean):
    """int16 round((x - mean) / scale), computed in float64; ValueError if it overflows."""
    q = np.round((np.asarray(x, np.float64) - np.float64(mean)) / np.float64(scale))
    if not np.isfinite(q).all():
        raise ValueError("non-finite values in field")
    if q.size and np.abs(q).max() > _I16:
        raise ValueError("field range exceeds int16 at this scale")
    return q.astype(np.int16)


def _f32_below(v):
    """Largest float32 <= v."""
    f = np.float32(v)
    return np.nextafter(f, np.float32(0)) if float(f) > v else f


def _spacing32(v):
    return float(np.spacing(np.float32(abs(v))))


def _channel_scale(dev, amax, bound):
    """Scale for one channel (max |x - mean| = dev, max |x| = amax) such that the float32
    reconstruction stays within `bound`."""
    if bound > 0:
        # q*s rounds to float32 (<= ulp/2 of |q*s|), then + mean rounds again (<= ulp/2 of |x_hat|)
        s = 2 * bound - _spacing32(dev + 2 * bound) - _spacing32(amax + bound)
        s *= 1 - 1e-9
        if s <= 0:
            raise ValueError(f"error bound {bound:.3g} is below the float32 resolution of "
                             f"values of magnitude {amax:.3g}")
        return _f32_below(s)
    return _f32_below(dev / (_I16 - 1)) if dev > 0 else np.float32(1.0)


class _Shrink(Exception):
    pass


def _encode(fields, scales, means, bounds, diff_std, err_frac, chunk, level):
    T, C = fields.shape[:2]
    cctx = zstd.ZstdCompressor(level=level)
    payloads, offsets, max_err = [], [0], np.zeros(C)
    for s in range(0, T, chunk):
        blk = fields[s:s + chunk]
        q = np.empty(blk.shape, np.int32)
        for c in range(C):
            q[:, c] = quantize_slab(blk[:, c], scales[c], means[c])
            # reconstruct exactly as GridPack.read_chunk does (float32), compare in float64
            xh = q[:, c].astype(np.float32) * scales[c] + means[c]
            err = float(np.abs(xh.astype(np.float64) - blk[:, c].astype(np.float64)).max())
            if err > bounds[c]:
                raise _Shrink(c)
            max_err[c] = max(max_err[c], err)
        d = q.copy()
        d[1:] -= q[:-1]                                # temporal delta, first slab absolute
        big = np.abs(d).reshape(len(d), C, -1).max(axis=(0, 2))
        if (big > _I16).any():
            c = int(np.argmax(big))
            why = (f"use err_frac >= {err_frac * int(big[c]) / _I16:.3g}" if diff_std[c] > 0
                   else "diff_std is 0 but the channel changes in time")
            raise ValueError(f"channel {c}: a one-step change of {int(big[c])} quantization "
                             f"steps exceeds int16; {why}")
        p = cctx.compress(_shuffle(d.astype(np.int16).ravel()))
        payloads.append(p)
        offsets.append(offsets[-1] + len(p))
    return payloads, offsets, max_err


def write(path, fields, diff_std, err_frac=0.05, chunk=64, level=10, extra=None):
    """fields: (T, C, H, W) float32; diff_std: (C,) std of one-step change per channel.

    Returns (payload bytes, per-channel max |x_hat - x|). `extra` adds JSON fields
    to the header (e.g. {"source": {...}}).
    """
    fields = np.asarray(fields, dtype=np.float32)
    if fields.ndim != 4:
        raise ValueError(f"fields must be (T, C, H, W), got shape {fields.shape}")
    T, C, H, W = fields.shape
    diff_std = np.asarray(diff_std, dtype=np.float64).reshape(-1)
    if len(diff_std) != C:
        raise ValueError(f"diff_std has {len(diff_std)} entries for {C} channels")
    if not (err_frac > 0 and np.isfinite(err_frac)):
        raise ValueError(f"err_frac must be positive, got {err_frac}")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    bad = [c for c in range(C) if not np.isfinite(fields[:, c]).all()]
    if bad:
        raise ValueError(f"non-finite values (NaN/inf) in channel(s) {bad}")
    if not np.isfinite(diff_std).all() or (diff_std < 0).any():
        raise ValueError(f"diff_std must be finite and >= 0, got {diff_std.tolist()}")
    # per-channel offset = channel mean (accumulated in float64, stored as float32)
    means = fields.reshape(T, C, -1).mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    dev, amax = np.zeros(C), np.zeros(C)
    for s in range(0, T, chunk):
        blk = fields[s:s + chunk].astype(np.float64)
        dev = np.maximum(dev, np.abs(blk - means.reshape(1, C, 1, 1)).max(axis=(0, 2, 3)))
        amax = np.maximum(amax, np.abs(blk).max(axis=(0, 2, 3)))
    bounds = err_frac * diff_std
    scales = np.zeros(C, np.float32)
    for c in range(C):
        scales[c] = _channel_scale(dev[c], amax[c], bounds[c])
        steps = dev[c] / float(scales[c])
        if steps > _I16 - 0.5:
            raise ValueError(f"channel {c}: values span +-{dev[c]:.4g} around the mean = "
                             f"{steps:.0f} int16 steps of {float(scales[c]):.3g}; use err_frac >= "
                             f"{err_frac * steps / (_I16 - 1):.3g}")
        if bounds[c] <= 0:   # range-based scale: bound = half a step + float32 rounding
            sc = float(scales[c])
            bounds[c] = sc / 2 + _spacing32(dev[c] + sc) + _spacing32(amax[c] + sc)
    for _ in range(8):
        try:
            payloads, offsets, max_err = _encode(fields, scales, means, bounds, diff_std,
                                                 err_frac, chunk, level)
            break
        except _Shrink as e:    # float32 rounding beat the margin: shrink that scale, redo
            c = e.args[0]
            scales[c] = _f32_below(float(scales[c]) * (1 - 1e-6))
    else:
        raise RuntimeError("GridPack could not meet its error bound")
    header = {"shape": [T, C, H, W], "chunk": chunk, "err_frac": err_frac,
              "scales": scales.tolist(), "means": means.tolist(),
              "max_abs_err": max_err.tolist(), "n_chunks": len(payloads),
              "pipeline": "int16-linear|temporal-delta|byte-shuffle|zstd",
              "bounds": bounds.tolist(), "bound_check": "float64"}
    for k, v in (extra or {}).items():
        if k in header:
            raise ValueError(f"extra header field {k!r} is reserved")
        header[k] = v
    hb = json.dumps(header).encode()
    tmp = os.fspath(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hb)))
        f.write(hb)
        f.write(np.asarray(offsets, dtype=np.uint64).tobytes())
        for p in payloads:
            f.write(p)
    os.replace(tmp, path)
    return offsets[-1], max_err


def _meta_for(path):
    meta = os.path.join(os.path.dirname(os.path.abspath(path)), "meta.json")
    if not os.path.exists(meta):
        return None
    try:
        with open(meta) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _source_mismatch(path, header, meta):
    """Why a pack does not match its directory's meta.json (prepare-era5), or None."""
    if meta is None:
        return None
    hint = "re-run `bantam pack-era5` or delete it"
    want = meta.get("prep_id")
    have = (header.get("source") or {}).get("prep_id")
    if want is not None and have != want:
        return f"{path} was packed from another prepare-era5 run (prep_id {have} != {want}); {hint}"
    T, C = header["shape"][:2]
    if "n_total" in meta and T != meta["n_total"]:
        return f"{path} holds {T} time steps but meta.json says n_total={meta['n_total']}; {hint}"
    if "channels" in meta and C != len(meta["channels"]):
        return f"{path} holds {C} channels but meta.json lists {len(meta['channels'])}; {hint}"
    return None


def stale_reason(path):
    """None if the GridPack at `path` matches DIR/meta.json, else a message saying why not."""
    gp = GridPack(path, verify=False)
    return _source_mismatch(gp.path, gp.h, _meta_for(gp.path))


class GridPack:
    """Reader. Thread-safe (one zstd decompressor per thread) and picklable.

    With `verify` (default) a pack that does not match its directory's meta.json
    (another prepare-era5 run, other T or C) raises StalePackError.
    """

    def __init__(self, path, verify=True):
        self.path = os.fspath(path)
        self.verify = verify
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        if bytes(self.mm[:6]) != MAGIC:
            raise ValueError(f"{self.path} is not a GridPack file")
        hlen = struct.unpack("<I", bytes(self.mm[6:10]))[0]
        self.h = json.loads(bytes(self.mm[10:10 + hlen]))
        pos = 10 + hlen
        nc = self.h["n_chunks"]
        self.offsets = np.frombuffer(self.mm[pos:pos + 8 * (nc + 1)].tobytes(), dtype=np.uint64)
        self.data_start = pos + 8 * (nc + 1)
        if self.data_start + int(self.offsets[-1]) > len(self.mm):
            raise ValueError(f"{self.path} is truncated")
        self.T, self.C, self.H, self.W = self.h["shape"]
        self.chunk = self.h["chunk"]
        self.scales = np.asarray(self.h["scales"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.means = np.asarray(self.h["means"], dtype=np.float32).reshape(1, -1, 1, 1)
        self._tl = threading.local()
        if verify:
            why = _source_mismatch(self.path, self.h, _meta_for(self.path))
            if why:
                raise StalePackError(why)

    def __getstate__(self):
        return {"path": os.path.abspath(self.path), "verify": self.verify}

    def __setstate__(self, state):
        self.__init__(**state)

    @property
    def dctx(self):
        """This thread's zstd decompressor (python-zstandard objects are not thread-safe)."""
        d = getattr(self._tl, "dctx", None)
        if d is None:
            d = self._tl.dctx = zstd.ZstdDecompressor()
        return d

    @property
    def compressed_bytes(self):
        return int(self.offsets[-1])

    def read_chunk(self, i):
        i = int(i)
        if not 0 <= i < self.h["n_chunks"]:
            raise IndexError(f"chunk {i} out of range ({self.h['n_chunks']} chunks)")
        a = self.data_start + int(self.offsets[i])
        b = self.data_start + int(self.offsets[i + 1])
        t = min(self.chunk, self.T - i * self.chunk)
        n = t * self.C * self.H * self.W
        d = _unshuffle(self.dctx.decompress(self.mm[a:b].tobytes()), n)
        q = np.cumsum(d.reshape(t, self.C, self.H, self.W).astype(np.int32), axis=0)
        return q.astype(np.float32) * self.scales + self.means

    def read(self, start, length):
        """Time steps [start, start + length) as float32 (T', C, H, W)."""
        start, length = int(start), int(length)
        if start < 0 or length < 0 or start + length > self.T:
            raise IndexError(f"read({start}, {length}) outside [0, {self.T})")
        if length == 0:
            return np.zeros((0, self.C, self.H, self.W), np.float32)
        c0, c1 = start // self.chunk, (start + length - 1) // self.chunk
        parts = [self.read_chunk(i) for i in range(c0, c1 + 1)]
        joined = parts[0] if len(parts) == 1 else np.concatenate(parts)
        off = start - c0 * self.chunk
        return joined[off:off + length]

    def read_all(self):
        return self.read(0, self.T)
