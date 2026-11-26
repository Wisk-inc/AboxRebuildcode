"""WeightPack: compact model checkpoints.

    save(path, state_dict, bits=8)   # 8: blockwise int8 matrices, 4: int4, 16: bf16, 32: exact
    state_dict = load(path)           # tensors in their original dtypes, ready for load_state_dict

bits=32 stores every tensor bit-exactly (any dtype: float, bf16, fp16, int, bool, complex,
0-d). bits=16 rounds fp32/fp64 tensors to bf16 and keeps fp16/bf16 and non-float tensors
exact. bits=8 quantizes every floating tensor with >= 2 dims and >= 4096 elements
(weight matrices, embeddings) with blockwise absmax int8 over the flattened tensor
(blocks of 256 elements, only the last block padded, one scale = absmax/127 per block:
fp16, or fp32 when a scale is outside the fp16 normal range, ~0.8 % overhead); bits=4
uses packed int4 codes with block 64 (bantamtrain.qweights), ~0.53 B/param. Everything
else (norm gains, biases, merge gates, small matrices) is kept exactly. Tensors that
share storage (tied embeddings) are stored once and come back shared. Everything is then
byte-shuffled and zstd-compressed.

File: magic "ZWPK2\\0", <u32 header length, <u32 CRC32 of the header, a JSON header
(version, byte order, per-tensor encodings), and one zstd frame with a content checksum.
Arrays are little-endian. Loading parses no pickle and validates names, dtypes, shapes
and sizes, so a damaged or hostile file raises instead of loading wrong weights.
Version-1 files ("ZWPK1") still load.

Sizes per matrix element: fp32 4 B, bf16 2 B, int8 ~1.01 B before zstd.
Quality is measured, not assumed: `roundtrip_delta()` reports the change in a
model's validation loss after save -> load.
"""
import io
import json
import math
import struct
import sys
import zlib

import numpy as np
import torch
import zstandard as zstd

from .qweights import dequantize_rows, quantize_rows

MAGIC_V1 = b"ZWPK1\0"
MAGIC = b"ZWPK2\0"
VERSION = 2
BLOCK = 256        # int8 block (elements)
BLOCK4 = 64        # int4 block
_DTYPES = {str(d).split(".")[-1]: d for d in (
    torch.float64, torch.float32, torch.float16, torch.bfloat16, torch.complex64, torch.complex128,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool,
    torch.uint16, torch.uint32, torch.uint64, torch.float8_e4m3fn, torch.float8_e5m2)}
_SCALES = {"float16": torch.float16, "float32": torch.float32}


def _le_bytes(t):
    """Raw little-endian bytes of a tensor (any dtype, any shape)."""
    t = t.detach().cpu().contiguous().reshape(-1)
    b = t.view(torch.uint8) if t.numel() else torch.empty(0, dtype=torch.uint8)
    if sys.byteorder == "big" and t.element_size() > 1:
        b = b.view(-1, t.element_size()).flip(1).reshape(-1)
    return b.numpy().tobytes()


def _from_le(buf, dtype, shape):
    raw = torch.frombuffer(bytearray(buf), dtype=torch.uint8) if len(buf) else \
        torch.empty(0, dtype=torch.uint8)
    size = torch.empty((), dtype=dtype).element_size()
    if sys.byteorder == "big" and size > 1:
        raw = raw.view(-1, size).flip(1).reshape(-1)
    return raw.view(dtype).reshape(shape)


def _flat_blocks(t, block):
    """t as (n_blocks, block) fp32 over the flattened tensor; only the last block is padded."""
    x = t.detach().reshape(-1)
    pad = (-x.numel()) % block
    if pad:
        x = torch.cat([x.float(), x.new_zeros(pad, dtype=torch.float32)])
    return x.view(-1, block), pad


def _storage_key(t):
    if t.numel() == 0 or t.device.type == "meta":
        return None
    return (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()),
            t.dtype, t.device.type)


def save(path, state_dict, bits=8, level=10):
    """Write state_dict to `path`; returns the file size in bytes."""
    if bits not in (4, 8, 16, 32):
        raise ValueError(f"bits must be 4, 8, 16 or 32, got {bits}")
    meta, blobs, first = [], [], {}
    for name, t in state_dict.items():
        t = t.detach()
        key = _storage_key(t)
        if key is not None and key in first:                 # tied / shared tensor: store once
            meta.append({"name": name, "enc": "alias", "alias": first[key]})
            continue
        if key is not None:
            first[key] = name
        t = t.cpu()
        dt = str(t.dtype).split(".")[-1]
        if dt not in _DTYPES:
            raise TypeError(f"{name}: unsupported dtype {t.dtype}")
        entry = {"name": name, "shape": list(t.shape), "dtype": dt}
        quant = t.is_floating_point() and t.dim() >= 2 and t.numel() >= 4096 and bits in (4, 8)
        if quant:
            blk = BLOCK if bits == 8 else BLOCK4
            x, pad = _flat_blocks(t, blk)
            q, s, _ = quantize_rows(x, bits, blk)
            entry.update(enc=f"int{bits}", pad=pad, block=blk, scale=str(s.dtype).split(".")[-1])
            parts = [_le_bytes(q), _le_bytes(s)]
        elif bits == 16 and t.dtype in (torch.float32, torch.float64):
            entry["enc"] = "bf16"
            parts = [_le_bytes(t.to(torch.bfloat16))]
        else:
            entry["enc"] = "raw"
            parts = [_le_bytes(t)]
        entry["sizes"] = [len(p) for p in parts]
        blobs += parts
        meta.append(entry)
    raw = b"".join(blobs)
    # byte-shuffle (2-byte stride helps bf16/fp16 data and scales; int8 codes are unaffected)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if len(arr) % 2:
        arr = np.concatenate([arr, np.zeros(1, np.uint8)])
    shuffled = np.ascontiguousarray(arr.reshape(-1, 2).T).tobytes()
    del raw, arr
    payload = zstd.ZstdCompressor(level=level, write_checksum=True).compress(shuffled)
    header = json.dumps({"version": VERSION, "byteorder": "little", "bits": bits, "tensors": meta,
                         "raw_len": sum(sum(e.get("sizes", ())) for e in meta)}).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", len(header), zlib.crc32(header)))
        f.write(header)
        f.write(payload)
    return len(MAGIC) + 8 + len(header) + len(payload)


def _check(cond, msg):
    if not cond:
        raise ValueError(f"corrupt WeightPack file: {msg}")


def _decompress(payload, raw_len):
    want = raw_len + raw_len % 2
    size = zstd.frame_content_size(payload)
    _check(size in (want, -1), f"payload is {size} bytes, header says {want}")
    data = zstd.ZstdDecompressor().decompress(payload, max_output_size=want)
    _check(len(data) == want, "payload size")
    shuffled = np.frombuffer(data, dtype=np.uint8)
    return np.ascontiguousarray(shuffled.reshape(2, -1).T).reshape(-1)[:raw_len]


def load(path, dtype=None):
    """WeightPack file -> {name: tensor}. Tensors come back in their saved dtype (quantized
    ones dequantized to it); dtype=torch.float32 casts every floating tensor instead."""
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic == MAGIC_V1:
            hlen = struct.unpack("<I", f.read(4))[0]
            h, crc = f.read(hlen), None
        elif magic == MAGIC:
            hlen, crc = struct.unpack("<II", f.read(8))
            h = f.read(hlen)
        else:
            raise ValueError(f"{path}: not a WeightPack file")
        _check(len(h) == hlen and (crc is None or zlib.crc32(h) == crc), "header checksum")
        h = json.loads(h)
        payload = f.read()
    try:
        raw = _decompress(payload, int(h["raw_len"]))
    except zstd.ZstdError as e:
        raise ValueError(f"corrupt WeightPack file: {e}") from e
    out = (_load_v1 if magic == MAGIC_V1 else _load_v2)(h, raw)
    if dtype is not None:
        out = {k: v.to(dtype) if v.is_floating_point() else v for k, v in out.items()}
    return out


def _entry(e, raw, pos, n_parts):
    sizes = e.get("sizes")
    _check(isinstance(sizes, list) and len(sizes) == n_parts
           and all(isinstance(s, int) and s >= 0 for s in sizes), f"sizes of {e.get('name')}")
    _check(pos + sum(sizes) <= len(raw), f"{e.get('name')} runs past the payload")
    parts, p = [], pos
    for s in sizes:
        parts.append(raw[p:p + s])
        p += s
    return parts, p


def _shape(e):
    shape = e.get("shape")
    _check(isinstance(shape, list) and all(isinstance(d, int) and d >= 0 for d in shape),
           f"shape of {e.get('name')}")
    return tuple(shape)


def _dtype(e):
    _check(e.get("dtype") in _DTYPES, f"dtype of {e.get('name')}")
    return _DTYPES[e["dtype"]]


def _load_v2(h, raw):
    _check(h.get("version") == VERSION and h.get("byteorder") == "little", "version")
    out, pos = {}, 0
    for e in h["tensors"]:
        name = e.get("name")
        _check(isinstance(name, str) and name not in out, "tensor names")
        enc = e.get("enc")
        if enc == "alias":
            _check(e.get("alias") in out, f"alias of {name}")
            out[name] = out[e["alias"]]
            continue
        shape, dt = _shape(e), _dtype(e)
        n = math.prod(shape)
        if enc in ("int8", "int4"):
            bits = 8 if enc == "int8" else 4
            block, pad, sdt = e.get("block"), e.get("pad"), _SCALES.get(e.get("scale"))
            _check(sdt is not None and isinstance(block, int) and block > 0 and block % 2 == 0
                   and isinstance(pad, int) and 0 <= pad < block and (n + pad) % block == 0,
                   f"blocks of {name}")
            nb = (n + pad) // block
            (bq, bs), pos = _entry(e, raw, pos, 2)
            ncode = nb * block if bits == 8 else nb * block // 2
            _check(len(bq) == ncode and len(bs) == nb * (2 if sdt == torch.float16 else 4),
                   f"sizes of {name}")
            q = _from_le(bq, torch.int8 if bits == 8 else torch.uint8,
                         (nb, 1, block) if bits == 8 else (nb, block // 2))
            s = _from_le(bs, sdt, (nb, 1))
            w = dequantize_rows(q, s, 0, bits, block, torch.float32).reshape(-1)
            out[name] = w[:n].reshape(shape).to(dt)
        elif enc in ("bf16", "raw"):
            (b,), pos = _entry(e, raw, pos, 1)
            src = torch.bfloat16 if enc == "bf16" else dt
            _check(len(b) == n * torch.empty((), dtype=src).element_size(), f"size of {name}")
            out[name] = _from_le(b, src, shape).to(dt)
        else:
            _check(False, f"encoding {enc!r} of {name}")
    _check(pos == len(raw), "payload length")
    return out


def _load_v1(h, raw):
    """Version-1 files: row-wise int8 (absmax fp16 scales) / int4, bf16 and native raw."""
    out, pos = {}, 0
    for e in h["tensors"]:
        name, enc, shape = e.get("name"), e.get("enc"), _shape(e)
        dt = _dtype(e)
        if enc == "int4":
            (bq, bs), pos = _entry(e, raw, pos, 2)
            q = _from_le(bq, torch.uint8, tuple(e["q_shape"]))
            s = _from_le(bs, torch.float16, tuple(e["scale_shape"]))
            out[name] = dequantize_rows(q, s, e["pad"], 4, 64).reshape(shape).to(dt)
        elif enc == "int8":
            (bq, bs), pos = _entry(e, raw, pos, 2)
            rows = math.prod(shape[:-1])
            q = _from_le(bq, torch.int8, (rows, -1, 256)).float()
            s = _from_le(bs, torch.float16, tuple(e["scale_shape"])).float()
            x = (q / 127 * s[..., None]).reshape(rows, -1)
            out[name] = (x[:, :x.shape[1] - e["pad"]] if e["pad"] else x).reshape(shape).to(dt)
        elif enc in ("bf16", "raw"):
            (b,), pos = _entry(e, raw, pos, 1)
            out[name] = _from_le(b, torch.bfloat16 if enc == "bf16" else dt, shape).to(dt)
        else:
            _check(False, f"encoding {enc!r} of {name}")
    return out


@torch.no_grad()
def roundtrip_delta(model, eval_fn, path, bits=8):
    """Save -> load -> reload into a copy of the model; return (loss_before, loss_after, bytes)."""
    before = eval_fn(model)
    nbytes = save(path, model.state_dict(), bits)
    sd = load(path)
    backup = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(backup[k].dtype) for k, v in sd.items()})
    after = eval_fn(model)
    model.load_state_dict(backup)
    return before, after, nbytes


def torch_save_bytes(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getbuffer().nbytes
