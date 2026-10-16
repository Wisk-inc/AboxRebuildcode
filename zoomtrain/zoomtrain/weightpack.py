"""WeightPack: compact model checkpoints.

    save(path, state_dict, bits=8)   # 8: blockwise int8 matrices, 16: bf16, 32: fp32
    state_dict = load(path)           # fp32 tensors, ready for load_state_dict

bits=8 quantizes every floating tensor with >= 2 dims and >= 4096 elements
(weight matrices, embeddings) with blockwise absmax int8 along the last axis
(block = 256 columns, one fp16 scale per block, 0.8 % overhead); bits=4 uses
packed int4 codes with block 64 (zoomtrain.qweights), ~0.53 B/param; vectors
(norm gains, biases, merge gates) are kept exactly in fp32 because they are
tiny and sensitive. Everything is then byte-shuffled and zstd-compressed.

Sizes per matrix element: fp32 4 B, bf16 2 B, int8 ~1.03 B before zstd.
Quality is measured, not assumed: `roundtrip_delta()` reports the change in a
model's validation loss after save -> load.
"""
import io
import json
import struct

import numpy as np
import torch
import zstandard as zstd

from .qweights import dequantize_rows, quantize_rows

MAGIC = b"ZWPK1\0"
BLOCK = 256


def _q8(t):
    x = t.float().reshape(-1, t.shape[-1])
    pad = (-x.shape[1]) % BLOCK
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    b = x.view(x.shape[0], -1, BLOCK)
    s = b.abs().amax(dim=2).clamp_min(1e-30)
    q = torch.round(b / s[..., None] * 127).to(torch.int8)
    return q.numpy(), s.to(torch.float16).numpy(), pad


def _dq8(q, s, pad, shape):
    x = torch.from_numpy(q.copy()).float() / 127 * torch.from_numpy(s.copy()).float()[..., None]
    x = x.reshape(x.shape[0], -1)
    if pad:
        x = x[:, :-pad]
    return x.reshape(shape)


def save(path, state_dict, bits=8, level=10):
    assert bits in (4, 8, 16, 32)
    meta, blobs = [], []
    for name, t in state_dict.items():
        t = t.detach().cpu()
        entry = {"name": name, "shape": list(t.shape), "dtype": str(t.dtype).split(".")[-1]}
        if t.is_floating_point() and bits == 4 and t.dim() >= 2 and t.numel() >= 4096:
            q, sc, pad = quantize_rows(t.reshape(-1, t.shape[-1]), 4, 64)
            entry.update(enc="int4", pad=pad, q_shape=list(q.shape), scale_shape=list(sc.shape))
            blobs += [q.numpy().tobytes(), sc.numpy().tobytes()]
            entry["sizes"] = [len(blobs[-2]), len(blobs[-1])]
        elif t.is_floating_point() and bits == 8 and t.dim() >= 2 and t.numel() >= 4096:
            q, s, pad = _q8(t)
            entry.update(enc="int8", pad=pad, scale_shape=list(s.shape))
            blobs += [q.tobytes(), s.tobytes()]
            entry["sizes"] = [len(blobs[-2]), len(blobs[-1])]
        elif t.is_floating_point() and bits <= 16:
            a = t.to(torch.bfloat16).view(torch.int16).numpy()
            entry["enc"] = "bf16"
            blobs.append(a.tobytes())
            entry["sizes"] = [len(blobs[-1])]
        else:
            a = t.contiguous().numpy() if t.is_floating_point() else t.numpy()
            entry["enc"] = "raw"
            blobs.append(a.tobytes())
            entry["sizes"] = [len(blobs[-1])]
        meta.append(entry)
    raw = b"".join(blobs)
    # byte-shuffle (2-byte stride helps bf16/fp16 scales; int8 codes are unaffected)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if len(arr) % 2:
        arr = np.concatenate([arr, np.zeros(1, np.uint8)])
    shuffled = np.ascontiguousarray(arr.reshape(-1, 2).T).tobytes()
    payload = zstd.ZstdCompressor(level=level).compress(shuffled)
    header = json.dumps({"bits": bits, "tensors": meta, "raw_len": len(raw)}).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        f.write(payload)
    return len(payload) + len(header) + 10


def load(path):
    with open(path, "rb") as f:
        assert f.read(6) == MAGIC, "not a WeightPack file"
        hlen = struct.unpack("<I", f.read(4))[0]
        h = json.loads(f.read(hlen))
        payload = f.read()
    shuffled = np.frombuffer(zstd.ZstdDecompressor().decompress(payload), dtype=np.uint8)
    raw = np.ascontiguousarray(shuffled.reshape(2, -1).T).tobytes()[:h["raw_len"]]
    out, pos = {}, 0
    for e in h["tensors"]:
        shape = tuple(e["shape"])
        if e["enc"] == "int4":
            n_q, n_s = e["sizes"]
            q = torch.from_numpy(np.frombuffer(raw[pos:pos + n_q], dtype=np.uint8).copy())
            sc = torch.from_numpy(np.frombuffer(raw[pos + n_q:pos + n_q + n_s],
                                                dtype=np.float16).copy())
            pos += n_q + n_s
            w = dequantize_rows(q.view(e["q_shape"]), sc.view(e["scale_shape"]), e["pad"], 4, 64)
            out[e["name"]] = w.reshape(shape)
        elif e["enc"] == "int8":
            n_q, n_s = e["sizes"]
            q = np.frombuffer(raw[pos:pos + n_q], dtype=np.int8)
            s = np.frombuffer(raw[pos + n_q:pos + n_q + n_s], dtype=np.float16)
            pos += n_q + n_s
            rows = int(np.prod(shape[:-1]))
            q = q.reshape(rows, -1, BLOCK)
            s = s.reshape(e["scale_shape"])
            out[e["name"]] = _dq8(q, s, e["pad"], shape)
        elif e["enc"] == "bf16":
            n = e["sizes"][0]
            a = np.frombuffer(raw[pos:pos + n], dtype=np.int16).copy()
            pos += n
            out[e["name"]] = torch.from_numpy(a).view(torch.bfloat16).float().reshape(shape)
        else:
            n = e["sizes"][0]
            dt = getattr(torch, e["dtype"])
            a = np.frombuffer(raw[pos:pos + n], dtype=np.dtype(str(dt).split(".")[-1]
                                                                 .replace("float", "float")
                                                                 if dt.is_floating_point
                                                                 else np.int64)).copy()
            pos += n
            out[e["name"]] = torch.from_numpy(a).reshape(shape).to(dt)
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
