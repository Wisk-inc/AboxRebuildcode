"""WeightPack v2: every dtype, exact small tensors, overflow-safe scales, no padding blowup,
tied tensors once, checksums and validated loading; version-1 files still load."""
import json
import os
import struct
import sys

import numpy as np
import pytest
import torch
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import qweights, weightpack  # noqa: E402
from bantamtrain.models import ZoomGPT  # noqa: E402

DTYPES = (torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
          torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64)


def _sample(dt, shape):
    g = torch.Generator().manual_seed(0)
    if dt == torch.bool:
        return torch.rand(shape, generator=g) > 0.5
    if dt.is_floating_point or dt.is_complex:
        return (torch.randn(shape, generator=g, dtype=torch.float32) * 3).to(dt)
    return torch.randint(-100 if dt != torch.uint8 else 0, 100, shape, generator=g).to(dt)


@pytest.mark.parametrize("bits", [32, 16, 8, 4])
def test_every_dtype_and_shape_round_trips(tmp_path, bits):
    """WP-1 / WP-2 / WP-3: non-float, bf16/fp16 and small tensors come back exactly."""
    sd = {}
    for dt in DTYPES:
        name = str(dt).split(".")[-1]
        sd[f"{name}.vec"] = _sample(dt, (7,))
        sd[f"{name}.scalar"] = _sample(dt, ())
        sd[f"{name}.empty"] = _sample(dt, (0, 3))
        sd[f"{name}.mat"] = _sample(dt, (64, 96))          # >= 4096 elements: quantized if float
    weightpack.save(tmp_path / "w.zwpk", sd, bits)
    back = weightpack.load(tmp_path / "w.zwpk")
    assert list(back) == list(sd)
    for k, v in sd.items():
        r = back[k]
        assert r.dtype == v.dtype and r.shape == v.shape, k
        lossy = (v.is_floating_point() and k.endswith(".mat") and bits in (4, 8)) or \
            (bits == 16 and v.dtype in (torch.float32, torch.float64))
        if not lossy:
            assert torch.equal(r, v), k
        elif bits == 16:
            assert torch.equal(r, v.to(torch.bfloat16).to(v.dtype)), k
        else:
            assert (r.float() - v.float()).abs().max() <= v.float().abs().max() / (100 if bits == 8 else 7)


def test_quantized_model_state_dict_round_trips(tmp_path):
    torch.manual_seed(0)
    m = ZoomGPT(512, 64, 2, 2).eval()
    qweights.quantize_model(m, 8)
    sd = m.state_dict()
    weightpack.save(tmp_path / "q.zwpk", sd, 32)
    m2 = ZoomGPT(512, 64, 2, 2).eval()
    qweights.quantize_model(m2, 8)
    m2.load_state_dict(weightpack.load(tmp_path / "q.zwpk"))
    x = torch.randint(0, 512, (2, 33))
    assert torch.equal(m(x[:, :32], x[:, 1:]), m2(x[:, :32], x[:, 1:]))


def test_imports_without_wide_unsigned_dtypes(monkeypatch, tmp_path):
    """torch < 2.3 has no uint16/32/64 (pyproject allows torch 2.2): weightpack still imports."""
    import importlib
    for name in ("uint16", "uint32", "uint64"):
        monkeypatch.delattr(torch, name)
    try:
        wp = importlib.reload(weightpack)
        assert "uint16" not in wp._DTYPES
        wp.save(tmp_path / "w.zwpk", {"w": torch.randn(64, 64)}, 8)
        assert wp.load(tmp_path / "w.zwpk")["w"].shape == (64, 64)
    finally:
        monkeypatch.undo()
        importlib.reload(weightpack)
    assert "uint16" in weightpack._DTYPES


def test_corruption_raises(tmp_path):
    """WP-4: payload, header and magic damage raise ValueError instead of loading."""
    sd = {"w": torch.randn(256, 64), "b": torch.randn(64)}
    p = tmp_path / "w.zwpk"
    weightpack.save(p, sd, 8)
    good = p.read_bytes()
    hlen = struct.unpack("<I", good[6:10])[0]
    start = 14 + hlen
    for pos in (start + 20, len(good) // 2, len(good) - 3, 20):   # payload, header
        bad = bytearray(good)
        bad[pos] ^= 0x10
        p.write_bytes(bytes(bad))
        with pytest.raises(ValueError):
            weightpack.load(p)
    for bad in (b"XXXXXX" + good[6:], good[:len(good) - 40]):
        p.write_bytes(bad)
        with pytest.raises(ValueError):
            weightpack.load(p)


def test_scales_never_overflow_or_underflow(tmp_path):
    """WP-5: int8 scales fall back to fp32 outside the fp16 range (was inf/NaN above 65504)."""
    torch.manual_seed(0)
    for scale in (1e5, 1e-7):
        for bits, tol in ((8, 0.01), (4, 0.15)):
            w = torch.randn(256, 256) * scale
            weightpack.save(tmp_path / "w.zwpk", {"w": w}, bits)
            r = weightpack.load(tmp_path / "w.zwpk")["w"]
            assert torch.isfinite(r).all()
            assert ((r - w).norm() / w.norm()).item() < tol


def test_narrow_last_axis_and_tied_tensors(tmp_path):
    """WP-6: blocks run over the flattened tensor; shared storage is written once."""
    torch.manual_seed(0)
    for shape in ((64, 64, 3, 3), (8192, 1)):
        t = {"k": torch.randn(shape)}
        assert weightpack.save(tmp_path / "a", t, 8) < weightpack.save(tmp_path / "b", t, 16)
    e = torch.randn(4096, 64)
    one = weightpack.save(tmp_path / "one", {"a": e}, 8)
    two = weightpack.save(tmp_path / "two", {"a": e, "b": e, "c": e[:10]}, 8)
    assert two < one * 1.1
    back = weightpack.load(tmp_path / "two")
    assert back["b"] is back["a"] and back["c"].shape == (10, 64)


def _save_v1(path, sd, bits):
    """The version-1 writer (commit 60759d3), kept here to prove old files still load."""
    meta, blobs = [], []
    for name, t in sd.items():
        e = {"name": name, "shape": list(t.shape), "dtype": str(t.dtype).split(".")[-1]}
        if t.is_floating_point() and bits == 8 and t.dim() >= 2 and t.numel() >= 4096:
            x = t.float().reshape(-1, t.shape[-1])
            pad = (-x.shape[1]) % 256
            x = torch.nn.functional.pad(x, (0, pad)).view(x.shape[0], -1, 256)
            s = x.abs().amax(dim=2).clamp_min(1e-30)
            q = torch.round(x / s[..., None] * 127).to(torch.int8)
            e.update(enc="int8", pad=pad, scale_shape=list(s.shape))
            parts = [q.numpy().tobytes(), s.to(torch.float16).numpy().tobytes()]
        elif t.is_floating_point() and bits == 4 and t.dim() >= 2 and t.numel() >= 4096:
            q, s, pad = qweights.quantize_rows(t.reshape(-1, t.shape[-1]), 4, 64, torch.float16)
            e.update(enc="int4", pad=pad, q_shape=list(q.shape), scale_shape=list(s.shape))
            parts = [q.numpy().tobytes(), s.numpy().tobytes()]
        elif t.is_floating_point() and bits <= 16:
            e["enc"] = "bf16"
            parts = [t.to(torch.bfloat16).view(torch.int16).numpy().tobytes()]
        else:
            e["enc"] = "raw"
            parts = [t.numpy().tobytes()]
        e["sizes"] = [len(b) for b in parts]
        blobs += parts
        meta.append(e)
    raw = b"".join(blobs)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if len(arr) % 2:
        arr = np.concatenate([arr, np.zeros(1, np.uint8)])
    payload = zstd.ZstdCompressor(level=3).compress(np.ascontiguousarray(arr.reshape(-1, 2).T).tobytes())
    header = json.dumps({"bits": bits, "tensors": meta, "raw_len": len(raw)}).encode()
    with open(path, "wb") as f:
        f.write(b"ZWPK1\0" + struct.pack("<I", len(header)) + header + payload)


@pytest.mark.parametrize("bits", [32, 16, 8, 4])
def test_version1_files_still_load(tmp_path, bits):
    torch.manual_seed(0)
    sd = ZoomGPT(512, 64, 2, 2, zoom_ks=[2]).state_dict()
    sd["mask"] = torch.tensor([True, False, True])
    sd["idx"] = torch.arange(5, dtype=torch.int32)
    _save_v1(tmp_path / "v1.zwpk", sd, bits)
    back = weightpack.load(tmp_path / "v1.zwpk")
    for k, v in sd.items():
        assert back[k].dtype == v.dtype and back[k].shape == v.shape
        if bits == 32 or not v.is_floating_point():
            assert torch.equal(back[k], v)
        else:
            assert (back[k] - v).abs().max() <= v.abs().max() / 7 + 1e-6
