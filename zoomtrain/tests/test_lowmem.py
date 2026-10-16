import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoomtrain import lowmem, memory, weightpack  # noqa: E402
from zoomtrain.models import ZoomGPT  # noqa: E402
from zoomtrain.optim import AdamWLowMem, Muon, build_optimizer  # noqa: E402


def test_signed_quant_error_bounds_and_small_values():
    torch.manual_seed(0)
    x = torch.randn(10_000) * torch.rand(10_000)       # heavy spread inside blocks
    for vmap in ("sqrt", "linear"):
        q, s = lowmem.quantize_signed(x, 256, vmap)
        y = lowmem.dequantize_signed(q, s, x.shape, vmap)
        assert q.dtype == torch.int8 and y.shape == x.shape
        assert (y - x).abs().max() <= s.max() / 127 * 1.01
    # sqrt map: better relative error on the small entries than the linear map
    small = x.abs() < 0.05
    e_sqrt = (lowmem.dequantize_signed(*lowmem.quantize_signed(x, 256, "sqrt"), x.shape, "sqrt") - x)[small]
    e_lin = (lowmem.dequantize_signed(*lowmem.quantize_signed(x, 256, "linear"), x.shape, "linear") - x)[small]
    assert e_sqrt.abs().mean() < e_lin.abs().mean()


def test_log_quant_relative_error_across_magnitudes():
    v = torch.logspace(-12, 2, 4096)
    q, lo, hi = lowmem.quantize_log(v, 256)
    y = lowmem.dequantize_log(q, lo, hi, v.shape)
    assert q.dtype == torch.uint8
    rel = ((y - v).abs() / v).max()
    assert rel < 0.05


def test_qstate_roundtrip_and_bytes():
    t = torch.randn(300, 70)
    st = lowmem.QState(t, "signed")
    st.set(t)
    assert (st.get() - t).abs().max() < t.abs().max() / 100
    assert st.nbytes() < t.numel() * 1.1


def test_stochastic_rounding_is_unbiased():
    torch.manual_seed(0)
    x = torch.full((200_000,), 1.0 + 2 ** -10)         # halfway below bf16 resolution at 1.0
    y = lowmem.to_bf16_stochastic(x).float()
    assert set(y.unique().tolist()) <= {1.0, 1.0 + 2 ** -7}
    assert abs(y.mean().item() - x[0].item()) < 2e-4   # expectation matches fp32 value
    neg = -x
    assert abs(lowmem.to_bf16_stochastic(neg).float().mean().item() - neg[0].item()) < 2e-4


def _quadratic_trajectory(opt_cls, lr, **kw):
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(64, 32))
    target = torch.randn(64, 32)
    opt = opt_cls([w], lr=lr, **kw)
    for _ in range(300):
        loss = ((w - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


def test_adamw_8bit_matches_32bit_on_quadratic():
    l32 = _quadratic_trajectory(AdamWLowMem, 3e-2, state_bits=32)
    l8 = _quadratic_trajectory(AdamWLowMem, 3e-2, state_bits=8)
    assert l32 < 0.05 and abs(l8 - l32) < 0.1 * l32 + 0.01


def test_muon_8bit_matches_32bit_on_quadratic():
    l32 = _quadratic_trajectory(Muon, 5e-2, state_bits=32)
    l8 = _quadratic_trajectory(Muon, 5e-2, state_bits=8)
    assert l32 < 0.5 and abs(l8 - l32) < 0.1 * l32 + 0.02


def test_bf16_params_train_with_stochastic_rounding():
    torch.manual_seed(0)
    m = ZoomGPT(128, 32, 1, 2).to(torch.bfloat16)
    opt = build_optimizer(m, "muon", 1e-2, 0.0, state_bits=8)
    x = torch.randint(0, 128, (4, 17))
    losses = []
    for _ in range(30):
        loss = m(x[:, :16], x[:, 1:17], k=1)
        opt.zero_grad()
        loss.float().backward()
        opt.step()
        losses.append(loss.item())
    assert all(p.dtype == torch.bfloat16 for p in m.parameters())
    assert losses[-1] < losses[0] - 0.5
    assert lowmem.state_bytes(opt) < sum(p.numel() for p in m.parameters()) * 2.2


def test_grad_checkpoint_same_loss_and_grads():
    torch.manual_seed(0)
    a = ZoomGPT(128, 32, 2, 2)
    b = ZoomGPT(128, 32, 2, 2, grad_ckpt=True)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 128, (2, 17))
    la, lb = a(x[:, :16], x[:, 1:], k=1), b(x[:, :16], x[:, 1:], k=1)
    la.backward()
    lb.backward()
    assert torch.allclose(la, lb)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6)


def test_weightpack_roundtrip_sizes_and_error(tmp_path):
    torch.manual_seed(0)
    m = ZoomGPT(512, 64, 2, 2, zoom_ks=[2])
    sd = m.state_dict()
    fp32 = weightpack.torch_save_bytes(sd)
    sizes = {}
    for bits in (32, 16, 8):
        p = tmp_path / f"w{bits}.zwpk"
        sizes[bits] = weightpack.save(p, sd, bits)
        back = weightpack.load(p)
        assert set(back) == set(sd)
        for k, v in sd.items():
            r = back[k]
            assert r.shape == v.shape
            if bits == 32:
                assert torch.equal(r, v)
            elif v.dim() < 2 or v.numel() < 4096:
                tol = 0.0 if bits == 32 else 2 ** -7 * v.abs().max() + 1e-6
                assert (r - v).abs().max() <= tol
            else:
                assert (r - v).abs().max() <= v.abs().max() / 100
    assert sizes[8] < sizes[16] < sizes[32] <= fp32 * 1.02
    assert sizes[8] < fp32 / 3


def test_memory_scoreboard_ladder():
    sb = memory.scoreboard()
    rows = sb["rows"]
    assert 4.9e9 < sb["params"] < 5.3e9
    std, eff = rows[0], rows[2]
    assert "ckpt" in std["config"]                  # standard recipe must recompute to fit
    assert std["bytes_per_param"] == 18 and eff["bytes_per_param"] < 5.2
    assert all(r["fits"] for r in rows)
    assert eff["max_micro"] >= 8
    days = [r["days"] for r in rows]
    assert days[0] > days[1] > days[2] > days[3]
    assert abs(days[0] / days[1] - 8 / 6) < 1e-6    # recompute removed
    assert abs(days[1] / days[2] - 1 / 0.52) < 1e-6



def test_qweights_int8_int4_shapes_error_and_bytes():
    from zoomtrain import qweights
    torch.manual_seed(0)
    w = torch.randn(96, 200)
    for bits, block in ((8, 128), (4, 64)):
        q, s, pad = qweights.quantize_rows(w, bits, block)
        r = qweights.dequantize_rows(q, s, pad, bits, block)
        assert r.shape == w.shape
        blk = torch.nn.functional.pad(w.abs(), (0, pad)).view(96, -1, block).amax(2)
        bound = (blk / qweights.QMAX[bits] / 2 * 1.01 + 1e-3).repeat_interleave(block, 1)[:, :200]
        assert ((r - w).abs() <= bound).all()


def test_quantize_model_in_place_keeps_outputs_close():
    from zoomtrain import qweights
    torch.manual_seed(0)
    m = ZoomGPT(512, 64, 2, 2).eval()
    x = torch.randint(0, 512, (2, 33))
    ref = m(x[:, :32], x[:, 1:], k=1).item()
    out = {}
    for bits in (8, 4):
        q = ZoomGPT(512, 64, 2, 2).eval()
        q.load_state_dict(m.state_dict())
        rep = qweights.quantize_model(q, bits)
        out[bits] = q(x[:, :32], x[:, 1:], k=1).item()
        assert rep["modules_quantized"] == 2 * 4 + 1          # 4 linears per block + embedding
        assert rep["ratio"] > (3.5 if bits == 8 else 6.0)
    assert abs(out[8] - ref) < 0.01 and abs(out[4] - ref) < 0.1


def test_weightpack_int4(tmp_path):
    torch.manual_seed(0)
    m = ZoomGPT(512, 64, 2, 2)
    sd = m.state_dict()
    p = tmp_path / "w4.zwpk"
    n4 = weightpack.save(p, sd, 4)
    n8 = weightpack.save(tmp_path / "w8.zwpk", sd, 8)
    back = weightpack.load(p)
    assert n4 < n8
    for k, v in sd.items():
        assert back[k].shape == v.shape
        if v.dim() >= 2 and v.numel() >= 4096:
            assert (back[k] - v).abs().max() <= v.abs().max() / 7
