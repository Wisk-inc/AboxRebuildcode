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


def test_memory_estimator_fits_logic():
    rows = memory.config_table(5e9, 4096, 32, 32000, 2048, 8, 96, 250, 0.4, 1e11)
    naive = rows[0]
    eff = [r for r in rows if "Muon 8-bit, ckpt" in r["config"]][0]
    assert not naive["fits"] and eff["fits"]
    assert eff["static_gb"] < 30 and naive["static_gb"] > 70
    assert eff["max_micro"] >= 8
