"""Regression tests for the ZoomGPT / zoom.py audit findings (model group)."""
import copy
import os
import sys
import warnings

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.models import ZoomGPT  # noqa: E402
from bantamtrain.optim import build_optimizer  # noqa: E402
from bantamtrain.zoom import SlotMerge, SlotUnmerge, ZoomSchedule, lr_at  # noqa: E402


def test_bf16_model_loss_is_fp32():
    """ZG-3 / GPU-17: the loss of a bf16-weight model is fp32 CE of its logits."""
    torch.manual_seed(0)
    for out_mode in ("slot", "bag"):
        m = ZoomGPT(256, 32, 1, 2, zoom_ks=[2], out_mode=out_mode).to(torch.bfloat16)
        x = torch.randint(0, 256, (4, 34))
        for k in (1, 2):
            loss = m(x[:, :32], x[:, k:32 + k], k=k)
            assert loss.dtype == torch.float32
            z = m(x[:, :32], k=k).float()
            tgt = x[:, k:32 + k]
            if k == 1 or out_mode == "slot":
                ref = F.cross_entropy(z.reshape(-1, 256), tgt.reshape(-1))
            else:
                ref = -F.log_softmax(z, -1).gather(-1, tgt.reshape(4, 16, 2)).mean()
            assert abs(loss.item() - ref.item()) < 1e-5


def test_rope_cache_after_inference_mode_and_bounded():
    """ZG-6: a table first built under inference_mode is usable for training; one
    table per (k, device), grown and sliced."""
    torch.manual_seed(0)
    m = ZoomGPT(64, 32, 2, 2, zoom_ks=[2])
    x = torch.randint(0, 64, (2, 26))
    with torch.inference_mode():
        for k in (1, 2):
            m(x[:, :24], x[:, k:24 + k], k=k)
    for k in (1, 2):
        m.zero_grad()
        m(x[:, :24], x[:, k:24 + k], k=k).backward()
        assert m.emb.weight.grad is not None
    for T in (8, 16, 12, 300, 24):
        m.hidden(x[:, :1].repeat(1, T))
    assert sorted(k for k, _ in m._rope_cache) == [1, 2]
    cos, _ = m._rope_cache[(1, x.device)]
    assert cos.shape[0] == 300
    a, _ = m._rope(5, 1)
    ref, _ = m._make_rope(5, 1, x.device)
    assert torch.equal(a, ref)


def test_compile_once_per_zoom_level():
    """GPU-18: the RoPE cache no longer invalidates the compiled graph's guards."""
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters
    dynamo.reset()
    counters.clear()
    torch.manual_seed(0)
    m = ZoomGPT(97, 32, 1, 2, zoom_ks=(4, 2))
    f = torch.compile(m, backend="eager", dynamic=False)
    x = torch.randint(0, 97, (2, 36))
    for k in (4, 2, 1):
        for _ in range(2):
            loss = f(x[:, :32], x[:, k:32 + k], k=k)
        assert torch.allclose(loss, m(x[:, :32], x[:, k:32 + k], k=k), atol=1e-5)
    assert counters["stats"]["unique_graphs"] == 3
    dynamo.reset()


@pytest.mark.parametrize("kw", [{}, {"merge_mode": "linear"}, {"out_mode": "bag"},
                                {"merge_mode": "linear", "grad_ckpt": True, "ce_chunk": 16}])
def test_exact_flops_match_flop_counter(kw):
    """ZG-1 / PRE-13: exact=True charges what the kernels execute."""
    from torch.utils.flop_counter import FlopCounterMode
    torch.manual_seed(0)
    m = ZoomGPT(96, 32, 2, 2, zoom_ks=[2, 4], **kw)
    x = torch.randint(0, 96, (2, 68))
    for k in (1, 2, 4):
        with FlopCounterMode(display=False) as fc:
            m(x[:, :64], x[:, k:64 + k], k=k).backward()
        measured = fc.get_total_flops()
        assert abs(m.flops_per_step(2, 64, k, exact=True) / measured - 1) < 0.01


def test_default_flops_unchanged_and_linear_merge_charged():
    """PRE-13: the budget numbers of gate/mean runs stay as recorded; the linear
    merge/unmerge maps (4kd^2 per group) are charged by default and by exact."""
    from torch.utils.flop_counter import FlopCounterMode
    gate = ZoomGPT(4096, 256, 6, 4, zoom_ks=[2, 4])
    lin = ZoomGPT(4096, 256, 6, 4, zoom_ks=[2, 4], merge_mode="linear")
    mean = ZoomGPT(4096, 256, 6, 4, zoom_ks=[2, 4], gated=False)
    d, L, V, seq, B = 256, 6, 4096, 256, 16
    for k in (1, 2, 4):
        n = seq // k
        old = 3 * B * n * (L * (24 * d * d + 2 * n * d) + 2 * d * V * (k if k > 1 else 1))
        assert gate.flops_per_step(B, seq, k) == old == mean.flops_per_step(B, seq, k)
        extra = 3 * B * n * 4 * k * d * d if k > 1 else 0
        assert lin.flops_per_step(B, seq, k) - old == extra
        assert lin.flops_per_step(B, seq, k, True) - gate.flops_per_step(B, seq, k, True) == extra
    # the charged difference is what the linear maps actually cost
    torch.manual_seed(0)
    small = {m: ZoomGPT(96, 32, 2, 2, zoom_ks=[2, 4], merge_mode=m) for m in ("gate", "linear")}
    x = torch.randint(0, 96, (2, 68))
    for k in (2, 4):
        measured = {}
        for mode, m in small.items():
            with FlopCounterMode(display=False) as fc:
                m(x[:, :64], x[:, k:64 + k], k=k).backward()
            measured[mode] = fc.get_total_flops()
        charged = small["linear"].flops_per_step(2, 64, k) - small["gate"].flops_per_step(2, 64, k)
        assert charged == measured["linear"] - measured["gate"]


def test_merge_mode_validated():
    """ZG-7: a typo no longer silently selects the fixed-mean ablation."""
    for bad in ("gated", "Linear", "gates"):
        with pytest.raises(ValueError):
            SlotMerge(2, 4, mode=bad)
        with pytest.raises(ValueError):
            SlotUnmerge(2, 4, mode=bad)
        with pytest.raises(ValueError):
            ZoomGPT(32, 16, 1, 2, merge_mode=bad)            # even without zoom levels
    assert SlotMerge(2, 4, gated=False).mode == "mean"
    assert SlotUnmerge(2, 4, mode="linear").mode == "linear"


def test_constructor_and_input_checks():
    for kw in ({"d": 30, "n_head": 2}, {"d": 32, "n_head": 3}, {"out_mode": "slots"},
               {"rope_pos": "abs"}, {"mlp": "swiglu"}, {"zoom_ks": [0]}, {"block_size": 0}):
        args = {"d": 32, "n_head": 2, **kw}
        with pytest.raises(ValueError):
            ZoomGPT(64, args.pop("d"), 1, args.pop("n_head"), **args)
    m = ZoomGPT(64, 32, 1, 2, zoom_ks=[2, 4])
    x = torch.randint(0, 64, (2, 30))
    with pytest.raises(ValueError, match="divisible"):
        m(x[:, :22], x[:, 4:26], k=4)
    for k in (3, 0):
        with pytest.raises(ValueError, match="zoom level"):
            m(x[:, :24], x[:, 3:27], k=k)


@pytest.mark.parametrize("spec", ["0:0.5,1:0.5", "4:1.5,1:-0.5", "4:0.5,1:0.2", "4:0.35,x,1:0.65",
                                  "2.5:0.5,1:0.5", "4:,1:1", "-2:0.5,1:0.5", "4:nan,1:1"])
def test_schedule_validation(spec):
    """ZG-9: bad schedules raise ValueError (not an assert that python -O strips)."""
    with pytest.raises(ValueError):
        ZoomSchedule.parse(spec)


def test_schedule_parse_robust_and_repr_round_trip():
    s = ZoomSchedule.parse(" 4:0.35, 2:0.25 ,1:0.4, ")
    assert s.stages == [(4, 0.35), (2, 0.25), (1, 0.4)] and repr(s) == "4:0.35,2:0.25,1:0.4"
    assert ZoomSchedule.parse(" 1 ").stages == [(1, 1.0)]
    assert repr(ZoomSchedule.parse("1")) == "1:1"
    with pytest.warns(UserWarning, match="not k=1"):
        assert ZoomSchedule.parse("4").stages == [(4, 1.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        thirds = ZoomSchedule([(4, 1 / 3), (2, 1 / 3), (1, 1 / 3)])
        assert ZoomSchedule.parse(repr(thirds)).stages == thirds.stages
        ZoomSchedule.parse("4:0.6,1:0.4,2:0")                # zero-length tail stage is fine
    with pytest.raises(ValueError):
        ZoomSchedule([])
    assert lr_at(1.0, 1.0, warmup=1.0) == 1.0               # no ZeroDivisionError


def test_muon_split_survives_deepcopy():
    """ZG-12: an optimizer built on a deep copy still splits qkv into q, k, v."""
    torch.manual_seed(0)
    m = ZoomGPT(64, 32, 1, 2)
    c = copy.deepcopy(m)
    assert getattr(c.blocks[0].qkv.weight, "muon_split", None) is None   # torch drops it
    names = dict(m.named_parameters())
    assert m.muon_splits() == {"blocks.0.qkv.weight": 3} and set(m.muon_splits()) <= set(names)
    x = torch.randint(0, 64, (2, 17))
    before = m.blocks[0].qkv.weight.detach().clone()
    for model in (m, c):
        opt = build_optimizer(model, "muon", 0.02, 0.0)
        model(x[:, :16], x[:, 1:]).backward()
        opt.step()
    assert c.blocks[0].qkv.weight.muon_split == 3
    assert torch.allclose(c.blocks[0].qkv.weight - before, m.blocks[0].qkv.weight - before,
                          atol=1e-6)
