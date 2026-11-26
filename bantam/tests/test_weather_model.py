"""ZoomViT: zoom-level validation, FLOP accounting vs torch's counter, Muon splits, config."""
import os
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.models import ZoomViT  # noqa: E402


def measured_flops(model, fn):
    with FlopCounterMode(display=False) as fc:
        fn().float().pow(2).mean().backward()
    model.zero_grad(set_to_none=True)
    return fc.get_total_flops()


def test_vit_rejects_non_square_k():
    for k in (2, 3, 8):
        with pytest.raises(ValueError):
            ZoomViT(12, 6, (32, 64), 2, 16, 1, 2, zoom_ks=(k,))
    m = ZoomViT(12, 6, (32, 64), 2, 16, 1, 2)          # lowres model: no merge modules
    for k in (2, 3):
        with pytest.raises(ValueError):
            m.flops_per_step(1, k, lowres=True)
    for k in (9, 0, -4):                               # 3 does not divide the 16x32 grid; k < 1
        with pytest.raises(ValueError):
            m.zoom_factor(k)
    assert [m.zoom_factor(k) for k in (1, 4, 16)] == [1, 2, 4]
    with pytest.raises(ValueError):                    # input is not the 2x-coarsened grid
        m.forward_lowres(torch.randn(1, 12, 32, 64), 2)
    assert m.forward_lowres(torch.randn(1, 12, 16, 32), 2).shape == (1, 6, 16, 32)


@pytest.mark.parametrize("mode,k", [("gate", 4), ("gate", 16), ("linear", 4), ("linear", 16),
                                    ("lowres", 4), ("lowres", 16)])
def test_vit_flops_match_flop_counter(mode, k):
    torch.manual_seed(0)
    lowres = mode == "lowres"
    m = ZoomViT(12, 6, (32, 64), 2, 128, 2, 2, zoom_ks=() if lowres else (k,),
                merge_mode=None if lowres else mode)
    x = torch.randn(2, 12, 32, 64)
    if lowres:
        f = m.zoom_factor(k)
        meas = measured_flops(m, lambda: m.forward_lowres(F.avg_pool2d(x, f), f))
    else:
        meas = measured_flops(m, lambda: m(x, k))
    assert 0.93 < m.flops_per_step(2, k, lowres=lowres) / meas < 1.07


def test_vit_linear_merge_flops_term():
    B, d = 3, 32
    lin = ZoomViT(12, 6, (32, 64), 2, d, 2, 2, zoom_ks=(4, 16), merge_mode="linear")
    gate = ZoomViT(12, 6, (32, 64), 2, d, 2, 2, zoom_ks=(4, 16))
    N = lin.gh * lin.gw
    for k in (4, 16):
        assert lin.flops_per_step(B, k) - gate.flops_per_step(B, k) == 3 * B * 4 * N * d * d
    assert lin.flops_per_step(B, 1) == gate.flops_per_step(B, 1)


def test_vit_muon_splits_and_config_roundtrip():
    torch.manual_seed(0)
    m = ZoomViT(12, 6, (32, 64), 2, 16, 3, 2, zoom_ks=(4,), merge_mode="linear")
    params = dict(m.named_parameters())
    sp = m.muon_splits()
    assert sp == {f"blocks.{i}.qkv.weight": 3 for i in range(3)}
    assert all(n in params and getattr(params[n], "muon_split", 1) == s for n, s in sp.items())
    m2 = ZoomViT(**m.config())
    m2.load_state_dict(m.state_dict())
    x = torch.randn(1, 12, 32, 64)
    assert torch.equal(m(x, 4), m2(x, 4))
