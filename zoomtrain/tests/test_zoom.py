import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoomtrain.models import ZoomGPT, ZoomViT  # noqa: E402
from zoomtrain.zoom import SlotMerge, SlotUnmerge, ZoomSchedule, lr_at  # noqa: E402


def test_schedule_parse_and_fractions():
    s = ZoomSchedule.parse("4:0.35,2:0.25,1:0.4")
    assert s.ks == [2, 4]
    assert s.k_at(0.0) == 4 and s.k_at(0.34) == 4
    assert s.k_at(0.35) == 2 and s.k_at(0.59) == 2
    assert s.k_at(0.6) == 1 and s.k_at(0.999) == 1 and s.k_at(1.0) == 1
    assert ZoomSchedule.parse("1").ks == []


def test_lr_schedule_shape():
    assert lr_at(0.0, 1.0) == 0.0
    assert abs(lr_at(0.05, 1.0) - 1.0) < 1e-9
    assert lr_at(0.5, 1.0) < 1.0
    assert abs(lr_at(1.0, 1.0) - 0.1) < 1e-6


def test_merge_unmerge_shapes_and_identity_init():
    m, u = SlotMerge(4, 8), SlotUnmerge(4, 8)
    x = torch.randn(2, 5, 4, 8)
    z = m(x)
    assert z.shape == (2, 5, 8)
    assert torch.allclose(z, x.mean(-2))          # init = plain mean
    y = u(z)
    assert y.shape == (2, 5, 4, 8)
    assert torch.allclose(y[:, :, 0], z)          # init = broadcast copy


def test_gpt_causality_at_every_zoom_level():
    torch.manual_seed(0)
    m = ZoomGPT(512, 32, 2, 2, zoom_ks=[2, 4]).eval()
    x = torch.randint(0, 512, (1, 32))
    x2 = x.clone()
    x2[0, 20:] = 7
    for k in (1, 2, 4):
        h1, h2 = m.hidden(x, k), m.hidden(x2, k)
        n_same = 20 // k                              # groups entirely before token 20
        assert torch.allclose(h1[:, :n_same], h2[:, :n_same], atol=1e-5)
        assert not torch.allclose(h1[:, n_same], h2[:, n_same])


def test_gpt_loss_and_grads_each_k():
    torch.manual_seed(0)
    for out_mode in ("slot", "bag"):
        m = ZoomGPT(512, 32, 2, 2, zoom_ks=[2, 4], out_mode=out_mode)
        x = torch.randint(0, 512, (3, 36))
        for k in (1, 2, 4):
            loss = m(x[:, :32], x[:, k:32 + k], k=k)
            assert torch.isfinite(loss)
            m.zero_grad()
            loss.backward()
            for kk, mod in m.merge.items():
                assert (mod.gate.grad is not None) == (int(kk) == k)


def test_gpt_flops_monotone_in_k():
    m = ZoomGPT(512, 32, 2, 2, zoom_ks=[2, 4])
    f = [m.flops_per_step(4, 64, k) for k in (1, 2, 4)]
    assert f[0] > f[1] > f[2]


def test_vit_shapes_and_grouping_locality():
    m = ZoomViT(5, 3, (16, 32), 2, 16, 1, 2, zoom_ks=[4, 16])
    x = torch.randn(2, 5, 16, 32)
    for k in (1, 4, 16):
        assert m(x, k).shape == (2, 3, 16, 32)
    t = torch.randn(1, 8, 16, 4)
    g = ZoomViT._group(t, 2)
    assert torch.equal(ZoomViT._ungroup(g, 8, 16, 2), t)
    assert torch.equal(g[0, 0], torch.stack([t[0, 0, 0], t[0, 0, 1], t[0, 1, 0], t[0, 1, 1]]))


def test_vit_lowres_path():
    m = ZoomViT(5, 3, (16, 32), 2, 16, 1, 2)
    x = torch.randn(2, 5, 8, 16)
    assert m.forward_lowres(x, 2).shape == (2, 3, 8, 16)
