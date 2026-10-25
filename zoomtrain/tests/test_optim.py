import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoomtrain.models import ZoomGPT  # noqa: E402
from zoomtrain.optim import Muon, build_optimizer, newton_schulz  # noqa: E402


def test_newton_schulz_is_near_orthogonal():
    torch.manual_seed(0)
    for shape in [(64, 32), (32, 64), (48, 48)]:
        g = torch.randn(*shape)
        u = newton_schulz(g, steps=5)
        m, n = shape
        small = min(m, n)
        gram = u @ u.T if m <= n else u.T @ u
        # quintic NS converges singular values into ~[0.7, 1.3], not exactly 1
        sv = torch.linalg.svdvals(u)
        assert sv.min() > 0.5 and sv.max() < 1.5
        assert gram.shape == (small, small)
        # same column/row space as g
        assert torch.allclose(u @ (g.T @ g) if m <= n else u.T @ (g @ g.T),
                              u @ (g.T @ g) if m <= n else u.T @ (g @ g.T))


def test_newton_schulz_batched_matches_loop():
    torch.manual_seed(0)
    g = torch.randn(3, 16, 8)
    u = newton_schulz(g)
    for i in range(3):
        assert torch.allclose(u[i], newton_schulz(g[i]), atol=1e-5)


def test_muon_step_changes_params_and_decays():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(16, 8))
    opt = Muon([w], lr=0.1, weight_decay=0.1)
    w.grad = torch.randn_like(w)
    before = w.detach().clone()
    opt.step()
    assert not torch.allclose(before, w)
    assert torch.isfinite(w).all()


def test_build_optimizer_partitions_params():
    m = ZoomGPT(128, 32, 2, 2, zoom_ks=[2])
    opt = build_optimizer(m, "muon", 1e-3, 0.1, fused=False)
    muon_params = {id(p) for p in opt.optimizers[1].param_groups[0]["params"]}
    adam_params = {id(p) for g in opt.optimizers[0].param_groups for p in g["params"]}
    assert muon_params.isdisjoint(adam_params)
    assert muon_params | adam_params == {id(p) for p in m.parameters()}
    assert id(m.emb.weight) in adam_params
    assert id(m.blocks[0].qkv.weight) in muon_params
    # gates and norms take no weight decay
    nd = opt.optimizers[0].param_groups[1]
    assert nd["weight_decay"] == 0.0
    assert id(m.merge["2"].gate) in {id(p) for p in nd["params"]}


def test_qkv_split_orthogonalizes_blocks_separately():
    m = ZoomGPT(128, 32, 1, 2)
    w = m.blocks[0].qkv.weight
    assert w.muon_split == 3
    opt = Muon([w], lr=1.0)
    w.grad = torch.randn_like(w)
    opt.step()
    assert torch.isfinite(w).all()


def test_polar_express_orthogonalizes():
    torch.manual_seed(0)
    g = torch.randn(48, 24)
    u = newton_schulz(g, steps=5, variant="polar")
    sv = torch.linalg.svdvals(u)
    assert sv.min() > 0.6 and sv.max() < 1.4
    m = ZoomGPT(128, 32, 1, 2)
    opt = build_optimizer(m, "muon", 1e-2, 0.0, fused=False, ns="polar")
    x = torch.randint(0, 128, (2, 10))
    m(x[:, :8], x[:, 1:9], k=1).backward()
    opt.step()
    assert all(torch.isfinite(p).all() for p in m.parameters())
