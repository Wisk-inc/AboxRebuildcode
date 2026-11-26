import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.models import ZoomGPT  # noqa: E402
from bantamtrain.optim import Muon, build_optimizer, newton_schulz  # noqa: E402


def test_newton_schulz_is_near_orthogonal():
    torch.manual_seed(0)
    for shape in [(64, 32), (32, 64), (48, 48)]:
        g = torch.randn(*shape)
        u = newton_schulz(g, steps=5)
        # quintic NS converges singular values into ~[0.7, 1.3], not exactly 1
        sv = torch.linalg.svdvals(u)
        assert sv.min() > 0.5 and sv.max() < 1.5
        _assert_polar_factor_of(u, g)


def _assert_polar_factor_of(u, g, rel=0.35):
    """u ~ U V^T of g = U S V^T: close to the polar factor, and diagonal with positive
    entries in g's singular bases (same singular vectors, no sign or subspace error)."""
    U, S, Vh = torch.linalg.svd(g, full_matrices=False)
    P = U @ Vh
    assert (u - P).norm() / P.norm() < rel
    D = U.T @ u @ Vh.T
    off = D - torch.diag(torch.diagonal(D))
    assert off.norm() < 1e-3 * D.norm()
    assert torch.diagonal(D).min() > 0 and torch.diagonal(D).mean() > 0.7


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


def test_linear_merge_maps_not_decayed():
    from bantamtrain.models.vit_weather import ZoomViT
    vit = ZoomViT(4, 4, (8, 16), 2, 32, 1, 2, zoom_ks=(4,), merge_mode="linear")
    gpt = ZoomGPT(128, 32, 1, 2, zoom_ks=[2], merge_mode="linear")
    for model, k in ((gpt, "2"), (vit, "4")):
        merge_ids = {id(model.merge[k].weight), id(model.unmerge[k].weight)}
        for name in ("adamw", "muon"):
            opt = build_optimizer(model, name, 1e-3, 0.1, fused=False)
            for g in opt.param_groups:
                if merge_ids & {id(p) for p in g["params"]}:
                    assert g["weight_decay"] == 0.0
            decayed = {id(p) for p in opt.optimizers[0].param_groups[0]["params"]}
            assert decayed.isdisjoint(merge_ids)


def test_qkv_split_orthogonalizes_blocks_separately():
    torch.manual_seed(0)
    m = ZoomGPT(128, 32, 1, 2)
    w = m.blocks[0].qkv.weight
    assert w.muon_split == 3
    opt = Muon([w], lr=1.0, momentum=0.0, nesterov=False)
    w.grad = torch.randn_like(w)
    before = w.detach().clone()
    opt.step()
    u = before - w.detach()
    # expected: each of q, k, v (32 x 32) orthogonalized on its own, Moonlight scale
    exp = torch.cat([newton_schulz(b) for b in w.grad.chunk(3, 0)])
    exp = exp * Muon._scale(32, 32, "moonlight")
    assert torch.allclose(u, exp, atol=1e-5)
    for blk, gb in zip(u.chunk(3, 0), w.grad.chunk(3, 0)):
        sv = torch.linalg.svdvals(blk)
        assert sv.max() / sv.min() < 10.0              # 30-800 if the split were ignored
        _assert_polar_factor_of(blk, gb)


def test_polar_express_orthogonalizes():
    torch.manual_seed(0)
    g = torch.randn(48, 24)
    u = newton_schulz(g, steps=5, variant="polar")
    sv = torch.linalg.svdvals(u)
    assert sv.min() > 0.6 and sv.max() < 1.4
    _assert_polar_factor_of(u, g)
    # the full 8-tuple schedule converges: more steps -> singular values at 1
    for steps in (8, 10):
        u = newton_schulz(torch.randn(256, 128), steps=steps, variant="polar")
        sv = torch.linalg.svdvals(u)
        assert sv.min() > 0.99 and sv.max() < 1.01
        assert (u.T @ u - torch.eye(128)).abs().max() < 0.02     # Gram matrix ~ I
    m = ZoomGPT(128, 32, 1, 2)
    opt = build_optimizer(m, "muon", 1e-2, 0.0, fused=False, ns="polar")
    x = torch.randint(0, 128, (2, 10))
    m(x[:, :8], x[:, 1:9], k=1).backward()
    opt.step()
    assert all(torch.isfinite(p).all() for p in m.parameters())
