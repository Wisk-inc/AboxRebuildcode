"""Parameter routing (Muon / AdamW, weight decay by role, lr_scale groups, Muon splits and
transposed Conv1D weights) and optimizer FLOPs."""
import copy
import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import lora  # noqa: E402
from bantamtrain.models import ZoomGPT  # noqa: E402
from bantamtrain.optim import (AdamWLowMem, Muon, build_optimizer, newton_schulz,  # noqa: E402
                               ns_flops, optimizer_flops)


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
                      tie_word_embeddings=False)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


def _tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=1,
                                      n_head=2, bos_token_id=0, eos_token_id=0))


def _wd_of(opt, model):
    name = {id(p): n for n, p in model.named_parameters()}
    return {name[id(p)]: g["weight_decay"] for g in opt.param_groups for p in g["params"]}


def test_weight_decay_by_role_on_hf_llama():
    m = _tiny_llama()
    wd = _wd_of(build_optimizer(m, "adamw", 1e-3, 0.1, fused=False), m)
    assert wd["model.layers.0.mlp.gate_proj.weight"] == 0.1        # not a "gate" of ZoomGPT
    assert wd["model.layers.0.mlp.up_proj.weight"] == 0.1
    assert wd["model.embed_tokens.weight"] == 0.1                  # default: embeddings decayed
    for n in ("model.layers.0.input_layernorm.weight", "model.norm.weight"):
        assert wd[n] == 0.0
    wd = _wd_of(build_optimizer(m, "adamw", 1e-3, 0.1, fused=False, decay_embeddings=False), m)
    assert wd["model.embed_tokens.weight"] == 0.0 and wd["lm_head.weight"] == 0.1
    lora.apply_lora(m, r=4, alpha=8, targets=("gate_proj", "up_proj"))
    wd = _wd_of(build_optimizer(m, "adamw", 1e-3, 0.1, fused=False), m)
    assert wd and all(v == 0.1 for v in wd.values())               # gate_proj adapters too


def test_lora_without_muon_lora_warns_and_muon_lora_routes_adapters():
    m = _tiny_llama()
    lora.apply_lora(m, r=4, alpha=8, targets=("q_proj", "v_proj"))
    with pytest.warns(UserWarning, match="no trainable parameter qualifies for Muon"):
        opt = build_optimizer(m, "muon", 1e-3, 0.0, state_bits=8, fused=False)
    assert [n for n, _ in opt.summary()] == ["AdamWLowMem"]
    opt = build_optimizer(m, "muon", 1e-3, 0.0, state_bits=8, fused=False, muon_lora=True)
    adapters = sum(p.numel() for p in lora.lora_parameters(m))
    assert dict(opt.summary())["Muon"] == adapters


def test_muon_routes_gpt2_conv1d_weights():
    m = _tiny_gpt2()
    opt = build_optimizer(m, "muon", 1e-3, 0.1, fused=False)
    muon = {id(p) for g in opt.optimizers[1].param_groups for p in g["params"]}
    blk = m.transformer.h[0]
    for mod in (blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj):
        assert id(mod.weight) in muon
    assert id(m.transformer.wte.weight) not in muon
    x = torch.randint(0, 64, (2, 16))
    m(input_ids=x, labels=x).loss.backward()
    opt.step()
    assert all(torch.isfinite(p).all() for p in m.parameters())


@pytest.mark.parametrize("split,scale", [(1, "jordan"), (2, "moonlight"), (2, "jordan")])
def test_muon_transposed_weight_matches_linear_layout(split, scale):
    """A (in, out) Conv1D weight gets exactly the update of the (out, in) Linear weight:
    blocks split along the output dim, scale computed on (out, in)."""
    torch.manual_seed(0)
    before = torch.randn(48, 24)                          # (out, in)
    w_lin = nn.Parameter(before.clone())
    w_conv = nn.Parameter(before.T.contiguous())          # (in, out)
    g = torch.randn(48, 24)
    w_lin.grad, w_conv.grad = g.clone(), g.T.contiguous()
    kw = dict(lr=0.1, momentum=0.0, nesterov=False, scale=scale)
    Muon([w_lin], splits={w_lin: split}, **kw).step()
    Muon([w_conv], splits={w_conv: split}, transposed=[w_conv], **kw).step()
    rows = 48 // split
    exp = torch.cat([newton_schulz(b) for b in g.chunk(split, 0)]) * Muon._scale(rows, 24, scale)
    assert torch.allclose(before - w_lin.detach(), 0.1 * exp, atol=1e-5)
    assert torch.allclose(w_conv.detach().T, w_lin.detach(), atol=1e-6)


def test_muon_splits_method_is_used():
    torch.manual_seed(0)
    m = ZoomGPT(64, 32, 1, 2)
    qkv = m.blocks[0].qkv.weight
    hidden = m.muon_params()
    m.muon_params = lambda: hidden                      # ZoomGPT's own would re-tag qkv
    del qkv.muon_split                                  # only the model-level map remains
    m.muon_splits = lambda: {"blocks.0.qkv.weight": 3}
    opt = build_optimizer(m, "muon", 0.1, 0.0, fused=False)
    assert not hasattr(qkv, "muon_split")
    assert opt.optimizers[1].layout(qkv) == (3, False)
    # the map survives a deep copy of the optimizer, whose param copies have no attribute
    twin = copy.deepcopy(opt)
    q2 = twin.optimizers[1].param_groups[0]["params"][0]
    assert q2 is not qkv and q2.shape == qkv.shape and twin.optimizers[1].layout(q2) == (3, False)
    x = torch.randint(0, 64, (2, 17))
    m(x[:, :16], x[:, 1:], k=1).backward()
    g, before = qkv.grad.clone(), qkv.detach().clone()
    opt.step()
    g = g.lerp(g * 0.05, 0.95)                          # first Nesterov step: 0.0975 g
    exp = torch.cat([newton_schulz(b) for b in g.chunk(3, 0)]) * Muon._scale(32, 32, "moonlight")
    assert torch.allclose(before - qkv.detach(), 0.1 * exp, atol=1e-5)


def test_lr_scale_groups():
    m = ZoomGPT(64, 32, 1, 2, zoom_ks=[2])

    def scale(n, p):
        return 0.5 if n == "emb.weight" else 2.0 if n == "blocks.0.fc.weight" else 1.0
    name = {id(p): n for n, p in m.named_parameters()}
    for optim in ("adamw", "muon"):
        for bits in (32, 8):
            opt = build_optimizer(m, optim, 1e-3, 0.1, lr_scale=scale, state_bits=bits,
                                  fused=False)
            seen = set()
            for g in opt.param_groups:
                assert g["lr"] == pytest.approx(1e-3 * g["lr_scale"])
                for p in g["params"]:
                    assert g["lr_scale"] == scale(name[id(p)], p)
                    seen.add(id(p))
            assert seen == {id(p) for p in m.parameters()}
            first = opt.optimizers[0].param_groups[:2]       # decay / no-decay at scale 1
            assert [g["lr_scale"] for g in first] == [1.0, 1.0]
            assert [g["weight_decay"] for g in first] == [0.1, 0.0]
    opt = build_optimizer(m, "muon", 1e-3, 0.1, fused=False)      # default layout unchanged
    assert [len(o.param_groups) for o in opt.optimizers] == [2, 1]
    assert all(g["lr_scale"] == 1.0 for g in opt.param_groups)


def test_ns_flops_match_counted_matmuls():
    from torch.utils.flop_counter import FlopCounterMode
    for shape in ((3, 32, 32), (64, 16), (16, 96)):
        with FlopCounterMode(display=False) as fc:
            newton_schulz(torch.randn(*shape), steps=5)
        rows, cols = shape[-2:]
        est = ns_flops(rows, cols, 5, shape[0] if len(shape) == 3 else 1)
        assert fc.get_total_flops() <= est < 1.15 * fc.get_total_flops()


def test_optimizer_flops():
    m = ZoomGPT(64, 32, 2, 2, zoom_ks=[2])
    opt = build_optimizer(m, "muon", 1e-3, 0.1, fused=False)
    adam_n = sum(p.numel() for g in opt.optimizers[0].param_groups for p in g["params"])
    muon_p = opt.optimizers[1].param_groups[0]["params"]
    exp = 12 * adam_n + sum(10 * p.numel() for p in muon_p)
    for p in muon_p:
        s = getattr(p, "muon_split", 1)
        exp += ns_flops(p.shape[0] // s, p.shape[1], 5, s)
    assert optimizer_flops(opt) == pytest.approx(exp)
    qkv = m.blocks[0].qkv.weight                        # q, k, v: three (32, 32) blocks
    assert opt.optimizers[1].param_flops(qkv, opt.optimizers[1].param_groups[0]) \
        == 3 * ns_flops(32, 32) + 10 * qkv.numel() != ns_flops(96, 32) + 10 * qkv.numel()
    assert optimizer_flops(opt, with_grad_only=True) == 0
    m.emb.weight.requires_grad_(False)                  # frozen params count 0
    assert optimizer_flops(opt) == pytest.approx(exp - 12 * m.emb.weight.numel())
    w = nn.Parameter(torch.randn(10, 10))
    assert optimizer_flops(torch.optim.AdamW([w])) == 1200
    assert optimizer_flops(AdamWLowMem([w])) == 1200
    assert optimizer_flops(torch.optim.SGD([w], lr=0.1)) == 200
    assert optimizer_flops(torch.optim.SGD([w], lr=0.1, momentum=0.9)) == 400
    assert isinstance(optimizer_flops(Muon([w])), float)


def test_adamw_lowmem32_matches_torch_adamw():
    torch.manual_seed(0)
    shapes = [(37, 19), (5,), (), (300, 3)]
    p1 = [nn.Parameter(torch.randn(s)) for s in shapes]
    p2 = [nn.Parameter(p.detach().clone()) for p in p1]

    def groups(ps):
        return [{"params": ps[:2], "weight_decay": 0.1}, {"params": ps[2:], "weight_decay": 0.0}]
    o1 = AdamWLowMem(groups(p1), lr=3e-3, state_bits=32, chunk=256)
    o2 = torch.optim.AdamW(groups(p2), lr=3e-3, betas=(0.9, 0.95), eps=1e-8)
    for t in range(60):
        for o in (o1, o2):
            for g in o.param_groups:
                g["lr"] = 3e-3 * (0.5 + 0.5 * math.cos(math.pi * t / 60))
        for a, b, s in zip(p1, p2, shapes):
            a.grad = torch.randn(s) * (t % 7 + 0.1)
            b.grad = a.grad.clone()
        if t % 10 == 3:                                  # a parameter without grad this step
            p1[1].grad = p2[1].grad = None
        o1.step()
        o2.step()
    assert max((a - b).abs().max().item() for a, b in zip(p1, p2)) < 1e-6
