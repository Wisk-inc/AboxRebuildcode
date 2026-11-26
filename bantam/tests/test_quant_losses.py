"""chunked_cross_entropy (contract C5): scale, softcap, bias, token weights, empty chunks,
quantized heads."""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import qweights  # noqa: E402
from bantamtrain.losses import chunked_cross_entropy  # noqa: E402


def _reference(h, w, y, bias, softcap, scale, tw):
    z = scale * F.linear(h, w, bias).float()
    if softcap:
        z = softcap * torch.tanh(z / softcap)
    per = F.cross_entropy(z.reshape(-1, z.shape[-1]), y.reshape(-1), ignore_index=-100,
                          reduction="none")
    wts = (y.reshape(-1) != -100).float() * (1.0 if tw is None else tw.reshape(-1))
    return (per * wts).sum() / wts.sum()


def test_scale_softcap_bias_and_token_weights_match_reference():
    torch.manual_seed(0)
    for softcap, scale, weighted in ((0.0, 1.0, False), (5.0, 0.25, True), (30.0, 8.0, True)):
        h = torch.randn(2, 37, 16, requires_grad=True)
        w = torch.randn(50, 16, requires_grad=True)
        b = torch.randn(50, requires_grad=True)
        y = torch.randint(0, 50, (2, 37))
        y[0, :20] = -100
        tw = torch.rand(2, 37) if weighted else None
        ref = _reference(h, w, y, b, softcap, scale, tw)
        grads = torch.autograd.grad(ref, (h, w, b))
        out = chunked_cross_entropy(h, w, y, chunk=8, bias=b, softcap=softcap, scale=scale,
                                    token_weights=tw)
        assert abs(out.item() - ref.item()) < 1e-5
        for g, r in zip(torch.autograd.grad(out, (h, w, b)), grads):
            assert torch.allclose(g, r, atol=1e-5)


def test_unsupervised_chunks_give_zero_not_nan():
    torch.manual_seed(0)
    h = torch.randn(40, 8, requires_grad=True)
    w = torch.randn(30, 8)
    y = torch.full((40,), -100)
    out = chunked_cross_entropy(h, w, y, chunk=16)
    out.backward()
    assert out.item() == 0.0 and torch.isfinite(h.grad).all()
    y[35:] = 3                                        # only the last chunk is supervised
    full = F.cross_entropy(F.linear(h, w), y, ignore_index=-100)
    assert abs(chunked_cross_entropy(h, w, y, chunk=16).item() - full.item()) < 1e-5
    zero_w = chunked_cross_entropy(h, w, y, chunk=16, token_weights=torch.zeros(40))
    assert zero_w.item() == 0.0
    with torch.no_grad():                             # eval path without checkpointing
        assert abs(chunked_cross_entropy(h, w, y, chunk=16).item() - full.item()) < 1e-5


def test_quantized_head_module_is_used_per_chunk():
    """A QLinear / tied head is dequantized per chunk instead of passing a float table around."""
    torch.manual_seed(0)
    head = qweights.QLinear(nn.Linear(32, 300), "nf4")
    h = torch.randn(3, 20, 32, requires_grad=True)
    y = torch.randint(0, 300, (3, 20))
    ref = chunked_cross_entropy(h, head.weight, y, chunk=7, bias=head.bias)
    (g_ref,) = torch.autograd.grad(ref, h)
    out = chunked_cross_entropy(h, head, y, chunk=7, bias=head.bias)
    (g,) = torch.autograd.grad(out, h)
    assert abs(out.item() - ref.item()) < 1e-5 and torch.allclose(g, g_ref, atol=1e-5)
    emb = qweights.QEmbedding(nn.Embedding(300, 32), 8)
    tied = qweights.QTiedHead(emb)
    assert torch.allclose(chunked_cross_entropy(h, tied, y, chunk=7),
                          chunked_cross_entropy(h, emb.weight, y, chunk=7), atol=1e-5)
    # a bf16 table with fp32 hidden states (no autocast) is cast, not an error
    out16 = chunked_cross_entropy(h, emb.weight.bfloat16(), y, chunk=7)
    assert torch.isfinite(out16)
