"""ZoomGPT inference API: logits from forward, KV-cache generate and its edge cases."""
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.models import ZoomGPT  # noqa: E402


def _model(vocab=64, seed=0, **kw):
    """Tiny model with well-separated logits (embedding std 1), so greedy decoding
    is not decided by float noise."""
    torch.manual_seed(seed)
    m = ZoomGPT(vocab, 32, 2, 2, **kw).eval()
    nn.init.normal_(m.emb.weight, std=1.0)
    return m


def _reference(m, idx, n_new, max_ctx):
    """No-cache greedy decoding: re-encode the last max_ctx tokens every step."""
    for _ in range(n_new):
        logits = m(idx[:, -max_ctx:])[:, -1]
        idx = torch.cat([idx, logits.argmax(-1, keepdim=True)], 1)
    return idx


def test_forward_without_targets_returns_logits():
    torch.manual_seed(0)
    for out_mode in ("slot", "bag"):
        m = ZoomGPT(50, 32, 1, 2, zoom_ks=[2], out_mode=out_mode)
        x = torch.randint(0, 50, (3, 18))
        z = m(x[:, :16])
        assert z.shape == (3, 16, 50)
        assert torch.allclose(z, m.logits(m.hidden(x[:, :16])))
        assert torch.allclose(F.cross_entropy(z.reshape(-1, 50), x[:, 1:17].reshape(-1)),
                              m(x[:, :16], x[:, 1:17]))
        z2 = m(x[:, :16], k=2)
        if out_mode == "slot":                 # row i predicts token i + k
            assert z2.shape == (3, 16, 50)
            ref = F.cross_entropy(z2.reshape(-1, 50), x[:, 2:18].reshape(-1))
        else:                                  # one distribution per group
            assert z2.shape == (3, 8, 50)
            tgt = x[:, 2:18].reshape(3, 8, 2)
            ref = -F.log_softmax(z2, -1).gather(-1, tgt).mean()
        assert torch.allclose(ref, m(x[:, :16], x[:, 2:18], k=2), atol=1e-6)


@pytest.mark.parametrize("kw", [{}, {"qk_norm": True, "softcap": 15.0, "mlp": "relu2"},
                                {"rope_pos": "patch", "zoom_ks": [2]}])
def test_generate_kv_cache_matches_no_cache_reference(kw):
    m = _model(**kw)
    prompt = torch.randint(0, 64, (3, 7))
    for max_ctx, n_new in ((40, 20), (12, 20)):          # within the window, then sliding
        ref = _reference(m, prompt, n_new, max_ctx)
        out = m.generate(prompt, n_new, temperature=0, max_ctx=max_ctx)
        assert torch.equal(out, ref)
        assert torch.equal(m.generate(prompt, n_new, temperature=0, max_ctx=max_ctx,
                                      use_cache=False), ref)
    # the cached step logits match a full re-encode, not just their argmax
    seq = _reference(m, prompt, 5, 64)
    from bantamtrain.models.gpt import KVCache
    caches = [KVCache(64) for _ in m.blocks]
    with torch.no_grad():
        m._decode(seq[:, :8], None, caches, 0)
        for t in range(8, seq.shape[1]):
            h = m._decode(seq[:, t:t + 1], None, caches, t)
            assert torch.allclose(m.logits(h[:, -1]), m(seq[:, :t + 1])[:, -1], atol=1e-4)


def test_generate_ragged_list_matches_single_rows():
    m = _model()
    prompts = [[5, 9, 2, 33, 7, 1], [4], [60, 61, 62]]
    out = m.generate(prompts, 9, temperature=0, max_ctx=10)
    for p, o in zip(prompts, out):
        assert o[:len(p)] == p and len(o) == len(p) + 9
        assert o == m.generate(torch.tensor([p]), 9, temperature=0, max_ctx=10)[0].tolist()


def test_generate_default_window_is_block_size():
    m = _model(block_size=8)
    prompt = torch.randint(0, 64, (1, 20))                # longer than block_size
    assert torch.equal(m.generate(prompt, 6, temperature=0), _reference(m, prompt, 6, 8))
    m2 = _model(block_size=300)
    long = torch.randint(0, 64, (1, 400))
    assert torch.equal(m2.generate(long, 2, temperature=0), _reference(m2, long, 2, 300))
    assert ZoomGPT(16, 16, 1, 2).block_size == 256        # old default window


def test_generate_edge_cases():
    m = _model(vocab=16)
    p = torch.randint(0, 16, (3, 4))
    g = torch.Generator().manual_seed(0)
    assert m.generate(p, 5, generator=g).shape == (3, 9)          # default top_k=40 > vocab
    for top_k in (None, 0):
        assert m.generate(p, 3, top_k=top_k, top_p=0.9, generator=g).shape == (3, 7)
    greedy = m.generate(p, 5, temperature=0)
    assert torch.equal(greedy, m.generate(p, 5, temperature=0.0, top_k=3))
    assert torch.equal(greedy, _reference(m, p, 5, 256))
    with pytest.raises(ValueError, match="empty"):
        m.generate(torch.zeros(1, 0, dtype=torch.long), 3)
    with pytest.raises(ValueError, match="empty"):
        m.generate([[1, 2], []], 3)
    with pytest.raises(ValueError):
        m.generate(torch.tensor([[1, 99]]), 3)                  # id outside the vocabulary
    # the RoPE cache stays one table per (k, device)
    m.generate(p, 30, temperature=0)
    assert len(m._rope_cache) == 1


def _fixed_logits(m, table):
    """Make the head return fixed per-row logits (rows of `table`), whatever the input."""
    table = torch.as_tensor(table, dtype=torch.float32)

    def logits(h):
        return table[:h.shape[0]].expand(*h.shape[:-1], table.shape[-1]).clone()
    m.logits = logits
    return m


def test_generate_eos_per_row():
    V = 8
    m = _model(vocab=V)
    big = torch.full((2, V), -10.0)
    big[0, 0] = 10.0                                   # row 0 wants eos (0) at once
    big[1, 3] = 10.0                                   # row 1 never emits eos
    _fixed_logits(m, big)
    p = torch.tensor([[1, 2], [2, 1]])
    out = m.generate(p, 4, temperature=0, eos_id=0)
    assert out.tolist() == [[1, 2, 0, 0, 0, 0], [2, 1, 3, 3, 3, 3]]  # finished row padded
    big[1] = big[0]                                    # now every row ends at once
    out = m.generate(p, 4, temperature=0.7, eos_id=0)
    assert out.tolist() == [[1, 2, 0], [2, 1, 0]]
    assert m.generate([[1, 2, 3], [4]], 4, temperature=0, eos_id=0) == [[1, 2, 3, 0], [4, 0]]


def test_generate_top_k_top_p_repetition_penalty():
    V = 8
    m = _model(vocab=V)
    row = torch.tensor([4.0, 3.9, 0.0, -1, -1, -1, -1, -1])
    _fixed_logits(m, row[None].repeat(64, 1))
    p = torch.tensor([[5]] * 64)
    g = torch.Generator().manual_seed(0)
    assert set(m.generate(p, 1, temperature=1.0, top_k=2, generator=g)[:, 1].tolist()) == {0, 1}
    assert set(m.generate(p, 1, temperature=1.0, top_p=0.3, generator=g)[:, 1].tolist()) == {0}
    assert m.generate(p[:1], 1, temperature=0)[0, 1] == 0
    # token 0 already in the prompt: penalised 4.0 / 2 < 3.9, so greedy picks token 1
    assert m.generate(torch.tensor([[0]]), 1, temperature=0, repetition_penalty=2.0)[0, 1] == 1
    assert m.generate(torch.tensor([[5]]), 1, temperature=0, repetition_penalty=2.0)[0, 1] == 0
