"""finetune loss: logit transforms on the chunked head, token weighting, NaN safety,
response-only eval, token-weighted grad accumulation and the DDP loss module."""
import inspect
import types

import pytest
import torch
import torch.nn.functional as F
from test_finetune_fixtures import make_tokenizer, rows, tiny_llama

from bantamtrain import finetune as ft
from bantamtrain import lora, losses, qweights
from bantamtrain.device import setup_runtime

KW = dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)


def _model(name):
    import transformers as tf
    torch.manual_seed(0)
    if name == "llama":
        return tf.LlamaForCausalLM(tf.LlamaConfig(**KW))
    if name == "gemma2":             # embeddings x40: logits of trained-model size, so the softcap bites
        m = tf.Gemma2ForCausalLM(tf.Gemma2Config(**KW, head_dim=16, final_logit_softcapping=30.0,
                                                 query_pre_attn_scalar=16))
        with torch.no_grad():
            m.model.embed_tokens.weight.mul_(40)
        return m
    if name == "granite":
        m = tf.GraniteForCausalLM(tf.GraniteConfig(**KW, logits_scaling=8.0))
    elif name == "cohere":
        m = tf.CohereForCausalLM(tf.CohereConfig(**KW, logit_scale=0.0625, tie_word_embeddings=False))
    elif name == "phi":
        cfg = {k: v for k, v in KW.items() if k != "num_key_value_heads"}
        m = tf.PhiForCausalLM(tf.PhiConfig(**cfg))
        with torch.no_grad():
            m.lm_head.bias.normal_(0, 1.0)
        return m
    else:
        return tf.GPT2LMHeadModel(tf.GPT2Config(vocab_size=300, n_positions=64, n_embd=64, n_layer=2, n_head=4))
    with torch.no_grad():
        m.lm_head.weight.mul_(30)
    return m


def _ref(m, x, y):
    logits = m(input_ids=x).logits.float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)


@pytest.mark.parametrize("name", ["llama", "gemma2", "granite", "cohere", "phi", "gpt2"])
def test_lm_loss_matches_model_logits_for_head_transforms(name):
    m = _model(name).eval()
    torch.manual_seed(1)
    x = torch.randint(0, 300, (2, 17))
    xs, ys = x[:, :-1], x[:, 1:].clone()
    ys[0, :4] = -100
    ref = _ref(m, xs, ys)
    loss = ft.lm_loss(m, xs, ys, 16)
    assert abs(loss.item() - ref.item()) < 1e-4 * max(1.0, ref.item())
    assert m.__dict__["_bantam_chunk_ok"]                 # the chunked head, not the fallback
    g = torch.autograd.grad(loss, m.model.norm.weight if hasattr(m, "model") and hasattr(m.model, "norm")
                            else next(m.parameters()))[0]
    assert torch.isfinite(g).all()


def test_head_transform_reads_each_family_config():
    def cfg(**kw):
        return types.SimpleNamespace(config=types.SimpleNamespace(**kw))
    assert ft.head_transform(cfg()) == (1.0, 0.0)
    assert ft.head_transform(cfg(final_logit_softcapping=30.0)) == (1.0, 30.0)
    assert ft.head_transform(cfg(final_logit_softcapping=None)) == (1.0, 0.0)
    assert ft.head_transform(cfg(logit_scale=0.0625)) == (0.0625, 0.0)
    assert ft.head_transform(cfg(logits_scaling=8.0)) == (0.125, 0.0)
    assert ft.head_transform(cfg(lm_head_multiplier=0.5)) == (0.5, 0.0)


def test_unknown_head_transform_falls_back_to_the_models_logits(capsys):
    from transformers import LlamaConfig, LlamaForCausalLM

    class Scaled(LlamaForCausalLM):            # a post-head step no config field describes
        def forward(self, *a, **k):
            out = super().forward(*a, **k)
            out.logits = out.logits * 3.0
            return out
    torch.manual_seed(0)
    m = Scaled(LlamaConfig(**KW)).eval()
    x = torch.randint(0, 300, (2, 12))
    assert abs(ft.lm_loss(m, x[:, :-1], x[:, 1:], 16).item() - _ref(m, x[:, :-1], x[:, 1:]).item()) < 1e-5
    assert m.__dict__["_bantam_chunk_ok"] is False and "full logits" in capsys.readouterr().out


def test_weighted_loss_chunked_equals_full_and_counts():
    m = _model("llama")
    torch.manual_seed(2)
    x = torch.randint(0, 300, (2, 20))
    y = x[:, 1:].clone()
    w = torch.ones_like(y, dtype=torch.float)
    w[:, :6] = 0.3
    y[1, -3:] = -100
    a = ft.lm_loss(m, x[:, :-1], y, 16, w)
    ga = torch.autograd.grad(a, m.model.layers[0].mlp.up_proj.weight)[0]
    b = ft.lm_loss(m, x[:, :-1], y, 0, w)
    gb = torch.autograd.grad(b, m.model.layers[0].mlp.up_proj.weight)[0]
    per = F.cross_entropy(m(input_ids=x[:, :-1]).logits.reshape(-1, 300), y.reshape(-1),
                          ignore_index=-100, reduction="none")
    wv = (w * (y != -100)).reshape(-1)
    assert abs(a.item() - (per * wv).sum().item() / wv.sum().item()) < 1e-5
    assert abs(a.item() - b.item()) < 1e-5 and torch.allclose(ga, gb, atol=1e-6)
    s, n = ft.lm_loss_sum(m, x[:, :-1], y, 16, w)
    assert abs(s.item() - a.item() * n.item()) < 1e-4 and abs(n.item() - wv.sum().item()) < 1e-5


@pytest.mark.skipif("token_weights" not in inspect.signature(losses.chunked_cross_entropy).parameters,
                    reason="losses.chunked_cross_entropy has no token_weights")
def test_weighted_loss_uses_the_chunked_head_when_supported(monkeypatch):
    m = _model("llama")
    calls = []
    real = ft.chunked_cross_entropy
    monkeypatch.setattr(ft, "chunked_cross_entropy", lambda *a, **k: calls.append(1) or real(*a, **k))
    x = torch.randint(0, 300, (2, 20))
    w = torch.full((2, 19), 0.3)
    ft.lm_loss(m, x[:, :-1], x[:, 1:], 16, w)
    assert calls


@pytest.mark.parametrize("chunk", [0, 16])
def test_all_ignored_batch_gives_zero_not_nan(chunk):
    m = _model("llama")
    x = torch.randint(0, 300, (2, 10))
    y = torch.full((2, 9), -100)
    loss = ft.lm_loss(m, x[:, :-1], y, chunk)
    loss.backward()
    assert loss.item() == 0.0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def _windows(tok, plw, mode="bfd", seq=32):
    items = ft.tokenize_examples(tok, rows(8, seed=3), seq, plw, truncate=mode != "wrapped")
    return ft.pack(items, seq, tok.pad_token_id, mode)


def test_eval_counts_response_tokens_only_and_is_nan_safe():
    tok = make_tokenizer("chatml")
    ft.resolve_template(tok)
    tok.pad_token = tok.eos_token
    m = tiny_llama(len(tok))
    rt = setup_runtime("cpu", "fp32", False, 1, 0)
    a = ft.evaluate(m, _windows(tok, 0.0), rt, 4, 16)
    b = ft.evaluate(m, _windows(tok, 0.3), rt, 4, 16)
    assert abs(a - b) < 1e-6                     # prompt tokens supervised at 0.3 are not counted
    long = [{"prompt": "what is " + " ".join(["the cat sat"] * 12), "response": "four"}] * 3
    items = ft.tokenize_examples(tok, long, 16, truncate=False)
    w = ft.pack(items, 16, tok.pad_token_id, "wrapped")
    assert any(all(t == -100 for t in x[1][1:]) for x in w)            # prompt-only windows
    e0, e1 = ft.evaluate(m, w, rt, 2, 0), ft.evaluate(m, w, rt, 2, 16)
    assert e0 == e0 and abs(e0 - e1) < 1e-5
    assert ft.evaluate(m, [], rt, 2, 16) != ft.evaluate(m, [], rt, 2, 16)      # NaN


def test_grad_accumulation_equals_the_large_batch():
    tok = make_tokenizer("chatml")
    ft.resolve_template(tok)
    tok.pad_token = tok.eos_token
    rt = setup_runtime("cpu", "fp32", False, 1, 0)
    ex = rows(2, seed=5) + [{"prompt": "what is two", "response": " ".join(["red green blue"] * 8)}] * 2
    items = ft.tokenize_examples(tok, ex, 64)
    w = ft.pack(items, 64, tok.pad_token_id, "none")
    targets = [sum(t != -100 for t in x[1][1:]) for x in w]
    assert max(targets) > 4 * min(targets)                  # very different supervised counts

    def grads(groups):
        m = tiny_llama(len(tok))
        lora.apply_lora(m, 4, 8, 0.0, ft.linear_targets(m))
        with torch.no_grad():
            for p in lora.lora_parameters(m):         # B != 0 so every adapter gets a gradient
                p.normal_(0, 0.1)
        loss, n = ft.accumulate(ft.LossModule(m), [ft._batch(w, g, "cpu") for g in groups], rt, 16)
        return loss, n, [p.grad.clone() for p in lora.lora_parameters(m)]

    l1, n1, g1 = grads([[0, 1, 2, 3]])
    l2, n2, g2 = grads([[0, 1], [2, 3]])
    assert n1 == n2 == sum(targets) and abs(l1 - l2) < 1e-6
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_quantized_head_is_dequantized_in_the_compute_dtype(monkeypatch):
    m = tiny_llama(300, 256, 512, 4, 1)
    qweights.quantize_model(m, 8, min_numel=1 << 16, embed_bits=8)
    assert isinstance(m.lm_head, qweights.QLinear)
    seen = []
    real = ft.chunked_cross_entropy
    monkeypatch.setattr(ft, "chunked_cross_entropy",
                        lambda h, wt, *a, **k: seen.append(wt) or real(h, wt, *a, **k))
    x = torch.randint(0, 300, (2, 12))
    ref = ft.lm_loss(m, x[:, :-1], x[:, 1:], 0).item()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = ft.lm_loss(m, x[:, :-1], x[:, 1:], 8)
    # the quantized head itself: each chunk dequantizes it block by block in the autocast
    # dtype (was the full fp32 head, 4 B/param, held across chunks)
    assert len(seen) == 1 and seen[0] is m.lm_head
    assert abs(loss.item() - ref) < 5e-2


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_compute_dtype_without_the_torch_2_4_autocast_api(monkeypatch):
    h = torch.zeros(2, 3)

    def dtype(legacy, autocast):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            with monkeypatch.context() as mp:          # autocast's own enter/exit uses the new API
                if legacy:                             # torch 2.2 / 2.3 (pyproject: torch>=2.2)
                    mp.delattr(torch, "get_autocast_dtype")
                return ft._compute_dtype(h)
    assert [dtype(legacy, a) for legacy in (False, True) for a in (False, True)] == \
        [torch.float32, torch.bfloat16] * 2


def _gloo(tmp_path):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    return dist


def test_ddp_loss_module_keeps_chunked_ce_syncs_once_and_skips_buffer_broadcast(tmp_path, monkeypatch):
    dist = _gloo(tmp_path)
    try:
        tok = make_tokenizer("chatml")
        ft.resolve_template(tok)
        tok.pad_token = tok.eos_token
        m = tiny_llama(len(tok), 256, 512, 4, 1)
        qweights.quantize_model(m, 8, min_numel=1 << 16, embed_bits=8)
        lora.apply_lora(m, 4, 8, 0.0, ft.linear_targets(m))
        rt = setup_runtime("cpu", "fp32", False, 1, 0)
        w = _windows(tok, 0.0)
        calls, bcast, hooks = [], [], []
        real_ce, real_b = ft.chunked_cross_entropy, dist._broadcast_coalesced
        monkeypatch.setattr(ft, "chunked_cross_entropy", lambda *a, **k: calls.append(1) or real_ce(*a, **k))
        ddp = ft.wrap_ddp(m, rt)
        monkeypatch.setattr(dist, "_broadcast_coalesced", lambda *a, **k: bcast.append(1) or real_b(*a, **k))

        def hook(state, bucket):
            hooks.append(1)
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut
        ddp.register_comm_hook(None, hook)
        mbs = [ft._batch(w, [i], "cpu") for i in range(3)]
        loss, n = ft.accumulate(ddp, mbs, rt, 16)
        assert len(calls) == 3                        # chunked CE inside DDP's forward
        assert bcast == []                            # quantized codes are not re-broadcast
        per_step = len(hooks)
        hooks.clear()
        ft.accumulate(ddp, mbs[:1], rt, 16)
        assert per_step == len(hooks) > 0             # one all-reduce round per step, not per micro-batch
        m.zero_grad()
        ref, n_ref = ft.accumulate(ft.LossModule(m), mbs, rt, 16)
        assert abs(loss - ref) < 1e-6 and n == n_ref
    finally:
        dist.destroy_process_group()
