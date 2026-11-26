import math
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import lora, qweights  # noqa: E402
from bantamtrain.losses import chunked_cross_entropy  # noqa: E402
from bantamtrain.models import ZoomGPT  # noqa: E402
from bantamtrain.optim import build_optimizer, default_muon_params  # noqa: E402


def _tiny_hf_llama():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=False)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_chunked_ce_matches_full_with_grads():
    torch.manual_seed(0)
    h = torch.randn(3, 70, 32, requires_grad=True)
    w = torch.randn(500, 32, requires_grad=True)
    y = torch.randint(0, 500, (3, 70))
    y[0, :10] = -100
    full = F.cross_entropy(F.linear(h, w).reshape(-1, 500), y.reshape(-1), ignore_index=-100)
    full.backward()
    gh, gw = h.grad.clone(), w.grad.clone()
    h.grad = w.grad = None
    ch = chunked_cross_entropy(h, w, y, chunk=33)
    ch.backward()
    assert abs(ch.item() - full.item()) < 1e-5
    assert torch.allclose(h.grad, gh, atol=1e-5) and torch.allclose(w.grad, gw, atol=1e-5)


def test_zoomgpt_ce_chunk_equals_full():
    torch.manual_seed(0)
    a = ZoomGPT(300, 32, 1, 2, zoom_ks=[2], softcap=10.0)
    b = ZoomGPT(300, 32, 1, 2, zoom_ks=[2], softcap=10.0, ce_chunk=16)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 300, (2, 34))
    for k in (1, 2):
        la, lb = a(x[:, :32], x[:, k:32 + k], k=k), b(x[:, :32], x[:, k:32 + k], k=k)
        assert abs(la.item() - lb.item()) < 1e-5


def test_lora_starts_at_base_and_trains_only_adapters():
    torch.manual_seed(0)
    m = _tiny_hf_llama()
    x = torch.randint(0, 256, (2, 16))
    ref = m(input_ids=x).logits
    rep = lora.apply_lora(m, r=4, alpha=8, targets=("q_proj", "v_proj", "down_proj"))
    assert rep["wrapped"] == 2 * 3
    assert torch.allclose(m(input_ids=x).logits, ref, atol=1e-6)        # B = 0 at init
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert rep["trainable_params"] == sum(p.numel() for p in lora.lora_parameters(m))
    loss = m(input_ids=x).logits.float().mean()
    loss.backward()
    assert all(p.grad is not None for p in lora.lora_parameters(m))


def test_lora_merge_equals_adapter_forward_and_adapter_io(tmp_path):
    torch.manual_seed(0)
    m = _tiny_hf_llama()
    lora.apply_lora(m, r=4, alpha=8, targets=("q_proj", "o_proj"))
    with torch.no_grad():
        for p in lora.lora_parameters(m):
            p.add_(torch.randn_like(p) * 0.1)
    x = torch.randint(0, 256, (2, 16))
    with_adapter = m(input_ids=x).logits
    sd = lora.lora_state_dict(m)
    assert all("lora_" in k for k in sd) and len(sd) == 2 * 2 * 2
    n = lora.merge_lora(m)
    assert n == 4
    merged = m(input_ids=x).logits
    assert torch.allclose(with_adapter, merged, atol=1e-4)
    lora.unwrap_lora(m)
    assert not any(isinstance(mod, lora.LoRALinear) for mod in m.modules())
    assert torch.allclose(m(input_ids=x).logits, merged, atol=1e-6)
    # load adapters into a fresh model
    m2 = _tiny_hf_llama()
    lora.apply_lora(m2, r=4, alpha=8, targets=("q_proj", "o_proj"))
    lora.load_lora_state_dict(m2, sd)
    assert torch.allclose(m2(input_ids=x).logits, with_adapter, atol=1e-5)


def test_qlora_int4_base_with_adapters_and_muon_routing():
    torch.manual_seed(0)
    m = _tiny_hf_llama()
    rep_q = qweights.quantize_model(m, 4, min_numel=1)
    assert rep_q["modules_quantized"] > 0
    rep = lora.apply_lora(m, r=4, alpha=8, targets=("q_proj", "k_proj", "v_proj", "o_proj"))
    assert rep["wrapped"] == 8
    x = torch.randint(0, 256, (2, 16))
    loss = m(input_ids=x).logits.float().mean()
    loss.backward()
    assert all(p.grad is not None for p in lora.lora_parameters(m))
    # Muon routing on a HF model: without muon_lora nothing is on Muon (base frozen)
    opt = build_optimizer(m, "muon", 1e-3, 0.0, state_bits=8, fused=False)
    assert len(opt.optimizers) == 1
    opt2 = build_optimizer(m, "muon", 1e-3, 0.0, state_bits=8, fused=False, muon_lora=True)
    assert len(opt2.optimizers) == 2
    opt2.step()


def test_default_muon_params_skips_embeddings_and_head():
    m = _tiny_hf_llama()
    ps = {id(p) for p in default_muon_params(m)}
    assert id(m.model.embed_tokens.weight) not in ps
    assert id(m.lm_head.weight) not in ps
    assert id(m.model.layers[0].self_attn.q_proj.weight) in ps
    assert id(m.model.layers[0].mlp.up_proj.weight) in ps


def test_finetune_data_pipeline_masks_prompt_and_packs():
    pytest.importorskip("transformers")
    from test_finetune_fixtures import make_tokenizer   # GPT-2 style byte-level BPE, built offline
    from bantamtrain import finetune as ft
    tok = make_tokenizer("bpe")
    exs = [{"prompt": "What is 2+2?", "response": "4"},
           {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
    items = ft.tokenize_examples(tok, exs, seq=64)
    assert len(items) == 2
    for ids, labels, weights in items:
        assert len(ids) == len(labels) == len(weights)
        assert labels[0] == -100 and labels[-1] != -100 and weights[-1] == 1.0
    # exactly the response is supervised, its first token included
    assert [tok.decode([t for t in l if t != -100]) for _, l, _ in items] == \
        ["4<|endoftext|>", " hello<|endoftext|>"]
    pad = tok.pad_token_id or 0
    for mode in ("bfd", "wrapped", "none"):
        w = ft.pack(items, 16, pad, mode)
        assert all(len(x) == len(y) == len(z) == len(g) == 17 for x, y, z, g in w)
    # wrapped: every response token is predicted exactly once (window position 0 is input only)
    w = ft.pack(items, 16, pad, "wrapped")
    n_targets = sum(1 for x, y, z, g in w for t in y[1:] if t != -100)
    n_resp = sum(1 for i, l, _ in items for t in l[1:] if t != -100)
    assert n_targets == n_resp
    # bfd never splits an example and fills bins best-first
    big = [(list(range(9)), list(range(9)), [1.0] * 9)] * 3 + [(list(range(5)), list(range(5)), [1.0] * 5)]
    w = ft.pack(big, 16, pad, "bfd")          # capacity 17: {9+5}, {9}, {9} -> 3 bins, no splits
    fill = sorted(sum(1 for v in z if v != 0.0) for _, _, z, _ in w)   # weights are 0 on padding
    assert len(w) == 3 and fill == [9, 9, 14]
    two = [x for x in w if max(x[3]) == 1][0]           # the {9 + 5} bin: segments 0 then 1, pad -1
    assert two[3] == [0] * 9 + [1] * 5 + [-1] * 3 and two[1][9] == -100   # no cross-example target
    # prompt_loss_weight > 0 supervises prompt tokens with that weight
    items2 = ft.tokenize_examples(tok, exs[:1], seq=64, prompt_loss_weight=0.3)
    ids, labels, weights = items2[0]
    assert labels[0] != -100 and weights[0] == 0.3 and weights[-1] == 1.0


def test_resolve_template_installs_chatml_for_base_vocab():
    from bantamtrain import finetune as ft

    class FakeTok:
        chat_template = None
        eos_token = "<|endoftext|>"

        def get_vocab(self):
            return {"<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 0}

    t = FakeTok()
    assert ft.resolve_template(t) == "chatml-installed"
    assert t.eos_token == "<|im_end|>" and "generation_prompt" in t.chat_template


def test_packed_examples_do_not_attend_to_each_other():
    """A packed window must give each example exactly the logits it gets alone."""
    from bantamtrain import finetune as ft
    m = _tiny_hf_llama()
    torch.manual_seed(1)
    a, b = torch.randint(0, 256, (11,)).tolist(), torch.randint(0, 256, (7,)).tolist()
    items = [(a, a, [1.0] * 11), (b, b, [1.0] * 7)]
    (x, y, w, g), = ft.pack(items, 20, 0, "bfd")
    xs, gs = torch.tensor([x]), torch.tensor([g])
    mask, pos = ft.segment_mask(gs)
    assert pos[0, :11].tolist() == list(range(11)) and pos[0, 11:18].tolist() == list(range(7))
    with torch.no_grad():
        packed = m(input_ids=xs, attention_mask=mask, position_ids=pos).logits[0]
        alone_a = m(input_ids=torch.tensor([a])).logits[0]
        alone_b = m(input_ids=torch.tensor([b])).logits[0]
        leaky = m(input_ids=xs).logits[0]
    assert torch.allclose(packed[:11], alone_a, atol=1e-5)
    assert torch.allclose(packed[11:18], alone_b, atol=1e-5)
    assert not torch.allclose(leaky[11:18], alone_b, atol=1e-3)      # what plain packing did
    # the training loss on the packed window equals the per-example losses combined
    ys = torch.tensor([y])
    loss = ft.lm_loss(m, xs[:, :-1], ys[:, 1:], 0, seg=gs[:, :-1])
    ref = torch.nn.functional.cross_entropy(
        torch.cat([alone_a[:-1], alone_b[:-1]]), torch.tensor(a[1:] + b[1:]))
    assert abs(loss.item() - ref.item()) < 1e-5
    assert ft.segment_mask(torch.zeros(2, 5, dtype=torch.long)) == (None, None)
