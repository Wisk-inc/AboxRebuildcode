import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoomtrain import lora, qweights  # noqa: E402
from zoomtrain.losses import chunked_cross_entropy  # noqa: E402
from zoomtrain.models import ZoomGPT  # noqa: E402
from zoomtrain.optim import build_optimizer, default_muon_params  # noqa: E402


def _tiny_hf_llama():
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
    from transformers import AutoTokenizer
    from zoomtrain import finetune as ft
    tok = AutoTokenizer.from_pretrained("gpt2") if os.environ.get("HF_HUB_OFFLINE") != "1" else None
    if tok is None:
        return
    exs = [{"prompt": "What is 2+2?", "response": "4"},
           {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
    items = ft.tokenize_examples(tok, exs, seq=64)
    assert len(items) == 2
    for ids, labels in items:
        assert len(ids) == len(labels)
        assert labels[0] == -100 and labels[-1] != -100
    w = ft.pack(items, 16, tok.pad_token_id or 0)
    assert all(len(x) == 17 and len(y) == 17 for x, y in w)
    # every response token is predicted exactly once (position 0 of a window is input only)
    n_targets = sum(1 for x, y in w for t in y[1:] if t != -100)
    n_resp = sum(1 for i, l in items for t in l[1:] if t != -100)
    assert n_targets == n_resp
