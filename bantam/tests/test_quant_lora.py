"""LoRA: adapter device/dtype, all-linear targets, DoRA, Conv1D, PEFT interop both ways."""
import copy
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import lora, qweights  # noqa: E402


def _llama(tied=False):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, tie_word_embeddings=tied)).eval()


def _gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=256, n_embd=32, n_layer=2, n_head=2,
                                      n_positions=64)).eval()


def _perturb(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in lora.lora_parameters(model):
            p.add_(torch.randn(p.shape, generator=g) * 0.1)


X = torch.randint(0, 256, (2, 12))


def test_adapters_follow_the_base_device():
    """GPU-14 / LORA-1: adapters are created where the base layer lives (meta stands in for cuda)."""
    for base in (nn.Linear(64, 32).to("meta"), qweights.QLinear(nn.Linear(64, 32), "nf4").to("meta"),
                 nn.Linear(64, 32, device="meta", dtype=torch.bfloat16)):
        m = lora.LoRALinear(base, 4, 8, dora=True)
        for p in (m.lora_A, m.lora_B, m.lora_magnitude_vector):
            assert p.device.type == "meta" and p.dtype == torch.float32
    base = nn.Linear(64, 32)                     # CPU base built before the default device changed
    torch.set_default_device("meta")
    try:
        m = lora.LoRALinear(base, 4, 8)
    finally:
        torch.set_default_device("cpu")
    assert m.lora_A.device.type == "cpu" and m.lora_B.device.type == "cpu"
    bf = lora.LoRALinear(nn.Linear(64, 32).bfloat16(), 4, 8)      # fp32 adapters on a bf16 base
    assert bf(torch.randn(2, 64, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_all_linear_targets_and_exclude():
    m = _llama()
    rep = lora.apply_lora(m, r=4, alpha=8, targets="all-linear")
    assert rep["wrapped"] == 2 * 7 and not isinstance(m.lm_head, lora.LoRALinear)
    m = _llama()
    rep = lora.apply_lora(m, r=4, alpha=8, targets="all-linear", exclude=("down_proj", "q_proj"))
    assert rep["wrapped"] == 2 * 5
    m = _llama()
    assert lora.apply_lora(m, 4, 8, targets=r".*\.layers\.0\..*_proj")["wrapped"] == 7
    g = _gpt2()                                          # Conv1D layers, tied nn.Linear head
    assert lora.apply_lora(g, 4, 8, targets="all-linear")["wrapped"] == 2 * 4
    assert not isinstance(g.lm_head, lora.LoRALinear)
    q = _llama(tied=True)                                # QLoRA: QLinear yes, tied head no
    qweights.quantize_model(q, "nf4", min_numel=1, embed_bits=8)
    assert lora.apply_lora(q, 4, 8, targets="all-linear")["wrapped"] == 2 * 7
    # a second call does not wrap the adapters' own base layers
    assert lora.apply_lora(q, 4, 8, targets="all-linear")["wrapped"] == 0


def test_dora_is_the_weight_decomposition_and_merges():
    torch.manual_seed(0)
    base = nn.Linear(48, 40)
    m = lora.LoRALinear(copy.deepcopy(base), r=4, alpha=8, dora=True).eval()
    x = torch.randn(3, 48)
    assert torch.allclose(m(x), base(x), atol=1e-5)           # starts at the base model
    _perturb(m)
    W = base.weight + m.scale * m.lora_B @ m.lora_A
    ref = F.linear(x, m.lora_magnitude_vector[:, None] * W / W.norm(dim=1, keepdim=True), base.bias)
    y = m(x)
    assert torch.allclose(y, ref, atol=1e-5)
    y.sum().backward()
    assert all(p.grad is not None for p in (m.lora_A, m.lora_B, m.lora_magnitude_vector))
    m.merge()
    assert torch.allclose(m(x), ref, atol=1e-5)
    # DoRA over a quantized base and a Conv1D base
    from transformers.pytorch_utils import Conv1D
    for b in (qweights.QLinear(nn.Linear(48, 40), "nf4"), Conv1D(40, 48)):
        d = lora.LoRALinear(b, 4, 8, dora=True).eval()
        assert torch.allclose(d(x), b(x), atol=1e-5)


def test_conv1d_lora_merges():
    g = _gpt2()
    lora.apply_lora(g, 4, 8, targets=("c_attn", "c_fc"))
    _perturb(g)
    ref = g(input_ids=X).logits
    assert lora.merge_lora(g) == 4
    assert torch.allclose(g(input_ids=X).logits, ref, atol=1e-5)


@pytest.mark.parametrize("arch", ["llama", "gpt2"])
@pytest.mark.parametrize("dora", [False, True])
def test_save_peft_adapter_loads_in_peft(tmp_path, arch, dora):
    from peft import PeftModel
    mk = {"llama": _llama, "gpt2": _gpt2}[arch]
    m = mk()
    lora.apply_lora(m, r=4, alpha=8, targets="all-linear", dora=dora, rslora=True)
    _perturb(m)
    ref = m(input_ids=X).logits
    cfg = lora.save_peft_adapter(m, tmp_path, "tiny-random")
    assert sorted(os.listdir(tmp_path)) == ["adapter_config.json", "adapter_model.safetensors"]
    assert cfg["use_dora"] == dora and cfg["fan_in_fan_out"] == (arch == "gpt2")
    pm = PeftModel.from_pretrained(mk(), tmp_path).eval()
    assert torch.allclose(pm(input_ids=X).logits, ref, atol=1e-5)
    m2 = mk()                                        # and back into a fresh model here
    lora.load_peft_adapter(m2, tmp_path)
    assert torch.allclose(m2(input_ids=X).logits, ref, atol=1e-6)


@pytest.mark.parametrize("arch,targets", [("llama", ["q_proj", "v_proj", "down_proj"]),
                                          ("gpt2", ["c_attn", "c_proj"])])
@pytest.mark.parametrize("dora", [False, True])
def test_load_peft_adapter_reads_peft_output(tmp_path, arch, targets, dora):
    from peft import LoraConfig, get_peft_model
    mk = {"llama": _llama, "gpt2": _gpt2}[arch]
    torch.manual_seed(2)
    pm = get_peft_model(mk(), LoraConfig(r=4, lora_alpha=16, target_modules=targets,
                                         init_lora_weights=False, use_dora=dora, use_rslora=True,
                                         rank_pattern={"v_proj": 2} if arch == "llama" else {}))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "magnitude" in n:
                p.mul_(1.1)
    pm.eval()
    ref = pm(input_ids=X).logits
    pm.save_pretrained(tmp_path)
    m = mk()
    rep = lora.load_peft_adapter(m, tmp_path)
    assert rep["trainable_params"] == sum(p.numel() for p in lora.lora_parameters(m))
    assert torch.allclose(m(input_ids=X).logits, ref, atol=1e-5)


def test_qlora_dora_trains_and_exports(tmp_path):
    m = _llama()
    qweights.quantize_model(m, "nf4", min_numel=1, embed_bits=8)
    rep = lora.apply_lora(m, 4, 8, targets="all-linear", dora=True)
    assert rep["trainable_params"] == sum(p.numel() for p in lora.lora_parameters(m))
    m.train()
    m(input_ids=X).logits.float().mean().backward()
    assert all(p.grad is not None for p in lora.lora_parameters(m))
    lora.save_peft_adapter(m, tmp_path, "tiny")
    with pytest.raises(RuntimeError):
        lora.merge_lora(m)
