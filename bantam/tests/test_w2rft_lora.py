"""Wave-2 end-to-end review of adapters: bantam chat loading, LoRA / DoRA over quantized bases."""
import pytest
import torch

from bantamtrain import lora

pytest.importorskip("transformers")


def _llama(seed=0):
    import transformers as tf
    torch.manual_seed(seed)
    return tf.LlamaForCausalLM(tf.LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                                              num_hidden_layers=1, num_attention_heads=4)).eval()


def test_chat_lora_alpha_override_loads_a_dora_adapter_pt(tmp_path):
    """--lora_alpha (an adapter.pt without adapter_meta.json / log.json) assumed plain LoRA, so a
    DoRA adapter's magnitudes were unexpected keys and loading failed."""
    from bantamtrain.cli import chat
    m = _llama()
    lora.apply_lora(m, 4, 8, targets=("q_proj", "v_proj"), dora=True)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
    torch.save(lora.lora_state_dict(m), tmp_path / "adapter.pt")
    m2 = _llama()
    assert chat.attach_adapter(m2, str(tmp_path / "adapter.pt"), lora_alpha=8) == "bantam"
    x = torch.randint(0, 128, (1, 9))
    with torch.no_grad():
        assert torch.allclose(m(input_ids=x).logits, m2.eval()(input_ids=x).logits, atol=1e-6)
