"""Wave-2 seams around quantized weights: no full dequantization of a quantized
embedding/head in ZoomGPT, default_muon_params / build_optimizer and finetune's loss;
finetune's quantized-parameter count, --param_dtype auto, import order; memory's
Newton-Schulz FLOPs from optim.ns_flops."""
import copy
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import finetune as ft  # noqa: E402
from bantamtrain import lora, memory, qweights  # noqa: E402
from bantamtrain.models import ZoomGPT  # noqa: E402
from bantamtrain.optim import (MUON_ELEMENTWISE_FLOPS, build_optimizer,  # noqa: E402
                               default_muon_params, optimizer_flops)


@pytest.fixture
def dequant_calls(monkeypatch):
    """Counts full-matrix dequantizations (QBase.dequantize, which .weight calls)."""
    calls = []
    real = qweights.QBase.dequantize

    def counting(self, dtype=None):
        calls.append(type(self).__name__)
        return real(self, dtype)
    monkeypatch.setattr(qweights.QBase, "dequantize", counting)
    return calls


def _llama(tied=False, inter=128):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=inter, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=tied)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_zoomgpt_with_a_quantized_embedding_never_dequantizes_it_whole(dequant_calls):
    """QNT: _rope read self.emb.weight.device and the tied head used emb.weight (full dequant)."""
    torch.manual_seed(0)
    m = ZoomGPT(256, 32, 1, 2, zoom_ks=[2]).eval()
    m.emb = qweights.QEmbedding(m.emb, 8)
    ref = copy.deepcopy(m)
    ref.emb = nn.Embedding(256, 32)
    ref.emb.weight.data = m.emb.dequantize(torch.float32)
    dequant_calls.clear()
    x = torch.randint(0, 256, (2, 18))
    with torch.no_grad():
        for k in (1, 2):
            assert torch.allclose(m(x[:, :16], x[:, k:16 + k], k=k),
                                  ref(x[:, :16], x[:, k:16 + k], k=k), atol=1e-5)
        m.ce_chunk = ref.ce_chunk = 8
        assert torch.allclose(m(x[:, :16], x[:, 1:17]), ref(x[:, :16], x[:, 1:17]), atol=1e-5)
        assert torch.allclose(m(x[:, :16]), ref(x[:, :16]), atol=1e-4)
        g = m.generate(x[:, :4], 3, temperature=0)
    assert torch.equal(g, ref.generate(x[:, :4], 3, temperature=0))
    assert dequant_calls == []


def test_muon_routing_skips_quantized_modules_without_dequantizing(dequant_calls):
    m = _llama()
    linears = {id(mod.weight) for n, mod in m.named_modules()
               if isinstance(mod, nn.Linear) and n != "lm_head"}
    assert {id(p) for p in default_muon_params(m)} == linears
    qweights.quantize_model(m, 8, min_numel=1, embed_bits=8)
    lora.apply_lora(m, r=4, targets="all-linear")
    dequant_calls.clear()
    assert default_muon_params(m) == []                      # frozen quantized base
    opt = build_optimizer(m, "muon", 1e-3, 0.0, muon_lora=True)
    assert dequant_calls == []
    muon = opt.optimizers[-1]
    assert type(muon).__name__ == "Muon" and all(
        p.dim() == 2 for g in muon.param_groups for p in g["params"])


@pytest.mark.parametrize("tied", [False, True])
def test_finetune_loss_hands_the_quantized_head_to_chunked_ce(tied, dequant_calls, monkeypatch):
    """OPT-4 / MEM-X1: no float vocab table is built (or kept across chunks) for the loss."""
    m = _llama(tied=tied)
    qweights.quantize_model(m, 8, min_numel=1, embed_bits=8)
    torch.manual_seed(1)
    x, y = torch.randint(0, 256, (2, 12)), torch.randint(0, 256, (2, 12))
    y[0, :3] = -100
    ft.lm_loss(m, x, y, 5)                                    # the one-time head probe
    seen = []
    real = ft.chunked_cross_entropy
    monkeypatch.setattr(ft, "chunked_cross_entropy", lambda h, w, *a, **k: seen.append(w)
                        or real(h, w, *a, **k))
    dequant_calls.clear()
    with torch.no_grad():
        got = ft.lm_loss(m, x, y, 5)
        assert dequant_calls == []
        full = ft.lm_loss(m, x, y, 0)                         # the model's own logits
    assert len(seen) == 1 and not torch.is_tensor(seen[0])
    assert type(seen[0]).__name__ == ("QTiedHead" if tied else "QLinear")
    assert abs(got.item() - full.item()) < 1e-4


def test_quantized_param_count_counts_a_tied_head_once_without_padding():
    m = _llama(tied=True, inter=134)                          # down_proj in=134: padded blocks
    want = {id(mod.weight): mod.weight.numel() for mod in m.modules()
            if isinstance(mod, (nn.Linear, nn.Embedding))}
    rep = qweights.quantize_model(m, 4, min_numel=1, embed_bits=8)
    assert rep["tied_head"] and any(getattr(mod, "pad", 0) for mod in m.modules())
    assert ft._quantized_params(m) == sum(want.values())


def test_param_dtype_auto_uses_native_bf16_only(monkeypatch):
    """GPU-8: is_bf16_supported() also says yes on sm70/sm75, where bf16 is emulated."""
    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (7, 5))
    assert ft.auto_param_dtype(cuda, 16) == "fp32" and ft.auto_param_dtype(cuda, 0) == "bf16"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (8, 0))
    assert ft.auto_param_dtype(cuda, 16) == "bf16"
    cpu = torch.device("cpu")
    assert ft.auto_param_dtype(cpu, 16) == "fp32" and ft.auto_param_dtype(cpu, 0) == "bf16"


def test_finetune_without_transformers_writes_nothing(tmp_path, monkeypatch):
    """PKG-5: a missing extra is reported before --out is created."""
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    monkeypatch.setitem(sys.modules, "transformers", None)
    out = tmp_path / "o"
    try:
        with pytest.raises(ImportError):
            ft.main(["--model", "x", "--out", str(out), "--threads", "1"])
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)
    assert not out.exists()


def test_finetune_rejects_a_bad_duty_before_loading(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        ft.main(["--model", "x", "--out", str(tmp_path / "o"), "--duty", "0"])
    assert e.value.code == 2 and "--duty" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_memory_ns_flops_match_the_optimizer_count():
    """MEM-10: the scoreboard's Newton-Schulz FLOPs are optim's, per Muon step."""
    m = ZoomGPT(64, 32, 2, 2)
    muon = build_optimizer(m, "muon", 1e-3, 0.0).optimizers[-1]
    n = sum(p.numel() for g in muon.param_groups for p in g["params"])
    ns = optimizer_flops(muon) - MUON_ELEMENTWISE_FLOPS * n
    assert memory.muon_ns_flops(32, 2) == pytest.approx(ns, rel=1e-12)
