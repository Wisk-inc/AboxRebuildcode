"""qweights: QLoRA memory path, dtypes, odd widths, scales, tied heads, subclasses, Conv1D,
streaming load_quantized."""
import copy
import os
import subprocess
import sys
import textwrap
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bantamtrain import lora, qweights  # noqa: E402
from bantamtrain.qweights import QEmbedding, QLinear, QTiedHead  # noqa: E402


def _llama(tied=False, vocab=256, layers=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=tied)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


# peak RSS of this process image (ru_maxrss would include the forked pytest parent's)
_PROC = textwrap.dedent(f"""
    import sys, torch
    sys.path.insert(0, {ROOT!r})
    def mem(key):
        return [int(l.split()[1]) * 1024 for l in open("/proc/self/status") if l.startswith(key)][0]
""")


class _Saved:
    """Tensors autograd saves for backward that are not module parameters or buffers."""

    def __init__(self, *modules):
        self.own = {t.data_ptr() for m in modules for t in list(m.parameters()) + list(m.buffers())}
        self.big = []

    def pack(self, t):
        if t.data_ptr() not in self.own:
            self.big.append(t.numel())
        return t

    def __enter__(self):
        self.ctx = torch.autograd.graph.saved_tensors_hooks(self.pack, lambda t: t)
        self.ctx.__enter__()
        return self

    def __exit__(self, *a):
        self.ctx.__exit__(*a)


@pytest.mark.parametrize("bits", [8, 4, "nf4"])
def test_qlinear_saves_only_codes_and_grads_match_float_path(bits):
    """QNT-2 / GPU-7: no dequantized weight is kept for backward; grads equal F.linear's."""
    torch.manual_seed(0)
    base = nn.Linear(256, 512)
    ql = QLinear(base, bits)
    ref_lin = nn.Linear(256, 512)
    with torch.no_grad():
        ref_lin.weight.copy_(ql.dequantize(torch.float32))
        ref_lin.bias.copy_(base.bias)
    a, b = lora.LoRALinear(ql, 4, 8), lora.LoRALinear(ref_lin, 4, 8)
    with torch.no_grad():
        b.lora_A.copy_(a.lora_A)
        a.lora_B.normal_()
        b.lora_B.copy_(a.lora_B)
    x = torch.randn(3, 5, 256, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    with _Saved(a) as saved:
        ya = a(x)
    assert max(saved.big) < 256 * 512 / 4                 # activations only, never W
    yb = b(xr)
    assert torch.allclose(ya, yb, atol=1e-5)
    g = torch.randn_like(ya)
    ya.backward(g)
    yb.backward(g)
    assert torch.allclose(x.grad, xr.grad, atol=1e-5)
    assert torch.allclose(a.lora_A.grad, b.lora_A.grad, atol=1e-4)
    assert torch.allclose(a.lora_B.grad, b.lora_B.grad, atol=1e-4)
    # bf16 autocast: dequantized in bf16, output bf16, still nothing big saved
    with _Saved(a) as saved, torch.autocast("cpu", dtype=torch.bfloat16):
        y = a(x)
    assert y.dtype == torch.bfloat16 and max(saved.big) < 256 * 512 / 4
    y.float().sum().backward()


def test_qlinear_row_chunks_match_single_shot(monkeypatch):
    torch.manual_seed(0)
    ql = QLinear(nn.Linear(96, 200), "nf4")
    x = torch.randn(4, 96, requires_grad=True)
    y = ql(x)
    (gx,) = torch.autograd.grad(y.square().sum(), x)
    monkeypatch.setattr(qweights, "CHUNK", 1000)          # ~10 rows per chunk
    x2 = x.detach().clone().requires_grad_(True)
    y2 = ql(x2)
    (gx2,) = torch.autograd.grad(y2.square().sum(), x2)
    assert torch.allclose(y, y2, atol=1e-5) and torch.allclose(gx, gx2, atol=1e-4)


def test_qembedding_gathers_rows_in_original_dtype(monkeypatch):
    """QNT-5: only the gathered rows are dequantized; a bf16 table returns bf16."""
    torch.manual_seed(0)
    emb = nn.Embedding(4096, 64).bfloat16()
    qe = QEmbedding(emb, 8)
    idx = torch.tensor([[3, 7, 4095], [0, 1, 2]])
    rows = []
    real = qweights.dequantize_rows
    monkeypatch.setattr(qweights, "dequantize_rows",
                        lambda q, *a, **k: rows.append(q.shape[0]) or real(q, *a, **k))
    out = qe(idx)
    assert rows == [6] and out.dtype == torch.bfloat16 and out.shape == (2, 3, 64)
    monkeypatch.setattr(qweights, "dequantize_rows", real)
    assert torch.equal(out, F.embedding(idx, qe.weight))
    assert qe.device == emb.weight.device
    # a bf16 model without autocast: quantized embedding feeds a float Linear left by min_numel
    m = nn.Sequential(nn.Embedding(4096, 64), nn.Linear(64, 64)).bfloat16()
    qweights.quantize_model(m, 8, min_numel=1 << 16)
    assert isinstance(m[0], QEmbedding) and m(idx).dtype == torch.bfloat16


def test_4bit_codebook_is_a_buffer_and_dequant_has_no_int64_temporaries(monkeypatch):
    """GPU-9 / QNT-7: the codebook moves with the module (no host->device copy per call);
    dequantization uses int32 indices; values equal the reference implementation."""
    from torch.utils._python_dispatch import TorchDispatchMode
    torch.manual_seed(0)
    q = torch.randint(0, 256, (16, 32), dtype=torch.uint8)
    s = torch.rand(16, 1, dtype=torch.float16) + 0.5
    for bits in ("nf4", 4):
        lo, hi = (q & 15).long(), (q >> 4).long()
        idx = torch.stack([lo, hi], -1).view(16, -1)
        vals = qweights.NF4[idx] if bits == "nf4" else (idx - 8).float()
        ref = vals * s.float()
        assert torch.equal(qweights.dequantize_rows(q, s, 0, bits, 64), ref)
        assert torch.equal(qweights.dequantize_rows(q, s, 0, bits, 64, cb=qweights.codebook(bits)), ref)

    class Created(TorchDispatchMode):
        big_int64 = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for o in out if isinstance(out, (tuple, list)) else [out]:
                if isinstance(o, torch.Tensor) and o.dtype == torch.int64 and o.numel() > 256:
                    Created.big_int64 += 1
            return out

    ql = QLinear(nn.Linear(256, 256), "nf4")
    assert "cb" in dict(ql.named_buffers()) and "cb" not in ql.state_dict()
    monkeypatch.setattr(qweights, "_lut", lambda *a: pytest.fail("global table used"))
    with Created():
        ql(torch.randn(2, 256))
    assert Created.big_int64 == 0
    ql.to("meta")
    assert ql.cb.device.type == "meta"
    assert ql(torch.empty(2, 256, device="meta")).device.type == "meta"


def test_nf4_codes_are_nearest_and_quantization_ram_is_small():
    """QNT-7: bucketize picks the nearest codebook value; no (R, C, 16) temporaries."""
    torch.manual_seed(0)
    w = torch.randn(256, 1024)
    q, s, pad = qweights.quantize_rows(w, "nf4", 64)
    codes = torch.stack([q & 15, q >> 4], -1).view(256, 16, 64).long()
    n = w.view(256, 16, 64) / s.float()[..., None]
    dist = (n[..., None] - qweights.NF4).abs()
    assert torch.allclose(dist.gather(-1, codes[..., None])[..., 0], dist.min(-1).values, atol=1e-6)
    code = _PROC + textwrap.dedent(f"""
        from bantamtrain.qweights import quantize_rows
        w = torch.randn(1024, 4096)
        quantize_rows(w[:8], "nf4", 64)
        r0 = mem("VmRSS")
        quantize_rows(w, "nf4", 64)
        print((mem("VmHWM") - r0) / w.numel())
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=120)
    assert float(out.stdout.split()[-1]) < 20, out.stdout + out.stderr     # was ~130 B/weight


@pytest.mark.parametrize("bits", [4, "nf4"])
@pytest.mark.parametrize("c_in,block", [(1, 64), (33, 64), (63, 64), (45, 15)])
def test_4bit_odd_widths(bits, c_in, block):
    """QNT-4: an odd number of codes per row is padded, not a crash."""
    torch.manual_seed(0)
    lin = nn.Linear(c_in, 252)
    ql = QLinear(lin, bits, block)
    assert ql.block % 2 == 0
    x = torch.randn(2, c_in)
    w = ql.dequantize(torch.float32)
    assert torch.allclose(ql(x), F.linear(x, w, lin.bias), atol=1e-5)
    rel = ((w - lin.weight).norm() / lin.weight.norm()).item()
    assert rel < (0.15 if c_in > 1 else 1e-3)
    m = nn.Sequential(nn.Linear(c_in, 252))
    qweights.quantize_model(m, bits, min_numel=1)
    assert m(x).shape == (2, 252)


def test_scales_fall_back_to_fp32_and_keep_dtype_through_to():
    """WP-5 (qweights) / QNT-8: tiny weights keep precision; .to(bf16) leaves scales alone."""
    torch.manual_seed(0)
    for std, sdt in ((1e-5, torch.float32), (2e-2, torch.float16)):
        lin = nn.Linear(256, 256, bias=False)
        nn.init.normal_(lin.weight, std=std)
        ql = QLinear(lin, 8)
        rel = ((ql.dequantize(torch.float32) - lin.weight).norm() / lin.weight.norm()).item()
        assert ql.s.dtype == sdt and rel < 0.01
    m = nn.Sequential(nn.Linear(576, 576), nn.Linear(576, 64))
    qweights.quantize_model(m, 8, min_numel=1)
    w0 = m[0].dequantize(torch.float32)
    m.to(torch.bfloat16)
    assert m[0].s.dtype == torch.float16 and torch.equal(m[0].dequantize(torch.float32), w0)
    assert m[0].dtype == torch.bfloat16
    assert m(torch.randn(2, 576, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_block_divides_common_widths():
    """QNT-8: 576 at block 128 uses 96 (no padding), NF4 keeps 64."""
    q8 = QLinear(nn.Linear(576, 576), 8)
    assert q8.pad == 0 and q8.block == 96
    assert (q8.q.numel() + q8.s.numel() * 2) / (576 * 576) < 1.03
    q4 = QLinear(nn.Linear(576, 576), "nf4", 64)
    assert q4.pad == 0 and q4.block == 64
    assert QLinear(nn.Linear(577, 8), 8).block == 128           # prime width: padded as before


def test_tied_embedding_quantized_once():
    """QNT-6 / OPT-3: the head reuses the embedding's codes; logits as an untied copy."""
    m = _llama(tied=True)
    assert m.lm_head.weight is m.model.embed_tokens.weight
    ref = copy.deepcopy(m)
    ref.lm_head.weight = nn.Parameter(ref.lm_head.weight.detach().clone())   # untie
    rep = qweights.quantize_model(m, "nf4", min_numel=1, embed_bits=8)
    rep_ref = qweights.quantize_model(ref, "nf4", min_numel=1, embed_bits=8)
    assert isinstance(m.lm_head, QTiedHead) and rep["tied_head"]
    assert m.lm_head.q.data_ptr() == m.model.embed_tokens.q.data_ptr()
    table = m.model.embed_tokens.nbytes()
    assert rep["bytes_after"] == rep_ref["bytes_after"] - table
    assert rep["modules_quantized"] == rep_ref["modules_quantized"] - 1
    x = torch.randint(0, 256, (2, 12))
    assert torch.allclose(m(input_ids=x).logits, ref(input_ids=x).logits, atol=1e-5)
    m2 = copy.deepcopy(m)
    assert m2.lm_head.emb is m2.model.embed_tokens
    assert not any(k.startswith("lm_head") for k in m.state_dict())
    # embeddings kept in float: the tie is kept, nothing duplicated
    m3 = _llama(tied=True)
    qweights.quantize_model(m3, "nf4", min_numel=1, embed_bits=None)
    assert m3.lm_head.weight is m3.model.embed_tokens.weight


def test_subclass_forward_is_kept_gemma_opt():
    """QNT-1: Gemma's scaled embedding keeps sqrt(d); OPT's learned positions still work."""
    import transformers as T
    small = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                 num_attention_heads=2, num_key_value_heads=1)
    torch.manual_seed(0)
    g = T.Gemma2ForCausalLM(T.Gemma2Config(**small, head_dim=32)).eval()
    x = torch.randint(0, 256, (2, 12))
    ref_logits, ref_emb = g(input_ids=x).logits, g.get_input_embeddings()(x)
    emb_cls = type(g.get_input_embeddings())
    qweights.quantize_model(g, 8, min_numel=1, embed_bits=8)
    e = g.get_input_embeddings()
    assert isinstance(e, emb_cls) and isinstance(e, qweights.QBase)
    ratio = (e(x).norm() / ref_emb.norm()).item()
    assert abs(ratio - 1) < 0.01                                   # was 1/16: scale lost
    rel = ((g(input_ids=x).logits - ref_logits).norm() / ref_logits.norm()).item()
    assert rel < 0.03
    torch.manual_seed(0)
    o = T.OPTForCausalLM(T.OPTConfig(vocab_size=128, hidden_size=64, ffn_dim=128, num_hidden_layers=2,
                                     num_attention_heads=2, word_embed_proj_dim=64,
                                     max_position_embeddings=64)).eval()
    ref = o(input_ids=x % 128).logits
    pos_cls = type(o.model.decoder.embed_positions)
    qweights.quantize_model(o, "nf4", min_numel=1, embed_bits=8)
    assert isinstance(o.model.decoder.embed_positions, pos_cls)
    out = o(input_ids=x % 128).logits
    assert ((out - ref).norm() / ref.norm()).item() < 0.15


def test_generic_subclasses_and_max_norm():
    class Doubled(nn.Linear):
        def forward(self, x):
            return super().forward(x) * 2

    torch.manual_seed(0)
    m = nn.Sequential(Doubled(64, 64), nn.Embedding(100, 64, max_norm=1.0))
    x = torch.randn(3, 64)
    ref = m[0](x)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rep = qweights.quantize_model(m, 8, min_numel=1)
    assert isinstance(m[0], Doubled) and isinstance(m[0], qweights.QBase)
    assert torch.allclose(m[0](x), ref, atol=0.05)
    assert type(m[1]) is nn.Embedding and rep["skipped"] == ["1"] and w


def test_conv1d_quantized_like_linear():
    """transformers Conv1D (GPT-2): quantize_model and QLinear handle the (in, out) weight."""
    import transformers as T
    from transformers.pytorch_utils import Conv1D
    torch.manual_seed(0)
    c = Conv1D(96, 32)
    ql = QLinear(c, 8)
    x = torch.randn(2, 5, 32)
    assert ql.in_features == 32 and ql.out_features == 96
    assert torch.allclose(ql(x), c(x), atol=0.02)
    g = T.GPT2LMHeadModel(T.GPT2Config(vocab_size=128, n_embd=32, n_layer=2, n_head=2,
                                       n_positions=64)).eval()
    ids = torch.randint(0, 128, (2, 10))
    ref = g(input_ids=ids).logits
    rep = qweights.quantize_model(g, 8, min_numel=1, embed_bits=8)
    assert isinstance(g.transformer.h[0].attn.c_attn, QLinear) and rep["tied_head"]
    out = g(input_ids=ids).logits
    assert ((out - ref).norm() / ref.norm()).item() < 0.02


@pytest.mark.parametrize("arch", ["llama_sharded", "llama_tied", "gpt2", "gemma2"])
def test_load_quantized_equals_quantize_model(tmp_path, arch):
    import transformers as T
    small = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                 num_attention_heads=2, num_key_value_heads=1)
    cfg = {"llama_sharded": T.LlamaConfig(**small, tie_word_embeddings=False),
           "llama_tied": T.LlamaConfig(**small, tie_word_embeddings=True),
           "gpt2": T.GPT2Config(vocab_size=256, n_embd=32, n_layer=2, n_head=2, n_positions=64),
           "gemma2": T.Gemma2Config(**small, head_dim=32)}[arch]
    torch.manual_seed(0)
    T.AutoModelForCausalLM.from_config(cfg).save_pretrained(
        tmp_path, max_shard_size="40KB" if arch == "llama_sharded" else "5GB")
    x = torch.randint(0, 256, (2, 12))
    for bits, eb in ((8, 8), ("nf4", 8), (4, None)):
        ref = T.AutoModelForCausalLM.from_pretrained(tmp_path).eval()
        rep = qweights.quantize_model(ref, bits, min_numel=1, embed_bits=eb)
        m = qweights.load_quantized(str(tmp_path), bits, embed_bits=eb, min_numel=1)
        assert not any(t.is_meta for t in list(m.parameters()) + list(m.buffers()))
        assert m.quant_report["bytes_after"] == rep["bytes_after"]
        assert torch.equal(m(input_ids=x).logits, ref(input_ids=x).logits)


def test_load_quantized_peak_ram_stays_near_quantized_size(tmp_path):
    """Streaming load: peak RSS growth is a fraction of the fp32 model (~160 MB here)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = dict(vocab_size=4096, hidden_size=512, intermediate_size=1536, num_hidden_layers=12,
               num_attention_heads=8, num_key_value_heads=4, tie_word_embeddings=False)
    torch.manual_seed(0)
    big = LlamaForCausalLM(LlamaConfig(**cfg))
    fp32 = sum(p.numel() for p in big.parameters()) * 4
    big.save_pretrained(tmp_path / "big")
    del big
    tiny = LlamaForCausalLM(LlamaConfig(**{**cfg, "vocab_size": 128, "hidden_size": 64,
                                           "intermediate_size": 64, "num_hidden_layers": 1,
                                           "num_attention_heads": 2, "num_key_value_heads": 1}))
    tiny.save_pretrained(tmp_path / "tiny")
    code = _PROC + textwrap.dedent(f"""
        from bantamtrain import qweights
        qweights.load_quantized({str(tmp_path / 'tiny')!r}, "nf4")      # warm up the imports
        rss0 = mem("VmRSS")
        m = qweights.load_quantized({str(tmp_path / 'big')!r}, "nf4")
        peak = mem("VmHWM")
        logits = m(input_ids=torch.tensor([[1, 2, 3]])).logits
        print(int(torch.isfinite(logits).all()), peak - rss0, qweights.model_bytes(m))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=300)
    finite, growth, qbytes = map(int, out.stdout.split()[-3:])
    assert finite and qbytes < fp32 / 5, out.stderr[-2000:]
    assert growth < 0.45 * fp32, (growth, fp32, qbytes)        # from_pretrained: >= 1.0 * fp32
