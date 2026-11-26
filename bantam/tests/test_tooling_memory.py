"""bantamtrain.memory: units, activation constants, recompute, fallbacks, QLoRA, FLOPs."""
import contextlib
import os
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import memory  # noqa: E402
from bantamtrain.memory import GIB  # noqa: E402
from bantamtrain.models import ZoomGPT  # noqa: E402


def _need(r, m):
    return r["fixed_bytes"] + m * r["per_seq_bytes"]


def test_capacity_is_binary_with_headroom():
    assert memory.capacity_bytes(96) == 96 * 2 ** 30
    sb = memory.scoreboard(headroom=1.0)
    assert sb["capacity_gb"] == 96
    assert memory.scoreboard()["capacity_gb"] == pytest.approx(96 * 0.85)
    # a configuration needing 100e9 B fits a 96 GiB card (103.1e9 B) but not 96e9 B
    assert 100e9 <= memory.capacity_bytes(96) and 100e9 > 96e9


def test_max_micro_is_tight_and_recompute_is_charged_per_sequence():
    for kw in ({}, {"headroom": 1.0}, {"gpu_gb": 100, "headroom": 1.0}, {"params": 1.5e10}):
        sb = memory.scoreboard(**kw)
        cap = sb["capacity_gb"] * GIB
        for r in sb["rows"]:
            m = r["max_micro"]
            assert (_need(r, m) <= cap or m == 0) and cap < _need(r, m + 1)
    sb = memory.scoreboard(headroom=1.0)
    std, L, d, T, V = sb["rows"][0], sb["layers"], sb["d"], 2048, 32000
    assert std["ckpt"] and std["fits"]
    # fp32 residual under AMP: 4 B stored per block input; the recomputed layer, and the
    # unchunked logits, scale with the micro-batch
    assert std["per_seq_bytes"] == pytest.approx((4 * L + 48.7) * T * d + 14 * T * V)
    assert std["fixed_bytes"] == pytest.approx(std["static_gb"] * GIB + 3 * GIB)


def test_activation_constants_match_zoomgpt():
    """Saved-for-backward bytes per layer*token*d of the real model vs memory.ZOOM_ACT."""
    def per_layer(amp, B=2, T=96, d=128):
        got, dtype = {}, {}
        for L in (2, 4):
            torch.manual_seed(0)
            m = ZoomGPT(512, d, L, 4)
            if not amp:
                m = m.to(torch.bfloat16)
            shapes = {s for p in m.parameters() for s in (tuple(p.shape), tuple(p.shape[::-1]))}
            x = torch.randint(0, 512, (B, T))

            def run():
                ctx = torch.autocast("cpu", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
                with ctx:
                    h = m.hidden(x, 1)
                dtype["resid"] = h.dtype
                h.float().sum()

            _, saved = memory.measure_saved_bytes(run, list(m.parameters()))
            got[L] = sum(b for shape, _, b in saved if shape not in shapes)   # weight casts are in B/param
        return (got[4] - got[2]) / (2 * B * T * d), dtype["resid"]

    for amp, key, resid in ((True, "amp", torch.float32), (False, "bf16", torch.bfloat16)):
        coef, dt = per_layer(amp)
        assert 0.9 * coef <= memory.ZOOM_ACT[key] <= 1.35 * coef, (key, coef)
        assert dt == resid                     # what checkpointing stores per block input


def test_every_row_falls_back_to_recompute():
    sb = memory.scoreboard(params=1.5e10, headroom=1.0)
    rows, N = sb["rows"], sb["params"]
    assert not rows[0]["fits"] and rows[0]["days"] is None
    assert rows[1]["fits"] and rows[1]["ckpt"] and "ckpt" in rows[1]["config"]
    assert rows[1]["flops"] == pytest.approx(8 * N * 1e11)
    out = memory.fmt(sb, 1e11)
    assert "n/a" in out and "reference = row 2" in out
    for r in rows:
        assert r["fits"] or r["days"] is None
    md = memory.fmt(sb, 1e11, markdown=True)
    assert md.count("\n") == len(rows) + 1 and "GiB" in md


def test_muon_newton_schulz_flops_charged():
    assert memory.muon_ns_flops(256, 6) == pytest.approx(3.02e10, rel=0.01)   # 5.8M-param model
    assert memory.muon_ns_flops(4096, 24) == pytest.approx(4.95e14, rel=0.01)
    sb = memory.scoreboard(global_batch_tokens=524288)
    N, r2, r3 = sb["params"], sb["rows"][1], sb["rows"][2]
    ns = sb["muon_ns_flops_per_step"] * 1e11 * 0.52 / 524288
    assert r3["flops"] == pytest.approx(6 * N * 1e11 * 0.52 + ns)
    assert r2["flops"] == pytest.approx(6 * N * 1e11)
    big_batch = memory.scoreboard(global_batch_tokens=4 * 524288)["rows"][2]["flops"]
    assert big_batch < r3["flops"]


def test_flop_convention_is_reported():
    sb = memory.scoreboard()
    N, L, d = sb["params"], sb["layers"], sb["d"]
    assert sb["flops_per_token"] == 6 * N
    assert sb["flops_per_token_nanochat"] == 6 * (N - 32000 * d) + 12 * L * d * 2048
    assert "nanochat" in memory.fmt(sb, 1e11)


def test_ce_chunk_term():
    sb = memory.scoreboard(headroom=1.0)
    r2 = sb["rows"][1]
    assert r2["fixed_bytes"] == pytest.approx(r2["static_gb"] * GIB + 3 * GIB + 2048 * 32000 * 14)
    small = memory.scoreboard(headroom=1.0, ce_chunk=512)["rows"][1]
    assert small["fixed_bytes"] < r2["fixed_bytes"]
    L, N = memory.model_dims(1e9, 2048, 32000)
    chunked = memory.finetune_table(N, 2048, L, 2048, 4, 24, ce_chunk=2048)
    full = memory.finetune_table(N, 2048, L, 2048, 4, 24, ce_chunk=0)
    assert all(b["total_gb"] > a["total_gb"] for a, b in zip(chunked, full))


def test_model_dims_warns_and_tied_head():
    with pytest.warns(UserWarning, match="closest model"):
        memory.scoreboard(params=1e7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        L, N = memory.model_dims(5e9, 4096, 32000)
        Lt, Nt = memory.model_dims(5e9, 4096, 32000, tied=True)
    assert N == 12 * L * 4096 ** 2 + 2 * 32000 * 4096
    assert Nt == 12 * Lt * 4096 ** 2 + 32000 * 4096


def test_storage_table_bits_flag():
    assert memory.storage_table(1e9, 1e11, bits_per_token=8)[-1][1] == 1e11
    assert "4k vocab" in memory.storage_table(1e9, 1e11)[-1][0]


def test_qlora_rows_charge_what_qweights_keeps():
    L, N = memory.model_dims(7e9, 4096, 32000)
    lin = N - 2 * 32000 * 4096
    keeps = {"dequant_bytes_per_param": 2.0, "embed_out_bytes": 4}
    fixed = {"dequant_bytes_per_param": 0.0, "embed_out_bytes": 2}
    a = memory.finetune_table(N, 4096, L, 2048, 4, 48, probe=keeps)
    b = memory.finetune_table(N, 4096, L, 2048, 4, 48, probe=fixed)
    qa, qb = [r for r in a if r["config"].startswith("QLoRA")], [r for r in b if r["config"].startswith("QLoRA")]
    assert (qa[0]["total_gb"] - qb[0]["total_gb"]) * GIB > 2.0 * lin    # dequantized weights + fp32 residual
    assert qa[1]["total_gb"] > qb[1]["total_gb"]                         # 4 B checkpointed inputs
    assert not qa[0]["fits"]                                              # 7B on 48 GB without ckpt: OOM
    lora_no = next(r for r in a if r["config"].startswith("LoRA") and r["ckpt"] == "no")
    assert qa[0]["total_gb"] > lora_no["total_gb"]                       # what the code really does
    # the probe measures the current qweights implementation
    p = memory.qlora_probe()
    assert p["measured"] and p["embed_out_bytes"] in (2, 4)
    assert p["dequant_bytes_per_param"] == pytest.approx(0, abs=0.05) or \
        p["dequant_bytes_per_param"] == pytest.approx(2, abs=0.05)


def test_finetune_table_matches_hf_llama():
    """HF Llama + LoRA on all projections: measured bytes per layer*token*d vs HF_ACT."""
    transformers = pytest.importorskip("transformers")
    from bantamtrain.lora import apply_lora
    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def per_layer(dropout, B=2, T=96, d=128):
        got = {}
        for L in (2, 4):
            torch.manual_seed(0)
            cfg = transformers.LlamaConfig(vocab_size=512, hidden_size=d, intermediate_size=344,
                                           num_hidden_layers=L, num_attention_heads=4,
                                           num_key_value_heads=4, max_position_embeddings=128)
            m = transformers.LlamaForCausalLM(cfg).to(torch.bfloat16)
            apply_lora(m, 8, 16, dropout, targets)
            shapes = {s for mod in m.modules() if hasattr(mod, "in_features")
                      for s in ((mod.out_features, mod.in_features), (mod.in_features, mod.out_features))}
            x = torch.randint(0, 512, (B, T))

            def run():
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    m.model(input_ids=x).last_hidden_state.float().sum()

            _, saved = memory.measure_saved_bytes(run, list(m.parameters()))
            got[L] = sum(b for shape, _, b in saved if shape not in shapes)
        return (got[4] - got[2]) / (2 * B * T * d)

    c0, c5 = per_layer(0.0), per_layer(0.05)
    assert abs(memory.HF_ACT["lora"] - c0) <= 0.15 * c0
    table = memory.HF_ACT["lora"] + memory.HF_ACT["lora_dropout"]
    assert 0.85 * c5 <= table <= 1.5 * c5          # never understates the dropout cost
    L, N = memory.model_dims(1e9, 2048, 32000)
    no_drop = memory.finetune_table(N, 2048, L, 2048, 4, 24, lora_dropout=0.0)
    drop = memory.finetune_table(N, 2048, L, 2048, 4, 24, lora_dropout=0.05)
    assert drop[4]["total_gb"] > no_drop[4]["total_gb"] * 1.3


def test_memory_cli_flags_and_returns_int(capsys):
    rc = memory.main(["--headroom", "1", "--overhead_gb", "2", "--ce_chunk", "1024", "--tied",
                      "--bits_per_token", "8", "--global_batch_tokens", "1e6", "--ns_steps", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "96.0 GiB usable" in out and "8.00 bits/token" in out
    assert memory.main(["--finetune", "--params", "1e9", "--gpu_gb", "24", "--lora_dropout", "0"]) == 0
    assert "20.4 GiB usable" in capsys.readouterr().out
