"""Wave-3 fine-tuning fixes: sliding-window layers in packed windows, full epochs, the
--examples_per_step accumulation, flush-to-zero only in main(), --low_ram with .bin-only
checkpoints, no --out dir on early failures, stale docstring / result metadata."""
import json
import math
import os
import random
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from test_finetune_fixtures import make_tokenizer, rows, run_args, save_model_dir, tiny_llama, write_jsonl

from bantamtrain import finetune as ft

pytest.importorskip("transformers")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def chatml_dir(tmp_path_factory):
    return save_model_dir(tmp_path_factory.mktemp("chatml"), "chatml")


def _run(model_dir, data, out, *extra):
    ft.main(run_args(model_dir, data, out, *extra))
    with open(os.path.join(out, "log.json")) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- sliding windows
def _windowed(name, attn):
    import transformers as tf
    torch.manual_seed(0)
    kw = {"vocab_size": 200, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
          "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 64,
          "initializer_range": 0.2}
    cfg = {"mistral": lambda: tf.MistralConfig(**kw, sliding_window=8),
           "gemma2": lambda: tf.Gemma2Config(**kw, head_dim=8, query_pre_attn_scalar=8, sliding_window=8),
           "gemma3": lambda: tf.Gemma3TextConfig(**kw, head_dim=8, query_pre_attn_scalar=8, sliding_window=8,
                                                 sliding_window_pattern=2),
           "qwen2": lambda: tf.Qwen2Config(**kw, use_sliding_window=True, sliding_window=8,
                                           max_window_layers=1)}[name]()
    cfg._attn_implementation = attn
    return tf.AutoModelForCausalLM.from_config(cfg).eval()


# Gemma-2 has no SDPA path in transformers 5.24 (it refuses attn_implementation="sdpa")
@pytest.mark.parametrize("name,attn,mode", [("mistral", "sdpa", "all"), ("mistral", "eager", "all"),
                                            ("gemma2", "eager", "mapping"), ("gemma3", "sdpa", "mapping"),
                                            ("gemma3", "eager", "mapping"), ("qwen2", "sdpa", "mapping")])
def test_packed_examples_keep_the_sliding_window(name, attn, mode):
    """Packed (and padded) windows passed one block-diagonal mask to every layer, so a
    sliding-window layer attended to the whole example: at --seq > window each packed
    example's logits differed from its logits alone (Mistral uniform window; Gemma-2/3 and
    Qwen2 alternating local/global layers)."""
    m = _windowed(name, attn)
    torch.manual_seed(1)
    exs = [torch.randint(1, 200, (k,)).tolist() for k in (14, 11, 5)]
    for pack in ("bfd", "none"):                    # "none": one example per window, padded
        windows = ft.pack([(e, e, [1.0] * len(e)) for e in exs], 31, 0, pack)
        assert len(windows) == (1 if pack == "bfd" else 3)
        for win in windows:
            x, y, w, g = (torch.tensor([v]) for v in win)
            seg = g[0, :-1]
            with torch.no_grad():
                alone, ref_sum, n_ref = {}, 0.0, 0
                for s in range(int(seg.max()) + 1):
                    i = (seg == s).nonzero()[:, 0]
                    alone[s] = i, m(input_ids=x[:, :-1][:, i]).logits[0]
                    ref_sum += F.cross_entropy(alone[s][1][:-1].float(), x[0, i[1:]], reduction="sum").item()
                    n_ref += len(i) - 1
                for chunk in (0, 8):                # the full logits and the chunked head
                    s_, n = ft.lm_loss_sum(m, x[:, :-1], y[:, 1:], chunk, w[:, 1:], g[:, :-1])
                    assert abs(s_.item() - ref_sum) < 1e-4 and int(n) == n_ref
                out = m(input_ids=x[:, :-1], **ft.packed_attention(m, g[:, :-1])).logits[0]
                for i, ref in alone.values():
                    assert torch.allclose(out[i], ref, atol=1e-5, rtol=0)
    assert ft.window_layout(m) == (8, mode)


def test_window_layout_without_a_window_or_a_known_use_changes_nothing():
    import transformers as tf
    m = tf.LlamaForCausalLM(tf.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                           num_attention_heads=2))
    assert ft.window_layout(m) == (None, None)
    q = tf.Qwen2ForCausalLM(tf.Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                           num_attention_heads=2, sliding_window=8, use_sliding_window=False))
    assert ft.window_layout(q) == (None, None)                     # use_sliding_window=False
    seg = torch.tensor([[0, 0, 0, 1, 1, -1]])
    mask, pos = ft.segment_mask(seg)
    kw = ft.packed_attention(m, seg)
    assert torch.equal(kw["attention_mask"], mask) and torch.equal(kw["position_ids"], pos)
    assert ft.packed_attention(m, torch.zeros(1, 6, dtype=torch.long)) == {}


def test_window_layout_notes_a_mix_it_cannot_pass(capsys):
    """Mixed sliding/full layers behind a decoder that takes one mask: noted once, and the
    plain block-diagonal mask is kept (what every layer got before)."""
    class Cfg:
        sliding_window, layer_types = 4, ["sliding_attention", "full_attention"]

    class Odd(torch.nn.Module):
        config = Cfg()

        def forward(self, input_ids=None, attention_mask=None, position_ids=None):
            return attention_mask

    m = Odd()
    assert ft.window_layout(m) == (4, "none") and ft.window_layout(m) == (4, "none")
    assert capsys.readouterr().out.count("cannot get their own mask") == 1
    seg = torch.tensor([[0] * 6 + [1] * 4])
    assert torch.equal(ft.packed_attention(m, seg)["attention_mask"], ft.segment_mask(seg)[0])


# ----------------------------------------------------------------------------- full epochs
@pytest.mark.parametrize("n,batch,accum", [(10, 4, 1), (10, 2, 2), (7, 3, 2), (235, 2, 2), (5, 8, 1), (12, 2, 3)])
def test_an_epoch_takes_every_window_once(n, batch, accum):
    """--epochs 1 used to stop at windows // (batch*accum) full steps: the windows that did
    not fill a last optimizer step were skipped (3 of 235 in the race)."""
    spe = ft.steps_per_epoch(n, batch, accum)
    assert spe == math.ceil(math.ceil(n / batch) / accum) and ft.plan_steps(n, 1, batch, accum, 2.0, 0) == (n, 2 * spe)
    rng, order, pos = random.Random(0), list(range(n)), 0
    rng.shuffle(order)
    for _ in range(3):                               # three epochs
        seen = []
        for k in range(spe):
            idx, pos = ft.step_windows(order, pos, rng, batch, accum)
            assert 1 <= len(idx) <= accum and all(1 <= len(mb) <= batch for mb in idx)
            if k < spe - 1:
                assert len(idx) == accum and all(len(mb) == batch for mb in idx)
            seen += [i for mb in idx for i in mb]
        assert sorted(seen) == list(range(n))         # every window exactly once
    assert ft.plan_steps(n, 1, batch, accum, 1.0, 3) == (n, 3)   # --max_steps still wins


def test_drop_last_is_the_old_stream():
    """--drop_last 1 (and checkpoints written before full epochs) keep the 0.2 schedule."""
    def old(order, rng, batch, accum, steps):
        pos, out = 0, []
        for _ in range(steps):
            for _ in range(accum):
                if pos + batch > len(order):
                    rng.shuffle(order)
                    pos = 0
                out.append(order[pos:pos + batch])
                pos += batch
        return out
    for n, b, a in ((10, 4, 1), (7, 3, 2), (235, 2, 2)):
        assert ft.steps_per_epoch(n, b, a, drop_last=True) == max(1, n // (b * a))
        r1, r2 = random.Random(3), random.Random(3)
        o1, o2 = list(range(n)), list(range(n))
        pos, new = 0, []
        for _ in range(7):
            idx, pos = ft.step_windows(o1, pos, r1, b, a, drop_last=True)
            new += idx
        assert new == old(o2, r2, b, a, 7)


def test_the_short_last_step_is_normalized_by_its_own_tokens():
    """A last step with fewer (and smaller) micro-batches is the mean over its own targets:
    micro-batches [w0, w1] + [w2] give the loss and gradients of one batch [w0, w1, w2]."""
    from bantamtrain.device import setup_runtime
    tok = make_tokenizer("chatml")
    ft.resolve_template(tok)
    tok.pad_token = tok.eos_token
    win = ft.pack(ft.tokenize_examples(tok, rows(9), 32), 32, tok.pad_token_id, "bfd")[:3]
    m = tiny_llama(len(tok))
    rt = setup_runtime("cpu", "auto", False, 1, 0)
    rt.amp_dtype = None
    fwd = ft.LossModule(m)
    res = []
    for groups in ([[0, 1], [2]], [[0, 1, 2]]):
        m.zero_grad()
        loss, n = ft.accumulate(fwd, [ft._batch(win, g, "cpu") for g in groups], rt, 0)
        res.append((loss, n, [p.grad.clone() for p in m.parameters()]))
    assert res[0][1] == res[1][1] and abs(res[0][0] - res[1][0]) < 1e-6
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(res[0][2], res[1][2]))


def test_finetune_epochs_train_every_window_and_log_it(chatml_dir, tmp_path, monkeypatch):
    data = write_jsonl(tmp_path / "d.jsonl", rows(40, seed=3))
    drawn, real = [], ft._batch
    monkeypatch.setattr(ft, "_batch", lambda w, idx, dev: (drawn.append(list(idx)) if isinstance(idx, list)
                                                           else None) or real(w, idx, dev))
    log = _run(chatml_dir, data, tmp_path / "o", "--accum", 2, "--epochs", 2, "--pack", "none")
    n = log["train_windows"]
    flat = [i for mb in drawn for i in mb]
    assert n % 4                                                # not a multiple of batch x accum
    assert len(flat) == 2 * n and sorted(flat[:n]) == sorted(flat[n:]) == list(range(n))
    spe = math.ceil(math.ceil(n / 2) / 2)
    assert log["steps_per_epoch"] == spe and log["steps"] == 2 * spe and log["windows_used"] == n
    assert log["examples_per_window"] == 1.0 and abs(log["examples_per_step"] - n / spe) < 1e-9
    old = _run(chatml_dir, data, tmp_path / "d", "--accum", 2, "--epochs", 1, "--pack", "none", "--drop_last", 1)
    assert old["steps"] == old["steps_per_epoch"] == n // 4 and old["windows_used"] == 4 * (n // 4)


def test_resuming_a_checkpoint_from_before_full_epochs_keeps_its_schedule(chatml_dir, tmp_path, capsys):
    """A checkpoint without the new options continues with the drop-last schedule it was
    planned with, so it still ends where the uninterrupted old run ends."""
    data = write_jsonl(tmp_path / "d.jsonl", rows(40, seed=3))
    common = ("--accum", 2, "--pack", "none", "--lr", 1e-2, "--save_peft", 0)
    ref = _run(chatml_dir, data, tmp_path / "ref", *common, "--drop_last", 1)
    ft.main(run_args(chatml_dir, data, tmp_path / "o", *common, "--drop_last", 1, "--save_every", 1,
                     "--stop_step", 2))
    p = tmp_path / "o" / "checkpoint" / "state.pt"
    st = torch.load(p, weights_only=False)
    for k in ("drop_last", "examples_per_step", "ftz"):        # as a 0.3.0.dev checkpoint
        st["args"].pop(k)
    torch.save(st, p)
    capsys.readouterr()
    log = _run(chatml_dir, data, tmp_path / "o", *common, "--resume", tmp_path / "o")
    assert "predates full epochs" in capsys.readouterr().out
    assert log["args"]["drop_last"] == 1 and log["steps"] == ref["steps"]
    assert [e["loss"] for e in log["train"]] == [e["loss"] for e in ref["train"]]
    assert log["final_eval_loss"] == ref["final_eval_loss"]
    with pytest.raises(SystemExit, match="examples_per_step"):
        ft.main(run_args(chatml_dir, data, tmp_path / "o", *common, "--examples_per_step", 9,
                         "--resume", tmp_path / "o"))


# ----------------------------------------------------------------------------- examples per step
def test_examples_per_step_picks_accum_and_steps_log_what_they_see(chatml_dir, tmp_path, capsys):
    data = write_jsonl(tmp_path / "d.jsonl", rows(60, seed=4))
    capsys.readouterr()
    base = _run(chatml_dir, data, tmp_path / "a", "--max_steps", 1)
    out = capsys.readouterr().out
    per_ex = base["examples_per_window"]
    assert per_ex > 2 and base["args"]["accum"] == 1          # several short examples per window
    assert abs(base["examples_per_step"] - per_ex * base["train_windows"] / base["steps_per_epoch"]) < 1e-9
    assert f"packing put {per_ex:.1f} examples per window" in out and "--examples_per_step" in out
    log = _run(chatml_dir, data, tmp_path / "b", "--examples_per_step", 16, "--max_steps", 1)
    want = max(1, round(16 / (2 * per_ex)))
    assert want > 1 and log["args"]["accum"] == want and log["args"]["examples_per_step"] == 16
    assert log["steps_per_epoch"] == ft.steps_per_epoch(log["train_windows"], 2, want)
    assert abs(log["examples_per_step"] - 16) < per_ex * 2      # within one micro-batch of N
    assert f"-> --accum {want}" in capsys.readouterr().out
    one = _run(chatml_dir, data, tmp_path / "c", "--pack", "none", "--max_steps", 1)
    assert one["examples_per_window"] == 1.0 and "packing put" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        ft.main(run_args(chatml_dir, data, tmp_path / "e", "--examples_per_step", -1))


# ----------------------------------------------------------------------------- flush-to-zero
_PROBE = "float(torch.tensor([1e-40], dtype=torch.float32) * 2) == 0.0"


def test_importing_finetune_or_chat_leaves_flush_to_zero_alone():
    """`import bantamtrain.finetune` called torch.set_flush_denormal(True): a process-wide
    side effect for anything importing its helpers (bantam chat, the race's HF arm)."""
    code = f"import torch; import bantamtrain.finetune, bantamtrain.cli.chat; print({_PROBE})"
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
                       check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def test_ftz_flag_sets_flush_to_zero_in_main(chatml_dir, tmp_path):
    before = eval(_PROBE)
    data = write_jsonl(tmp_path / "d.jsonl", rows(24))
    try:
        for v in (0, 1):
            log = _run(chatml_dir, data, tmp_path / str(v), "--ftz", v, "--eval_only", 1)
            assert log["args"]["ftz"] == v and eval(_PROBE) == bool(v)
    finally:
        torch.set_flush_denormal(before)


# ----------------------------------------------------------------------------- --low_ram, .bin
def _bin_only_dir(path):
    """A tiny model (big enough for --quant) whose weights are pytorch_model.bin only; transformers
    5 saves safetensors whatever safe_serialization says, so the .bin is written directly."""
    tok = make_tokenizer("chatml")
    m = tiny_llama(len(tok), 256, 512, 4, 1)
    m.config.bos_token_id = m.config.eos_token_id = tok.eos_token_id
    m.save_pretrained(path, safe_serialization=False)
    for f in os.listdir(path):
        if f.endswith((".safetensors", ".safetensors.index.json")):
            os.remove(os.path.join(path, f))
    torch.save(m.state_dict(), os.path.join(path, "pytorch_model.bin"))
    tok.save_pretrained(path)
    return str(path), m


def test_load_quantized_falls_back_for_bin_only_checkpoints(tmp_path):
    from bantamtrain import qweights
    d, ref = _bin_only_dir(tmp_path / "m")
    with pytest.warns(UserWarning, match="no .safetensors"):
        m = qweights.load_quantized(d, 8)
    assert m.quant_report["low_ram"] is False and "from_pretrained" in m.quant_report["fallback"]
    qweights.quantize_model(ref.eval(), 8, min_numel=1 << 16, embed_bits=8)
    x = torch.randint(0, 100, (1, 9))
    with torch.no_grad():
        assert torch.equal(m(input_ids=x).logits, ref(input_ids=x).logits)
    with pytest.raises(FileNotFoundError):
        qweights.load_quantized(d, 8, fallback=False)


def test_low_ram_finetune_runs_on_a_bin_only_checkpoint(tmp_path, capsys):
    d, _ = _bin_only_dir(tmp_path / "m")
    data = write_jsonl(tmp_path / "d.jsonl", rows(24))
    with pytest.warns(UserWarning):
        log = _run(d, data, tmp_path / "o", "--quant", "nf4", "--low_ram", 1, "--max_steps", 1)
    assert log["report"]["quant"]["low_ram"] is False and log["quantized_base_params"] > 0
    assert "--low_ram: no .safetensors weights" in capsys.readouterr().out


# ----------------------------------------------------------------------------- no empty --out
def test_cli_without_transformers_reports_the_extra_and_creates_nothing(tmp_path, monkeypatch, capsys):
    from bantamtrain import cli
    monkeypatch.setitem(sys.modules, "transformers", None)
    threads = torch.get_num_threads()
    try:
        assert cli.main(["finetune", "--model", "gpt2", "--out", str(tmp_path / "D"), "--threads", "1"]) == 1
    finally:
        torch.set_num_threads(threads)
    assert "pip install 'bantamtrain[finetune]'" in capsys.readouterr().err
    assert not (tmp_path / "D").exists()


def test_failures_before_training_leave_no_out_dir(chatml_dir, tmp_path, monkeypatch, capsys):
    """--out was created right after importing transformers, so a missing `datasets`, an
    unreadable --model or a missing data source left an empty dir behind."""
    from bantamtrain import cli
    data = write_jsonl(tmp_path / "d.jsonl", rows(24))
    with pytest.raises(SystemExit) as e:                        # no --dataset / --jsonl: usage error
        ft.main(["--model", chatml_dir, "--out", str(tmp_path / "a"), "--threads", "1"])
    assert e.value.code == 2 and not (tmp_path / "a").exists()
    with pytest.raises(OSError):
        ft.main(run_args(str(tmp_path / "no_such_model"), data, tmp_path / "b"))
    assert not (tmp_path / "b").exists()
    with pytest.raises(SystemExit, match="no usable rows"):
        ft.main(run_args(chatml_dir, write_jsonl(tmp_path / "bad.jsonl", [{"text": "x"}]), tmp_path / "c"))
    assert not (tmp_path / "c").exists()
    monkeypatch.setitem(sys.modules, "datasets", None)
    capsys.readouterr()
    assert cli.main(["finetune", "--model", chatml_dir, "--dataset", "x/y", "--out", str(tmp_path / "d"),
                     "--threads", "1"]) == 1
    assert "needs `datasets`" in capsys.readouterr().err and not (tmp_path / "d").exists()


# ----------------------------------------------------------------------------- stale strings
def test_docstring_example_parses_and_recommends_no_muon():
    doc = ft.__doc__
    cmd = doc[doc.index("python -m bantamtrain.finetune"):].split("\n\n")[0].replace("\\\n", " ").split()[3:]
    assert "--quant" in cmd and cmd[cmd.index("--quant") + 1] == "nf4" and "muon" not in cmd
    with pytest.raises(SystemExit) as e:                        # every flag and value is accepted
        ft.main(cmd + ["--help"])
    assert e.value.code == 0


def test_quantization_quality_metadata_matches_its_logs():
    res = os.path.join(ROOT, "results", "finetune")
    with open(os.path.join(res, "quantization_quality.json")) as f:
        meta = json.load(f)
    with open(os.path.join(res, "quant_eval_0", "log.json")) as f:
        log = json.load(f)
    n_eval = int(log["examples"] * log["args"]["eval_frac"])
    assert n_eval == 88 and meta["eval"].startswith(f"{n_eval} held-out")
    assert f"{log['eval_windows']} packed windows" in meta["eval"]
