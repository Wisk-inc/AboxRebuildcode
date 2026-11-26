"""experiments/race_finetune.py: the fine-tuning race harness (HF Trainer + PEFT against
bantam finetune). Offline, CPU, tiny models from test_finetune_fixtures."""
import json
import os
import random
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("transformers")
import race_finetune as race  # noqa: E402
from test_finetune_fixtures import WORDS, rows, save_model_dir, write_jsonl  # noqa: E402


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return save_model_dir(tmp_path_factory.mktemp("race_model"), "chatml", big=True)


def dolly_rows(n, seed=0):
    r = random.Random(seed)

    def text(k):
        return " ".join(r.choice(WORDS) for _ in range(k))
    return [{"instruction": f"  what is {text(r.randint(2, 6))} ",
             "context": text(r.randint(0, 3)) if i % 3 else "",
             "response": f"\n{text(r.randint(1, 8))}  ", "category": "qa"} for i in range(n)]


def test_split_is_deterministic_disjoint_and_stripped():
    rs = dolly_rows(50) + [{"instruction": "x", "context": "", "response": "  "}]   # unusable last row
    a, b = race.split_rows(rs, 30, 10, 1234), race.split_rows(rs, 30, 10, 1234)
    assert a == b and len(a[0]) == 30 and len(a[1]) == 10
    assert not {e["id"] for e in a[0]} & {e["id"] for e in a[1]}
    assert [e["id"] for e in a[0]] != [e["id"] for e in race.split_rows(rs, 30, 10, 1)[0]]
    for e in a[0] + a[1]:
        r = rs[e["id"]]
        assert e["response"] == r["response"].strip()
        ctx = r["context"].strip()
        assert e["prompt"] == r["instruction"].strip() + (f"\n\n{ctx}" if ctx else "")
    with pytest.raises(SystemExit):
        race.split_rows(rs, 45, 10, 1234)


def test_hf_side_does_not_import_bantam_finetune(model_dir):
    """The HF arm's process loads none of bantam (0.3.0.dev's bantamtrain.finetune turned on
    flush-to-zero at import); race_tokenizer keeps its own copy of the ChatML template."""
    import subprocess

    from bantamtrain.finetune import CHATML
    assert race.CHATML == CHATML
    code = (f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'experiments')!r}); import race_finetune as r; "
            f"r.race_tokenizer({model_dir!r}); print('bantamtrain.finetune' in sys.modules)")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)
    assert p.stdout.strip().splitlines()[-1] == "False"


def test_split_skips_examples_with_nothing_to_supervise(model_dir):
    tok = race.race_tokenizer(model_dir)
    rs = dolly_rows(40, seed=4)
    for r in rs[::5]:
        r["context"] = " ".join(["sky"] * 100)                  # the response falls beyond seq
    st = {}
    train, judge = race.split_rows(rs, 25, 6, 1234, lambda e: race.encode(tok, e, 48) is not None, st)
    assert st["unusable"] >= 1 and all(race.encode(tok, e, 48) for e in train + judge)
    assert len(train) == 25 and len(judge) == 6


def test_encode_is_exactly_what_bantam_finetune_trains_on(model_dir, tmp_path):
    """The HF arm's own tokenization (chat template + prompt masking) equals bantam
    finetune's load_examples + tokenize_examples, also for examples cut at seq and examples
    whose response is cut away entirely (dropped by both)."""
    exs = [{"id": i, **race.to_example(r)} for i, r in enumerate(dolly_rows(30, seed=3))]
    exs[4]["prompt"] = " ".join(["two"] * 80)                     # nothing of the response is left
    exs[5]["response"] = " ".join(["red"] * 80)                   # truncated inside the response
    path = str(tmp_path / "d.jsonl")
    race.write_jsonl(path, [race.as_messages(e) for e in exs])
    tok = race.race_tokenizer(model_dir)
    assert tok.eos_token == "<|im_end|>" and tok.pad_token_id is not None
    par = race.token_parity(tok, path, 40)
    assert par == {"examples": 29, "bantam_examples": 29, "mismatched": 0, "ok": True}
    st = {}
    enc = [race.encode(tok, e, 40, st) for e in exs]
    assert enc[4] is None and len(enc[5][0]) == 40 and st["no_target"] == 1 and st["truncated"] >= 2
    ids, labels = enc[0]
    k = labels.index(next(t for t in labels if t != -100))
    assert tok.decode(ids[:k]).endswith("<|im_start|>assistant\n")
    assert tok.decode(ids[k:]) == exs[0]["response"] + "<|im_end|>\n"


def test_judge_equals_bantam_evaluate_on_the_same_examples(model_dir):
    """One example at a time through the model's own forward gives bantam's packed,
    block-diagonal, chunked-CE evaluate() loss: the cross-check the race reports."""
    from transformers import AutoModelForCausalLM

    from bantamtrain import finetune as ft
    from bantamtrain.device import setup_runtime
    tok = race.race_tokenizer(model_dir)
    exs = [race.to_example(r) for r in dolly_rows(12, seed=5)]
    items = [race.encode(tok, e, 48) for e in exs]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    with torch.no_grad():                                      # a non-trivial head and body
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    sc = race.judge_scores(model, items)
    assert [n for _, n in sc] == [sum(1 for t in l[1:] if t != -100) for _, l in items]
    rt = setup_runtime("cpu", "auto", False, 1, 0)
    rt.amp_dtype = None
    windows = ft.pack([(i, l, [1.0] * len(i)) for i, l in items], 48, tok.pad_token_id, "bfd")
    assert len(windows) < len(items)                           # really packed
    ref = ft.evaluate(model, windows, rt, 2, 2048)
    assert race.pooled(sc) == pytest.approx(ref, abs=1e-5)
    lo, hi = race.bootstrap_ci(sc)
    assert lo <= race.pooled(sc) <= hi
    d_lo, d_hi = race.bootstrap_ci(sc, sc)
    assert d_lo == d_hi == 0.0


def test_bantam_plan_replays_the_finetune_run(model_dir, tmp_path):
    """The real (non-pad) tokens and targets the race credits a bantam arm with come from
    replaying its data plan; the replay must agree with finetune's own log.json."""
    from bantamtrain import finetune as ft
    p = {"seq": 64, "batch": 2, "accum": 2, "epochs": 1.0, "seed": 7}
    data = write_jsonl(tmp_path / "t.jsonl", [race.as_messages({"id": i, **r})
                                               for i, r in enumerate(rows(45, seed=2))])
    ev = write_jsonl(tmp_path / "e.jsonl", [race.as_messages({"id": i, **r}) for i, r in enumerate(rows(4))])
    out = tmp_path / "run"
    ft.main(["--model", model_dir, "--jsonl", data, "--eval_jsonl", ev, "--out", str(out), "--seq", "64",
             "--batch", "2", "--accum", "2", "--seed", "7", "--threads", "1", "--eval_every", "0", "--lora", "4",
             "--pack", "bfd", "--save_peft", "0"])
    log = json.load(open(out / "log.json"))
    tok = race.race_tokenizer(model_dir)
    plan = race.bantam_plan(tok, data, p)
    assert race.check_plan(plan, log)
    assert plan["windows_used"] == plan["windows"] == log["windows_used"]       # one epoch: every window
    assert plan["supervised_tokens"] <= plan["supervised_tokens_all"] == log["supervised_tokens"]
    assert sum(plan["step_targets"]) == plan["supervised_tokens"]
    assert not race.check_plan(race.bantam_plan(tok, data, {**p, "seed": 8}), log)


def test_run_measured_sees_the_child_memory_and_cpu(tmp_path):
    code = "import time; b = b'x' * (300 << 20); time.sleep(0.8); print('done')"
    m = race.run_measured([sys.executable, "-c", code], str(tmp_path / "log"), echo=False)
    assert m["returncode"] == 0 and open(tmp_path / "log").read().strip() == "done"
    assert m["peak_rss_mb"] > 300 and m["maxrss_mb"] > 300 and m["samples"] >= 3
    assert m["wall_s"] >= 0.8 and m["cpu_s"] > 0
    bad = race.run_measured([sys.executable, "-c", "raise SystemExit(3)"], str(tmp_path / "log2"), echo=False)
    assert bad["returncode"] == 3


def test_proc_fallback_matches_psutil(monkeypatch):
    if race.psutil is None:
        pytest.skip("psutil not installed")
    with_ps = race.tree_rss(os.getpid())
    monkeypatch.setattr(race, "psutil", None)
    assert race.tree_rss(os.getpid()) == pytest.approx(with_ps, rel=0.2)


def test_dry_run_prints_the_commands_and_one_race_per_dir(model_dir, tmp_path, capsys):
    import fcntl
    data = write_jsonl(tmp_path / "rows.jsonl", dolly_rows(30, seed=1))
    argv = ["--model", model_dir, "--data_jsonl", data, "--n_train", "16", "--n_judge", "6", "--seq", "64",
            "--out", str(tmp_path / "race"), "--dry_run"]
    assert race.main(argv) == 0
    out = capsys.readouterr().out
    assert "bantam_nf4 train: " in out and "--quant nf4 --low_ram 1" in out and "hf_peft judge: " in out
    assert not os.path.exists(tmp_path / "race" / "summary.json")
    with open(tmp_path / "race" / "race.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="another race"):
            race.main(argv)


@pytest.mark.slow
def test_race_end_to_end_and_resume(model_dir, tmp_path):
    """Every arm on a tiny model: records, summary, fairness checks, skip on rerun, and a
    judge-only rerun when only a judge setting changes."""
    data = write_jsonl(tmp_path / "rows.jsonl", dolly_rows(40, seed=1))
    out = tmp_path / "race"
    argv = ["--model", model_dir, "--data_jsonl", data, "--n_train", "24", "--n_judge", "8", "--judge_n", "8",
            "--bantam_eval_n", "4", "--seq", "64", "--threads", "1", "--gen_tokens", "4", "--out", str(out),
            "--quiet", "--max_examples", "20"]
    assert race.main(argv) == 0
    s = json.load(open(out / "summary.json"))
    assert set(s["arms"]) == set(race.DEFAULT_ARMS.split(",")) and s["smoke"]
    f = s["fairness"]
    assert f["token_parity"]["ok"] and f["same_lora_modules"]
    assert len(set(f["supervised_tokens"].values())) == 1
    assert all(v for v in f["bantam_plan_matches_log"].values() if v is not None)
    for n, a in s["arms"].items():
        assert a["judge_examples"] == 8 and len(a["generations"]) == len(race.GEN_PROMPTS)
        if n != "base":
            assert a["peak_rss_mb"] > 50 and a["tokens_per_s"] > 0 and a["real_tokens"] > 0
            assert a["lora_modules"] == 14 and a["est_full_train_min"] > 0
        if n.startswith("bantam"):
            assert abs(a["bantam_eval_minus_judge_subset"]) < 1e-4       # the judge agrees with bantam
    assert "judge_loss_float_base" in s["arms"]["bantam_nf4"]
    assert s["arms"]["hf_peft"]["steps"] == 5 and s["arms"]["bantam"]["steps"] >= 1
    rec = json.load(open(out / "bantam.json"))
    assert rec["versions"]["bantamtrain"] and "--optim muon" in rec["commands"]["train"]
    assert (out / "summary.md").read_text().count("\n| ") == 1 + len(race.DEFAULT_ARMS.split(","))
    mtime = os.path.getmtime(out / "runs" / "hf_peft" / "train_result.json")
    assert race.main(argv) == 0                                             # everything skipped
    assert os.path.getmtime(out / "runs" / "hf_peft" / "train_result.json") == mtime
    assert race.main([*argv, "--judge_n", "6", "--arms", "hf_peft"]) == 0  # re-judged, not retrained
    assert os.path.getmtime(out / "runs" / "hf_peft" / "train_result.json") == mtime
    s2 = json.load(open(out / "summary.json"))
    assert list(s2["arms"]) == ["hf_peft"] and s2["arms"]["hf_peft"]["judge_examples"] == 6
    assert race.main([*argv, "--judge_n", "6", "--arms", "hf_peft", "--force", "hf_peft"]) == 0
    assert os.path.getmtime(out / "runs" / "hf_peft" / "train_result.json") > mtime
