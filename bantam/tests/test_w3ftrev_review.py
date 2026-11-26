"""Review of the wave-3 fine-tuning fixes: bantam chat names only the settings it took from
a run that quantized its base, and the race harness replays `--drop_last 1` runs (the
schedule of the published race)."""
import json
import os
import sys

import pytest
import torch
from test_finetune_fixtures import rows, run_args, save_model_dir, write_jsonl

from bantamtrain import finetune as ft

pytest.importorskip("transformers")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A float-base LoRA run and one over an int8 base with float embeddings."""
    root = tmp_path_factory.mktemp("rev")
    d = save_model_dir(root / "m", "chatml", big=True)
    data = write_jsonl(root / "d.jsonl", rows(24))
    ft.main(run_args(d, data, root / "float", "--max_steps", 1, "--save_peft", 0))
    ft.main(run_args(d, data, root / "q8", "--quant", 8, "--embed_quant", "none", "--max_steps", 1,
                     "--save_peft", 0))
    return d, str(root / "float"), str(root / "q8")


@pytest.mark.parametrize("which,extra,note", [
    ("float", ["--quant", "nf4"], None),                        # nothing came from that float run
    ("q8", ["--quant", "nf4"], "--embed_quant none"),           # only the embeddings did
    ("q8", [], "--quant 8 --embed_quant none"),
    ("q8", ["--embed_quant", "8"], "--quant 8"),
])
def test_chat_names_only_the_settings_taken_from_training(runs, which, extra, note, capsys):
    """A float run records embed_quant 8 (finetune's default); `--quant nf4` on its adapter
    was reported as 'quantized as in training: --quant nf4 --embed_quant 8'."""
    from bantamtrain.cli import chat
    d, flt, q8 = runs
    capsys.readouterr()
    assert chat.main(["--model", d, "--adapter", flt if which == "float" else q8, "--prompt", "hi",
                      "--max_new", "2", "--temperature", "0", *extra]) == 0
    err = [ln for ln in capsys.readouterr().err.splitlines() if "as in training" in ln]
    if note is None:
        assert err == []
    else:
        assert len(err) == 1 and err[0].startswith(f"[chat] base quantized as in training: {note} (")


@pytest.fixture(scope="module")
def race():
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import race_finetune
    return race_finetune


def test_race_plan_replays_a_drop_last_run(race, tmp_path):
    """bantam_plan replayed only the full-epoch schedule, so a race arm run with
    `--drop_last 1` (to reproduce the published runs) failed its plan check."""
    d = save_model_dir(tmp_path / "m", "chatml", big=True)
    p = {"seq": 64, "batch": 2, "accum": 2, "epochs": 1.0, "seed": 7}
    data = write_jsonl(tmp_path / "t.jsonl", [race.as_messages({"id": i, **r}) for i, r in enumerate(rows(45, seed=2))])
    ev = write_jsonl(tmp_path / "e.jsonl", [race.as_messages({"id": i, **r}) for i, r in enumerate(rows(4))])
    out, ftz = tmp_path / "run", float(torch.tensor([1e-40]) * 2) == 0.0
    try:
        ft.main(["--model", d, "--jsonl", data, "--eval_jsonl", ev, "--out", str(out), "--seq", "64", "--batch",
                 "2", "--accum", "2", "--seed", "7", "--threads", "1", "--eval_every", "0", "--lora", "4",
                 "--pack", "bfd", "--save_peft", "0", "--drop_last", "1", "--ftz", "0"])
    finally:
        torch.set_flush_denormal(ftz)
    with open(out / "log.json") as f:
        log = json.load(f)
    tok = race.race_tokenizer(d)
    plan = race.bantam_plan(tok, data, p, "bfd", drop_last=True)
    assert log["train_windows"] % 4 and plan["windows_used"] == log["windows_used"] < plan["windows"]
    assert race.check_plan(plan, log) and not race.check_plan(race.bantam_plan(tok, data, p), log)
    s = race.bantam_train_result(str(out), plan)["settings"]
    assert s["drop_last"] == 1 and s["flush_denormal"] is False          # from the run's own args
