"""summarize.py marks completed runs whose log says needs_rerun; the committed results keep
their unfinished and bf16-scored runs flagged."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import summarize  # noqa: E402

ARGS = {"data": "/x/tok4k", "zoom": "1", "budget": 1.5e13, "optim": "muon", "lr": 0.016, "seed": 0,
        "state_bits": 8, "param_dtype": "bf16"}


def _run(final, **kw):
    return {"args": ARGS, "schedule": "1:1", "evals": [], "final_val_loss": final, "steps": 10,
            "tokens": 1e5, "train_time": 1.0, **kw}


def test_needs_rerun_is_marked_in_tables_and_summary(tmp_path):
    d = tmp_path / "results" / "llm_lowmem"
    d.mkdir(parents=True)
    (d / "ok.json").write_text(json.dumps(_run(3.58)))
    (d / "bf16.json").write_text(json.dumps(_run(3.5751953125, needs_rerun=True,
                                                 needs_rerun_reason="bf16 CE")))
    assert summarize.main(["--results", str(tmp_path / "results")]) == 0
    md = (tmp_path / "results" / "SUMMARY.md").read_text()
    assert "| 3.5752 * |" in md and "| 3.5800 |" in md
    assert "needs a rerun: bf16: bf16 CE" in md
    summ = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert summ["needs_rerun"] == {"llm_lowmem": {"bf16": "bf16 CE"}}


def test_committed_unfinished_and_bf16_runs_are_flagged():
    res = os.path.join(ROOT, "results")
    long10 = json.load(open(os.path.join(res, "published", "long_muon_10x.json")))
    assert long10["status"] == "killed" and long10["in_progress"] and "final_val_loss" not in long10
    for name in ("lm_muon8_bf16", "full_muon8_bf16"):
        r = json.load(open(os.path.join(res, "llm_lowmem", f"{name}.json")))
        assert r["args"]["param_dtype"] == "bf16" and r["needs_rerun"] and r["eval_arithmetic"] == "bf16"
