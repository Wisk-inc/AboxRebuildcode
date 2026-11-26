"""Wave-2 review (rdw): experiments/summarize.py and plots.py on logs of the 0.3 CLIs."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tooling_summarize import llm_run, summarize, weather_run  # noqa: E402

WX_ARGS = {"data": "/d/era5_64x32", "zoom": "4:0.6,1:0.4", "coarse_mode": "merge", "budget": 5.5e13,
           "optim": "muon", "state_bits": 32, "d": 128, "layers": 6, "heads": 4, "batch": 16,
           "lr": 0.016, "warmup": 0.05, "wd": 0.05, "gated": 1, "merge_mode": "", "seed": 0,
           "name": "main_muon_zoom_s0", "out": "results/weather", "threads": 2}


def test_recipe_pools_older_logs_with_0_3_cli_defaults():
    """Logs written by the 0.3 CLIs carry new args at their defaults (weather: flops_mode,
    ic_stride, max_ics and the --forecast options, among them a non-zero --steps; pretrain:
    data_format, flops_mode, auto_batch). They formed a different recipe from older logs,
    so seed groups and baseline curves split old and new runs of the same recipe."""
    old = {**weather_run(0.16), "args": dict(WX_ARGS)}
    new = {**weather_run(0.15), "args": {**WX_ARGS, "seed": 1, "flops_mode": "model",
                                         "ic_stride": 12, "max_ics": 96, "val_year": None,
                                         "forecast": False, "load": "", "init": "", "steps": 12,
                                         "forecast_out": "", "ckpt": "results/weather/x.ckpt",
                                         "ckpt_every": 0, "resume": "", "stop_step": 0}}
    assert summarize.recipe(old) == summarize.recipe(new)
    g, = summarize.seed_groups({"a_s0": old, "a_s1": new}, lambda r: r["final"]["val_loss"])
    assert g["n"] == 2
    for k, v in (("ic_stride", 14), ("max_ics", 0), ("flops_mode", "total"), ("val_year", 2017)):
        assert summarize.recipe(old) != summarize.recipe({**new, "args": {**new["args"], k: v}})
    lo, ln = llm_run(2.7), llm_run(2.7)
    ln["args"].update(data_format="auto", flops_mode="model", auto_batch=True,
                      allow_arg_change=True)
    assert summarize.recipe(lo) == summarize.recipe(ln)
    ln["args"]["data_format"] = "bin"                 # the raw corpus is another dataset
    assert summarize.recipe(lo) != summarize.recipe(ln)


def test_seed_aggregates_count_distinct_seeds():
    """Two runs of the same seed (results/llm main_muon_base_s0 and up_base_s0) were pooled
    as two samples: n=4 'seeds' for 3 seeds, and an sd from a duplicated sample."""
    runs = {"base_s0": llm_run(2.70, seed=0), "rep_s0": llm_run(2.72, seed=0),
            "base_s1": llm_run(2.80, seed=1)}
    g, = summarize.seed_groups(runs, lambda r: r["final_val_loss"])
    assert g["n"] == 2 and g["mean"] == pytest.approx((2.71 + 2.80) / 2)
    assert g["sd"] == pytest.approx(0.09 / 2 ** 0.5) and len(g["runs"]) == 3
    assert summarize.seed_groups({k: runs[k] for k in ("base_s0", "rep_s0")},
                                 lambda r: r["final_val_loss"]) == []
    zoom = llm_run(2.75, schedule="2:0.5,1:0.5")
    row, = [r for r in summarize.compute_to_target({**runs, "z_s0": zoom}) if r["run"] == "z_s0"]
    assert row["baseline_points"] == [(1e13, pytest.approx((2.71 + 2.80) / 2))]


def test_plot_weather_without_coarse_mode(tmp_path):
    pytest.importorskip("matplotlib")
    import plots
    r = weather_run(0.2)
    r["schedule"] = "4:0.6,1:0.4"                      # a zoom run logged before --coarse_mode
    plots.plot_weather({"main_muon_zoom_s0": r}, ["main_muon_zoom_s0"], "w.png", "t",
                       out=str(tmp_path))
    assert (tmp_path / "w.png").exists()


def test_run_queue_dry_run_and_weather_resume(tmp_path, monkeypatch, capsys):
    """A killed `bantam weather --ckpt_every N` run (no JSON yet: weather writes it at the
    end) restarted from scratch although weather resumes bit-exactly since 0.3; --dry_run
    did not exist."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_pretrain_queue import _write, rq
    root = tmp_path / "root"
    monkeypatch.setattr(rq, "ROOT", str(root))
    monkeypatch.setattr(rq.subprocess, "run", lambda *a, **k: pytest.fail("ran a command"))
    wx = root / "results" / "weather"
    _write(str(wx / "done.json"), {"args": {}, "final": {"val_loss": 0.1}})
    _write(str(wx / "killed.ckpt"), "x")
    q = tmp_path / "q.txt"
    q.write_text("D=/data\n" + "".join(f"bantam weather --data $D --budget 1 --name {n} "
                                       "--ckpt_every 50\n" for n in ("done", "killed", "fresh")))
    assert rq.main([str(q), "--dry_run", "--threads", "3"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "done: finished -> skip"
    assert out[1] == ("killed: missing -> resume: bantam weather --data /data --budget 1 --name "
                      f"killed --ckpt_every 50 --threads 3 --resume {wx / 'killed.ckpt'}")
    assert out[2].startswith("fresh: missing -> run: ") and "--resume" not in out[2]


def test_summarize_lists_diverged_runs_apart(tmp_path):
    """A diverged pretrain log ("status": "diverged", no final) was listed as running, and a
    pre-0.3 NaN final_val_loss was tabulated, so its seed group's mean became NaN."""
    import json
    from test_tooling_summarize import write
    res = tmp_path / "results"
    write(str(res / "llm" / "a_s0.json"), llm_run(2.7, seed=0))
    write(str(res / "llm" / "a_s1.json"), llm_run(2.8, seed=1))
    write(str(res / "llm" / "a_s2.json"), json.dumps(llm_run(2.9, seed=2)).replace("2.9", "NaN"))
    d = llm_run(1.0)                                   # as pretrain writes it: no final loss
    del d["final_val_loss"]
    write(str(res / "llm" / "d_s0.json"), {**d, "status": "diverged", "diverged": True,
                                           "diverged_at_step": 37})
    assert summarize.main(["--results", str(res)]) == 0
    summ = json.loads((res / "summary.json").read_text())
    assert summ["diverged"] == {"llm": {"a_s2": 100, "d_s0": 37}} and summ["running"] == {}
    g, = summ["llm_seed_groups"]
    assert g["n"] == 2 and g["mean"] == pytest.approx(2.75)
    md = (res / "SUMMARY.md").read_text()
    assert "## Diverged" in md and "| llm/d_s0 | 37 |" in md and "| a_s2 |" not in md
