"""experiments/summarize.py and experiments/plots.py on synthetic and committed results."""
import json
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import summarize  # noqa: E402

BASE_ARGS = {"data": "/x/tok4k", "zoom": "1", "budget": 1e13, "optim": "muon", "d": 256, "layers": 6,
             "heads": 4, "seq": 256, "batch": 16, "lr": 0.016, "warmup": 0.05, "wd": 0.1,
             "gated": 1, "out_mode": "slot", "rope_pos": "token", "eval_seqs": 256,
             "final_eval_seqs": 2048, "seed": 0, "threads": 2, "name": "x", "out": "results/llm"}


def llm_run(final, budget=1e13, seed=0, schedule="1:1", **kw):
    """A completed run whose eval curve sits 0.28 below its final loss (different val subset)."""
    a = {**BASE_ARGS, "budget": budget, "seed": seed, "zoom": schedule if schedule != "1:1" else "1", **kw}
    evals = [{"frac": f, "flops": f * budget, "val_loss": final - 0.28 + (1 - f)} for f in (0.5, 1.0)]
    return {"args": a, "schedule": schedule, "evals": evals, "final_val_loss": final, "steps": 100,
            "tokens": 1e6, "train_time": 10.0}


def weather_run(mse):
    fin = {"val_loss": mse, **{f"rmse_{c}_{h}h": 1.0 for c in ("z500", "t850", "t2m") for h in (6, 24, 72)},
           **{f"persist_{c}_{h}h": 2.0 for c in ("z500", "t850", "t2m") for h in (6, 24, 72)}}
    return {"args": {"optim": "muon", "lr": 0.016, "seed": 0, "budget": 1e13, "zoom": "1"},
            "schedule": "1:1", "final": fin, "steps": 10, "epochs": 0.5, "train_time": 1.0}


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(obj if isinstance(obj, str) else json.dumps(obj))


def test_summarize_skips_running_and_non_run_files(tmp_path):
    res = tmp_path / "results"
    write(str(res / "llm" / "main_muon_base_s0.json"), llm_run(3.0))
    write(str(res / "llm" / "up_polar_s0.json"),
          {"args": BASE_ARGS, "schedule": "1:1", "evals": [{"frac": 0.601, "val_loss": 3}],
           "in_progress": True})
    write(str(res / "llm" / "zoom_compute_to_target.json"), {"baseline_mean": 2.7})
    write(str(res / "llm" / "truncated.json"), '{"args": {')
    write(str(res / "weather" / "main_muon_base_s0.json"), weather_run(0.25))
    assert summarize.main(["--results", str(res)]) == 0
    md = (res / "SUMMARY.md").read_text()
    assert "| main_muon_base_s0 |" in md and "0.2500" in md
    assert "llm/up_polar_s0 | 0.601" in md                      # listed as running, not tabulated
    assert "llm/zoom_compute_to_target.json" in md and "llm/truncated.json" in md
    summ = json.loads((res / "summary.json").read_text())
    assert summ["running"] == {"llm": {"up_polar_s0": 0.601}}


def test_compute_to_target_uses_final_losses_and_matched_baseline():
    runs = {"main_muon_base_s0": llm_run(3.0, 1e13), "long_muon_base_2x_s0": llm_run(2.8, 2e13),
            "zoom_s0": llm_run(2.9, 1e13, schedule="2:0.5,1:0.5", batch_scale_k=1),
            "main_adamw_base_s0": llm_run(2.0, 1e13, optim="adamw", lr=1e-3),
            "up_relu2_s0": llm_run(1.0, 3e13, mlp="relu2")}
    rows = {r["run"]: r for r in summarize.compute_to_target(runs)}
    z = rows["zoom_s0"]
    assert z["baseline_points"] == [(1e13, 3.0), (2e13, 2.8)]   # no AdamW, no relu2, no curve points
    assert z["baseline_flops"] == pytest.approx(1e13 * math.sqrt(2))
    assert z["speedup"] == pytest.approx(math.sqrt(2)) and z["status"] == "ok"
    assert rows["main_muon_base_s0"]["speedup"] == 1.0                   # never below 1x vs itself
    assert rows["main_adamw_base_s0"]["speedup"] == 1.0
    assert rows["up_relu2_s0"]["baseline_points"] == [(1e13, 3.0), (2e13, 2.8)]
    assert rows["up_relu2_s0"]["status"] == "beyond"
    # the committed numbers: 1.0x (2.7341) and 1.25x (2.6366) seed-0 baselines, target 2.6851
    runs = {"b1": llm_run(2.7341, 6.8e13), "b125": llm_run(2.6366, 8.5e13),
            "z": llm_run(2.6851, 6.8e13, schedule="2:0.5,1:0.5")}
    z = {r["run"]: r for r in summarize.compute_to_target(runs)}["z"]
    assert z["speedup"] == pytest.approx(1.1187, abs=1e-3)


def test_flops_to_reach_first_crossing():
    curve = [(1e13, 0.9), (2e13, 1.1), (3e13, 0.95)]
    assert summarize.flops_to_reach(curve, 1.0) == (1e13, "below")
    f, s = summarize.flops_to_reach([(1e13, 3.0), (4e13, 2.0)], 2.5)
    assert s == "ok" and f == pytest.approx(2e13)
    assert summarize.flops_to_reach([(1e13, 3.0)], 2.0) == (None, "beyond")


def test_seed_aggregation_mean_sd():
    runs = {f"fu_s{i}": llm_run(v, seed=i, schedule="2:0.5,1:0.5") for i, v in enumerate((2.6, 2.7, 2.8))}
    runs["other_s0"] = llm_run(3.0)
    g, = summarize.seed_groups(runs, lambda r: r["final_val_loss"])
    assert g["label"] == "fu" and g["n"] == 3
    assert g["mean"] == pytest.approx(2.7) and g["sd"] == pytest.approx(0.1)
    rows = {r["run"]: r for r in summarize.compute_to_target(runs)}
    assert "fu (mean of 3)" in rows and rows["fu (mean of 3)"]["val_loss"] == pytest.approx(2.7)


def test_recipe_fills_defaults_of_older_logs():
    old = llm_run(2.7)
    new = llm_run(2.7)
    new["args"].update(sched="cosine", mlp="gelu", softcap=0.0, ns="quintic", some_new_flag=0)
    assert summarize.recipe(old) == summarize.recipe(new)
    new["args"]["mlp"] = "relu2"
    assert summarize.recipe(old) != summarize.recipe(new)
    assert summarize.recipe(old, reset=True) == summarize.recipe(new, reset=True)


def test_summarize_on_committed_results(tmp_path):
    res = os.path.join(ROOT, "results")
    if not os.path.isdir(os.path.join(res, "llm")):
        pytest.skip("no committed results")
    assert summarize.main(["--results", res, "--out", str(tmp_path / "SUMMARY.md")]) == 0
    summ = json.loads((tmp_path / "summary.json").read_text())
    rows = {r["run"]: r for r in summ["compute_to_target"]}
    for r in rows.values():
        if r["run"].startswith("main_muon_base"):
            assert r["speedup"] == 1.0
    if "fu_2-50_bk (mean of 3)" in rows:
        assert rows["fu_2-50_bk (mean of 3)"]["speedup"] == pytest.approx(1.12, abs=0.02)


def test_plot_llm_labels_final_loss(tmp_path):
    pytest.importorskip("matplotlib")
    import plots
    runs = {"main_muon_base_s0": llm_run(2.7341), "main_muon_zoom_s0": llm_run(2.8, schedule="2:0.5,1:0.5")}
    labels = plots.plot_llm(runs, list(runs), "x.png", "t", out=str(tmp_path))
    assert labels == ["2.734", "2.800"]                      # final_val_loss, not evals[-1]
    assert (tmp_path / "x.png").exists()
    assert plots.COLORS["adamw/lowres"] != plots.COLORS["muon/lowres"]
    plots.make_plots(runs, {"main_muon_lowres_s0": weather_run(0.2)}, out=str(tmp_path))
    assert (tmp_path / "llm_main.png").exists()
