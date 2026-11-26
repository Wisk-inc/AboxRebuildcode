"""Wave-2 seams: ZoomSchedule.stage_at past the end, summarize.py run filtering and
weather columns."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.zoom import ZoomSchedule  # noqa: E402
from test_tooling_summarize import llm_run, summarize, weather_run, write  # noqa: E402


def test_stage_at_the_end_is_the_last_stage_with_a_share():
    """A trailing zero-length stage is never the answer (the end-at-k warning agrees)."""
    s = ZoomSchedule([(4, 0.6), (1, 0.4), (2, 0.0)])
    assert s.stage_at(0.3) == (4, 0.0, 0.6)
    assert s.stage_at(1.0) == (1, 0.6, 1.0) and s.stage_at(1.0 - 1e-13)[0] == 1
    assert s.stage_at(math.nan)[0] == 1 and s.stage_at(2.0)[0] == 1
    assert ZoomSchedule.parse("4:0.5,1:0.5").stage_at(1.0) == (1, 0.5, 1.0)


def test_summarize_does_not_tabulate_unfinished_status(tmp_path):
    """PRE-1: a log whose status is not "finished" (e.g. diverged) is not a result."""
    res = str(tmp_path)
    write(os.path.join(res, "llm", "ok_s0.json"), llm_run(2.9))
    write(os.path.join(res, "llm", "div_s0.json"), {**llm_run(9.9), "status": "diverged"})
    write(os.path.join(res, "llm", "fin_s0.json"), {**llm_run(2.8), "status": "finished"})
    runs, running, _ = summarize.scan("llm", res)
    assert set(runs) == {"ok_s0", "fin_s0"} and set(running) == {"div_s0"}


def test_weather_tables_show_the_val_split_and_acc(tmp_path):
    """--val_year runs: LR sweeps select on final_val (valsplit_mse), not on the test year."""
    res = str(tmp_path)
    r = weather_run(0.5)
    r["final"]["acc_z500_72h"] = 0.875
    write(os.path.join(res, "weather_sweep", "v_lr1.json"), {**r, "final_val": {"val_loss": 0.25}})
    write(os.path.join(res, "weather", "w_s0.json"), weather_run(0.4))
    assert summarize.main(["--results", res]) == 0
    md = open(os.path.join(res, "SUMMARY.md")).read()
    sweep = md.split("## Weather learning-rate sweep")[1].split("##")[0]
    assert "valsplit_mse" in sweep and "| 0.2500 |" in sweep
    main = md.split("## Weather (ERA5")[1]
    assert "acc z500@72h" in main and "w_s0" in main
    rows = summarize.weather_rows({"v": {**r, "final_val": {"val_loss": 0.25}}})
    assert rows[0]["valsplit_mse"] == "0.2500" and rows[0]["acc z500@72h"] == "0.875"
    assert json.load(open(os.path.join(res, "summary.json")))["running"] == {}
