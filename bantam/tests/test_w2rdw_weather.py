"""Wave-2 review (rdw): bantam weather data-source choice and resumed run logs."""
import json
import os
import shutil
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weather_cli import CPU, W, hours6, make_era5, run, train  # noqa: E402

from bantamtrain.data import gridpack  # noqa: E402


def _pack(d, x, meta, **kw):
    gridpack.write(os.path.join(d, "fields.gridpack"), x, np.asarray(meta["diff_std"]), chunk=16,
                   **kw)


def _age(d, name, seconds):
    t = time.time() + seconds
    os.utime(os.path.join(d, name), (t, t))


def test_copied_data_dir_keeps_its_gridpack(tmp_path):
    """cp/rsync can leave fields.gridpack older than fields.npy: a pack that holds the
    data of fields.npy must still be used (it was skipped as 'stale' by mtime)."""
    d = str(tmp_path / "legacy")
    x, meta = make_era5(d, hours6("2017-12-25", 64))
    _pack(d, x, meta)                                   # 0.2 pack: no source prep_id
    _age(d, "fields.gridpack", -100)                    # e.g. `cp src/* dst/` copies it first
    assert W.ERA5(d, CPU).source == "gridpack"
    d2 = str(tmp_path / "new")                          # 0.3: prep_id in meta and pack header
    shutil.copytree(d, d2)
    meta["prep_id"] = "p1"
    json.dump(meta, open(os.path.join(d2, "meta.json"), "w"))
    _pack(d2, x, meta, extra={"source": {"prep_id": "p1"}})
    _age(d2, "fields.gridpack", -100)
    assert W.field_source(d2, meta)[0] == "gridpack"
    assert W.pack_holds_npy(os.path.join(d2, "fields.gridpack"), "/nonexistent.npy", meta)


def test_same_shape_stale_gridpack_is_detected(tmp_path, capsys):
    """A pack left over from another prepare-era5 run with the same shape was used
    silently when it looked newer than fields.npy."""
    d = str(tmp_path)
    times = hours6("2017-12-25", 64)
    x, meta = make_era5(d, times)
    _pack(d, x, meta)
    x2, _ = make_era5(d, times, seed=1)                 # re-prepared in place, same shape
    _age(d, "fields.gridpack", +100)
    with pytest.raises(gridpack.StalePackError, match="does not hold the data of fields.npy"):
        W.ERA5(d, CPU)
    _age(d, "fields.gridpack", -100)
    ds = W.ERA5(d, CPU)
    assert ds.source == "npy" and "stale pack" in capsys.readouterr().out
    assert np.allclose((ds.x * ds.std + ds.mean).numpy(), x2, rtol=1e-6, atol=1e-2)


def test_resumed_log_records_the_resuming_args(tmp_path):
    """The final JSON of a resumed run kept the first invocation's args (--stop_step,
    no --resume); pretrain records the resuming args plus a resume_history."""
    d = str(tmp_path / "era5")
    make_era5(d)
    run(d, tmp_path, "r", "--ckpt_every", "20", "--stop_step", "20", budget="1e10")
    ck = str(tmp_path / "r.ckpt")
    log = train(d, tmp_path, "r", "--resume", ck, budget="1e10")
    assert log["args"]["stop_step"] == 0 and log["args"]["resume"] == ck
    assert log["resume_history"] == [{"step": 20, "args": {**log["resume_history"][0]["args"],
                                                           "stop_step": 20, "resume": ""}}]
    assert log["steps"] > 20


def test_resume_refuses_schedule_changes(tmp_path):
    """--resume with another --lr / --budget silently changed the schedule midway (only
    model options were compared)."""
    d = str(tmp_path / "era5")
    make_era5(d)
    run(d, tmp_path, "r", "--ckpt_every", "10", "--stop_step", "10", budget="1e10")
    ck = str(tmp_path / "r.ckpt")
    for extra, budget in ((("--lr", "2e-3"), "1e10"), ((), "2e10"), (("--seed", "1"), "1e10")):
        with pytest.raises(SystemExit):
            run(d, tmp_path, "r", "--resume", ck, *extra, budget=budget)
    run(d, tmp_path, "r", "--resume", ck, "--threads", "2", "--evals", "3", budget="1e10")
