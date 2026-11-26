"""Wave-2 seams in bantam weather: meta.json val/test splits, stale GridPacks, empty
splits, --duty, forecasts across data gaps, safe checkpoint loading, ZoomViT muon_split."""
import copy
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bantamtrain.cli.weather as W  # noqa: E402
from bantamtrain.data import gridpack  # noqa: E402
from bantamtrain.device import setup_runtime  # noqa: E402
from bantamtrain.models import ZoomViT  # noqa: E402
from test_weather_cli import CH, CPU, SMALL, hours6, make_era5  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_torch_state():
    n, rng = torch.get_num_threads(), torch.get_rng_state()
    yield
    torch.set_num_threads(n)
    torch.set_rng_state(rng)


def train(data, out, name, *extra, budget="3e9"):
    W.main(["--data", data, "--out", str(out), "--name", name, "--budget", budget, *SMALL, *extra])
    with open(os.path.join(str(out), f"{name}.json")) as f:
        return json.load(f)


def _three_years(path):
    """120 times in 2016, 100 in 2017, 80 in 2018, with gaps between the segments."""
    times = np.concatenate([hours6("2016-11-01", 120), hours6("2017-03-01", 100),
                            hours6("2018-01-01", 80)])
    return make_era5(path, times)


def test_meta_val_split_is_used_without_val_year(tmp_path):
    """DATA-5: prepare-era5 --val_year writes n_val/splits; evals run on val, test at the end."""
    d = str(tmp_path / "d")
    x, meta = _three_years(d)
    st = W.train_stats(x, 120)
    meta.update(n_train=120, n_val=100, n_test=80, **st,
                splits={"train": [0, 120], "val": [120, 220], "test": [220, 300]})
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    log = train(d, tmp_path, "m")
    sp = log["split"]
    assert sp["eval"] == "val" and sp["val_from"] == "meta.json"
    assert (sp["train"]["n"], sp["val"]["n"], sp["test"]["n"]) == (120, 100, 80)
    assert sp["test"]["first"] == "2018-01-01T00" and sp["norm_stats_rows"] == 120
    assert "final_val" in log
    assert W.first_test_row({"n_train": 7, "n_total": 9}) == 7      # pre-0.3 meta.json


def test_stale_gridpack_from_another_prepare_run_falls_back_to_npy(tmp_path):
    """DATA-X1: GridPack raises StalePackError for a pack of another prep_id: read fields.npy."""
    d = str(tmp_path)
    x, meta = make_era5(d, hours6("2017-12-25", 64))
    json.dump({**meta, "prep_id": "new"}, open(os.path.join(d, "meta.json"), "w"))
    gridpack.write(os.path.join(d, "fields.gridpack"), x, np.asarray(meta["diff_std"]), chunk=16,
                   extra={"source": {"prep_id": "old"}})          # same shape, newer file
    ds = W.ERA5(d, CPU)
    assert ds.source == "npy" and np.allclose(ds.x[0, 0].numpy() * ds.std[0, 0].item()
                                              + ds.mean[0, 0].item(), x[0, 0], atol=1e-2)
    os.remove(os.path.join(d, "fields.npy"))
    with pytest.raises(gridpack.StalePackError):
        W.ERA5(d, CPU)


def test_empty_eval_splits_raise_clearly(tmp_path):
    """WX-4: no ZeroDivisionError from an empty split."""
    d = str(tmp_path)
    make_era5(d, hours6("2017-12-25", 64))
    ds = W.ERA5(d, CPU)
    rt = setup_runtime("cpu", "fp32", False, 1, 0)
    m = ZoomViT(len(CH) + 6, len(CH), (32, 64), 2, 16, 1, 2)
    empty = torch.arange(0)
    with pytest.raises(ValueError, match="empty split"):
        W.quick_eval(m, ds, empty, rt)
    with pytest.raises(ValueError, match="empty split"):
        W.full_eval(m, ds, torch.arange(3), empty, rt)


def test_duty_is_validated_before_training(tmp_path, capsys):
    d = str(tmp_path)
    make_era5(d, hours6("2017-11-01", 320))
    with pytest.raises(SystemExit) as e:
        train(d, tmp_path, "duty", "--duty", "0")
    assert e.value.code == 2 and "--duty" in capsys.readouterr().err


def test_forecast_scores_only_times_present_in_the_data(tmp_path):
    """A rollout across a gap in meta.json times stops scoring at the gap."""
    d = str(tmp_path / "d")
    _three_years(d)
    ds = W.ERA5(d, CPU)
    m = ZoomViT(len(CH) + 6, len(CH), (32, 64), 2, 16, 1, 2)
    save = str(tmp_path / "m.pt")
    W.save_model(save, m, ds)
    out = str(tmp_path / "f.npy")
    fc = W.forecast(save, d, "2016-11-30T06", steps=4, out=out)   # data jumps to 2017-03-01
    side = json.load(open(tmp_path / "f.json"))
    assert fc.shape[0] == 5 and side["rmse_lead_hours"] == [0, 6, 12]
    assert len(side["rmse"]["z500"]) == 3


class _NotPlainData:
    pass


def test_checkpoint_loading_never_unpickles_arbitrary_objects(tmp_path):
    path = str(tmp_path / "c.ckpt")
    torch.save({"model": {}, "x": _NotPlainData()}, path)
    with pytest.raises(Exception):
        W.load_ckpt(path)


def test_zoomvit_muon_params_retag_qkv_after_deepcopy():
    m = copy.deepcopy(ZoomViT(len(CH) + 6, len(CH), (32, 64), 2, 16, 2, 2))
    for b in m.blocks:
        b.qkv.weight.__dict__.pop("muon_split", None)
    m.muon_params()
    assert all(b.qkv.weight.muon_split == 3 for b in m.blocks)
