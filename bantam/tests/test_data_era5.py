"""prepare-era5 / pack-era5: splits follow --start/--end/--test_year/--val_year, stats come
from training steps only, and re-preparing invalidates an earlier GridPack."""
import contextlib
import io
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
xr = pytest.importorskip("xarray")
pd = pytest.importorskip("pandas")
from bantamtrain.cli import pack_era5, prepare_era5  # noqa: E402
from bantamtrain.data import gridpack  # noqa: E402


@pytest.fixture
def fake_era5(monkeypatch):
    times = pd.date_range("2015-01-01", "2019-12-31T18", freq="6h")
    rng = np.random.default_rng(0)
    T = len(times)
    trend = np.linspace(0, 5, T).reshape(T, 1, 1)          # makes the per-split stats differ

    def f(base, scale, *shape):
        return (base + scale * (rng.standard_normal((T,) + shape) + trend.reshape((T,) + (1,) * len(shape))))

    ds = xr.Dataset({
        "geopotential": (("time", "level", "latitude", "longitude"), f(5.4e4, 300.0, 2, 4, 8).astype(np.float32)),
        "temperature": (("time", "level", "latitude", "longitude"), f(280.0, 5.0, 2, 4, 8).astype(np.float32)),
        "2m_temperature": (("time", "latitude", "longitude"), f(285.0, 5.0, 4, 8).astype(np.float32)),
        "10m_u_component_of_wind": (("time", "latitude", "longitude"), f(0.0, 3.0, 4, 8).astype(np.float32)),
        "10m_v_component_of_wind": (("time", "latitude", "longitude"), f(0.0, 3.0, 4, 8).astype(np.float32)),
        "mean_sea_level_pressure": (("time", "latitude", "longitude"), f(1.0e5, 800.0, 4, 8).astype(np.float32)),
        "land_sea_mask": (("latitude", "longitude"), rng.random((4, 8))),
        "geopotential_at_surface": (("latitude", "longitude"), rng.random((4, 8))),
    }, coords={"time": times, "level": [500, 850], "latitude": np.linspace(-80, 80, 4),
               "longitude": np.arange(0, 360, 45.0)})
    monkeypatch.setattr(xr, "open_zarr", lambda *a, **k: ds)
    return ds


def _prep(out, *argv):
    with contextlib.redirect_stdout(io.StringIO()):
        prepare_era5.main(["--out", str(out), *argv])
    return json.load(open(os.path.join(out, "meta.json")))


def test_prepare_era5_split_follows_arguments(fake_era5, tmp_path):
    for rng in (("2019-01-01", "2019-01-10T18"),      # after the test year: no training data
                ("2016-01-01", "2016-12-31T18"),      # before it: no test data
                ("2017-09-01", "2017-12-31T18"),
                ("2018-01-01", "2018-03-31T18"),
                ("2016-01-01", "2019-06-30T18")):     # 2019 would fall in no split
        with pytest.raises(SystemExit):
            _prep(tmp_path / "bad", "--start", rng[0], "--end", rng[1])
    assert not (tmp_path / "bad").exists()                       # failed before downloading
    with pytest.raises(SystemExit, match="before"):
        _prep(tmp_path / "bad", "--val_year", "2018")

    m = _prep(tmp_path / "a", "--start", "2015-01-01", "--end", "2018-12-31T18")
    x = np.load(tmp_path / "a" / "fields.npy")
    n_tr = sum(t < "2018" for t in m["times"])
    assert (m["n_train"], m["n_val"], m["n_test"], m["n_total"]) == (n_tr, 0, len(x) - n_tr, len(x))
    assert m["splits"] == {"train": [0, n_tr], "val": [n_tr, n_tr], "test": [n_tr, len(x)]}
    assert np.allclose(m["mean"], x[:n_tr].mean(axis=(0, 2, 3)), rtol=1e-6)
    assert np.allclose(m["diff_std"], (x[1:n_tr] - x[:n_tr - 1]).std(axis=(0, 2, 3)), rtol=1e-5)
    assert not np.allclose(m["mean"], x.mean(axis=(0, 2, 3)), rtol=1e-4)   # test year excluded
    assert m["test_year"] == 2018 and m["val_year"] is None and len(m["prep_id"]) == 32

    m = _prep(tmp_path / "b", "--start", "2015-01-01", "--end", "2018-12-31T18", "--val_year", "2017")
    years = [int(t[:4]) for t in m["times"]]
    lo, hi = m["splits"]["val"]
    assert set(years[:lo]) == {2015, 2016} and set(years[lo:hi]) == {2017}
    assert set(years[hi:]) == {2018} and m["n_val"] == hi - lo > 0
    x = np.load(tmp_path / "b" / "fields.npy")
    assert np.allclose(m["std"], x[:m["n_train"]].std(axis=(0, 2, 3)), rtol=1e-5)

    m = _prep(tmp_path / "c", "--start", "2016-01-01", "--end", "2019-12-31T18", "--test_year", "2019")
    assert m["n_test"] == sum(t.startswith("2019") for t in m["times"]) and m["n_train"] > 0


def test_reprepare_era5_invalidates_gridpack(fake_era5, tmp_path):
    out = tmp_path / "d"
    m = _prep(out, "--start", "2016-01-01", "--end", "2018-12-31T18")
    with contextlib.redirect_stdout(io.StringIO()):
        pack_era5.main(["--data", str(out), "--err_frac", "0.05"])
    gp = gridpack.GridPack(out / "fields.gridpack")
    assert gp.h["source"]["prep_id"] == m["prep_id"] and gp.T == m["n_total"]
    st = json.load(open(out / "pack_stats.json"))
    assert all(c["max_abs_err"] <= c["bound"] for c in st["per_channel"].values())
    _prep(out, "--start", "2017-01-01", "--end", "2018-12-31T18")
    assert not (out / "fields.gridpack").exists() and not (out / "pack_stats.json").exists()


def test_failed_reprepare_era5_keeps_existing_pack(fake_era5, tmp_path, monkeypatch):
    """A download that fails halfway leaves the earlier preparation and its GridPack intact."""
    out = tmp_path / "d"
    _prep(out, "--start", "2016-01-01", "--end", "2018-12-31T18")
    with contextlib.redirect_stdout(io.StringIO()):
        pack_era5.main(["--data", str(out)])
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    monkeypatch.setattr(xr, "open_zarr", lambda *a, **k: fake_era5.drop_vars("land_sea_mask"))
    with pytest.raises(KeyError):
        _prep(out, "--start", "2017-01-01", "--end", "2018-12-31T18")
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert gridpack.GridPack(out / "fields.gridpack").T == json.load(open(out / "meta.json"))["n_total"]
