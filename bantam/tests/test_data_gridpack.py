"""GridPack: exact error bound, explicit errors instead of asserts, bounds checks,
thread safety, pickling and stale-pack detection."""
import json
import os
import pickle
import subprocess
import sys
import threading

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bantamtrain.data import gridpack  # noqa: E402


def _walk(T, C, H, W, base, step, seed=0):
    rng = np.random.default_rng(seed)
    base = np.asarray(base, np.float64).reshape(1, C, 1, 1)
    step = np.asarray(step, np.float64).reshape(1, C, 1, 1)
    x = (base + np.cumsum(rng.standard_normal((T, C, H, W)), 0) * step).astype(np.float32)
    return x, np.diff(x, axis=0).std(axis=(0, 2, 3)).astype(np.float32)


def test_gridpack_bound_holds_exactly_in_float64(tmp_path):
    """|x_hat - x| <= err_frac * diff_std with no tolerance, on msl-like 1e5 fields
    (the float32 reconstruction overshot the bound by ~1 ulp before)."""
    worst = 0.0
    for seed in range(12):
        x, dstd = _walk(40, 3, 6, 8, [1.0e5, 5.4e4, -9.0e4], [250.0, 230.0, 10.0], seed)
        ef = [0.05, 0.02, 0.1][seed % 3]
        _, max_err = gridpack.write(tmp_path / "f.gp", x, dstd, ef, chunk=7)
        y = gridpack.GridPack(tmp_path / "f.gp").read_all()
        err = np.abs(y.astype(np.float64) - x.astype(np.float64)).max(axis=(0, 2, 3))
        bound = ef * dstd.astype(np.float64)
        assert (err <= bound).all() and (max_err <= bound).all(), (seed, err / bound)
        assert np.allclose(err, max_err)
        worst = max(worst, float((err / bound).max()))
    assert worst > 0.99                                   # the bound is tight, not loose


def test_gridpack_rejects_nonfinite_and_overflow(tmp_path):
    x, dstd = _walk(20, 2, 4, 8, [1.0e5, 280.0], [250.0, 1.0])
    for bad in (np.nan, np.inf):
        y = x.copy()
        y[3, 1, 2, 2] = bad
        with pytest.raises(ValueError, match="non-finite.*\\[1\\]"):
            gridpack.write(tmp_path / "f.gp", y, dstd)
    with pytest.raises(ValueError, match="int16.*err_frac >="):
        gridpack.write(tmp_path / "f.gp", x, dstd, err_frac=1e-4)    # range needs > 32767 steps
    jumpy = x.copy()
    jumpy[10:, 1] += 5000.0                                          # one huge 6 h change
    with pytest.raises(ValueError, match="one-step change"):
        gridpack.write(tmp_path / "f.gp", jumpy, dstd, err_frac=0.05)
    with pytest.raises(ValueError, match="float32 resolution"):
        gridpack.write(tmp_path / "f.gp", x, np.array([1e-6, 1.0], np.float32))
    with pytest.raises(ValueError):
        gridpack.write(tmp_path / "f.gp", x, np.array([np.nan, 1.0]))
    with pytest.raises(ValueError):
        gridpack.write(tmp_path / "f.gp", x[0], dstd)
    assert not (tmp_path / "f.gp").exists()


def test_gridpack_zero_diff_std_channels(tmp_path):
    x, dstd = _walk(20, 3, 4, 8, [1.0e5, 0.0, 0.0], [250.0, 1.0, 1.0])
    x[:, 1] = 7.25                                        # constant forcing: exact
    x[:, 2] = np.linspace(0, 1, 32, dtype=np.float32).reshape(4, 8)   # static field
    dstd[1:] = 0
    _, max_err = gridpack.write(tmp_path / "f.gp", x, dstd)
    gp = gridpack.GridPack(tmp_path / "f.gp")
    y = gp.read_all()
    assert np.array_equal(y[:, 1], x[:, 1]) and max_err[1] == 0
    assert np.abs(y[:, 2] - x[:, 2]).max() <= gp.h["bounds"][2] < 1e-4


def test_gridpack_errors_survive_python_O(tmp_path):
    code = (f"import sys, numpy as np; sys.path.insert(0, {ROOT!r})\n"
            "from bantamtrain.data import gridpack\n"
            "x = (1e5 + 5000 * np.random.default_rng(0).standard_normal((20, 1, 4, 4))).astype(np.float32)\n"
            "y = x.copy(); y[0, 0, 0, 0] = np.nan\n"
            "for f, ef in ((x, 1e-4), (y, 0.05)):\n"
            "    try:\n"
            f"        gridpack.write({str(tmp_path / 'o.gp')!r}, f, np.array([250.0]), ef); print('wrote')\n"
            "    except ValueError:\n        print('raised')\n")
    out = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert out.stdout.split() == ["raised", "raised"], out.stderr


def test_gridpack_read_bounds(tmp_path):
    x, dstd = _walk(40, 2, 4, 8, [280.0, 0.0], [1.0, 1.0])
    gridpack.write(tmp_path / "f.gp", x, dstd, chunk=16)
    gp = gridpack.GridPack(tmp_path / "f.gp")
    y = gp.read_all()
    for s, n in ((38, 5), (-1, 1), (0, 41), (5, -1)):
        with pytest.raises(IndexError):
            gp.read(s, n)
    assert gp.read(16, 0).shape == (0, 2, 4, 8)
    for s in range(0, 40, 3):
        for n in {1, 7, 16, 40 - s} & set(range(41 - s)):
            assert np.array_equal(gp.read(s, n), y[s:s + n])
    with pytest.raises(IndexError):
        gp.read_chunk(3)


def test_gridpack_threadsafe_and_picklable(tmp_path):
    x, dstd = _walk(256, 2, 16, 32, [280.0, 1.0e5], [1.0, 200.0])
    gridpack.write(tmp_path / "f.gp", x, dstd, chunk=8, level=3)
    gp = gridpack.GridPack(tmp_path / "f.gp")
    ref = [gp.read_chunk(i) for i in range(32)]
    bad, errors = [0], []

    def job(seed):
        r = np.random.default_rng(seed)
        for _ in range(150):
            i = int(r.integers(0, 32))
            try:
                bad[0] += not np.array_equal(gp.read_chunk(i), ref[i])
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

    ts = [threading.Thread(target=job, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert bad[0] == 0 and not errors
    assert np.array_equal(pickle.loads(pickle.dumps(gp)).read(30, 20), np.concatenate(ref)[30:50])


def test_era5_stale_gridpack_detected(tmp_path):
    old, dstd = _walk(40, 2, 4, 8, [280.0, 0.0], [1.0, 1.0])
    gridpack.write(tmp_path / "fields.gridpack", old, dstd, extra={"source": {"prep_id": "a"}})
    meta = {"channels": ["a", "b"], "n_total": 40, "prep_id": "a"}
    json.dump(meta, open(tmp_path / "meta.json", "w"))
    assert gridpack.stale_reason(tmp_path / "fields.gridpack") is None
    json.dump({**meta, "prep_id": "b"}, open(tmp_path / "meta.json", "w"))
    with pytest.raises(gridpack.StalePackError, match="prep_id"):
        gridpack.GridPack(tmp_path / "fields.gridpack")
    json.dump({"channels": ["a", "b"], "n_total": 20}, open(tmp_path / "meta.json", "w"))  # pre-0.3
    with pytest.raises(gridpack.StalePackError, match="n_total=20"):
        gridpack.GridPack(tmp_path / "fields.gridpack")
    assert gridpack.GridPack(tmp_path / "fields.gridpack", verify=False).T == 40
