"""Download a small ERA5 subset (WeatherBench2, 5.625 deg, 64x32, 6-hourly).

Writes one float32 array (time, channel, lat, lon) plus normalization stats, so
training reads contiguous local memory instead of a remote zarr store. (float16
is too coarse here: at z500 ~ 5e4 its step is 32 m2/s2, the size of a 6 h error.)

Channels: z500, t850, t2m, u10, v10, msl.

Splits come from the requested range: --test_year (default 2018) is the test
set, the optional --val_year the validation set, and every earlier time step
in --start..--end is training data. The layout is contiguous
[train | val | test] (meta.json: n_train, n_val, n_test, splits) and the
normalization statistics use the training steps only. The default range
2010-01-01..2018-12-31T18 gives train 2010-2017, test 2018.
"""
import argparse
import hashlib
import json
import os
import uuid

import numpy as np

URL = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr")
CHANNELS = [
    ("z500", "geopotential", 500),
    ("t850", "temperature", 850),
    ("t2m", "2m_temperature", None),
    ("u10", "10m_u_component_of_wind", None),
    ("v10", "10m_v_component_of_wind", None),
    ("msl", "mean_sea_level_pressure", None),
]
ROLLOUT = 13          # steps a held-out split needs: one start + 72 h (12 steps) of rollout
STALE = ("fields.gridpack", "pack_stats.json", "meta.json")


def split_times(times, test_year, val_year=None):
    """(n_train, n_val, n_test) for sorted time strings, or SystemExit with the reason.

    train = every step before the first held-out year, val = val_year, test =
    test_year; all of the range must fall in one of them, in that order.
    """
    years = np.array([int(str(t)[:4]) for t in times], dtype=np.int64)
    if val_year is not None and val_year >= test_year:
        raise SystemExit(f"--val_year ({val_year}) must come before --test_year ({test_year})")
    first = test_year if val_year is None else val_year
    n_train = int((years < first).sum())
    n_val = int((years == val_year).sum()) if val_year is not None else 0
    n_test = int((years == test_year).sum())
    other = sorted(set(years.tolist()) - set(range(years.min(initial=0), first))
                   - {val_year, test_year}) if len(years) else []
    rng = f"{times[0]}..{times[-1]}" if len(times) else "an empty range"
    if other:
        raise SystemExit(f"years {other} of {rng} fall in no split (train < "
                         f"{'val_year < ' if val_year is not None else ''}test_year): "
                         f"adjust --start/--end or --test_year/--val_year")
    if n_train < 2:
        raise SystemExit(f"{rng} has {n_train} training steps before {first}; "
                         f"need >= 2: move --start earlier or the held-out year later")
    if n_test < ROLLOUT + 1:
        raise SystemExit(f"{rng} has {n_test} steps in --test_year {test_year}; need > {ROLLOUT}: "
                         f"extend --end or pick another --test_year")
    if val_year is not None and n_val < ROLLOUT + 1:
        raise SystemExit(f"{rng} has {n_val} steps in --val_year {val_year}; need > {ROLLOUT}")
    return n_train, n_val, n_test


def train_stats(x, n_train):
    """Per-channel mean, std and one-step-change std over the training steps x[:n_train]."""
    xt = x[:n_train]
    mean = xt.mean(axis=(0, 2, 3))
    std = xt.std(axis=(0, 2, 3))
    diff = x[1:n_train] - x[:n_train - 1]
    diff_std = diff.std(axis=(0, 2, 3))
    for name, v in (("mean", mean), ("std", std), ("diff_std", diff_std)):
        if not np.isfinite(v).all() or (name != "mean" and (v <= 0).any()):
            raise SystemExit(f"bad training {name} {v.tolist()}: missing or constant data?")
    return mean, std, diff_std


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bantam prepare-era5", description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2018-12-31T18")
    ap.add_argument("--test_year", type=int, default=2018)
    ap.add_argument("--val_year", type=int, default=None,
                    help="hold out this year for validation (e.g. 2017); default: none")
    args = ap.parse_args(argv)
    try:
        import xarray as xr
    except ImportError:
        raise SystemExit("prepare-era5 needs xarray and zarr: pip install 'bantamtrain[weather]'")

    ds = xr.open_zarr(URL, consolidated=True, chunks=None)
    ds = ds.sel(time=slice(args.start, args.end))
    times = ds.time.values.astype("datetime64[h]").astype(str)
    n_train, n_val, n_test = split_times(times, args.test_year, args.val_year)  # before download
    os.makedirs(args.out, exist_ok=True)
    for name in STALE:   # files of an earlier preparation no longer match
        if os.path.exists(os.path.join(args.out, name)):
            os.remove(os.path.join(args.out, name))
    fields = []
    for name, var, level in CHANNELS:
        da = ds[var]
        if level is not None:
            da = da.sel(level=level)
        arr = da.transpose("time", "latitude", "longitude").values.astype(np.float32)
        print(name, arr.shape, float(arr.mean()), float(arr.std()), flush=True)
        fields.append(arr)
    x = np.stack(fields, axis=1)  # (T, C, lat, lon)
    del fields
    lat = ds.latitude.values
    lon = ds.longitude.values
    lsm = ds["land_sea_mask"].transpose("latitude", "longitude").values
    oro = ds["geopotential_at_surface"].transpose("latitude", "longitude").values
    mean, std, diff_std = train_stats(x, n_train)

    np.save(os.path.join(args.out, "fields.npy"), x)
    np.save(os.path.join(args.out, "static.npy"),
            np.stack([lsm, (oro - oro.mean()) / oro.std()]).astype(np.float32))
    files = {}
    for name in ("fields.npy", "static.npy"):
        path, h = os.path.join(args.out, name), hashlib.blake2b()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        files[name] = {"bytes": os.path.getsize(path), "blake2b": h.hexdigest()}
    n_total = len(times)
    meta = {"channels": [c[0] for c in CHANNELS], "times": [str(t) for t in times],
            "lat": lat.tolist(), "lon": lon.tolist(), "mean": mean.tolist(),
            "std": std.tolist(), "diff_std": diff_std.tolist(),
            "n_train": n_train, "n_total": n_total, "n_val": n_val, "n_test": n_test,
            "splits": {"train": [0, n_train], "val": [n_train, n_train + n_val],
                       "test": [n_train + n_val, n_total]},
            "test_year": args.test_year, "val_year": args.val_year,
            "start": args.start, "end": args.end, "prep_id": uuid.uuid4().hex, "files": files}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print({k: v for k, v in meta.items() if k not in ("times", "lat", "lon", "files")})


if __name__ == "__main__":
    main()
