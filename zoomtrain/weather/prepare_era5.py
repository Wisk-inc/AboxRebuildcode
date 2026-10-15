"""Download a small ERA5 subset (WeatherBench2, 5.625 deg, 64x32, 6-hourly).

Writes one float32 array (time, channel, lat, lon) plus normalization stats, so
training reads contiguous local memory instead of a remote zarr store. (float16
is too coarse here: at z500 ~ 5e4 its step is 32 m2/s2, the size of a 6 h error.)

Channels: z500, t850, t2m, u10, v10, msl.
"""
import argparse
import json
import os

import numpy as np
import xarray as xr

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2018-12-31T18")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = xr.open_zarr(URL, consolidated=True, chunks=None)
    ds = ds.sel(time=slice(args.start, args.end))
    fields = []
    for name, var, level in CHANNELS:
        da = ds[var]
        if level is not None:
            da = da.sel(level=level)
        arr = da.transpose("time", "latitude", "longitude").values.astype(np.float32)
        print(name, arr.shape, float(arr.mean()), float(arr.std()), flush=True)
        fields.append(arr)
    x = np.stack(fields, axis=1)  # (T, C, lat, lon)
    times = ds.time.values.astype("datetime64[h]").astype(str)
    lat = ds.latitude.values
    lon = ds.longitude.values
    lsm = ds["land_sea_mask"].transpose("latitude", "longitude").values
    oro = ds["geopotential_at_surface"].transpose("latitude", "longitude").values

    train_mask = np.array([t < "2018" for t in times])
    mean = x[train_mask].mean(axis=(0, 2, 3))
    std = x[train_mask].std(axis=(0, 2, 3))
    diff = x[1:][train_mask[1:]] - x[:-1][train_mask[1:]]
    diff_std = diff.std(axis=(0, 2, 3))

    np.save(os.path.join(args.out, "fields.npy"), x)
    np.save(os.path.join(args.out, "static.npy"),
            np.stack([lsm, (oro - oro.mean()) / oro.std()]).astype(np.float32))
    meta = {"channels": [c[0] for c in CHANNELS], "times": [str(t) for t in times],
            "lat": lat.tolist(), "lon": lon.tolist(), "mean": mean.tolist(),
            "std": std.tolist(), "diff_std": diff_std.tolist(),
            "n_train": int(train_mask.sum()), "n_total": len(times)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print({k: v for k, v in meta.items() if k not in ("times", "lat", "lon")})


if __name__ == "__main__":
    main()
