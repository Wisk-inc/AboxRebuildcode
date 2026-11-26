"""Pack ERA5 fields.npy into an error-bounded GridPack and report compression.

    bantam pack-era5 --data DIR [--err_frac 0.05]

Also reports, per channel, the quantization error next to the one-step
change std and the float16 error, so the bound can be judged against what a
6 h forecast model can resolve. The pack records the prep_id of DIR/meta.json,
so a later prepare-era5 into DIR invalidates it.
"""
import argparse
import json
import os
import time

import numpy as np

from ..data import gridpack  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bantam pack-era5", description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--err_frac", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--level", type=int, default=10)
    args = ap.parse_args(argv)
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    x = np.load(os.path.join(args.data, "fields.npy"), mmap_mode="r")
    if "n_total" in meta and x.shape[0] != meta["n_total"]:
        raise SystemExit(f"fields.npy has {x.shape[0]} steps but meta.json says n_total="
                         f"{meta['n_total']}: re-run prepare-era5")
    x = np.ascontiguousarray(x, dtype=np.float32)
    dstd = np.asarray(meta["diff_std"], dtype=np.float32)
    path = os.path.join(args.data, "fields.gridpack")
    t0 = time.time()
    nbytes, max_err = gridpack.write(path, x, dstd, args.err_frac, args.chunk, args.level,
                                     extra={"source": {"prep_id": meta.get("prep_id")}})
    wdt = time.time() - t0
    t0 = time.time()
    y = gridpack.GridPack(path).read_all()
    rdt = time.time() - t0
    per = {}
    for i, c in enumerate(meta["channels"]):
        with np.errstate(over="ignore"):      # float16 overflow is what is being measured
            f16 = x[:, i].astype(np.float16).astype(np.float32)
        per[c] = {"diff_std": float(dstd[i]),
                  "max_abs_err": float(np.abs(y[:, i].astype(np.float64) - x[:, i]).max()),
                  "rms_err": float(np.sqrt(((y[:, i] - x[:, i]) ** 2).mean())),
                  "bound": float(args.err_frac * np.float64(dstd[i])),
                  "float16_max_abs_err": float(np.nan_to_num(np.abs(f16 - x[:, i]),
                                                             posinf=1e30).max()),
                  "float16_overflow": bool(np.isinf(f16).any())}
    stats = {"float32_bytes": int(x.nbytes), "float16_bytes": int(x.nbytes // 2),
             "gridpack_bytes": int(nbytes), "ratio_vs_float32": x.nbytes / nbytes,
             "bits_per_value": 8 * nbytes / x.size, "err_frac": args.err_frac,
             "write_seconds": wdt, "read_all_seconds": rdt,
             "read_MB_per_s": x.nbytes / 1e6 / max(rdt, 1e-9), "per_channel": per,
             "prep_id": meta.get("prep_id")}
    with open(os.path.join(args.data, "pack_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
