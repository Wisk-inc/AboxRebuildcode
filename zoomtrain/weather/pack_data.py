"""Pack ERA5 fields.npy into an error-bounded GridPack and report compression.

    python weather/pack_data.py --data DIR [--err_frac 0.05]

Also reports, per channel, the quantization error next to the one-step
change std and the float16 error, so the bound can be judged against what a
6 h forecast model can resolve.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from zoomtrain.data import gridpack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--err_frac", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--level", type=int, default=10)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    x = np.load(os.path.join(args.data, "fields.npy"))
    dstd = np.asarray(meta["diff_std"], dtype=np.float32)
    path = os.path.join(args.data, "fields.gridpack")
    t0 = time.time()
    nbytes, max_err = gridpack.write(path, x, dstd, args.err_frac, args.chunk, args.level)
    wdt = time.time() - t0
    t0 = time.time()
    y = gridpack.GridPack(path).read_all()
    rdt = time.time() - t0
    f16 = x.astype(np.float16).astype(np.float32)
    per = {}
    for i, c in enumerate(meta["channels"]):
        per[c] = {"diff_std": float(dstd[i]), "max_abs_err": float(np.abs(y[:, i] - x[:, i]).max()),
                  "rms_err": float(np.sqrt(((y[:, i] - x[:, i]) ** 2).mean())),
                  "float16_max_abs_err": float(np.nan_to_num(np.abs(f16[:, i] - x[:, i]),
                                                             posinf=1e30).max()),
                  "float16_overflow": bool(np.isinf(x[:, i].astype(np.float16)).any())}
    stats = {"float32_bytes": int(x.nbytes), "float16_bytes": int(x.nbytes // 2),
             "gridpack_bytes": int(nbytes), "ratio_vs_float32": x.nbytes / nbytes,
             "bits_per_value": 8 * nbytes / x.size, "err_frac": args.err_frac,
             "write_seconds": wdt, "read_all_seconds": rdt,
             "read_MB_per_s": x.nbytes / 1e6 / rdt, "per_channel": per}
    with open(os.path.join(args.data, "pack_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
