"""Aggregate results/*.json into Markdown tables (and optional PNG plots).

    python experiments/summarize.py [--plots]

Writes results/SUMMARY.md. Groups: LLM sweep, LLM main, LLM ablations,
LLM speedup-to-target, weather. Speedup-to-target: for a reference run
(baseline at the same budget) and a Zoom run, interpolate the baseline's
val-loss-vs-FLOPs curve over the longer baseline runs to find the FLOPs at
which the baseline first reaches the Zoom run's final loss.
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(sub):
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, sub, "*.json"))):
        out[os.path.basename(p)[:-5]] = json.load(open(p))
    return out


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def flops_to_reach(curve_runs, target):
    """Smallest FLOPs at which any baseline eval curve is <= target (linear interp)."""
    best = None
    for r in curve_runs:
        ev = r["evals"]
        f = np.array([e["flops"] for e in ev])
        v = np.array([e["val_loss"] for e in ev])
        for i in range(1, len(f)):
            if v[i] <= target:
                if v[i - 1] <= target:
                    x = f[i - 1]
                else:
                    x = f[i - 1] + (f[i] - f[i - 1]) * (v[i - 1] - target) / (v[i - 1] - v[i])
                best = x if best is None else min(best, x)
                break
    return best


def llm_rows(runs):
    rows = []
    for name, r in runs.items():
        a = r["args"]
        rows.append({"run": name, "optim": a["optim"], "zoom": r["schedule"],
                     "out": a.get("out_mode", ""), "gated": a.get("gated", ""),
                     "rope": a.get("rope_pos", ""), "lr": a["lr"], "seed": a["seed"],
                     "budget": f"{a['budget']:.2e}", "steps": r["steps"],
                     "tokens(M)": f"{r['tokens']/1e6:.2f}",
                     "epochs": f"{r.get('data_epochs', 0):.3f}",
                     "train_s": f"{r['train_time']:.0f}",
                     "val_loss": f"{r['final_val_loss']:.4f}"})
    return rows


def weather_rows(runs):
    rows = []
    for name, r in runs.items():
        a, fin = r["args"], r["final"]
        rows.append({"run": name, "optim": a["optim"], "zoom": r["schedule"],
                     "mode": a["coarse_mode"], "lr": a["lr"], "seed": a["seed"],
                     "budget": f"{a['budget']:.2e}", "steps": r["steps"],
                     "epochs": f"{r['epochs']:.2f}", "train_s": f"{r['train_time']:.0f}",
                     "val_mse": f"{fin['val_loss']:.4f}",
                     "z500@6h": f"{fin['rmse_z500_6h']:.1f}",
                     "z500@24h": f"{fin['rmse_z500_24h']:.1f}",
                     "z500@72h": f"{fin['rmse_z500_72h']:.1f}",
                     "t850@24h": f"{fin['rmse_t850_24h']:.2f}",
                     "t2m@24h": f"{fin['rmse_t2m_24h']:.2f}"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    parts = ["# Results summary (auto-generated)\n"]

    sweep = load("llm_sweep")
    if sweep:
        parts.append("## LLM learning-rate sweep (2.3e13 FLOPs)\n")
        parts.append(md_table(llm_rows(sweep), ["run", "optim", "lr", "steps", "val_loss"]))

    wsweep = load("weather_sweep")
    if wsweep:
        parts.append("\n\n## Weather learning-rate sweep (1.8e13 FLOPs)\n")
        parts.append(md_table(weather_rows(wsweep), ["run", "optim", "lr", "steps", "val_mse",
                                                     "z500@6h", "z500@24h", "z500@72h"]))

    main_runs = load("llm")
    if main_runs:
        parts.append("\n\n## LLM main runs and ablations\n")
        cols = ["run", "optim", "zoom", "out", "gated", "rope", "lr", "seed", "budget", "steps",
                "tokens(M)", "epochs", "train_s", "val_loss"]
        parts.append(md_table(llm_rows(main_runs), cols))
        base_long = [r for n, r in main_runs.items() if r["schedule"] == "1:1"]
        if base_long:
            parts.append("\n\n### Compute-to-target (baseline FLOPs needed to reach each run's final loss)\n")
            rows = []
            for n, r in main_runs.items():
                need = flops_to_reach(base_long, r["final_val_loss"])
                rows.append({"run": n, "val_loss": f"{r['final_val_loss']:.4f}",
                             "run_flops": f"{r['args']['budget']:.2e}",
                             "baseline_flops_to_match": f"{need:.2e}" if need else "not reached",
                             "speedup": f"{need / r['args']['budget']:.2f}x" if need else "> curve"})
            parts.append(md_table(rows, ["run", "val_loss", "run_flops", "baseline_flops_to_match",
                                         "speedup"]))

    wx = load("weather")
    if wx:
        parts.append("\n\n## Weather (ERA5 5.625 deg, 6 h step, test year 2018)\n")
        cols = ["run", "optim", "zoom", "mode", "lr", "seed", "budget", "steps", "epochs",
                "train_s", "val_mse", "z500@6h", "z500@24h", "z500@72h", "t850@24h", "t2m@24h"]
        parts.append(md_table(weather_rows(wx), cols))
        any_run = next(iter(wx.values()))["final"]
        parts.append("\n\nPersistence RMSE on the same initial conditions: "
                     + ", ".join(f"{c}@{h}h={any_run[f'persist_{c}_{h}h']:.2f}"
                                 for h in (6, 24, 72) for c in ("z500", "t850", "t2m")))

    with open(os.path.join(RES, "SUMMARY.md"), "w") as f:
        f.write("\n".join(parts) + "\n")
    print("\n".join(parts))

    if args.plots:
        from plots import make_plots  # noqa: F401
        make_plots(main_runs, wx)


if __name__ == "__main__":
    main()
