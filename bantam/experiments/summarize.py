"""Aggregate results/*.json into Markdown tables (and optional PNG plots).

    python experiments/summarize.py [--plots] [--results DIR] [--out SUMMARY.md]

Writes results/SUMMARY.md (raw tables; RESULTS.md is curated from it) and
results/summary.json (seed aggregates and compute-to-target, machine-readable).
Only completed runs are tabulated: logs still marked in_progress, or with a status
other than "finished" (e.g. "killed"), are listed as running or interrupted, diverged
runs (status "diverged", or a NaN final loss from before 0.3) as diverged, and JSON
files that are not run logs (derived tables) are skipped. Weather tables add the
validation-split MSE of --val_year runs (valsplit_mse: select LRs on it, not on the
test-year val_mse) and the z500 72 h anomaly correlation.

Seed aggregates: runs with the same recipe (args minus seed, name, paths, threads,
eval cadence) and budget are pooled into n, mean and sample sd over distinct seeds
(replicate runs of one seed are averaged into one sample first).

Compute-to-target: the baseline curve is (budget, final_val_loss) of COMPLETED
baseline runs, each with its own full LR schedule, averaged over seeds at each
budget. The baseline recipe of a run is its own recipe with every Zoom and upgrade
option reset to the default (same optimizer, LR, model, data and eval set). The
FLOPs at which the curve reaches a run's final loss come from linear interpolation
in log-FLOPs. Eval-curve points (eval_seqs windows) are never compared with final
losses (final_eval_seqs windows): they are different metrics. Runs of the baseline
recipe are 1.00x by definition.
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
# args that never change what is trained or measured
NON_RECIPE = {"seed", "name", "out", "threads", "device", "precision", "compile", "duty", "save",
              "save_every_eval", "ckpt_every", "ckpt", "resume", "stop_step", "evals", "eval_seqs",
              "grad_ckpt", "auto_batch", "allow_arg_change",
              # bantam weather --forecast options (not used when training)
              "forecast", "load", "init", "steps", "forecast_out"}
# flags added after the first runs, at the value older logs implicitly used
DEFAULTS = {"llm": {"state_bits": 32, "param_dtype": "fp32", "accum": 1, "batch_scale": 0.0,
                    "batch_scale_k": 0, "sched": "cosine", "decay_frac": 0.45, "lr_restart": 0,
                    "mix_fine": 0.0, "merge_mode": "", "qk_norm": 0, "mlp": "gelu", "softcap": 0.0,
                    "zero_init_proj": 0, "ns": "quintic", "warmup": 0.05, "gated": 1,
                    "out_mode": "slot", "rope_pos": "token", "data_format": "auto",
                    "flops_mode": "model"},
            "weather": {"state_bits": 32, "coarse_mode": "merge", "batch_scale_k": 0,
                        "sched": "cosine", "decay_frac": 0.45, "ns": "quintic", "lr_restart": 0,
                        "mix_fine": 0.0, "merge_mode": "", "gated": 1, "warmup": 0.05,
                        "flops_mode": "model", "ic_stride": 12, "max_ics": 96}}
# Zoom and upgrade options: the baseline recipe has all of them at their default
VARIANT = {"zoom": "1", "batch_scale_k": 0, "batch_scale": 0.0, "mix_fine": 0.0, "lr_restart": 0,
           "merge_mode": "", "gated": 1, "out_mode": "slot", "rope_pos": "token", "mlp": "gelu",
           "softcap": 0.0, "zero_init_proj": 0, "sched": "cosine", "decay_frac": 0.45,
           "ns": "quintic", "qk_norm": 0, "coarse_mode": "merge"}
CTT_METHOD = ("baseline = completed runs with the run's optimizer, LR, model, data and eval set and "
              "every Zoom/upgrade option at its default; final_val_loss averaged over seeds at each "
              "budget; FLOPs to reach the target by linear interpolation in log-FLOPs; eval-curve "
              "points are never mixed with final losses; baseline-recipe runs are 1.00x")


def _diverged(r):
    if r.get("status") == "diverged" or r.get("diverged"):
        return True
    fin = r.get("final_val_loss", (r.get("final") or {}).get("val_loss"))
    return isinstance(fin, float) and not math.isfinite(fin)


def scan(sub, res=None, diverged=None):
    """(complete runs, {running name: last eval frac}, [skipped non-run files]) in res/sub.
    Diverged runs go into the `diverged` dict ({name: step}) when one is passed, else
    they are listed as running; they are never complete."""
    runs, running, skipped = {}, {}, []
    for p in sorted(glob.glob(os.path.join(res or RES, sub, "*.json"))):
        name = os.path.basename(p)[:-5]
        try:
            with open(p) as f:
                r = json.load(f)
        except (OSError, ValueError):
            skipped.append(name)
            continue
        if not isinstance(r, dict) or "args" not in r:
            skipped.append(name)                    # derived table, not a run log
        elif _diverged(r) and diverged is not None:
            diverged[name] = r.get("diverged_at_step", r.get("steps"))
        elif (r.get("in_progress") or _diverged(r) or r.get("status", "finished") != "finished"
              or ("final_val_loss" not in r and "final" not in r)):    # running, killed
            ev = r.get("evals") or [{}]
            running[name] = ev[-1].get("frac")
        else:
            runs[name] = r
    return runs, running, skipped


def load(sub, res=None):
    """Completed runs only."""
    return scan(sub, res)[0]


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def _f(x, spec):
    return "" if x is None else format(x, spec)


def kind(r):
    return "weather" if "final" in r and "final_val_loss" not in r else "llm"


def recipe(r, budget=True, reset=False):
    """Hashable recipe key: args minus NON_RECIPE, older logs filled with DEFAULTS, falsy
    values dropped (a flag at its off value is the same recipe as a log without it)."""
    a = {**DEFAULTS[kind(r)], **r["args"]}
    if reset:
        a.update({k: v for k, v in VARIANT.items() if k in a})
    if not budget:
        a.pop("budget", None)
    if "data" in a:
        a["data"] = os.path.basename(os.path.normpath(str(a["data"])))
    return tuple(sorted((k, json.dumps(v)) for k, v in a.items()
                        if k not in NON_RECIPE and v not in (None, 0, 0.0, "", False, "0")))


def group_label(names):
    stems = {re.sub(r"_s\d+$", "", n) for n in names}
    return stems.pop() if len(stems) == 1 else " / ".join(sorted(names))


def per_seed(rs, metric):
    """metric of each distinct seed (replicates of a seed averaged), sorted by seed."""
    by = {}
    for r in rs:
        by.setdefault(json.dumps(r["args"].get("seed")), []).append(metric(r))
    return [float(np.mean(by[k])) for k in sorted(by)]


def seed_groups(runs, metric):
    """[{label, n, mean, sd, runs}] for recipes run with more than one seed; n counts
    distinct seeds (two runs of the same seed are one sample, not two)."""
    groups = {}
    for n, r in runs.items():
        groups.setdefault(recipe(r), []).append(n)
    out = []
    for names in groups.values():
        v = np.array(per_seed([runs[n] for n in names], metric))
        if len(v) < 2:
            continue
        out.append({"label": group_label(names), "n": len(v), "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)), "runs": sorted(names)})
    return sorted(out, key=lambda g: g["label"])


def flops_to_reach(curve, target):
    """Smallest compute at which a (compute, loss) curve sorted by compute reaches target,
    by linear interpolation in log-compute. Returns (flops, status) with status 'ok',
    'below' (the cheapest point already reaches it: an upper bound) or 'beyond' (None)."""
    if not curve:
        return None, "beyond"
    if curve[0][1] <= target:
        return curve[0][0], "below"
    for (c0, l0), (c1, l1) in zip(curve, curve[1:]):
        if l1 <= target < l0:
            f = (l0 - target) / (l0 - l1)
            return math.exp(math.log(c0) + f * (math.log(c1) - math.log(c0))), "ok"
    return None, "beyond"


def baseline_curve(runs, key):
    """[(budget, mean final loss over seeds)] of the completed runs whose recipe is `key`."""
    pts = {}
    for r in runs.values():
        if recipe(r, budget=False) == key:
            pts.setdefault(r["args"]["budget"], []).append(r)
    return [(b, float(np.mean(per_seed(v, lambda r: r["final_val_loss"]))))
            for b, v in sorted(pts.items())]


def compute_to_target(runs):
    """One row per completed run, plus one per multi-seed group (target = seed mean)."""
    rows = []
    targets = [(n, [n]) for n in runs]
    targets += [(f"{g['label']} (mean of {g['n']})", g["runs"])
                for g in seed_groups(runs, lambda r: r["final_val_loss"])]
    for label, names in targets:
        r = runs[names[0]]
        target = float(np.mean(per_seed([runs[n] for n in names], lambda r: r["final_val_loss"])))
        budget = r["args"]["budget"]
        base_key = recipe(r, budget=False, reset=True)
        row = {"run": label, "val_loss": target, "run_flops": budget, "baseline_flops": None,
               "speedup": None, "status": "", "baseline_points": []}
        if recipe(r, budget=False) == base_key:
            row.update(baseline_flops=budget, speedup=1.0, status="baseline")
        else:
            curve = baseline_curve(runs, base_key)
            need, status = flops_to_reach(curve, target)
            row.update(baseline_flops=need, status=status, baseline_points=curve,
                       speedup=need / budget if need else None)
        rows.append(row)
    return rows


def _ctt_cells(row):
    s, x = row["status"], row["speedup"]
    if s == "beyond":
        return ("not reached" if row["baseline_points"] else "no baseline runs",
                "> curve" if row["baseline_points"] else "n/a")
    sp = f"{x:.2f}x" if s != "below" else f"<= {x:.2f}x"
    return f"{row['baseline_flops']:.2e}", sp


def llm_rows(runs):
    rows = []
    for name, r in runs.items():
        a = r["args"]
        rows.append({"run": name, "optim": a["optim"], "zoom": r.get("schedule", a.get("zoom")),
                     "out": a.get("out_mode", ""), "gated": a.get("gated", ""),
                     "rope": a.get("rope_pos", ""), "lr": a["lr"], "seed": a.get("seed", ""),
                     "budget": _f(a.get("budget"), ".2e"), "steps": r.get("steps", ""),
                     "tokens(M)": _f(r.get("tokens", 0) / 1e6, ".2f"),
                     "epochs": _f(r.get("data_epochs", 0), ".3f"),
                     "train_s": _f(r.get("train_time"), ".0f"),
                     "bits": a.get("state_bits", 32), "dtype": a.get("param_dtype", "fp32"),
                     "val_loss": _f(r["final_val_loss"], ".4f")})
    return rows


def weather_rows(runs):
    rows = []
    for name, r in runs.items():
        a, fin = r["args"], r["final"]
        rows.append({"run": name, "optim": a["optim"], "zoom": r.get("schedule", a.get("zoom")),
                     "mode": a.get("coarse_mode", "merge"), "lr": a["lr"], "seed": a.get("seed", ""),
                     "budget": _f(a.get("budget"), ".2e"), "steps": r.get("steps", ""),
                     "epochs": _f(r.get("epochs"), ".2f"), "train_s": _f(r.get("train_time"), ".0f"),
                     "val_mse": _f(fin.get("val_loss"), ".4f"),
                     # --val_year runs: the validation-split eval, the one to select LRs on
                     "valsplit_mse": _f((r.get("final_val") or {}).get("val_loss"), ".4f"),
                     "acc z500@72h": _f(fin.get("acc_z500_72h"), ".3f"),
                     "z500@6h": _f(fin.get("rmse_z500_6h"), ".1f"),
                     "z500@24h": _f(fin.get("rmse_z500_24h"), ".1f"),
                     "z500@72h": _f(fin.get("rmse_z500_72h"), ".1f"),
                     "t850@24h": _f(fin.get("rmse_t850_24h"), ".2f"),
                     "t2m@24h": _f(fin.get("rmse_t2m_24h"), ".2f")})
    return rows


def _seed_table(groups, fmt=".4f"):
    rows = [{"recipe": g["label"], "n": g["n"], "mean +- sd": f"{g['mean']:{fmt}} +- {g['sd']:{fmt}}",
             "runs": ", ".join(g["runs"])} for g in groups]
    return md_table(rows, ["recipe", "n", "mean +- sd", "runs"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--results", default=RES, help="results directory to read")
    ap.add_argument("--out", default="", help="Markdown output (default <results>/SUMMARY.md)")
    args = ap.parse_args(argv)
    res = args.results
    parts = ["# Results summary (auto-generated by experiments/summarize.py)\n"]
    summary = {"compute_to_target_method": CTT_METHOD, "running": {}, "diverged": {},
               "skipped": {}}

    def scan_(sub):
        diverged = {}
        runs, running, skipped = scan(sub, res, diverged)
        if running:
            summary["running"][sub] = running
        if diverged:
            summary["diverged"][sub] = diverged
        if skipped:
            summary["skipped"][sub] = skipped
        return runs

    sweep = scan_("llm_sweep")
    if sweep:
        parts.append("## LLM learning-rate sweep (2.3e13 FLOPs)\n")
        parts.append(md_table(llm_rows(sweep), ["run", "optim", "lr", "steps", "val_loss"]))

    wsweep = scan_("weather_sweep")
    if wsweep:
        parts.append("\n\n## Weather learning-rate sweep (1.8e13 FLOPs)\n")
        parts.append(md_table(weather_rows(wsweep), ["run", "optim", "lr", "steps", "val_mse",
                                                     "valsplit_mse", "z500@6h", "z500@24h",
                                                     "z500@72h"]))

    main_runs = scan_("llm")
    if main_runs:
        parts.append("\n\n## LLM main runs and ablations\n")
        cols = ["run", "optim", "zoom", "out", "gated", "rope", "lr", "seed", "budget", "steps",
                "tokens(M)", "epochs", "train_s", "val_loss"]
        parts.append(md_table(llm_rows(main_runs), cols))
        groups = seed_groups(main_runs, lambda r: r["final_val_loss"])
        summary["llm_seed_groups"] = groups
        if groups:
            parts.append("\n\n### Seed aggregates (final val loss, sample sd)\n")
            parts.append(_seed_table(groups))
        ctt = compute_to_target(main_runs)
        summary["compute_to_target"] = ctt
        parts.append("\n\n### Compute-to-target (baseline FLOPs needed to reach each run's final loss)\n")
        parts.append(f"Method: {CTT_METHOD}.\n")
        rows = []
        for row in ctt:
            need, sp = _ctt_cells(row)
            rows.append({"run": row["run"], "val_loss": f"{row['val_loss']:.4f}",
                         "run_flops": f"{row['run_flops']:.2e}", "baseline_flops_to_match": need,
                         "speedup": sp})
        parts.append(md_table(rows, ["run", "val_loss", "run_flops", "baseline_flops_to_match",
                                     "speedup"]))

    lowmem = scan_("llm_lowmem")
    if lowmem:
        parts.append("\n\n## LLM low-memory runs\n")
        parts.append(md_table(llm_rows(lowmem), ["run", "optim", "bits", "dtype", "lr", "budget",
                                                 "steps", "train_s", "val_loss"]))

    wx = scan_("weather")
    if wx:
        parts.append("\n\n## Weather (ERA5 5.625 deg, 6 h step, test year 2018)\n")
        cols = ["run", "optim", "zoom", "mode", "lr", "seed", "budget", "steps", "epochs",
                "train_s", "val_mse", "valsplit_mse", "z500@6h", "z500@24h", "z500@72h",
                "t850@24h", "t2m@24h", "acc z500@72h"]
        parts.append(md_table(weather_rows(wx), cols))
        groups = seed_groups(wx, lambda r: r["final"]["val_loss"])
        summary["weather_seed_groups"] = groups
        if groups:
            parts.append("\n\n### Seed aggregates (val MSE, sample sd)\n")
            parts.append(_seed_table(groups))
        fin = next(iter(wx.values()))["final"]
        persist = [f"{c}@{h}h={fin[f'persist_{c}_{h}h']:.2f}" for h in (6, 24, 72)
                   for c in ("z500", "t850", "t2m") if f"persist_{c}_{h}h" in fin]
        if persist:
            parts.append("\n\nPersistence RMSE on the same initial conditions: " + ", ".join(persist))

    if summary["running"]:
        parts.append("\n\n## Running or interrupted (not tabulated)\n")
        parts.append(md_table([{"run": f"{sub}/{n}", "last eval frac": _f(fr, ".3f")}
                               for sub, d in summary["running"].items() for n, fr in d.items()],
                              ["run", "last eval frac"]))
    if summary["diverged"]:
        parts.append("\n\n## Diverged (not tabulated)\n")
        parts.append(md_table([{"run": f"{sub}/{n}", "step": "" if st is None else st}
                               for sub, d in summary["diverged"].items() for n, st in d.items()],
                              ["run", "step"]))
    if summary["skipped"]:
        parts.append("\n\nSkipped (not run logs): " + ", ".join(
            f"{sub}/{n}.json" for sub, names in summary["skipped"].items() for n in names))

    out = args.out or os.path.join(res, "SUMMARY.md")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(parts) + "\n")
    with open(os.path.join(os.path.dirname(os.path.abspath(out)), "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n".join(parts))

    if args.plots:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from plots import make_plots
        make_plots(main_runs, wx, out=os.path.join(res, "plots"))
    return 0


if __name__ == "__main__":
    main()
