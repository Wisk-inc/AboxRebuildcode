"""Quality vs training compute: our runs against published TinyStories models.

    python experiments/plot_published.py results/published/compare_clean1000.json [--png P]

Reads the bits-per-byte scores from compare_published.py and the training
compute table (experiments/published_compute.json), prints a table and the
"compute to reach the same quality" ratios, and writes <results>_table.json and
results/plots/published_bpb_vs_compute.png.

Compute convention: 6 * N_total * D for every model, ours included. Models with
no entry in published_compute.json are listed as "compute unknown" and left out
of the chart. Our curve is the running-minimum frontier of completed runs; a
compute-to-match read off a single point (not an interpolated crossing) is an
upper bound, so its ratio is printed as ">=". Files written by the current
compare_published.py carry a scoring signature that every row must share.
"""
import argparse
import json
import math
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
COL = {"ours": "#2a78d6", "published": "#eb6834", "published-llama2c": "#1baf7a"}
LABEL = {"ours": "Bantam (this repo, CPU)", "published": "TinyStories (Eldan & Li 2023)",
         "published-llama2c": "llama2.c tinyllamas (Karpathy 2023)"}


def compute_of(name, r, table):
    """(compute, (lo, hi)) with 6*N*D; (None, (None, None)) when the training tokens are unknown."""
    if r["family"] == "ours":
        if r.get("train_tokens") is None:
            warnings.warn(f"{name}: training tokens unknown (in-progress run?)")
            return None, (None, None)
        c = 6 * r["params"] * r["train_tokens"]
        return c, (c, c)
    t = table.get(name)
    if t is None:
        warnings.warn(f"{name}: not in published_compute.json; compute unknown")
        return None, (None, None)
    lo_hi = t["tokens_range"] or [t["tokens"], t["tokens"]]
    return (6 * r["params"] * t["tokens"],
            (6 * r["params"] * lo_hi[0], 6 * r["params"] * lo_hi[1]))


def frontier(points):
    """Running-minimum (Pareto) frontier of (compute, bpb) points, sorted by compute."""
    out = []
    for c, b in sorted(points):
        if not out or b < out[-1][1]:
            out.append((c, b))
    return out


def first_reach(curve, target, with_kind=False):
    """Smallest compute at which a (compute, bpb) curve sorted by compute reaches target (log
    interp). kind: 'crossing' (interpolated) or 'upper bound' (a point already below target)."""
    res = None, None
    for i, (c, b) in enumerate(curve):
        if b <= target:
            if i and curve[i - 1][1] > target:
                c0, b0 = curve[i - 1]
                f = (b0 - target) / (b0 - b)
                res = math.exp(math.log(c0) + f * (math.log(c) - math.log(c0))), "crossing"
            else:
                res = c, ("crossing" if b == target else "upper bound")
            break
    return res if with_kind else res[0]


def check_signature(ev, res):
    """Every row of a signed file must carry the file's signature id."""
    sid = ev.get("signature_id")
    if sid is None:
        print("note: legacy results file, scoring convention not recorded per row")
        return
    bad = [n for n, r in res.items() if r.get("signature_id") != sid]
    if bad:
        raise SystemExit(f"rows scored with another convention than _eval: {', '.join(bad)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?",
                    default=os.path.join(ROOT, "results", "published", "compare_clean1000.json"))
    ap.add_argument("--png", default=os.path.join(ROOT, "results", "plots", "published_bpb_vs_compute.png"))
    ap.add_argument("--table", default="", help="default: <path>_table.json")
    args = ap.parse_args(argv)
    path = args.path
    res = json.load(open(path))
    table = json.load(open(os.path.join(ROOT, "experiments", "published_compute.json")))
    ev = res.pop("_eval")
    check_signature(ev, res)
    rows = []
    for name, r in res.items():
        c, (lo, hi) = compute_of(name, r, table)
        rows.append({"name": name, "family": r["family"], "params": r["params"], "bpb": r["bpb"],
                     "compute": c, "lo": lo, "hi": hi})
    rows.sort(key=lambda x: (x["compute"] is None, x["compute"] or 0))
    # "_fNN" checkpoints were saved mid-schedule (LR not yet decayed); they understate what a
    # run *finished* at that budget reaches, so they are drawn hollow and never used for
    # compute-to-match. Completed runs form the curve.
    for x in rows:
        x["mid"] = x["family"] == "ours" and "_f0." in x["name"]
    ours = frontier([(x["compute"], x["bpb"]) for x in rows
                     if x["family"] == "ours" and not x["mid"] and x["compute"] is not None])
    print(f"eval: {ev['stories']} {'clean ' if ev.get('clean_subset', True) else ''}validation "
          f"stories, {ev['bytes']} bytes, end-of-story scored={ev['end_of_story_scored']}")
    print(f"\n{'model':32s} {'params':>8s} {'train FLOPs':>11s} {'BPB':>7s}  ours reaches this BPB at")
    out_rows = []
    for x in rows:
        reach, kind = (first_reach(ours, x["bpb"], with_kind=True)
                       if x["family"] != "ours" and x["compute"] is not None else (None, None))
        ratio = (x["compute"] / reach) if reach else None
        ge = ">= " if kind == "upper bound" else ""
        s = (f"{'<= ' if ge else ''}{reach:.2e} FLOPs = {ge}{ratio:,.0f}x less" if reach else
             "compute unknown" if x["compute"] is None else
             ("(mid-schedule checkpoint)" if x["mid"] else
              "—" if x["family"] == "ours" else "not reached yet"))
        comp = "?" if x["compute"] is None else f"{x['compute']:.2e}"
        print(f"{x['name']:32s} {x['params']/1e6:7.2f}M {comp:>11s} {x['bpb']:7.4f}  {s}")
        out_rows.append({**x, "ours_compute_to_match": reach, "compute_ratio": ratio,
                         "match_kind": kind})
    with open(args.table or path.replace(".json", "_table.json"), "w") as f:
        json.dump({"eval": ev, "rows": out_rows}, f, indent=1)
    rows = [x for x in rows if x["compute"] is not None]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for fam in ("published", "published-llama2c", "ours"):
        pts = [x for x in rows if x["family"] == fam]
        if not pts:
            continue
        xs = np.array([p["compute"] for p in pts])
        ys = np.array([p["bpb"] for p in pts])
        err = np.array([[p["compute"] - p["lo"] for p in pts], [p["hi"] - p["compute"] for p in pts]])
        if fam == "ours":
            done = [p for p in pts if not p["mid"]]
            mid = [p for p in pts if p["mid"]]
            ax.plot([p["compute"] for p in done], [p["bpb"] for p in done], color=COL[fam],
                    linewidth=2, marker="o", markersize=5, label=LABEL[fam] + " (completed runs)",
                    zorder=3)
            if mid:
                ax.plot([p["compute"] for p in mid], [p["bpb"] for p in mid], color=COL[fam],
                        linewidth=0, marker="o", markersize=5, markerfacecolor="none",
                        label="Bantam mid-schedule checkpoints", zorder=3)
            pts = done
        else:
            ax.errorbar(xs, ys, xerr=err, fmt="o", color=COL[fam], markersize=7, capsize=3,
                        linewidth=1.2, label=LABEL[fam], zorder=3)
        offsets = {"stories15M": (-22, 8), "roneneldan/TinyStories-33M": (7, -2),
                   "roneneldan/TinyStories-28M": (-16, -14)}
        for p in pts:
            if fam != "ours" or p is pts[-1]:
                short = p["name"].split("/")[-1].replace("TinyStories-", "TS-")
                if fam == "ours":
                    short = f"Bantam {p['params']/1e6:.1f}M"
                ax.annotate(short, (p["compute"], p["bpb"]), textcoords="offset points",
                            xytext=offsets.get(p["name"], (6, 4)), fontsize=7.5, color=INK)
    ax.set_xscale("log")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(True, color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("training compute (FLOPs, 6·N·D)", color=INK, fontsize=10)
    ax.set_ylabel("bits per byte on unseen stories (lower is better)", color=INK, fontsize=10)
    seen = "stories no model saw" if ev.get("clean_subset", True) else "validation stories, some seen in training"
    ax.set_title(f"TinyStories: quality vs training compute ({ev['stories']} {seen})",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    out = args.png
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    main()
