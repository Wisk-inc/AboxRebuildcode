"""Quality vs training compute: our runs against published TinyStories models.

    python experiments/plot_published.py results/published/compare_clean1000.json

Reads the bits-per-byte scores from compare_published.py and the training
compute table (experiments/published_compute.json), prints a table and the
"compute to reach the same quality" ratios, and writes
results/plots/published_bpb_vs_compute.png.

Compute convention: 6 * N_total * D for every model, ours included.
"""
import json
import math
import os
import sys

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
    if r["family"] == "ours":
        c = 6 * r["params"] * r["train_tokens"]
        return c, (c, c)
    t = table[name]
    lo_hi = t["tokens_range"] or [t["tokens"], t["tokens"]]
    return (6 * r["params"] * t["tokens"],
            (6 * r["params"] * lo_hi[0], 6 * r["params"] * lo_hi[1]))


def first_reach(curve, target):
    """Smallest compute at which a (compute, bpb) curve sorted by compute reaches target (log interp)."""
    for (c0, b0), (c1, b1) in zip(curve, curve[1:]):
        if b1 <= target <= b0:
            f = (b0 - target) / (b0 - b1) if b0 != b1 else 0.0
            return math.exp(math.log(c0) + f * (math.log(c1) - math.log(c0)))
    return curve[0][0] if curve and curve[0][1] <= target else None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "published",
                                                              "compare_clean1000.json")
    res = json.load(open(path))
    table = json.load(open(os.path.join(ROOT, "experiments", "published_compute.json")))
    ev = res.pop("_eval")
    rows = []
    for name, r in res.items():
        c, (lo, hi) = compute_of(name, r, table)
        rows.append({"name": name, "family": r["family"], "params": r["params"], "bpb": r["bpb"],
                     "compute": c, "lo": lo, "hi": hi})
    rows.sort(key=lambda x: x["compute"])
    ours = [(x["compute"], x["bpb"]) for x in rows if x["family"] == "ours"]
    print(f"eval: {ev['stories']} clean validation stories, {ev['bytes']} bytes, "
          f"end-of-story scored={ev['end_of_story_scored']}")
    print(f"\n{'model':32s} {'params':>8s} {'train FLOPs':>11s} {'BPB':>7s}  ours reaches this BPB at")
    out_rows = []
    for x in rows:
        reach = first_reach(ours, x["bpb"]) if x["family"] != "ours" else None
        ratio = (x["compute"] / reach) if reach else None
        s = (f"{reach:.2e} FLOPs = {ratio:,.0f}x less" if reach else
             ("—" if x["family"] == "ours" else "not reached yet"))
        print(f"{x['name']:32s} {x['params']/1e6:7.2f}M {x['compute']:11.2e} {x['bpb']:7.4f}  {s}")
        out_rows.append({**x, "ours_compute_to_match": reach, "compute_ratio": ratio})
    with open(path.replace(".json", "_table.json"), "w") as f:
        json.dump({"eval": ev, "rows": out_rows}, f, indent=1)

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
            ax.plot(xs, ys, color=COL[fam], linewidth=2, marker="o", markersize=5, label=LABEL[fam],
                    zorder=3)
        else:
            ax.errorbar(xs, ys, xerr=err, fmt="o", color=COL[fam], markersize=7, capsize=3,
                        linewidth=1.2, label=LABEL[fam], zorder=3)
        for p in pts:
            if fam != "ours" or p is pts[-1]:
                short = p["name"].split("/")[-1].replace("TinyStories-", "TS-")
                ax.annotate(short, (p["compute"], p["bpb"]), textcoords="offset points",
                            xytext=(6, 4), fontsize=7.5, color=INK)
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
    ax.set_title(f"TinyStories: quality vs training compute ({ev['stories']} stories no model saw)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    os.makedirs(os.path.join(ROOT, "results", "plots"), exist_ok=True)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "plots", "published_bpb_vs_compute.png")
    fig.savefig(out)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
