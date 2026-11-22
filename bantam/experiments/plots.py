"""PNG plots for results/: val loss vs FLOPs (LLM) and RMSE vs lead time (weather).

Colors follow a fixed categorical order (blue, orange, aqua, yellow) assigned
by entity, never by rank; one y-axis per chart; recessive grid; direct labels.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "plots")
COLORS = {"adamw/base": "#2a78d6", "adamw/zoom": "#eb6834",
          "muon/base": "#1baf7a", "muon/zoom": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _style(ax, xlabel, ylabel, title):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left")


def _kind(run):
    a = run["args"]
    return f"{a['optim']}/{'base' if run['schedule'] == '1:1' else 'zoom'}"


def plot_llm(runs, names, fname, title):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for n in names:
        r = runs[n]
        ev = r["evals"][1:]
        x = [e["flops"] for e in ev]
        y = [e["val_loss"] for e in ev]
        c = COLORS.get(_kind(r), "#6250d6")
        ax.plot(x, y, color=c, linewidth=2, marker="o", markersize=4,
                label=f"{_kind(r)} ({r['schedule']})")
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color=INK, va="center")
    _style(ax, "training FLOPs", "validation loss (nats/token, k=1)", title)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def plot_weather(runs, names, fname, title):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    hours = [6, 24, 72]
    for n in names:
        r = runs[n]
        a = r["args"]
        kind = f"{a['optim']}/{'base' if r['schedule'] == '1:1' else a['coarse_mode']}"
        c = COLORS.get(kind.replace("merge", "zoom"), "#6250d6")
        y = [r["final"][f"rmse_z500_{h}h"] for h in hours]
        ax.plot(hours, y, color=c, linewidth=2, marker="o", markersize=5,
                label=f"{kind} ({r['schedule']})")
        ax.annotate(f"{y[-1]:.0f}", (hours[-1], y[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color=INK, va="center")
    any_run = runs[names[0]]["final"]
    ax.plot(hours, [any_run[f"persist_z500_{h}h"] for h in hours], color=MUTED,
            linewidth=1.5, linestyle="--", label="persistence")
    _style(ax, "lead time (h)", "z500 RMSE (m²/s²), lat-weighted, 2018", title)
    ax.set_xticks(hours)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def make_plots(llm_runs, wx_runs):
    if llm_runs:
        mains = [n for n in llm_runs if n.startswith("main_") and "_s0" in n]
        if mains:
            plot_llm(llm_runs, mains, "llm_main.png",
                     "TinyStories, 5.8M params: validation loss vs compute")
    if wx_runs:
        mains = [n for n in wx_runs if n.startswith("main_") and "_s0" in n]
        if mains:
            plot_weather(wx_runs, mains, "weather_main.png",
                         "ERA5 5.625°, 1M-param ViT: z500 RMSE vs lead time")
