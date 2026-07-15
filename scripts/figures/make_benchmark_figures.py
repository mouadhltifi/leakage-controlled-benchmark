#!/usr/bin/env python3
"""Benchmark-paper figures (Paper B) — regenerated from committed results.

Two demonstrations, in the benchmark's own visual voice (distinct from the
companion findings paper's figures by palette, orientation, and label
register):
  fig-selection.pdf  — the cost of relaxing C3 (test-insulated selection):
                       one MSGCA training run, the released rule's running
                       maximum vs the validation-selected readout.
  fig-split.pdf      — the cost of relaxing C2 (chronological splits):
                       CAMEF three-step audit, horizontal log bars.

Data: results/analysis/msgca_diagnostic_rerun_history.json (per-epoch
histories, shipped) and the audited CAMEF constants (audits/camef/).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INK = "#1a1a1a"
SLATE = "#3d5a6c"      # the benchmark's base series color
ORANGE = "#c2571a"     # the violation/cost accent
GREY = "#9a9a9a"
GRID = "#e3e3e3"

plt.rcParams.update({
    "font.size": 7.5, "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "font.family": "sans-serif",
    # TrueType, not Type-3: ACM TAPS rejects Type-3 fonts at camera-ready
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

CAMEF = {"checkpoint": 0.000431, "retrain_random": 0.00249,
         "retrain_chrono": 15.0, "reported": 0.000489}


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def fig_selection():
    h = json.load(open(ROOT / "results/analysis/msgca_diagnostic_rerun_history.json"))
    runs = h["arm_b"]
    gaps = [max(r2["test_mcc"] for r2 in run["history"])
            - run["history"][int(np.argmax([r2["val_mcc"] for r2 in run["history"]]))]["test_mcc"]
            for run in runs]
    mean_gap = float(np.mean(gaps))
    rb = runs[2]
    hist = rb["history"]
    ep = [r["epoch"] for r in hist]
    test = [r["test_mcc"] for r in hist]
    val = [r["val_mcc"] for r in hist]
    run_max = np.maximum.accumulate(test)
    best_ep, best = int(np.argmax(test)), max(test)
    sel = int(np.argmax(val))
    at_sel = test[sel]

    fig, ax = plt.subplots(figsize=(3.2, 1.95))
    # the violation band: everything the released rule "earns" above the
    # insulated readout, shaded
    ax.fill_between(ep, at_sel, run_max, where=run_max > at_sel,
                    color=ORANGE, alpha=0.14, lw=0, zorder=1)
    ax.plot(ep, test, color=SLATE, lw=0.6, alpha=0.35, zorder=2)
    ax.plot(ep, run_max, color=ORANGE, lw=1.5, zorder=4)
    ax.axhline(at_sel, color=SLATE, lw=1.4, zorder=3)
    ax.axhline(0, color=GREY, lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.annotate(f"C3 relaxed: report max over epochs → {best:.3f}",
                xy=(best_ep, best), xytext=(52, 0.135), fontsize=6.8,
                color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))
    ax.annotate(f"C3 enforced: validation-selected → {at_sel:.3f}",
                xy=(320, at_sel), xytext=(52, -0.052), fontsize=6.8,
                color=SLATE,
                arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6))
    ax.text(398, (best + at_sel) / 2,
            f"+{best - at_sel:.3f} this run\n+{mean_gap:.3f} mean of 8",
            fontsize=6.8, color=INK, ha="right", va="center")
    ax.set_xlabel("training epoch (instrumented run of the released pipeline)", fontsize=7.2)
    ax.set_ylabel("test-set MCC", fontsize=7.2)
    ax.set_xlim(0, 400)
    ax.set_ylim(-0.10, 0.155)
    ax.grid(color=GRID, lw=0.4, zorder=0)
    despine(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig-selection.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"LEDGER fig-selection: best_on_test={best:.4f}@ep{best_ep} "
          f"val_selected={at_sel:.4f}@ep{sel} gap=+{best-at_sel:.4f} mean8=+{mean_gap:.4f}")


def fig_split():
    fig, ax = plt.subplots(figsize=(3.2, 1.35))
    labels = ["released checkpoint,\nauthors' own test",
              "shipped-default retrain,\nnon-temporal split (as shipped)",
              "same retrain,\nchronological split"]
    vals = [CAMEF["checkpoint"], CAMEF["retrain_random"], CAMEF["retrain_chrono"]]
    colors = [SLATE, SLATE, ORANGE]
    y = [2, 1, 0]
    # dot plot, not bars: bar length is meaningless on a log axis (no zero
    # baseline); position carries the value. Labels stay short and inside
    # the axes so the tight bbox equals the axes and the figure fills the
    # column at true size; the editorial reading lives in the caption.
    for yi, v, c in zip(y, vals, colors):
        ax.plot([v], [yi], marker="o", ms=6.5, color=c, zorder=3)
        ax.axhline(yi, color=GRID, lw=0.5, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([1e-3, 1e-1, 1e1, 1e3])
    ax.axvline(CAMEF["reported"], color=GREY, ls=(0, (4, 3)), lw=0.9, zorder=2)
    ax.text(CAMEF["reported"] * 1.35, 2.58, "reported 0.000489", fontsize=6.4,
            color=INK, ha="left")
    for yi, v, t, ha in zip(y, vals,
                            ["0.00043 — reproduces", "0.0025", "≈15"],
                            ["left", "left", "right"]):
        xt = v * 1.9 if ha == "left" else v / 1.9
        ax.text(xt, yi, t, va="center", ha=ha, fontsize=6.8, color=INK)
    ax.set_ylim(-0.5, 2.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.6)
    ax.set_xlabel("test MSE (log scale)", fontsize=7.2)
    ax.set_xlim(1.2e-4, 9e3)
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    ax.tick_params(axis="y", length=0)
    despine(ax, keep=("bottom",))
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig-split.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"LEDGER fig-split: checkpoint={CAMEF['checkpoint']} "
          f"retrain_random={CAMEF['retrain_random']} chrono~{CAMEF['retrain_chrono']} "
          f"reported={CAMEF['reported']}")


if __name__ == "__main__":
    fig_selection()
    fig_split()
