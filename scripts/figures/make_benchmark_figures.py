#!/usr/bin/env python3
"""Benchmark-paper figures (Paper B) — regenerated from committed results.

Two demonstrations, in the benchmark's own visual voice (distinct from the
companion findings paper's figures by palette, orientation, and label
register):
  fig-selection.pdf  — the cost of relaxing C3 (test-insulated selection):
                       one MSGCA training run, the released rule's running
                       maximum vs the validation-selected readout.
  fig-split.pdf      — the cost of relaxing C2 (chronological splits):
                       CAMEF three-step audit, horizontal log dot plot.
  fig-forest.pdf     — the reference floor drawn: paired dMCC + fold-block
                       bootstrap CIs per configuration (Table 3's rows).
  fig-timeline.pdf   — coverage/fold timeline: five family lanes, warm-up,
                       expanding folds with calendar-tail validation.
  fig-sectorband.pdf — FF12-vs-GICS deltas against the pre-registered
                       +/-0.005 equivalence band (Appendix B).

Data: results/analysis/msgca_diagnostic_rerun_history.json (per-epoch
histories, shipped) and the audited CAMEF constants (audits/camef/).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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
    ax.plot([best_ep], [best], marker="o", ms=4.5, color=ORANGE, zorder=6)
    ax.plot([sel], [at_sel], marker="o", ms=4.5, color=SLATE, zorder=6)
    ax.text(8, 0.022, "per-epoch\ntest MCC", fontsize=6.0, color=GREY,
            ha="left", va="bottom", linespacing=1.35, zorder=5)
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
    ax.set_xlabel("training epoch", fontsize=7.2)
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




FOREST = [  # tables/ref_direction_v2.tex, verbatim; last field = flag side
    ("P+N",     +0.0022, -0.0033, +0.0076, False, None, None),
    ("P+M",     -0.0035, -0.0198, +0.0110, False, None, None),
    ("P+S",     -0.0047, -0.0081, -0.0015, True,  "harm", "left"),
    ("P+G",     +0.0090, +0.0004, +0.0176, True,  "lean (§4.1)", "right"),
    # P+N+M's low whisker sits at the axis edge: flag goes right of the
    # high whisker or it collides with the row label in the tick gutter
    ("P+N+M",   -0.0142, -0.0268, -0.0051, True,  "harm", "right"),
    ("P+M+S",   -0.0032, -0.0199, +0.0077, False, None, None),
    ("P+M+G",   -0.0017, -0.0234, +0.0165, False, None, None),
    ("P+N+M+S", -0.0054, -0.0154, +0.0021, False, None, None),
]


def fig_forest():
    fig, ax = plt.subplots(figsize=(3.2, 1.95))
    n = len(FOREST)
    for i, (name, d, lo, hi, excl, flag, side) in enumerate(FOREST):
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color=SLATE, lw=1.1, zorder=3,
                solid_capstyle="butt")
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.16, y + 0.16], color=SLATE, lw=1.1,
                    zorder=3)
        ax.plot([d], [y], marker="o", ms=4.6, mfc=SLATE if excl else "white",
                mec=SLATE, mew=1.0, zorder=4)
        if flag:
            if side == "left":
                ax.text(lo - 0.0012, y, flag, fontsize=6.0, color=INK,
                        va="center", ha="right")
            else:
                ax.text(hi + 0.0012, y, flag, fontsize=6.0, color=INK,
                        va="center", ha="left")
    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(FOREST)], fontsize=6.6)
    ax.set_xlabel("paired ΔMCC vs. the tuned price-only baseline",
                  fontsize=7.0)
    ax.set_xlim(-0.030, 0.024)
    ax.set_ylim(-1.6, n - 0.4)
    ax.text(-0.029, -1.25,
            "no configuration clears the corrected bar "
            "(all $p_{\\mathrm{bonf}} \\geq 0.57$)",
            fontsize=6.2, color=INK, style="italic", va="center")
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    ax.tick_params(axis="y", length=0)
    despine(ax, keep=("bottom",))
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig-forest.pdf", bbox_inches="tight")
    plt.close(fig)
    print("LEDGER fig-forest: 8 rows; zero-excluding = P+S, P+G, P+N+M")


TRAIN0 = 2016.01     # 2016-01-04 feature start
FOLDS = [  # (test_start, test_end, regime)  July-June windows
    (2019.50, 2020.50, "COVID crash"),
    (2020.50, 2021.50, "recovery"),
    (2021.50, 2022.50, "bull-to-bear"),
    (2022.50, 2023.50, "2022 bear"),
    (2023.50, 2024.00, "2023 rally"),
]
LANES = [  # (name, start, end, note)
    ("Price",  2015.00, 2024.00, None),
    ("News",   2015.00, 2023.96, "53/55 names"),
    ("Social", 2015.00, 2022.99, "ends 2022-12-30"),
    ("Macro",  2015.00, 2024.00, None),
    ("Graph",  2015.00, 2024.00, None),
]
TRAIN_C, VAL_C = "#dbe4ea", "#9fb4c0"


def fig_timeline():
    fig, ax = plt.subplots(figsize=(3.2, 2.05))
    yl = 12.6                       # family lanes occupy y 9.0..12.6
    for i, (name, s, e, note) in enumerate(LANES):
        y = yl - 0.9 * i
        ax.barh(y, e - s, left=s, height=0.52, color=SLATE, zorder=3)
        if e < 2024.0:              # explicit gap, neutral hatch
            ax.barh(y, 2024.0 - e, left=e, height=0.52, fc="white",
                    ec=GREY, lw=0.5, hatch="////", zorder=2)
        ax.text(2014.86, y, name, fontsize=6.0, ha="right", va="center",
                color=INK)
        if note:
            ax.text(e - 0.12, y, note, fontsize=5.2, ha="right",
                    va="center", color="white", zorder=4)
    # warm-up band (prices from 2015-01-02 warm the trailing indicators)
    ax.axvspan(2015.0, TRAIN0, ymin=0.0, ymax=1.0, fc=GRID, alpha=0.45,
               zorder=1)
    ax.text(2015.5, 13.75, "warm-up", fontsize=5.6, color=GREY, ha="center",
            va="center")
    # fold ribbons: expanding train, calendar-tail val (last 20%), test
    for i, (ts, te, regime) in enumerate(FOLDS):
        y = 6.5 - 0.9 * i
        v0 = ts - 0.2 * (ts - TRAIN0)
        ax.barh(y, v0 - TRAIN0, left=TRAIN0, height=0.52, color=TRAIN_C,
                zorder=3)
        ax.barh(y, ts - v0, left=v0, height=0.52, color=VAL_C, zorder=3)
        ax.barh(y, te - ts, left=ts, height=0.52, color=SLATE, zorder=3)
        ax.text(2014.86, y, f"F{i}", fontsize=6.0, ha="right", va="center",
                color=INK)
        ax.text(te + 0.12, y, regime, fontsize=5.2, ha="left", va="center",
                color=GREY)
    ax.set_xlim(2013.7, 2026.4)
    ax.set_ylim(1.6, 14.6)
    ax.set_xticks([2015, 2017, 2019, 2021, 2023])
    ax.set_xticklabels(["2015", "2017", "2019", "2021", "2023"],
                       fontsize=6.4)
    ax.set_yticks([])
    # inline swatch legend between the two blocks
    ly = 7.85
    for x0, c, lab in ((2016.01, TRAIN_C, "train (expanding)"),
                       (2019.7, VAL_C, "validation (calendar tail)"),
                       (2024.2, SLATE, "test")):
        ax.barh(ly, 0.32, left=x0, height=0.42, color=c, zorder=3)
        ax.text(x0 + 0.48, ly, lab, fontsize=5.2, color=GREY, ha="left",
                va="center")
    despine(ax, keep=("bottom",))
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig-timeline.pdf", bbox_inches="tight")
    plt.close(fig)
    print("LEDGER fig-timeline: 5 lanes, 5 folds, warm-up 2015->2016-01")


SECTOR = [  # tables/appendix_sector.tex: config, arch, delta(FF12-GICS)
    ("P+G · FF",     +0.0102),
    ("P+G · LSTM",   +0.0051),
    ("P+M+G · FF",   +0.0015),
    ("P+M+G · LSTM", -0.0033),
]


def fig_sectorband():
    fig, ax = plt.subplots(figsize=(3.2, 1.25))
    ax.axvspan(-0.005, 0.005, fc=SLATE, alpha=0.10, zorder=1)
    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    n = len(SECTOR)
    for i, (name, d) in enumerate(SECTOR):
        y = n - 1 - i
        ax.plot([d], [y], marker="o", ms=5.5, color=SLATE, zorder=4)
        ax.axhline(y, color=GRID, lw=0.5, zorder=0)
    ax.annotate("outside the pre-registered band:\nequivalence FAILED",
                xy=(0.0102, 3), xytext=(0.0125, 1.55), fontsize=6.0,
                color=INK, ha="left", va="center", linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6))
    ax.text(0.0048, 3.62, "±0.005 band", fontsize=5.6, color=GREY,
            ha="right", va="bottom")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(SECTOR)], fontsize=6.6)
    ax.set_xlabel("ΔMCC, released FF12 graph − proprietary GICS",
                  fontsize=7.0)
    ax.set_xlim(-0.0075, 0.0215)
    ax.set_ylim(-0.6, 4.3)
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    ax.tick_params(axis="y", length=0)
    despine(ax, keep=("bottom",))
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig-sectorband.pdf", bbox_inches="tight")
    plt.close(fig)
    print("LEDGER fig-sectorband: 4 rows; P+G FF +0.0102 outside +/-0.005")


def fig_split():
    # Option A (R15, author-approved): taller than the old 1.35in so labels
    # breathe; each value carries a one-word verdict; the collapse (orange)
    # and the clean elbow show "same recipe, only the split changed".
    # Dot plot, not bars: bar length is meaningless on a log axis (no zero
    # baseline); position carries the value.
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    rows = [("checkpoint,\nauthors' own eval", CAMEF["checkpoint"], SLATE,
             "0.000431 — reproduces"),
            ("retrain,\nshipped split", CAMEF["retrain_random"], SLATE,
             "0.0025 — matches"),
            ("retrain,\nchronological split", CAMEF["retrain_chrono"], ORANGE,
             "≈15 — collapses")]
    y = [2, 1, 0]
    for yi, (lab, v, c, txt) in zip(y, rows):
        ax.axhline(yi, color=GRID, lw=0.5, zorder=1)
        ax.plot([v], [yi], marker="o", ms=6.5, color=c, zorder=4)
        ax.text(v * 1.9, yi + 0.22, txt, va="bottom", ha="left",
                fontsize=6.6, color=INK, zorder=5)
    # reported reference: subtle line + label RIGHT-ALIGNED just left of the
    # line so the line never cuts through the text
    ax.axvline(CAMEF["reported"], color=GREY, ls=(0, (4, 3)), lw=0.9, zorder=2)
    ax.text(CAMEF["reported"] * 0.82, 2.5, "reported\n0.000489", fontsize=5.6,
            color=GREY, ha="right", va="center", linespacing=1.15, zorder=3)
    # the price, drawn: one clean elbow from the shipped dot down to the
    # chronological dot (same recipe, only the split changed)
    arr = FancyArrowPatch((CAMEF["retrain_random"], 0.82),
                          (CAMEF["retrain_chrono"], 0.18),
                          connectionstyle="angle,angleA=-90,angleB=180,rad=6",
                          arrowstyle="-|>", mutation_scale=8, color=ORANGE,
                          lw=1.0, zorder=3)
    ax.add_patch(arr)
    ax.text(0.09, 0.5, "same recipe,\nsplit made chronological", fontsize=5.8,
            color=ORANGE, ha="center", va="center", linespacing=1.2, zorder=5)
    ax.set_xscale("log")
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    ax.set_ylim(-0.55, 3.1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xlabel("test MSE (log scale)", fontsize=7.2)
    ax.set_xlim(1.2e-4, 3e2)
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
    fig_forest()
    fig_timeline()
    fig_sectorband()
