#!/usr/bin/env python3
"""The paper's overview figure (Figure 1) — the instrument, drawn.

Schematic, no data: five availability-timed source families enter the
protocol-locked harness whose five controls (C1-C5) gate the evaluation
path end to end; the harness emits the claim (paired delta vs the tuned
price-only floor, fold-level corrected intervals), which lands at audit
Level 1 and climbs the graduated standard to "established". The orange
strip prices the demonstrated violations of Section 4 - same visual
voice as fig-selection/fig-split (slate machinery, orange violation).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INK = "#1a1a1a"
SLATE = "#3d5a6c"
SLATE_FILL = "#eef2f5"
ORANGE = "#c2571a"
ORANGE_FILL = "#fdf3ec"
GREY = "#8a8a8a"

plt.rcParams.update({
    "font.size": 7.0, "text.color": INK, "font.family": "sans-serif",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def box(ax, x, y, w, h, fc, ec, lw=0.7, r=0.8, z=2):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(b)
    return b


def arrow(ax, x0, y0, x1, y1, color=SLATE, lw=1.0, z=3, ms=7):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                        color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(7.05, 2.52))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    # ---- column headers -------------------------------------------------
    heads = [
        (8.5, "FIVE SOURCE FAMILIES", "availability-timed features"),
        (38.0, "EVALUATION PROTOCOL, FIXED IN CODE", "five controls, enforced jointly"),
        (81.0, "GRADUATED AUDIT STANDARD", "what it takes for a gain to count"),
    ]
    for x, t, s in heads:
        ax.text(x, 34.7, t, fontsize=6.3, color=SLATE, ha="center",
                va="center", fontweight="bold")
        ax.text(x, 33.0, s, fontsize=5.8, color=GREY, ha="center",
                va="center", style="italic")

    # ---- source family tiles --------------------------------------------
    fams = [
        ("Price", "10 indicators"),
        ("News", "day-level sentiment"),
        ("Social", "17 aggregates"),
        ("Macro", "5 federal series"),
        ("Graph", "sector + correlation"),
    ]
    tile_h, gap = 4.2, 1.0
    for i, (name, sub) in enumerate(fams):
        y = 29.6 - tile_h - i * (tile_h + gap)
        box(ax, 1.5, y, 15.2, tile_h, "white", SLATE, lw=0.6, r=0.5)
        ax.text(2.9, y + tile_h / 2 + 0.85, name, fontsize=6.3, va="center",
                fontweight="bold", color=INK)
        ax.text(2.9, y + tile_h / 2 - 1.05, sub, fontsize=5.3, va="center",
                color=GREY)
        arrow(ax, 16.9, y + tile_h / 2, 21.6, y + tile_h / 2, lw=0.6, ms=5)

    # ---- harness box with control gates ---------------------------------
    box(ax, 22.0, 4.6, 32.2, 25.0, SLATE_FILL, SLATE, lw=0.9, r=1.0)
    box(ax, 22.0, 26.2, 32.2, 3.4, SLATE, SLATE, lw=0.9, r=1.0)
    ax.add_patch(plt.Rectangle((22.0, 26.2), 32.2, 1.4, fc=SLATE, ec=SLATE,
                               lw=0, zorder=2))
    ax.text(38.0, 28.1, "PROTOCOL-LOCKED HARNESS", fontsize=6.6, ha="center",
            va="center", color="white", fontweight="bold")
    ax.text(38.0, 25.3, "55 names · 5 chronological folds · 2015–2023",
            fontsize=5.2, ha="center", va="center", color=GREY)
    controls = [
        ("C1", "tuned price-only reference floor"),
        ("C2", "availability-timed chronology"),
        ("C3", "selection never reads the test set"),
        ("C4", "fold-level corrected statistics"),
        ("C5", "liquid-universe scope"),
    ]
    ch, cgap = 3.35, 0.72
    for i, (cid, txt) in enumerate(controls):
        y = 24.6 - ch - i * (ch + cgap)
        box(ax, 23.2, y, 29.6, ch, "white", SLATE, lw=0.6, r=0.5)
        box(ax, 23.8, y + 0.45, 3.2, ch - 0.9, SLATE, SLATE, lw=0.5, r=0.4, z=3)
        ax.text(25.4, y + ch / 2, cid, fontsize=6.0, va="center", ha="center",
                color="white", fontweight="bold", zorder=4)
        ax.text(28.0, y + ch / 2, txt, fontsize=5.6, va="center", color=INK)

    # ---- ladder rows (bottom-up), claim enters Level 1 --------------------
    lx, lw_ = 64.0, 34.5
    rows = [
        (5.8, "Level 1 · reported", "numbers in a paper", None),
        (13.4, "Level 2 · auditable", "code, config, seeds released", None),
        (21.0, "Level 3 · audited", "independent re-run + conformance read",
         "→ established"),
    ]
    rh = 5.6
    for ybot, lvl, sub, tag in rows:
        box(ax, lx, ybot, lw_, rh, "white", SLATE,
            lw=1.0 if tag else 0.7, r=0.6)
        ax.text(lx + 1.6, ybot + rh - 1.6, lvl, fontsize=6.1, va="center",
                color=SLATE, fontweight="bold")
        ax.text(lx + 1.6, ybot + 1.5, sub, fontsize=5.5, va="center",
                color=INK)
        if tag:
            box(ax, lx + lw_ - 13.4, ybot + rh - 2.8, 12.2, 2.3, SLATE, SLATE,
                lw=0.5, r=0.6, z=3)
            ax.text(lx + lw_ - 7.3, ybot + rh - 1.65, tag, fontsize=5.9,
                    va="center", ha="center", color="white", fontweight="bold",
                    zorder=4)
    for y0, y1 in ((11.4, 13.4), (19.0, 21.0)):
        arrow(ax, lx + lw_ / 2, y0, lx + lw_ / 2, y1, lw=0.8, ms=6)

    # ---- claim arrow: harness -> Level 1 ----------------------------------
    arrow(ax, 54.4, 8.6, 63.8, 8.6, lw=1.2, ms=8)
    ax.text(59.1, 10.3, "the claim", fontsize=6.0, ha="center", color=INK,
            fontweight="bold")
    ax.text(58.2, 6.5, "paired per-fold Δ\nvs the tuned floor", fontsize=4.7,
            ha="center", va="center", color=GREY)

    # ---- priced-violation strip (renderer-measured placement) -------------
    box(ax, 1.5, 0.2, 97.0, 3.2, ORANGE_FILL, ORANGE, lw=0.7, r=0.5)
    items = [
        ("RELAXING A CONTROL, PRICED", ORANGE, "bold", 5.6),
        ("skip C3 → +0.04–0.07 MCC in-run", INK, "normal", 5.2),
        ("skip C2 → MSE 0.0025 → ≈15", INK, "normal", 5.2),
        ("untuned C1 → spurious +0.014", INK, "normal", 5.2),
    ]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    x = 3.2
    for i, (s, c, w, fs) in enumerate(items):
        txt = ax.text(x, 1.8, s, fontsize=fs, color=c, va="center",
                      fontweight=w, zorder=4)
        bb = txt.get_window_extent(renderer=rend)
        (x0d, _), (x1d, _) = inv.transform([[bb.x0, 0], [bb.x1, 0]])
        wdt = x1d - x0d
        if i > 0:
            box(ax, x - 0.8, 0.55, wdt + 1.6, 2.5, "white", ORANGE, lw=0.5,
                r=0.5, z=3)
            txt.set_zorder(5)
        x += wdt + 2.3
    if x - 2.4 > 98.0:
        print(f"WARN: strip overflows to x={x - 2.4:.1f}")

    fig.savefig(OUT / "fig-overview.pdf", bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT / "fig-overview.pdf")


if __name__ == "__main__":
    main()
