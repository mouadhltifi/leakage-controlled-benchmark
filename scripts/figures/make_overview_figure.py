#!/usr/bin/env python3
"""The paper's overview figure (Figure 1) — the instrument, drawn.

Schematic, no data. Built from an explicit design spec (v16 rebuild;
v26 removes the priced-violation strip — its content lives in §4 /
Table 5 / Figs. 3-4, and an overview does one job):
  grid    x: margin 1.5 | tiles 1.5-17.0 | gutter 17-21.5 (family arrows)
          | harness 21.5-54.5 | gutter 54.5-67.5 (the claim) | ladder
          67.5-98.5 | margin 1.5.  y: content band 5.2-30.6 shared by
          ALL columns (tops and bottoms aligned) | headers 32.6 / 34.6.
  type    4 sizes: 7.4 band title / 6.6 labels+headers / 6.0 body /
          5.4 fine print. Bold marks identity, never body text.
  weight  borders 0.7 (emphasis 1.0, chips 0.55); arrows 0.6/0.9/1.3.
  radius  0.8 outer boxes, 0.45 inner chips/badges.
  color   slate = machinery (all-slate since v26: orange is reserved
          for violations and none are depicted), grey = fine print,
          white-on-slate = the two brackets (band title, established
          chip).

Set OVERVIEW_VARIANT=B for the ladder-in-panel candidate.
"""
from __future__ import annotations

import os
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

F_BAND, F_LABEL, F_BODY, F_FINE = 7.4, 6.6, 6.0, 5.4
# ladder-rung headings sit a step below the three column headers so the
# longest one ("Level 3 . independently audited") fits its box with margin
F_RUNG = 6.2
W_CHIP, W_BOX, W_EMPH = 0.55, 0.7, 1.0
R_OUT, R_IN = 0.8, 0.45

plt.rcParams.update({
    "font.size": F_BODY, "text.color": INK, "font.family": "sans-serif",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# grid
X_TIL0, X_TIL1 = 1.5, 17.0
X_HAR0, X_HAR1 = 21.5, 54.5
X_LAD0, X_LAD1 = 67.5, 99.4
Y_BAND0, Y_BAND1 = 5.2, 30.6


def box(ax, x, y, w, h, fc, ec, lw=W_BOX, r=R_OUT, z=2):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(b)
    return b


def arrow(ax, x0, y0, x1, y1, color=SLATE, lw=0.9, z=3, ms=7):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                        color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def main():
    variant = os.environ.get("OVERVIEW_VARIANT", "A").upper()
    fig, ax = plt.subplots(figsize=(7.05, 3.02))
    ax.set_xlim(0, 100)
    ax.set_ylim(4.4, 37.2)   # v26: band bottom is the canvas bottom (strip removed)
    ax.axis("off")

    # ---- column headers (each centered over its column) -------------------
    heads = [
        (X_TIL0, "SOURCE FAMILIES", "availability-timed features"),
        (X_HAR0, "EVALUATION PROTOCOL", "four controls + a fixed universe, fixed in the release"),
        (X_LAD0, "GRADUATED AUDIT STANDARD", "what it takes for a gain to count"),
    ]
    for x, t, s in heads:
        ax.text(x, 34.6, t, fontsize=F_LABEL, color=SLATE, ha="left",
                va="center", fontweight="bold")
        ax.text(x, 32.6, s, fontsize=F_FINE, color=GREY, ha="left",
                va="center", style="italic")

    # ---- source family tiles (fill the band exactly) ----------------------
    fams = [
        ("Price", "10 indicators"),
        ("News", "day-level sentiment"),
        ("Social", "17 aggregates"),
        ("Macro", "6 series"),
        ("Graph", "sector + correlation"),
    ]
    n = len(fams)
    t_gap = 1.2
    t_h = (Y_BAND1 - Y_BAND0 - (n - 1) * t_gap) / n
    for i, (name, sub) in enumerate(fams):
        y = Y_BAND1 - t_h - i * (t_h + t_gap)
        box(ax, X_TIL0, y, X_TIL1 - X_TIL0, t_h, "white", SLATE, lw=W_BOX, r=R_IN)
        ax.text(X_TIL0 + 1.4, y + t_h / 2 + 0.85, name, fontsize=F_LABEL,
                va="center", fontweight="bold", color=INK)
        ax.text(X_TIL0 + 1.4, y + t_h / 2 - 1.05, sub, fontsize=F_FINE,
                va="center", color=GREY)
        arrow(ax, X_TIL1 + 0.2, y + t_h / 2, X_HAR0 - 0.2, y + t_h / 2,
              lw=0.6, ms=5)

    # ---- harness: emphasis container with slate header band ---------------
    box(ax, X_HAR0, Y_BAND0, X_HAR1 - X_HAR0, Y_BAND1 - Y_BAND0,
        SLATE_FILL, SLATE, lw=W_EMPH, r=R_OUT)
    band_h = 3.4
    box(ax, X_HAR0, Y_BAND1 - band_h, X_HAR1 - X_HAR0, band_h, SLATE, SLATE,
        lw=W_EMPH, r=R_OUT)
    ax.add_patch(plt.Rectangle((X_HAR0, Y_BAND1 - band_h), X_HAR1 - X_HAR0,
                               band_h / 2, fc=SLATE, ec=SLATE, lw=0, zorder=2))
    ax.text((X_HAR0 + X_HAR1) / 2 - 0.11, Y_BAND1 - band_h / 2 - 0.13, "PROTOCOL-LOCKED HARNESS",
            fontsize=F_BAND, ha="center", va="center", color="white",
            fontweight="bold", zorder=3)
    inset = 1.1
    ax.text(X_HAR0 + inset, Y_BAND1 - band_h - 1.15,
            "55 names · 5 chronological folds · 2015–2023",
            fontsize=5.2, ha="left", va="center", color=GREY, zorder=3)

    controls = [
        ("C1", "tuned price-only baseline"),
        ("C2", "availability-timed chronology"),
        ("C3", "selection never reads the test set"),
        ("C4", "fold-level corrected statistics"),
        ("C5", "scope: the fixed liquid universe"),
    ]
    c_top = Y_BAND1 - band_h - 2.3
    c_bot = Y_BAND0 + 0.9
    c_gap = 0.8
    scope_gap = 1.2   # C5 is the scope condition, set apart from the four controls
    c_h = (c_top - c_bot - (n - 1) * c_gap - scope_gap) / n
    for i, (cid, txt) in enumerate(controls):
        is_scope = (i == n - 1)   # C5: scope, not a fourth-wall control
        y = c_top - c_h - i * (c_h + c_gap) - (scope_gap if is_scope else 0)
        box(ax, X_HAR0 + inset, y, (X_HAR1 - X_HAR0) - 2 * inset, c_h,
            "white", SLATE, lw=W_CHIP, r=R_IN, z=3)
        # the four controls carry a filled badge; the scope condition an
        # outline one, so the figure reads "four controls + a fixed universe"
        badge_fill = "white" if is_scope else SLATE
        badge_text = SLATE if is_scope else "white"
        box(ax, X_HAR0 + inset + 0.55, y + 0.45, 3.1, c_h - 0.9, badge_fill, SLATE,
            lw=W_CHIP, r=R_IN, z=4)
        ax.text(X_HAR0 + inset + 2.1, y + c_h / 2 - 0.11, cid, fontsize=F_BODY,
                va="center", ha="center", color=badge_text, fontweight="bold",
                zorder=5)
        ax.text(X_HAR0 + inset + 4.5, y + c_h / 2, txt, fontsize=F_BODY,
                va="center", color=INK, zorder=5)

    # ---- audit ladder (fills the band; B adds a peer container) -----------
    lad_x0, lad_x1 = X_LAD0, X_LAD1
    if variant == "B":
        box(ax, X_LAD0, Y_BAND0, X_LAD1 - X_LAD0, Y_BAND1 - Y_BAND0,
            "#f7f9fa", SLATE, lw=W_BOX, r=R_OUT)
        lad_x0, lad_x1 = X_LAD0 + 1.0, X_LAD1 - 1.0
    rows = [
        ("Level 1 · reported", "numbers in a paper", None),
        ("Level 2 · auditable", "code, config, seeds released", None),
        ("Level 3 · independently audited", "independent re-run\n+ conformance read",
         "→ established"),
    ]
    r_gap = 3.1
    band_h_l = (Y_BAND1 - Y_BAND0) if variant == "A" else (Y_BAND1 - Y_BAND0 - 2.0)
    y_base = Y_BAND0 if variant == "A" else Y_BAND0 + 1.0
    r_h = (band_h_l - 2 * r_gap) / 3
    row_y = {i: y_base + i * (r_h + r_gap) for i in range(3)}
    for i, (lvl, sub, tag) in enumerate(rows):
        ybot = row_y[i]
        box(ax, lad_x0, ybot, lad_x1 - lad_x0, r_h, "white", SLATE,
            lw=W_EMPH if tag else W_BOX, r=R_IN, z=3)
        ax.text(lad_x0 + 1.3, ybot + r_h - 1.8, lvl, fontsize=F_RUNG,
                va="center", color=SLATE, fontweight="bold", zorder=4)
        ax.text(lad_x0 + 1.3, ybot + 1.8, sub, fontsize=F_BODY,
                va="center", color=INK, zorder=4)
        if tag:
            cw, chh = 11.8, 2.5
            cx = lad_x1 - 1.6 - cw
            cy = ybot + 1.15
            box(ax, cx, cy, cw, chh, SLATE, SLATE, lw=W_CHIP, r=R_IN, z=4)
            ax.text(cx + cw / 2, cy + chh / 2 - 0.05, tag, fontsize=5.8,
                    va="center", ha="center", color="white",
                    fontweight="bold", zorder=5)
    xc = (lad_x0 + lad_x1) / 2
    for i in (0, 1):
        arrow(ax, xc, row_y[i] + r_h + 0.15, xc, row_y[i + 1] - 0.15,
              lw=1.1, ms=6)

    # ---- the claim: one anchored group at Level-1 mid ----------------------
    y_claim = row_y[0] + r_h / 2
    arrow(ax, X_HAR1 + 0.2, y_claim, X_LAD0 - 0.2, y_claim, lw=1.6, ms=9)
    gx = (X_HAR1 + X_LAD0) / 2
    box(ax, gx - 4.8, y_claim - 1.45, 9.6, 2.9, "white", SLATE, lw=W_EMPH,
        r=R_IN, z=4)
    ax.text(gx, y_claim - 0.11, "the claim", fontsize=F_LABEL, ha="center",
            va="center", color=INK, fontweight="bold", zorder=5)
    ax.text(gx, y_claim + 3.6, "paired per-fold Δ\nvs the tuned baseline",
            fontsize=F_FINE, ha="center", va="center", color=GREY,
            linespacing=1.35)

    suffix = "" if variant == "A" else "-B"
    fig.savefig(OUT / f"fig-overview{suffix}.pdf", bbox_inches="tight",
                pad_inches=0.02)
    print("wrote", OUT / f"fig-overview{suffix}.pdf", f"(variant {variant})")


if __name__ == "__main__":
    main()
