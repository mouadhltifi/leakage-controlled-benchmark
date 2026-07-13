"""Shared figure style for the manuscript figures.

Style guide:
- Font: Latin Modern Roman / Computer Modern Roman (matches the LaTeX body),
  STIXGeneral as fallback (NEVER DejaVu Sans).
- Palette: Okabe-Ito 8-colour, colour-blind safe and greyscale-distinguishable.
- Sizing: single source THESIS_TEXTWIDTH_IN = 5.5; matches the manuscript body
  textwidth at 12pt. Figures are sized once and not re-rescaled.
- Output: vector PDF; raster fallback at 300 DPI.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.font_manager as fm

# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
THESIS_TEXTWIDTH_IN = 5.5  # manuscript body textwidth, ~5.5 in at 12pt


def fig_size(rel_width: float = 1.0, aspect: float = 0.65):
    """Figure size in inches at the given fraction of textwidth and aspect.

    aspect is height / width. 0.65 is a comfortable wide rectangle; use 1.0
    for square; use 0.5 for very wide.
    """
    w = THESIS_TEXTWIDTH_IN * rel_width
    return (w, w * aspect)


# ---------------------------------------------------------------------------
# Palette: Okabe-Ito 8-colour
# ---------------------------------------------------------------------------
OKABE_ITO = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "red":     "#D55E00",  # vermillion / orange-red
    "purple":  "#CC79A7",
}

# Semantic colour assignments — used uniformly across figures
COLOUR_FF       = OKABE_ITO["blue"]     # FF / classifier feedforward
COLOUR_LSTM     = OKABE_ITO["red"]      # LSTM (was originally teal — review wants D55E00)
COLOUR_TFT      = OKABE_ITO["green"]    # TFT (forecaster-shaped)
COLOUR_MACRO    = OKABE_ITO["orange"]   # macro modality
COLOUR_NEWS     = OKABE_ITO["purple"]   # news modality
COLOUR_SOCIAL   = OKABE_ITO["skyblue"]  # social modality
COLOUR_GRAPH    = OKABE_ITO["yellow"]   # graph modality
COLOUR_BASELINE = "#999999"             # price-only baseline / null reference

# Misc helpers used by many figures
COLOUR_DARK      = "#222222"
COLOUR_GREY      = "#777777"
COLOUR_LIGHT     = "#cccccc"
COLOUR_HIGHLIGHT = OKABE_ITO["orange"]


# ---------------------------------------------------------------------------
# Font selection — try LaTeX-default fonts first, fall back to STIXGeneral
# ---------------------------------------------------------------------------
def _select_serif_font() -> list[str]:
    """Return the matplotlib font.serif list with the best available font first."""
    available = {f.name for f in fm.fontManager.ttflist}
    preferred = [
        "Latin Modern Roman",
        "Computer Modern Roman",
        "CMU Serif",
        "STIX Two Text",
        "STIXGeneral",
    ]
    chosen = [f for f in preferred if f in available]
    # Always pad the list with reasonable fallbacks
    return chosen + ["STIXGeneral", "DejaVu Serif"]


def apply_style() -> None:
    """Apply the unified style to matplotlib rcParams."""
    serif_list = _select_serif_font()

    mpl.rcParams.update({
        # Font — match study body
        "font.family": "serif",
        "font.serif": serif_list,
        "mathtext.fontset": "stix",  # closest stretch of CM-style maths in matplotlib
        "text.usetex": False,
        # Sizes (per review §style guide)
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        # Output
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.10,  # was 0.05 — too tight, caused clipping
        # Axes
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.labelpad": 6,
        # Lines + markers
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,
    })


__all__ = [
    "THESIS_TEXTWIDTH_IN",
    "fig_size",
    "OKABE_ITO",
    "COLOUR_FF",
    "COLOUR_LSTM",
    "COLOUR_TFT",
    "COLOUR_MACRO",
    "COLOUR_NEWS",
    "COLOUR_SOCIAL",
    "COLOUR_GRAPH",
    "COLOUR_BASELINE",
    "COLOUR_DARK",
    "COLOUR_GREY",
    "COLOUR_LIGHT",
    "COLOUR_HIGHLIGHT",
    "apply_style",
]
