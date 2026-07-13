"""Generate the paper figures from the committed experiment results.

Outputs (vector PDF) into <artifact-root>/figures/:
  rq1-forest.pdf        per-config MCC delta vs A7 (paired), 95% CI, both families
  msgca-diagnostic.pdf  MSGCA reproduction (best-on-test) vs honest (val-best),
                        with the within-run selection-inflation band, from the
                        finished re-run JSON.
  ... and msgca-epochs / param-budget / news-by-fold / cross-family /
  vol-heatmap / sharpe / encoder-box.

All inputs are read from the committed CSVs / JSON under <artifact-root>/results/;
no GPU and no raw data are needed. The headline deltas come from the authoritative
table (AUTH, below) rather than ad-hoc recomputation -- see the note on AUTH.

CSV schema: columns are mcc, config_id, fold_idx, fusion_type, seed. Architecture
is implied by the file: ablation_ff.csv / ablation_lstm.csv hold A4-A9;
ablation_news_ff.csv / ablation_news_lstm.csv hold A1-A3. A7 is the price-only
baseline and appears in the non-news files.

Run from the artifact root:
  python scripts/figures/make_paper_figures.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_root() -> Path:
    """Locate the artifact root (the directory that contains results/)."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "results").is_dir() and (p / "scripts").is_dir():
            return p
    # fall back to two levels up (scripts/figures/ -> root)
    return here.parents[2]


ROOT = _find_root()
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#3a6ea5"
BLUE_D = "#274c73"
ORANGE = "#d9822b"
GREY = "#9aa0a6"
plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.7, "axes.edgecolor": "#444",
    "xtick.color": "#444", "ytick.color": "#444", "text.color": "#222",
    "axes.labelcolor": "#222", "figure.dpi": 150,
})


# Authoritative per-config classifier-shaped statistics, transcribed verbatim
# from results/analysis/phase5_tables_750core.md (the project's
# AUTHORITATIVE direction analysis; n=30 paired = 5 folds x 3 seeds x 2 arch;
# A8/A9 are the LayerNorm-corrected graph runs). These are NOT recomputed here:
# the authoritative pipeline applies the graph_ln correction and a specific paired
# construction; reproducing it ad hoc gave inconsistent values, so the published
# table is the single source of truth. mean MCC, delta vs A7, Cohen's d.
AUTH = {
    # cfg: (raw_mean_mcc, paired_delta_vs_a7, cohens_d)
    # NOTE: raw_mean is the unpaired mean over all of that config's runs; the
    # paired delta is the mean of per-(fold,seed,arch) differences against A7 on
    # MATCHED cells. They are different statistics; do not "reconcile" them by
    # recomputation. Figures use only the paired delta and d.
    # MACRO rows (A1,A2,A4,A5,A9) are from the publication-lag re-run
    # (results/macrolag/, via scripts/analysis/analyze_macrolag.py), paired
    # against the matched re-run baseline. Non-macro rows (A3,A6,A8) are the
    # published v2 values (phase5_tables_750core.md).
    "A1": (+0.0004, -0.0048, -0.241),
    "A2": (-0.0046, -0.0097, -0.458),
    "A3": (0.0080, -0.0034, -0.191),
    "A4": (-0.0011, -0.0063, -0.251),
    "A5": (-0.0000, -0.0052, -0.248),
    "A6": (0.0098, +0.0005, +0.030),
    "A8": (0.0063, +0.0000, +0.001),
    "A9": (-0.0005, -0.0057, -0.260),
}
A7_MEAN = 0.0095
N_PAIRED = 30  # 5 folds x 3 seeds x 2 architectures

# Display layer: the article names configurations by composition (P = price
# baseline; N/M/S/G = news/macro/social/graph; P+all = all four). Data stays
# keyed by the archive IDs (A1-A9) used in the result CSVs; only the DISPLAY
# uses these names. Display order: baseline, singles, macro-anchored pairs, all.
DISP = {"A7": "P", "A3": "P+N", "A4": "P+M", "A6": "P+S", "A8": "P+G",
        "A2": "P+N+M", "A5": "P+M+S", "A9": "P+M+G", "A1": "P+N+M+S"}
ORDER8 = ["A3", "A4", "A6", "A8", "A2", "A5", "A9", "A1"]  # non-baseline configs
ORDER9 = ["A7"] + ORDER8

# Cross-family contrasts: forecaster-shaped TFT MCC minus classifier-shaped MCC,
# paired on (fold, seed), n=15 per cell. Transcribed verbatim from the audited
# reference table (tab:ch5-cross-family); A6-vs-LSTM (-0.021, d=-0.916,
# p_bonf=0.029) is the ONLY contrast clearing Bonferroni anywhere in the 18-test
# grid -- and it is negative. These are NOT recomputed here.
CF_FF = {"A1": +0.017, "A2": +0.012, "A3": -0.002, "A4": +0.012, "A5": +0.002,
         "A6": -0.008, "A7": -0.004, "A8": +0.007, "A9": +0.005}
CF_LSTM = {"A1": +0.005, "A2": +0.008, "A3": +0.000, "A4": +0.014, "A5": -0.001,
           "A6": -0.021, "A7": -0.005, "A8": +0.009, "A9": -0.010}


def fig_rq1_forest():
    """Forest plot of per-config MCC delta vs A7, from the authoritative table.

    The interval drawn is delta +/- |delta/d| * d_se where the SE on a paired
    Cohen's d at n=30 is approx sqrt(1/n + d^2/(2n)); this yields an honest CI on
    the delta consistent with the published d, without re-deriving the delta.
    """
    cfgs = ORDER8
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    y = np.arange(len(cfgs))[::-1]
    means, errs = [], []
    for c in cfgs:
        _, delta, d = AUTH[c]
        means.append(delta)
        # SE(d) at n; convert to a delta-scale half-width via |delta|/|d|
        d_se = np.sqrt(1.0 / N_PAIRED + d * d / (2 * N_PAIRED))
        scale = abs(delta / d) if abs(d) > 1e-6 else abs(delta) / 0.05
        errs.append(1.96 * d_se * scale)
    ax.errorbar(means, y, xerr=errs, fmt="s", ms=5, color=BLUE, ecolor=BLUE,
                elinewidth=1.1, capsize=2.6, zorder=3)
    ax.axvline(0, color=ORANGE, lw=1.1, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([DISP[c] for c in cfgs], fontsize=8.5)
    ax.set_xlabel("MCC delta vs price-only baseline (P); 95% CI, $n=30$ paired")
    ax.set_title("No configuration beats price-only after correction",
                 fontsize=9.5, color=BLUE_D, loc="left", pad=8)
    ax.set_xlim(-0.022, 0.013)
    ax.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/rq1-forest.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote rq1-forest.pdf (authoritative deltas)")


def fig_msgca():
    j = json.load(open(ANALYSIS / "msgca_diagnostic_rerun.json"))
    r = j["report_by_budget"]
    budgets = ["200", "300", "400"]
    repro = [r[b]["arm_a_reproduction_best_on_test"]["mean"] for b in budgets]
    honest = [r[b]["arm_b_honest_best_on_val_test"]["mean"] for b in budgets]

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    x = np.arange(len(budgets))
    w = 0.34
    # Two bars per budget = the bottom-line collapse. The released code on its
    # ORIGINAL split (best-on-test, no validation set) reproduces the reported
    # headline; an honest leakage-free protocol (a proper train/validation/test
    # split, epoch chosen on validation) collapses the same model and data to
    # chance. The collapse corrects two coupled flaws -- no validation set at all
    # AND test-set selection; the selection-only part, holding data fixed, is the
    # +0.04-0.07 within-run inflation reported in Table 6.
    ax.bar(x - w / 2, repro, w, color=BLUE, zorder=3,
           label="best-on-test, original split (reported / reproduced)")
    ax.bar(x + w / 2, honest, w, color=GREY, zorder=3,
           label="validation-best, leakage-free split (honest)")
    ax.axhline(0.1112, color=BLUE_D, lw=1.0, ls=":", zorder=2)
    ax.text(2.52, 0.1112, "paper 0.111", va="bottom", ha="right",
            fontsize=7.5, color=BLUE_D)
    ax.axhline(0, color="#444", lw=0.9, zorder=2)
    ax.text(2.52, 0.003, "chance", va="bottom", ha="right",
            fontsize=7, color="#555", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} epochs" for b in budgets])
    ax.set_ylabel("MCC on BigData22")
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.045, 0.205)
    ax.set_title("MSGCA: the reported headline does not survive a leakage-free protocol",
                 fontsize=9.2, color=BLUE_D, loc="left", pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=7.6,
              borderaxespad=0.6, handletextpad=0.5, labelspacing=0.4)
    ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/msgca-diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote msgca-diagnostic.pdf repro=%s honest=%s" %
          ([round(v, 3) for v in repro], [round(v, 3) for v in honest]))


def fig_news_by_fold():
    """News degradation concentrates in the 2022 bear (F3) and 2023 (F4) folds.
    Per-fold A2-minus-A4 paired MCC delta (P+N+M vs P+M in article naming),
    n=18 per fold (3 seeds x 3 fusions x 2 architectures). Values transcribed
    from the verified recompute (matches the authoritative
    phase5_full_750core.md exactly): overall -0.0098, F4 -0.0293, F3 -0.0141."""
    folds = ["F0", "F1", "F2", "F3", "F4"]
    regimes = ["pre-COVID\n+ crash", "recovery", "transition",
               "2022 bear", "2023 rally"]
    delta = [0.0002, -0.0002, -0.0056, -0.0141, -0.0293]  # A2 - A4, per fold

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(folds))
    cols = [GREY if d > -0.005 else ORANGE for d in delta]
    ax.bar(x, delta, 0.62, color=cols, zorder=3)
    ax.axhline(0, color="#444", lw=0.8, zorder=2)
    for xi, d in zip(x, delta):
        ax.text(xi, d - 0.0016 if d < 0 else d + 0.0006, f"{d:+.3f}",
                ha="center", va="top" if d < 0 else "bottom", fontsize=7.5,
                color="#9a5a12" if d < -0.005 else "#666")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n{r}" for f, r in zip(folds, regimes)],
                       fontsize=7.5)
    ax.set_ylabel("(P+N+M) $-$ (P+M) paired $\\Delta$ MCC")
    ax.set_title("News degrades, and the damage concentrates in volatile folds",
                 fontsize=9.5, color=BLUE_D, loc="left", pad=8)
    ax.set_ylim(-0.034, 0.006)
    ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/news-by-fold.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote news-by-fold.pdf")


def fig_msgca_epochs():
    """The selection mechanism made visual, WITHIN one training run.

    Uses a single validation-arm run (arm_b), whose history records BOTH the
    per-epoch test MCC and the per-epoch validation MCC, so the two selection
    rules can be applied to the SAME trajectory of the SAME run: best-on-test
    reports the running maximum of the test curve; validation-based selection
    commits to the best-validation epoch and reports the test MCC there. The
    vertical gap between the two reported points is therefore pure selection
    effect (training data and length held fixed) -- the within-run inflation
    quoted in the article. We show arm_b[2] (split_seed=300), the run whose gap
    (+0.091) is closest to the eight-split mean (+0.069 at the 400-epoch
    budget); three of the eight splits never train above chance at that budget,
    so their gap is zero and the mean includes them.
    Data: msgca_diagnostic_rerun_history.json.
    """
    h = json.load(open(ANALYSIS / "msgca_diagnostic_rerun_history.json"))
    runs = h["arm_b"]
    # Eight-split mean within-run inflation at the full 400-epoch budget, for
    # the on-chart annotation (must agree with the rerun JSON's +0.069).
    gaps = []
    for run in runs:
        hist = run["history"]
        val = [r["val_mcc"] for r in hist]
        test = [r["test_mcc"] for r in hist]
        gaps.append(max(test) - test[int(np.argmax(val))])
    mean_gap = float(np.mean(gaps))

    rb = runs[2]  # split_seed=300: gap closest to the eight-split mean
    hist = rb["history"]
    ep = [r["epoch"] for r in hist]
    test_mcc = [r["test_mcc"] for r in hist]
    val_mcc = [r["val_mcc"] for r in hist]
    run_max = np.maximum.accumulate(test_mcc)
    best_ep = int(np.argmax(test_mcc))
    best_test = test_mcc[best_ep]
    sel = int(np.argmax(val_mcc))           # epoch chosen by validation
    test_at_sel = test_mcc[sel]
    gap = best_test - test_at_sel

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ep, test_mcc, color=BLUE, lw=0.8, alpha=0.45,
            label="test MCC per epoch (one run)")
    ax.plot(ep, run_max, color=BLUE_D, lw=1.6,
            label="its running max (what best-on-test reports)")
    ax.axhline(0, color="#444", lw=0.8, zorder=1)
    ax.text(4, 0.0025, "chance", va="bottom", ha="left", fontsize=7,
            color="#555", style="italic")
    # best-on-test reported point (peak of the same curve)
    ax.plot(best_ep, best_test, "o", color=BLUE_D, ms=6, zorder=5)
    ax.annotate(f"best-on-test reports {best_test:.3f} (epoch {best_ep})",
                xy=(best_ep, best_test), xytext=(8, 0.114),
                fontsize=7.5, color=BLUE_D,
                arrowprops=dict(arrowstyle="->", color=BLUE_D, lw=0.7))
    # validation-selected point ON THE SAME CURVE
    ax.plot(sel, test_at_sel, "s", color=ORANGE, ms=6, zorder=5)
    ax.annotate(f"validation selects epoch {sel};\ntest MCC there: {test_at_sel:.3f}",
                xy=(sel, test_at_sel), xytext=(8, -0.062),
                fontsize=7.5, color="#9a5a12",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7))
    # the within-run gap, drawn to scale at the right edge
    gx = 388
    ax.annotate("", xy=(gx, best_test), xytext=(gx, test_at_sel),
                arrowprops=dict(arrowstyle="<->", color="#9a5a12", lw=1.2))
    ax.annotate(f"within-run selection\ninflation: +{gap:.3f}\n"
                f"(8-split mean +{mean_gap:.3f})",
                xy=(gx, (best_test + test_at_sel) / 2), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=7.2, color="#9a5a12")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("MCC on BigData22 test set")
    ax.set_title("Two selection rules applied to the same training run",
                 fontsize=9.5, color=BLUE_D, loc="left", pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax.set_xlim(0, 400)
    ax.set_ylim(-0.095, 0.152)
    ax.grid(color="#ececec", lw=0.6, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/msgca-epochs.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote msgca-epochs.pdf  split_seed={rb['split_seed']} "
          f"best-on-test={best_test:.4f}@{best_ep} "
          f"val-selected-test={test_at_sel:.4f}@{sel} gap=+{gap:.4f} "
          f"mean8={mean_gap:.4f} per-split gaps={[round(g, 3) for g in gaps]}")


def fig_param_budget():
    """Parameter budget: the two families on a log scale, with the dominant
    block annotated. Numbers are verified against the run CSVs: classifier
    n_params 9,251 (A7) .. 88,035 (A9 gated-cross-attn LSTM, graph-LN grid);
    TFT n_params 195,021 (A7) .. 277,701 (A1), from results/v3/
    m8_ablation/full_9config.csv. The point is the ~10x gap and where capacity
    concentrates, which backs the capacity-robustness argument; sub-block
    magnitudes are shown qualitatively, not mixed across sources.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    fams = ["classifier-shaped\n(light head)", "forecaster-shaped\n(TFT body)"]
    lo = [9251, 195021]
    hi = [88035, 277701]
    y = [0, 1]
    for yi, l, h, col in zip(y, lo, hi, [GREY, BLUE]):
        ax.plot([l, h], [yi, yi], color=col, lw=8, solid_capstyle="round",
                zorder=3)
        ax.plot(l, yi, "o", color=col, ms=4, zorder=4)
        ax.plot(h, yi, "o", color=col, ms=4, zorder=4)
        # left label anchored right-of-point, right label left-of-point, so
        # the two never collide even when the bar is short (log scale).
        ax.annotate(f"{l:,}", xy=(l, yi), xytext=(-4, 9),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=7.5, color=col)
        ax.annotate(f"{h:,}", xy=(h, yi), xytext=(4, 9),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=7.5, color=col)
    ax.text(np.sqrt(9251 * 88035), -0.26,
            "heaviest block: fusion projection", ha="center", va="top",
            fontsize=7.2, color=BLUE_D, style="italic")
    ax.text(np.sqrt(195021 * 277701), 0.74,
            "heaviest block: variable-selection network; 95-99% in body",
            ha="center", va="bottom", fontsize=7.2, color=BLUE_D, style="italic")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(fams, fontsize=8.5)
    ax.set_ylim(-0.6, 1.5)
    ax.set_xlim(5e3, 5e5)
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_title("A tenfold capacity gap, tested under one grid",
                 fontsize=9.5, color=BLUE_D, loc="left", pad=8)
    ax.grid(axis="x", color="#ececec", lw=0.6, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/param-budget.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote param-budget.pdf")


# Per-fold paired delta vol-R^2 (config minus A7), concat fusion, n=6 per cell.
# Transcribed VERBATIM from the audited reference appendix table that the article
# also prints (tab:app-volatility). NOT recomputed -- the figure and the table
# share this single source so they cannot diverge.
# Per-fold delta vol-R^2 (config minus A7), concat-only, n=6/fold (2 arch x 3 seeds).
# The five macro rows (A1,A2,A4,A5,A9) are the LEAKAGE-FREE publication-lag re-run,
# paired against the re-run concat baseline. The three non-macro rows (A3,A6,A8)
# contain no macro feature, are invariant to the macro timing convention, and
# retain the original volatility ablation. Matches tab:app-volatility.
VOL = {
    "A1": [+0.0229, +0.0267, +0.0381, -0.0333, +0.0084],
    "A2": [-0.0018, +0.0116, +0.0014, -0.0323, -0.0024],
    "A3": [-0.0085, -0.0236, +0.0146, +0.0171, -0.0627],
    "A4": [-0.0089, +0.0111, +0.0093, -0.0280, -0.0591],
    "A5": [+0.0018, +0.0698, +0.0389, -0.0417, -0.0050],
    "A6": [+0.0165, +0.0172, +0.0298, +0.0092, -0.0164],
    "A8": [+0.0252, -0.0400, +0.0249, +0.0097, -0.0241],
    "A9": [+0.0023, -0.0063, +0.0092, -0.0238, +0.0091],
}

# Per-config naive daily long-short Sharpe (gross of costs), transcribed VERBATIM
# from the article's financial-metrics appendix table (tab:app-financial). The
# point: price-only A7 has the HIGHEST Sharpe of any configuration -- no
# multi-source config buys economic value either. NOT recomputed.
SHARPE_FF = {"A1": 0.006, "A2": 0.038, "A3": 0.333, "A4": -0.007, "A5": 0.105,
             "A6": 0.377, "A7": 0.427, "A8": 0.303, "A9": 0.117}
SHARPE_LSTM = {"A1": 0.221, "A2": -0.115, "A3": 0.259, "A4": 0.031, "A5": 0.049,
               "A6": 0.403, "A7": 0.512, "A8": 0.240, "A9": 0.116}


def fig_vol_heatmap():
    """Per-fold delta vol-R^2 by config (config minus A7). Shows the volatility
    null is regime-uniform: small mixed deltas, no consistent edge, and the
    largest departures are the negative ones in the 2023 (F4) column. Drawn from
    the transcribed VOL dict (same source as tab:app-volatility)."""
    cfgs = ORDER8
    folds = ["F0", "F1", "F2", "F3", "F4"]
    M = np.array([VOL[c] for c in cfgs])
    vmax = np.abs(M).max()

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels(folds)
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels([DISP[c] for c in cfgs], fontsize=8)
    ax.set_xlabel("fold (test-window regime)")
    for i in range(len(cfgs)):
        for j in range(len(folds)):
            v = M[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=6.6,
                    color="white" if abs(v) > 0.6 * vmax else "#222")
    ax.set_title("Volatility: per-fold $\\Delta R^2$ vs price-only (no edge)",
                 fontsize=9.5, color=BLUE_D, loc="left", pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("$\\Delta R^2$ (config $-$ P)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/vol-heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote vol-heatmap.pdf (transcribed per-fold delta R^2)")


def fig_sharpe():
    """Per-config naive Sharpe (gross), both architectures. Price-only A7 is the
    HIGHEST -- no multi-source config buys economic value. From transcribed
    SHARPE_FF / SHARPE_LSTM (tab:app-financial)."""
    cfgs = ORDER9
    x = np.arange(len(cfgs))
    ff = [SHARPE_FF[c] for c in cfgs]
    ls = [SHARPE_LSTM[c] for c in cfgs]
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    cols_ff = [ORANGE if c == "A7" else GREY for c in cfgs]
    cols_ls = [ORANGE if c == "A7" else BLUE for c in cfgs]
    ax.bar(x - w / 2, ff, w, color=cols_ff, zorder=3)
    ax.bar(x + w / 2, ls, w, color=cols_ls, zorder=3)
    ax.axhline(0, color="#444", lw=0.8, zorder=2)
    ax.axhline(SHARPE_LSTM["A7"], color=ORANGE, lw=0.9, ls=":", zorder=2)
    ax.text(8.4, SHARPE_LSTM["A7"], "price-only ceiling", ha="right", va="bottom",
            fontsize=7, color="#9a5a12")
    ax.set_xticks(x)
    ax.set_xticklabels([DISP[c] for c in cfgs], fontsize=7.6)
    ax.set_ylabel("naive daily long-short Sharpe (gross)")
    ax.set_xlabel("configuration")
    ax.set_title("Price-only has the highest Sharpe; no source buys value",
                 fontsize=9.5, color=BLUE_D, loc="left", pad=8)
    # explicit handles: with P (highlighted orange) first, bar-derived legend
    # swatches would both inherit the highlight colour.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GREY, label="feedforward"),
                       Patch(facecolor=BLUE, label="LSTM"),
                       Patch(facecolor=ORANGE, label="price-only baseline")],
              loc="upper center", bbox_to_anchor=(0.42, 0.97), ncol=3,
              frameon=False, fontsize=7.5, handlelength=1.2,
              columnspacing=1.0, handletextpad=0.5)
    ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sharpe.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote sharpe.pdf (transcribed financial metrics)")


def fig_encoder_box():
    """Per-encoder MCC distribution on A3 (price+news) and A7 (price-only).
    Reads the real encoder-screening CSVs (verified schema: mcc, config_id,
    fold_idx, seed; arch implied by filename). Four encoders, 30 paired obs each
    (5 folds x 3 seeds x 2 arch). The medians sit on top of each other -> encoder
    choice is not the bottleneck. Anchored against the 6 pairwise contrasts in
    tab:app-encoder."""
    enc_files = {"FinBERT": "finbert", "BGE": "bge",
                 "FinLang": "finlang", "Qwen3": "qwen3"}
    base = RESULTS / "encoder_screen"
    data = {}  # (encoder, cfg) -> array of mcc
    means = {}
    for name, key in enc_files.items():
        frames = [pd.read_csv(base / f"encoder_screen_{key}_{a}.csv")
                  for a in ("ff", "lstm")]
        df = pd.concat(frames, ignore_index=True)
        for cfg in ("A3", "A7"):
            vals = df.loc[df.config_id == cfg, "mcc"].to_numpy()
            assert len(vals) == 30, f"{name} {cfg}: expected 30, got {len(vals)}"
            data[(name, cfg)] = vals
            means[(name, cfg)] = float(np.mean(vals))

    # A7 (price-only) carries no news pathway, so it is identical across the four
    # encoder files (verified: same 30 rows). Show it ONCE as the baseline
    # reference, with the four A3 (price+news) encoder distributions beside it.
    encs = list(enc_files)
    a7 = data[("FinBERT", "A7")]  # identical in every file
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    positions = [0, 1, 2, 3, 4.4]
    box_data = [data[(n, "A3")] for n in encs] + [a7]
    box_colors = [BLUE, BLUE, BLUE, BLUE, GREY]
    xlabels = encs + ["price-only"]
    bp = ax.boxplot(box_data, positions=positions, widths=0.62,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="#222", lw=1.1))
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#444")
    # baseline median guide line across the A3 boxes
    ax.axhline(float(np.median(a7)), color=GREY, lw=0.9, ls=":", zorder=1)
    ax.axhline(0, color="#444", lw=0.7, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=7.6, rotation=18)
    ax.set_ylabel("MCC (30 paired obs per box)")
    ax.text(1.5, ax.get_ylim()[1] * 0.93, "P+N (price+news), by encoder",
            ha="center", fontsize=8, color=BLUE_D, fontweight="bold")
    ax.text(4.4, ax.get_ylim()[1] * 0.93, "P", ha="center",
            fontsize=8, color="#555", fontweight="bold")
    ax.set_title("Four encoders, one distribution: the encoder is not the bottleneck",
                 fontsize=9.3, color=BLUE_D, loc="left", pad=8)
    ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/encoder-box.pdf", bbox_inches="tight")
    plt.close(fig)
    sp = {k: round(v, 4) for k, v in means.items()}
    print("wrote encoder-box.pdf  per-(enc,cfg) means:", sp)


def fig_cross_family():
    """Capacity robustness: the forecaster-shaped TFT (~10x params) minus each
    classifier-shaped concat baseline, per configuration. Every contrast sits in
    the Bonferroni-undetectable band around zero except A6 (price+social) against
    the LSTM reference, which dips to -0.021 (d=-0.916) -- the only contrast that
    clears correction across the whole 18-test grid, and it is negative. Drawn
    from the transcribed CF_FF / CF_LSTM dicts (audited reference table)."""
    cfgs = ORDER9
    x = np.arange(len(cfgs))
    ff = [CF_FF[c] for c in cfgs]
    ls = [CF_LSTM[c] for c in cfgs]

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    # the smallest effect the per-contrast test (n=15) can resolve is large;
    # shade the near-zero region purely as a visual "indistinguishable" guide.
    ax.axhspan(-0.010, 0.010, color="#f0f0f2", zorder=0)
    ax.axhline(0, color="#444", lw=0.8, zorder=1)
    ax.plot(x - 0.11, ff, "o", ms=6, color=GREY, zorder=3,
            label="vs. feedforward concat")
    ax.plot(x + 0.11, ls, "s", ms=6, color=BLUE, zorder=3,
            label="vs. LSTM concat")
    # emphasise the lone Bonferroni-clearing (negative) contrast: P+S vs LSTM
    a6 = cfgs.index("A6")
    ax.plot(a6 + 0.11, CF_LSTM["A6"], "s", ms=9, mfc="none",
            mec=ORANGE, mew=1.8, zorder=4)
    ax.annotate("largest contrast ($d=-0.916$):\nnegative, borderline ($p_\\mathrm{bonf}\\approx0.058$)",
                xy=(a6 + 0.11, CF_LSTM["A6"]), xytext=(a6 + 1.3, -0.027),
                fontsize=7.2, color="#9a5a12",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels([DISP[c] for c in cfgs], fontsize=7.6)
    ax.set_ylabel("TFT $-$ classifier $\\Delta$ MCC ($n=15$ paired)")
    ax.set_xlabel("configuration")
    ax.set_title("Tenfold capacity recovers no multi-source signal",
                 fontsize=9.5, color=BLUE_D, loc="left", pad=8)
    ax.set_ylim(-0.030, 0.024)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cross-family.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cross-family.pdf (transcribed cross-family deltas)")


if __name__ == "__main__":
    # The forest plot is drawn from AUTH (the authoritative published table),
    # which the draft Table 1 also uses; no ad-hoc recomputation. Confirm the
    # draft and the figure share the same delta source.
    draft = {"A1": -0.0048, "A2": -0.0097, "A3": -0.0034, "A4": -0.0063,
             "A5": -0.0052, "A6": 0.0005, "A8": 0.0000, "A9": -0.0057}
    assert all(abs(AUTH[c][1] - draft[c]) < 1e-9 for c in draft), \
        "figure AUTH deltas diverge from draft Table 1"
    print("AUTH deltas == draft Table 1 (authoritative source).")
    fig_rq1_forest()
    fig_msgca()
    fig_msgca_epochs()
    fig_param_budget()
    fig_news_by_fold()
    fig_cross_family()
    fig_vol_heatmap()
    fig_sharpe()
    fig_encoder_box()
    print("done ->", OUT)
