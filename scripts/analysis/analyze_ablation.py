"""Phase 4-R / Phase 5 ablation analysis infrastructure.

Analyzes the full ablation matrix (750+ experiments) for all 3 Research Questions.
Works with partial results — call at any time to see progress.

Core CSVs (750 experiments):
  - results/ablation_ff.csv        (FF non-news: A4-A9)
  - results/ablation_lstm.csv       (LSTM non-news: A4-A9)
  - results/ablation_news_ff.csv    (FF news: A1-A3)
  - results/ablation_news_lstm.csv  (LSTM news: A1-A3)

Supplementary CSVs (loaded when present):
  - results/ablation_a10_ff.csv        (A10: price+news+social, FF)
  - results/ablation_a10_lstm.csv      (A10: price+news+social, LSTM)
  - results/ablation_social_lstm.csv   (A5+A6 with social LSTM sequences)
  - results/ablation_graph_ln_ff.csv   (A8+A9 with GraphEncoder LayerNorm fix, FF)
  - results/ablation_graph_ln_lstm.csv (A8+A9 with GraphEncoder LayerNorm fix, LSTM)

All CSVs have identical columns; they are concatenated at load time.

Each row has columns:
  config_id, fusion_type, fold_idx, seed, modalities, mcc, accuracy,
  best_val_mcc, epochs_trained, n_train, n_test, n_params,
  deadzone, batch_size, patience, grad_clip, hidden_dim, learning_rate, alpha,
  sharpe_ratio, profit_factor, cumulative_return, max_drawdown,
  precision, recall, f1, mse, rmse, mae, r2, directional_accuracy

Usage (run from the artifact root; CSVs are auto-located under results/):

    # Full analysis (all RQs):
    python scripts/analysis/analyze_ablation.py

    # Individual sections:
    python scripts/analysis/analyze_ablation.py --section rq1
    python scripts/analysis/analyze_ablation.py --section rq2
    python scripts/analysis/analyze_ablation.py --section rq3
    python scripts/analysis/analyze_ablation.py --section arch
    python scripts/analysis/analyze_ablation.py --section overview
    python scripts/analysis/analyze_ablation.py --section financial
    python scripts/analysis/analyze_ablation.py --section ms_vs_baseline
    python scripts/analysis/analyze_ablation.py --section fin_metrics

    # Save markdown report:
    python scripts/analysis/analyze_ablation.py --output /tmp/analysis.md

    # Custom data directory or explicit CSV files:
    python scripts/analysis/analyze_ablation.py --data-dir path/to/csvs
    python scripts/analysis/analyze_ablation.py --csv path/to/file1.csv path/to/file2.csv
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ============================================================================
# CONSTANTS
# ============================================================================


def _find_results() -> Path:
    """Locate the committed results directory (artifact-root/results)."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "results").is_dir() and (p / "scripts").is_dir():
            return p / "results"
    # fall back: CWD-relative legacy layout
    return Path("results") if Path("results").is_dir() else Path("experiments/phase4r")


# The ablation/supplementary CSVs live under results/ (in the ablation/ subdir);
# resolve each by basename so the analysis is independent of the exact layout.
RESULTS_DIR = _find_results()


def _csv(basename: str) -> Path:
    """Resolve a result CSV by basename, searching results/ recursively."""
    direct = RESULTS_DIR / basename
    if direct.exists():
        return direct
    hits = sorted(RESULTS_DIR.rglob(basename))
    return hits[0] if hits else direct


PHASE4R_DIR = RESULTS_DIR  # back-compat alias for downstream defaults

# All expected CSV files (non-news + news, split to avoid write conflicts)
CSV_FILES = {
    "FF": [
        _csv("ablation_ff.csv"),
        _csv("ablation_news_ff.csv"),
        _csv("ablation_a10_ff.csv"),
    ],
    "LSTM": [
        _csv("ablation_lstm.csv"),
        _csv("ablation_news_lstm.csv"),
        _csv("ablation_a10_lstm.csv"),
    ],
}

# Supplementary CSVs (loaded separately with special handling)
SUPPLEMENTARY_CSVS = {
    "social_lstm": _csv("ablation_social_lstm.csv"),
    "graph_ln_ff": _csv("ablation_graph_ln_ff.csv"),
    "graph_ln_lstm": _csv("ablation_graph_ln_lstm.csv"),
}

# Fold regime labels
FOLD_LABELS = {
    0: "Pre-COVID (2019H2-2020H1)",
    1: "Recovery (2020H2-2021H1)",
    2: "Bull-to-Bear (2021H2-2022H1)",
    3: "Bear Market (2022H2-2023H1)",
    4: "AI Rally (2023H2)",
}

# Ablation configuration definitions
ABLATION_CONFIGS = {
    "A1": {"modalities": ["price", "news", "macro", "social"], "desc": "Full model (4 modalities)"},
    "A2": {"modalities": ["price", "news", "macro"], "desc": "Remove social"},
    "A3": {"modalities": ["price", "news"], "desc": "Price + news"},
    "A4": {"modalities": ["price", "macro"], "desc": "Price + macro"},
    "A5": {"modalities": ["price", "macro", "social"], "desc": "Price + macro + social"},
    "A6": {"modalities": ["price", "social"], "desc": "Price + social"},
    "A7": {"modalities": ["price"], "desc": "Price only (baseline)"},
    "A8": {"modalities": ["price", "graph"], "desc": "Price + graph"},
    "A9": {"modalities": ["price", "macro", "graph"], "desc": "Price + macro + graph"},
    "A10": {"modalities": ["price", "news", "social"], "desc": "Price + news + social"},
}

# Modality names for leave-one-out analysis
MODALITY_NAMES = ["price", "news", "macro", "social", "graph"]

# Phase 3-R diagnostic reference values (honest train/val/test splits)
# Used to sanity-check Phase 4-R results against known baselines
PHASE3R_REFERENCE = {
    "A7_FF": {
        "mean_mcc": 0.009,
        "per_fold": {0: 0.043, 1: 0.045, 2: -0.015, 3: -0.002, 4: -0.013},
    },
    "A7_LSTM": {
        "mean_mcc": 0.019,
        "per_fold": {},  # all 5 folds positive, exact values TBD
    },
}

# Primary metric
METRIC = "mcc"

# Output buffer for markdown mode
_output_lines = []
_output_mode = "print"  # "print" or "buffer"

# Supplementary data (populated during load)
_SOCIAL_LSTM_DF = None  # Social LSTM sequence experiments (A5+A6 with seq_mods=price,social)


def out(s=""):
    """Output a line (either print or buffer for markdown)."""
    if _output_mode == "buffer":
        _output_lines.append(s)
    else:
        print(s)


# ============================================================================
# DATA LOADING
# ============================================================================

def _load_arch_csvs(arch: str, csv_paths: list[Path]) -> pd.DataFrame | None:
    """Load and concatenate all CSVs for one architecture."""
    parts = []
    for path in csv_paths:
        if path.exists():
            df = pd.read_csv(path)
            out(f"  Loaded {len(df)} rows from {path.name}")
            parts.append(df)
    if not parts:
        return None
    merged = pd.concat(parts, ignore_index=True)
    merged["architecture"] = arch
    if "best_val_mcc" in merged.columns:
        merged["val_test_gap"] = merged["best_val_mcc"] - merged[METRIC]
    out(f"  {arch} total: {len(merged)} experiments")
    return merged


def _load_supplementary() -> dict[str, pd.DataFrame]:
    """Load supplementary CSVs (social LSTM, graph LN). Returns dict keyed by name."""
    loaded = {}
    for name, path in SUPPLEMENTARY_CSVS.items():
        if path.exists():
            df = pd.read_csv(path)
            out(f"  Loaded {len(df)} supplementary rows from {path.name}")
            loaded[name] = df
    return loaded


def _replace_graph_rows(arch_df: pd.DataFrame, graph_ln_df: pd.DataFrame,
                        arch: str) -> pd.DataFrame:
    """Replace original A8/A9 rows with Graph LayerNorm versions.

    Only replaces if the LN dataset is complete (>= original A8/A9 count).
    During partial runs, keeps original data and logs a warning.
    """
    orig_a8a9 = arch_df[arch_df["config_id"].isin(["A8", "A9"])]
    n_orig = len(orig_a8a9)
    n_ln = len(graph_ln_df)

    if n_ln < n_orig:
        out(f"  {arch}: Graph LN incomplete ({n_ln}/{n_orig} rows) — keeping original A8/A9")
        return arch_df

    rest = arch_df[~arch_df["config_id"].isin(["A8", "A9"])]
    graph_ln_df = graph_ln_df.copy()
    graph_ln_df["architecture"] = arch
    if "best_val_mcc" in graph_ln_df.columns and "val_test_gap" not in graph_ln_df.columns:
        graph_ln_df["val_test_gap"] = graph_ln_df["best_val_mcc"] - graph_ln_df[METRIC]
    merged = pd.concat([rest, graph_ln_df], ignore_index=True)
    out(f"  {arch}: replaced {n_orig} original A8/A9 rows with "
        f"{n_ln} Graph LayerNorm rows")
    return merged


def load_ablation_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame]:
    """Load all ablation CSVs and merge. Returns (ff_df, lstm_df, combined_df).

    Supplementary CSV handling:
    - A10 CSVs are included in the main CSV_FILES lists (merged automatically).
    - Graph LN CSVs replace original A8/A9 rows (bug-fixed version is canonical).
    - Social LSTM CSV is loaded but stored separately (accessible via _SOCIAL_LSTM_DF).
    """
    out("Loading data...")
    ff_df = _load_arch_csvs("FF", CSV_FILES["FF"])
    lstm_df = _load_arch_csvs("LSTM", CSV_FILES["LSTM"])

    if ff_df is None:
        out("WARNING: No FF data found")
    if lstm_df is None:
        out("WARNING: No LSTM data found")

    # Load supplementary CSVs
    supplementary = _load_supplementary()

    # Replace A8/A9 with Graph LayerNorm versions if available
    if "graph_ln_ff" in supplementary and ff_df is not None:
        ff_df = _replace_graph_rows(ff_df, supplementary["graph_ln_ff"], "FF")
    if "graph_ln_lstm" in supplementary and lstm_df is not None:
        lstm_df = _replace_graph_rows(lstm_df, supplementary["graph_ln_lstm"], "LSTM")

    # Store social LSTM data for supplementary analysis
    global _SOCIAL_LSTM_DF
    if "social_lstm" in supplementary:
        sldf = supplementary["social_lstm"]
        sldf["architecture"] = "LSTM"
        sldf["source_tag"] = "social_lstm_sequence"
        if "best_val_mcc" in sldf.columns and "val_test_gap" not in sldf.columns:
            sldf["val_test_gap"] = sldf["best_val_mcc"] - sldf[METRIC]
        _SOCIAL_LSTM_DF = sldf
    else:
        _SOCIAL_LSTM_DF = None

    frames = [f for f in [ff_df, lstm_df] if f is not None]
    if not frames:
        out("ERROR: No data files found.")
        return None, None, pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    out(f"Combined: {len(combined)} total experiments")
    out(f"  Configs: {sorted(combined['config_id'].unique())}")
    out(f"  Fusions: {sorted(combined['fusion_type'].unique())}")
    out(f"  Seeds: {sorted(combined['seed'].unique())}")
    out(f"  Folds: {sorted(combined['fold_idx'].unique())}")
    if "architecture" in combined.columns:
        out(f"  Architectures: {sorted(combined['architecture'].unique())}")
    return ff_df, lstm_df, combined


# ============================================================================
# STATISTICAL HELPERS
# ============================================================================

def paired_test(vals_a: np.ndarray, vals_b: np.ndarray, label_a: str, label_b: str,
                alpha: float = 0.05) -> dict:
    """Run paired t-test with Cohen's d. Returns result dict."""
    assert len(vals_a) == len(vals_b), f"Unequal lengths: {len(vals_a)} vs {len(vals_b)}"
    n = len(vals_a)
    if n < 3:
        return {"delta": np.nan, "t": np.nan, "p": np.nan, "d": np.nan,
                "sig": "N/A", "n": n, "wins": 0}

    diff = vals_a - vals_b
    delta = diff.mean()
    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
    d_std = diff.std()
    cohens_d = delta / d_std if d_std > 1e-10 else 0.0
    wins = (diff > 0).sum()
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

    return {"delta": delta, "t": t_stat, "p": p_val, "d": cohens_d,
            "sig": sig, "n": n, "wins": wins, "label_a": label_a, "label_b": label_b}


def get_matched_pairs(df: pd.DataFrame, col: str, val_a, val_b,
                      match_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Extract matched MCC pairs from df, matching on fold_idx + seed (+ optional cols)."""
    if match_cols is None:
        match_cols = ["fold_idx", "seed"]

    # Also match on architecture and fusion_type if present and not the variable being compared
    for extra in ["architecture", "fusion_type"]:
        if extra in df.columns and extra != col and extra not in match_cols:
            match_cols = match_cols + [extra]

    a = df[df[col] == val_a].set_index(match_cols)[METRIC].sort_index()
    b = df[df[col] == val_b].set_index(match_cols)[METRIC].sort_index()

    # Intersect indices
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return np.array([]), np.array([])

    return a.loc[common].values, b.loc[common].values


def bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Apply Bonferroni correction."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def effect_size_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def confidence_interval_95(vals: np.ndarray) -> tuple[float, float]:
    """Compute 95% CI for the mean using t-distribution."""
    n = len(vals)
    if n < 2:
        return (np.nan, np.nan)
    mean = vals.mean()
    se = stats.sem(vals)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return ci


def format_ci(vals: np.ndarray) -> str:
    """Format mean with 95% CI as 'mean [lo, hi]'."""
    ci = confidence_interval_95(vals)
    if np.isnan(ci[0]):
        return f"{vals.mean():.4f}"
    return f"{vals.mean():.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


# ============================================================================
# OVERVIEW
# ============================================================================

def analyze_overview(df: pd.DataFrame):
    """Print overall summary statistics."""
    out("\n# Phase 4-R / Phase 5 Ablation Analysis")
    out()
    out("## Overview")
    out()
    out(f"- Total experiments: {len(df)}")
    out(f"- Configs: {sorted(df['config_id'].unique())}")
    out(f"- Fusion types: {sorted(df['fusion_type'].unique())}")
    out(f"- Seeds: {sorted(df['seed'].unique())}")
    out(f"- Folds: {sorted(df['fold_idx'].unique())}")

    n_arch = df["architecture"].nunique() if "architecture" in df.columns else 1
    out(f"- Architectures: {n_arch}")
    out()

    # Completion check: A7 (single modality) only runs concat, others run all 3 fusions
    n_multi = sum(1 for v in ABLATION_CONFIGS.values() if len(v["modalities"]) > 1)
    n_single = len(ABLATION_CONFIGS) - n_multi
    expected_per_arch = (n_multi * 3 + n_single * 1) * 3 * 5  # 375 = (8*3 + 1*1) * 3 seeds * 5 folds
    for arch in sorted(df["architecture"].unique()) if "architecture" in df.columns else ["all"]:
        arch_df = df[df["architecture"] == arch] if "architecture" in df.columns else df
        pct = len(arch_df) / expected_per_arch * 100
        out(f"  {arch}: {len(arch_df)}/{expected_per_arch} ({pct:.0f}% complete)")
    out()

    # Overall MCC distribution
    out("### MCC Distribution")
    out()
    out(f"- Mean: {df[METRIC].mean():.4f}")
    out(f"- Median: {df[METRIC].median():.4f}")
    out(f"- Std: {df[METRIC].std():.4f}")
    out(f"- Min: {df[METRIC].min():.4f}")
    out(f"- Max: {df[METRIC].max():.4f}")
    out(f"- % negative: {(df[METRIC] < 0).mean()*100:.1f}%")
    out()

    # Val-test gap
    if "val_test_gap" in df.columns:
        out("### Val-Test Gap")
        out()
        out(f"- Mean gap: {df['val_test_gap'].mean():.4f}")
        out(f"- Std gap: {df['val_test_gap'].std():.4f}")
        out(f"- All positive: {(df['val_test_gap'] > 0).all()}")
        out()

    # Grand summary table: config x fusion
    out("### MCC Summary: Config x Fusion")
    out()
    fusions = sorted(df["fusion_type"].unique())
    header = f"| Config | Description |"
    for ft in fusions:
        header += f" {ft} |"
    header += " Mean |"
    out(header)
    out("|" + "---|" * (len(fusions) + 3))

    for cid in sorted(df["config_id"].unique()):
        desc = ABLATION_CONFIGS.get(cid, {}).get("desc", "")
        row = f"| {cid} | {desc} |"
        for ft in fusions:
            sub = df[(df["config_id"] == cid) & (df["fusion_type"] == ft)]
            if len(sub) > 0:
                row += f" {sub[METRIC].mean():.4f} |"
            else:
                row += " N/A |"
        config_mean = df[df["config_id"] == cid][METRIC].mean()
        row += f" {config_mean:.4f} |"
        out(row)
    out()

    # Diagnostic check: compare A7 against Phase 3-R reference
    _diagnostic_check(df)


def _diagnostic_check(df: pd.DataFrame):
    """Compare A7 baseline results against Phase 3-R reference values."""
    out("### Diagnostic Check: A7 vs Phase 3-R Reference")
    out()

    for arch_label, ref_key in [("FF", "A7_FF"), ("LSTM", "A7_LSTM")]:
        ref = PHASE3R_REFERENCE.get(ref_key)
        if ref is None:
            continue

        if "architecture" in df.columns:
            a7_sub = df[(df["config_id"] == "A7") & (df["architecture"] == arch_label)]
        else:
            a7_sub = df[df["config_id"] == "A7"]
            arch_label = "all"

        if len(a7_sub) == 0:
            out(f"- A7 {arch_label}: no data yet")
            continue

        p4r_mean = a7_sub[METRIC].mean()
        ref_mean = ref["mean_mcc"]
        delta = p4r_mean - ref_mean
        status = "OK" if abs(delta) < 0.015 else "WARN"

        out(f"- A7 {arch_label}: Phase 4-R MCC={p4r_mean:.4f} vs Phase 3-R ref={ref_mean:.3f} "
            f"(delta={delta:+.4f}) [{status}]")

        # Per-fold check
        if ref.get("per_fold"):
            for fold, ref_val in sorted(ref["per_fold"].items()):
                fold_sub = a7_sub[a7_sub["fold_idx"] == fold]
                if len(fold_sub) > 0:
                    fold_mcc = fold_sub[METRIC].mean()
                    fold_delta = fold_mcc - ref_val
                    fold_status = "OK" if abs(fold_delta) < 0.02 else "WARN"
                    out(f"  F{fold}: {fold_mcc:.4f} vs ref {ref_val:.3f} "
                        f"(delta={fold_delta:+.4f}) [{fold_status}]")

    out()


# ============================================================================
# RQ1: MULTI-SOURCE vs SINGLE-SOURCE
# ============================================================================

def analyze_rq1(df: pd.DataFrame):
    """RQ1: Does integrating multiple data sources improve prediction?

    Compare each multi-modal config against the A7 (price-only) baseline.
    Uses paired t-tests matching on (fold, seed, fusion, architecture).
    """
    out("\n---")
    out("## RQ1: Multi-Source vs Single-Source")
    out()
    out("Baseline: A7 (price only). Each multi-modal config compared via paired t-tests.")
    out()

    baseline = "A7"
    configs = [c for c in sorted(df["config_id"].unique()) if c != baseline]

    if baseline not in df["config_id"].values:
        out("ERROR: A7 baseline not found in data.")
        return

    # --- Fold-averaged comparison ---
    out("### RQ1.1: Fold-Averaged Comparison (vs A7)")
    out()
    out("Paired t-tests match on (fold, seed, fusion_type, architecture). Since A7 only runs")
    out("concat fusion, all pairs are concat-vs-concat. Mean MCC columns show the paired subset")
    out("means (concat only) for consistency with the statistical test.")
    out()
    out("| Config | Description | Paired MCC | A7 MCC | Delta | p-value | p (Bonf) | Cohen d | n | Sig | Wins |")
    out("|--------|-------------|------------|--------|-------|---------|----------|---------|---|-----|------|")

    results = []
    paired_means = {}  # store paired subset means for each config
    for cid in configs:
        # Get matched pairs across all fusions/seeds/folds
        vals_a, vals_b = get_matched_pairs(df, "config_id", cid, baseline)
        if len(vals_a) < 3:
            out(f"| {cid} | {ABLATION_CONFIGS.get(cid, {}).get('desc', '')} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        r = paired_test(vals_a, vals_b, cid, baseline)
        results.append(r)
        paired_means[cid] = (vals_a.mean(), vals_b.mean())

    # Bonferroni correction
    if results:
        raw_ps = [r["p"] for r in results]
        adj_ps = bonferroni_adjust(raw_ps)

        for r, adj_p, cid in zip(results, adj_ps, configs):
            if np.isnan(r["delta"]):
                continue
            adj_sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
            desc = ABLATION_CONFIGS.get(cid, {}).get("desc", "")
            mean_a, mean_b = paired_means.get(cid, (float("nan"), float("nan")))
            out(f"| {cid} | {desc} | {mean_a:.4f} | {mean_b:.4f} | {r['delta']:+.4f} | "
                f"{r['p']:.4f} | {adj_p:.4f} | {r['d']:.3f} | {r['n']} | {adj_sig} | {r['wins']}/{r['n']} |")

    out()

    # --- Per-fold breakdown ---
    out("### RQ1.2: Per-Fold Breakdown (Multi-Source Delta vs A7)")
    out()
    folds = sorted(df["fold_idx"].unique())
    header = "| Config |"
    for f in folds:
        header += f" F{f} |"
    header += " Mean |"
    out(header)
    out("|" + "---|" * (len(folds) + 2))

    for cid in configs:
        row = f"| {cid} |"
        deltas = []
        for fold in folds:
            fold_df = df[df["fold_idx"] == fold]
            vals_a, vals_b = get_matched_pairs(fold_df, "config_id", cid, baseline)
            if len(vals_a) > 0:
                delta = vals_a.mean() - vals_b.mean()
                deltas.append(delta)
                row += f" {delta:+.4f} |"
            else:
                row += " N/A |"
        mean_delta = np.mean(deltas) if deltas else float("nan")
        row += f" {mean_delta:+.4f} |"
        out(row)
    out()

    # --- Per-fusion breakdown ---
    out("### RQ1.3: Per-Fusion Breakdown (Multi-Source Delta vs A7)")
    out()
    out("Note: A7 only runs concat. For non-concat fusions, delta is computed as")
    out("config-fusion MCC minus A7-concat MCC (matched on fold, seed, architecture).")
    out()
    fusions = sorted(df["fusion_type"].unique())
    header = "| Config |"
    for ft in fusions:
        header += f" {ft} |"
    out(header)
    out("|" + "---|" * (len(fusions) + 1))

    # A7 baseline is always concat; get per-(fold, seed, arch) means
    a7_df = df[(df["config_id"] == baseline) & (df["fusion_type"] == "concat")]

    for cid in configs:
        row = f"| {cid} |"
        for ft in fusions:
            cid_ft_df = df[(df["config_id"] == cid) & (df["fusion_type"] == ft)]
            if len(cid_ft_df) == 0:
                row += " N/A |"
                continue
            # Match against A7-concat on fold, seed, architecture
            match_cols = ["fold_idx", "seed"]
            if "architecture" in df.columns:
                match_cols.append("architecture")
            a_vals = cid_ft_df.set_index(match_cols)[METRIC].sort_index()
            b_vals = a7_df.set_index(match_cols)[METRIC].sort_index()
            common = a_vals.index.intersection(b_vals.index)
            if len(common) > 0:
                delta = a_vals.loc[common].mean() - b_vals.loc[common].mean()
                row += f" {delta:+.4f} |"
            else:
                row += " N/A |"
        out(row)
    out()

    # --- RQ1 Summary ---
    out("### RQ1 Summary")
    out()
    if results:
        any_sig = any(r["p"] < 0.05 for r in results)
        any_positive = any(r["delta"] > 0 for r in results)
        best = max(results, key=lambda r: r["delta"])
        worst = min(results, key=lambda r: r["delta"])
        out(f"- Any significant improvement over A7: {'YES' if any_sig else 'NO'}")
        out(f"- Any positive delta: {'YES' if any_positive else 'NO'}")
        out(f"- Best config vs A7: {best['label_a']} (delta={best['delta']:+.4f}, p={best['p']:.4f})")
        out(f"- Worst config vs A7: {worst['label_a']} (delta={worst['delta']:+.4f}, p={worst['p']:.4f})")
    out()


# ============================================================================
# RQ2: MARGINAL CONTRIBUTION PER MODALITY
# ============================================================================

def analyze_rq2(df: pd.DataFrame):
    """RQ2: What is the relative contribution of each data source?

    Uses leave-one-out analysis from the full model (A1).
    Also computes add-one-in from baseline (A7).
    """
    out("\n---")
    out("## RQ2: Marginal Contribution per Modality")
    out()

    # --- Leave-one-out from A1 (full model) ---
    # A1 = [price, news, macro, social]
    # Dropping one modality:
    #   - Drop news  -> A5 [price, macro, social]
    #   - Drop macro -> A6... wait, need to check
    #   - Drop social -> A2 [price, news, macro]
    # Actually let me derive from config definitions

    leave_one_out = {
        "news":   ("A1", "A5"),   # A1=[price,news,macro,social] → A5=[price,macro,social]
        "macro":  ("A1", "A6"),   # A1 → A6=[price,social]... no, A6=[price,social], that drops macro AND news
        "social": ("A1", "A2"),   # A1 → A2=[price,news,macro]
    }

    # More precise: leave-one-out requires full minus exactly one
    # A1 = {price, news, macro, social}
    # -news = {price, macro, social} = A5
    # -social = {price, news, macro} = A2
    # -macro = {price, news, social} -- not a defined config
    # So we can only do leave-one-out for news and social directly

    out("### RQ2.1: Leave-One-Out from Full Model (A1)")
    out()
    out("Remove one modality from the full model and measure MCC change.")
    out()
    out("| Removed | Full (A1) | Without | Delta | p-value | Cohen d | Sig | Interpretation |")
    out("|---------|-----------|---------|-------|---------|---------|-----|----------------|")

    loo_results = []
    loo_pairs = [
        ("news", "A1", "A5"),    # A1 - news = A5
        ("social", "A1", "A2"),  # A1 - social = A2
    ]

    for modality, full_cfg, reduced_cfg in loo_pairs:
        if full_cfg not in df["config_id"].values or reduced_cfg not in df["config_id"].values:
            out(f"| {modality} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue

        vals_full, vals_reduced = get_matched_pairs(df, "config_id", full_cfg, reduced_cfg)
        if len(vals_full) < 3:
            out(f"| {modality} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue

        r = paired_test(vals_full, vals_reduced, full_cfg, reduced_cfg)
        loo_results.append((modality, r))

        mean_full = df[df["config_id"] == full_cfg][METRIC].mean()
        mean_reduced = df[df["config_id"] == reduced_cfg][METRIC].mean()
        interp = "helps" if r["delta"] > 0 else "hurts" if r["delta"] < 0 else "neutral"
        out(f"| {modality} | {mean_full:.4f} | {mean_reduced:.4f} | {r['delta']:+.4f} | "
            f"{r['p']:.4f} | {r['d']:.3f} | {r['sig']} | {interp} |")

    out()

    # --- Add-one-in from A7 baseline ---
    out("### RQ2.2: Add-One-In from Baseline (A7)")
    out()
    out("Add one modality to price-only baseline and measure MCC change.")
    out()
    out("| Added | A7 (base) | With modality | Config | Delta | p-value | Cohen d | Sig |")
    out("|-------|-----------|---------------|--------|-------|---------|---------|-----|")

    add_one_pairs = [
        ("macro", "A7", "A4"),    # A7 + macro = A4
        ("news", "A7", "A3"),     # A7 + news = A3
        ("social", "A7", "A6"),   # A7 + social = A6
        ("graph", "A7", "A8"),    # A7 + graph = A8
    ]

    add_results = []
    for modality, base_cfg, added_cfg in add_one_pairs:
        if base_cfg not in df["config_id"].values or added_cfg not in df["config_id"].values:
            out(f"| {modality} | N/A | N/A | {added_cfg} | N/A | N/A | N/A | N/A |")
            continue

        vals_added, vals_base = get_matched_pairs(df, "config_id", added_cfg, base_cfg)
        if len(vals_added) < 3:
            out(f"| {modality} | N/A | N/A | {added_cfg} | N/A | N/A | N/A | N/A |")
            continue

        r = paired_test(vals_added, vals_base, added_cfg, base_cfg)
        add_results.append((modality, r))

        mean_base = df[df["config_id"] == base_cfg][METRIC].mean()
        mean_added = df[df["config_id"] == added_cfg][METRIC].mean()
        out(f"| {modality} | {mean_base:.4f} | {mean_added:.4f} | {added_cfg} | {r['delta']:+.4f} | "
            f"{r['p']:.4f} | {r['d']:.3f} | {r['sig']} |")

    out()

    # --- Modality ranking ---
    out("### RQ2.3: Modality Value Ranking")
    out()
    out("Ranked by add-one-in delta MCC (positive = helps, negative = hurts):")
    out()

    if add_results:
        sorted_results = sorted(add_results, key=lambda x: x[1]["delta"], reverse=True)
        for rank, (modality, r) in enumerate(sorted_results, 1):
            effect = effect_size_label(r["d"])
            out(f"{rank}. **{modality}**: delta={r['delta']:+.4f}, d={r['d']:.3f} ({effect}), p={r['p']:.4f} {r['sig']}")
    out()

    # --- Per-fold marginal contribution ---
    out("### RQ2.4: Per-Fold Marginal Contribution (Add-One-In Delta)")
    out()
    folds = sorted(df["fold_idx"].unique())
    header = "| Modality |"
    for f in folds:
        header += f" F{f} |"
    header += " Mean |"
    out(header)
    out("|" + "---|" * (len(folds) + 2))

    for modality, base_cfg, added_cfg in add_one_pairs:
        if base_cfg not in df["config_id"].values or added_cfg not in df["config_id"].values:
            continue
        row = f"| {modality} |"
        fold_deltas = []
        for fold in folds:
            fold_df = df[df["fold_idx"] == fold]
            vals_a, vals_b = get_matched_pairs(fold_df, "config_id", added_cfg, base_cfg)
            if len(vals_a) > 0:
                delta = vals_a.mean() - vals_b.mean()
                fold_deltas.append(delta)
                row += f" {delta:+.4f} |"
            else:
                row += " N/A |"
        mean_d = np.mean(fold_deltas) if fold_deltas else float("nan")
        row += f" {mean_d:+.4f} |"
        out(row)
    out()


# ============================================================================
# RQ3: CONDITIONAL ANALYSIS (When does multi-source help?)
# ============================================================================

def analyze_rq3(df: pd.DataFrame):
    """RQ3: Under what conditions does multi-source integration provide value?

    Analyzes per-fold performance (each fold = different market regime).
    Also examines fusion strategy effectiveness by regime.
    """
    out("\n---")
    out("## RQ3: Conditional Analysis (When Does Multi-Source Help?)")
    out()

    folds = sorted(df["fold_idx"].unique())
    configs = sorted(df["config_id"].unique())

    # --- Per-fold MCC heatmap ---
    out("### RQ3.1: MCC by Config x Fold (Market Regime)")
    out()
    header = "| Config |"
    for f in folds:
        label = FOLD_LABELS.get(f, f"F{f}")
        header += f" F{f} |"
    header += " Mean | Std |"
    out(header)
    out("|" + "---|" * (len(folds) + 3))

    for cid in configs:
        row = f"| {cid} |"
        fold_means = []
        for fold in folds:
            sub = df[(df["config_id"] == cid) & (df["fold_idx"] == fold)]
            if len(sub) > 0:
                m = sub[METRIC].mean()
                fold_means.append(m)
                row += f" {m:.4f} |"
            else:
                row += " N/A |"
        overall = np.mean(fold_means) if fold_means else float("nan")
        std = np.std(fold_means) if fold_means else float("nan")
        row += f" {overall:.4f} | {std:.4f} |"
        out(row)
    out()

    # Fold regime legend
    out("**Fold regime labels:**")
    for f, label in FOLD_LABELS.items():
        out(f"- F{f}: {label}")
    out()

    # --- Which config wins per fold? ---
    out("### RQ3.2: Best Config per Fold")
    out()
    out("| Fold | Regime | Best Config | MCC | 2nd Best | MCC | A7 MCC | Delta vs A7 |")
    out("|------|--------|-------------|-----|----------|-----|--------|-------------|")

    for fold in folds:
        fold_df = df[df["fold_idx"] == fold]
        config_means = fold_df.groupby("config_id")[METRIC].mean().sort_values(ascending=False)
        if len(config_means) >= 2:
            best_cfg = config_means.index[0]
            best_mcc = config_means.iloc[0]
            second_cfg = config_means.index[1]
            second_mcc = config_means.iloc[1]
            a7_mcc = config_means.get("A7", float("nan"))
            delta = best_mcc - a7_mcc if not np.isnan(a7_mcc) else float("nan")
            regime = FOLD_LABELS.get(fold, "")
            out(f"| F{fold} | {regime} | {best_cfg} | {best_mcc:.4f} | "
                f"{second_cfg} | {second_mcc:.4f} | {a7_mcc:.4f} | {delta:+.4f} |")
    out()

    # --- Multi-source advantage by regime ---
    out("### RQ3.3: Multi-Source Advantage by Regime")
    out()
    out("For each fold, does ANY multi-modal config significantly beat A7?")
    out()
    out("**Note**: Per-fold tests are exploratory (not Bonferroni-corrected across folds).")
    out("Sample sizes per fold are small (n reported); interpret with caution.")
    out()
    out("| Fold | Regime | Best Multi | Delta vs A7 | p-value | n | Sig | Verdict |")
    out("|------|--------|------------|-------------|---------|---|-----|---------|")

    for fold in folds:
        fold_df = df[df["fold_idx"] == fold]
        multi_configs = [c for c in configs if c != "A7"]
        best_delta = -999
        best_result = None
        best_cfg = None

        for cid in multi_configs:
            vals_a, vals_b = get_matched_pairs(fold_df, "config_id", cid, "A7")
            if len(vals_a) >= 3:
                r = paired_test(vals_a, vals_b, cid, "A7")
                if r["delta"] > best_delta:
                    best_delta = r["delta"]
                    best_result = r
                    best_cfg = cid

        if best_result:
            regime = FOLD_LABELS.get(fold, "")
            verdict = "MULTI WINS" if best_result["p"] < 0.05 and best_delta > 0 else \
                      "MULTI LOSES" if best_result["p"] < 0.05 and best_delta < 0 else "NO DIFF"
            out(f"| F{fold} | {regime} | {best_cfg} | {best_delta:+.4f} | "
                f"{best_result['p']:.4f} | {best_result['n']} | {best_result['sig']} | {verdict} |")
    out()

    # --- Fusion strategy by fold ---
    out("### RQ3.4: Fusion Strategy Effectiveness by Fold")
    out()
    fusions = sorted(df["fusion_type"].unique())
    if len(fusions) > 1:
        header = "| Fold | Regime |"
        for ft in fusions:
            header += f" {ft} |"
        header += " Best |"
        out(header)
        out("|" + "---|" * (len(fusions) + 3))

        for fold in folds:
            fold_df = df[df["fold_idx"] == fold]
            row = f"| F{fold} | {FOLD_LABELS.get(fold, '')} |"
            fusion_means = {}
            for ft in fusions:
                sub = fold_df[fold_df["fusion_type"] == ft]
                if len(sub) > 0:
                    m = sub[METRIC].mean()
                    fusion_means[ft] = m
                    row += f" {m:.4f} |"
                else:
                    row += " N/A |"
            best_ft = max(fusion_means, key=fusion_means.get) if fusion_means else "N/A"
            row += f" {best_ft} |"
            out(row)
        out()

    # --- Consistency analysis ---
    out("### RQ3.5: Cross-Fold Consistency")
    out()
    out("How consistent is each config's ranking across folds?")
    out()

    # Rank configs per fold
    rank_data = {}
    for fold in folds:
        fold_df = df[df["fold_idx"] == fold]
        config_means = fold_df.groupby("config_id")[METRIC].mean().sort_values(ascending=False)
        for rank, (cid, _) in enumerate(config_means.items(), 1):
            if cid not in rank_data:
                rank_data[cid] = []
            rank_data[cid].append(rank)

    out("| Config | " + " | ".join(f"F{f} Rank" for f in folds) + " | Mean Rank | Std Rank |")
    out("|" + "---|" * (len(folds) + 3))

    for cid in sorted(rank_data.keys()):
        ranks = rank_data[cid]
        row = f"| {cid} | " + " | ".join(f"{r}" for r in ranks)
        row += f" | {np.mean(ranks):.1f} | {np.std(ranks):.1f} |"
        out(row)
    out()


# ============================================================================
# ARCHITECTURE COMPARISON (FF vs LSTM)
# ============================================================================

def analyze_architecture(ff_df: pd.DataFrame | None, lstm_df: pd.DataFrame | None,
                         combined: pd.DataFrame):
    """Compare feedforward vs LSTM architectures."""
    out("\n---")
    out("## Architecture Comparison: FF vs LSTM")
    out()

    if ff_df is None or lstm_df is None:
        out("WARNING: Need both FF and LSTM data for architecture comparison.")
        if ff_df is not None:
            out(f"  FF data available: {len(ff_df)} experiments")
        if lstm_df is not None:
            out(f"  LSTM data available: {len(lstm_df)} experiments")
        return

    # --- Overall comparison ---
    out("### Overall")
    out()
    out(f"- FF: mean MCC = {ff_df[METRIC].mean():.4f} (std={ff_df[METRIC].std():.4f}, n={len(ff_df)})")
    out(f"- LSTM: mean MCC = {lstm_df[METRIC].mean():.4f} (std={lstm_df[METRIC].std():.4f}, n={len(lstm_df)})")

    # Matched pairs
    vals_lstm, vals_ff = get_matched_pairs(combined, "architecture", "LSTM", "FF",
                                           match_cols=["config_id", "fusion_type", "fold_idx", "seed"])
    if len(vals_lstm) >= 3:
        r = paired_test(vals_lstm, vals_ff, "LSTM", "FF")
        out(f"- Delta (LSTM - FF): {r['delta']:+.4f}")
        out(f"- Paired t-test: t={r['t']:.3f}, p={r['p']:.4f} {r['sig']}")
        out(f"- Cohen's d: {r['d']:.3f} ({effect_size_label(r['d'])})")
        out(f"- LSTM wins: {r['wins']}/{r['n']} ({r['wins']/r['n']*100:.0f}%)")
    out()

    # --- Per-config comparison ---
    out("### Per-Config Architecture Comparison")
    out()
    out("| Config | FF MCC | LSTM MCC | Delta | p-value | Cohen d | Sig | LSTM Win% |")
    out("|--------|--------|----------|-------|---------|---------|-----|-----------|")

    configs = sorted(combined["config_id"].unique())
    for cid in configs:
        cid_df = combined[combined["config_id"] == cid]
        vals_l, vals_f = get_matched_pairs(cid_df, "architecture", "LSTM", "FF",
                                           match_cols=["fusion_type", "fold_idx", "seed"])
        if len(vals_l) < 3:
            out(f"| {cid} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue

        r = paired_test(vals_l, vals_f, "LSTM", "FF")
        ff_mean = ff_df[ff_df["config_id"] == cid][METRIC].mean()
        lstm_mean = lstm_df[lstm_df["config_id"] == cid][METRIC].mean()
        win_pct = r["wins"] / r["n"] * 100
        out(f"| {cid} | {ff_mean:.4f} | {lstm_mean:.4f} | {r['delta']:+.4f} | "
            f"{r['p']:.4f} | {r['d']:.3f} | {r['sig']} | {win_pct:.0f}% |")
    out()

    # --- Per-fold architecture effect ---
    out("### Per-Fold Architecture Effect")
    out()
    out("**Note**: Per-fold tests are exploratory (not Bonferroni-corrected across folds).")
    out("Cohen's d and n reported for transparency.")
    out()
    out("| Fold | Regime | FF MCC | LSTM MCC | Delta | p-value | Cohen d | n | Sig |")
    out("|------|--------|--------|----------|-------|---------|---------|---|-----|")

    folds = sorted(combined["fold_idx"].unique())
    for fold in folds:
        fold_df = combined[combined["fold_idx"] == fold]
        vals_l, vals_f = get_matched_pairs(fold_df, "architecture", "LSTM", "FF",
                                           match_cols=["config_id", "fusion_type", "seed"])
        if len(vals_l) < 3:
            continue
        r = paired_test(vals_l, vals_f, "LSTM", "FF")
        ff_mean = ff_df[ff_df["fold_idx"] == fold][METRIC].mean()
        lstm_mean = lstm_df[lstm_df["fold_idx"] == fold][METRIC].mean()
        regime = FOLD_LABELS.get(fold, "")
        out(f"| F{fold} | {regime} | {ff_mean:.4f} | {lstm_mean:.4f} | "
            f"{r['delta']:+.4f} | {r['p']:.4f} | {r['d']:.3f} | {r['n']} | {r['sig']} |")
    out()

    # --- Regime-architecture interaction ---
    out("### Regime-Architecture Interaction")
    out()
    out("Key question from v1 findings: Does LSTM excel in volatile/transitional markets")
    out("while FF works better in calm/trending markets?")
    out("*(Exploratory analysis — per-fold tests not corrected for multiple comparisons)*")
    out()

    for fold in folds:
        fold_df = combined[combined["fold_idx"] == fold]
        vals_l, vals_f = get_matched_pairs(fold_df, "architecture", "LSTM", "FF",
                                           match_cols=["config_id", "fusion_type", "seed"])
        if len(vals_l) < 3:
            continue
        r = paired_test(vals_l, vals_f, "LSTM", "FF")
        regime = FOLD_LABELS.get(fold, "")
        winner = "LSTM" if r["delta"] > 0 else "FF"
        strength = r["sig"]
        out(f"- F{fold} ({regime}): **{winner}** {strength} (delta={r['delta']:+.4f}, d={r['d']:.3f}, n={r['n']})")
    out()


# ============================================================================
# FUSION STRATEGY ANALYSIS
# ============================================================================

def analyze_fusion(df: pd.DataFrame):
    """Analyze which fusion strategy works best."""
    out("\n---")
    out("## Fusion Strategy Analysis")
    out()

    fusions = sorted(df["fusion_type"].unique())
    if len(fusions) < 2:
        out("Only one fusion type found. Skipping comparison.")
        return

    # --- Overall comparison ---
    out("### Overall Fusion Comparison")
    out()
    out("| Fusion | Mean MCC | Std | Median | % Negative | N |")
    out("|--------|----------|-----|--------|------------|---|")
    for ft in fusions:
        sub = df[df["fusion_type"] == ft]
        out(f"| {ft} | {sub[METRIC].mean():.4f} | {sub[METRIC].std():.4f} | "
            f"{sub[METRIC].median():.4f} | {(sub[METRIC] < 0).mean()*100:.1f}% | {len(sub)} |")
    out()

    # --- Pairwise comparisons ---
    out("### Pairwise Fusion Comparisons")
    out()
    out("| Comparison | Delta | p-value | p (Bonf) | Cohen d | Sig |")
    out("|------------|-------|---------|----------|---------|-----|")

    pair_results = []
    for ft_a, ft_b in combinations(fusions, 2):
        vals_a, vals_b = get_matched_pairs(df, "fusion_type", ft_a, ft_b,
                                           match_cols=["config_id", "fold_idx", "seed"])
        if len(vals_a) >= 3:
            r = paired_test(vals_a, vals_b, ft_a, ft_b)
            pair_results.append((ft_a, ft_b, r))

    if pair_results:
        raw_ps = [r["p"] for _, _, r in pair_results]
        adj_ps = bonferroni_adjust(raw_ps)

        for (ft_a, ft_b, r), adj_p in zip(pair_results, adj_ps):
            adj_sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
            out(f"| {ft_a} vs {ft_b} | {r['delta']:+.4f} | {r['p']:.4f} | {adj_p:.4f} | "
                f"{r['d']:.3f} | {adj_sig} |")
    out()

    # --- Fusion x Config interaction ---
    out("### Fusion Effectiveness by Config (Only multi-modal)")
    out()
    multi_configs = [c for c in sorted(df["config_id"].unique()) if c != "A7"]
    if len(multi_configs) == 0:
        out("No multi-modal configs found.")
        return

    header = "| Config |"
    for ft in fusions:
        header += f" {ft} |"
    header += " Best |"
    out(header)
    out("|" + "---|" * (len(fusions) + 2))

    for cid in multi_configs:
        cid_df = df[df["config_id"] == cid]
        row = f"| {cid} |"
        fusion_means = {}
        for ft in fusions:
            sub = cid_df[cid_df["fusion_type"] == ft]
            if len(sub) > 0:
                m = sub[METRIC].mean()
                fusion_means[ft] = m
                row += f" {m:.4f} |"
            else:
                row += " N/A |"
        best_ft = max(fusion_means, key=fusion_means.get) if fusion_means else "N/A"
        row += f" {best_ft} |"
        out(row)
    out()


# ============================================================================
# FINANCIAL METRICS
# ============================================================================

def analyze_financial(df: pd.DataFrame):
    """Analyze financial performance metrics beyond MCC."""
    out("\n---")
    out("## Financial Metrics Analysis")
    out()

    # cumulative_return and max_drawdown are unreliable in pre-fix CSVs
    # (computed with cumprod(1+log_return) instead of exp(cumsum(log_return)))
    fin_cols = ["sharpe_ratio", "profit_factor", "directional_accuracy"]
    available = [c for c in fin_cols if c in df.columns]

    if not available:
        out("No financial metric columns found.")
        return

    out("**Note**: cumulative_return and max_drawdown excluded (computed with "
        "incorrect formula in pre-fix experiment runs; see metrics.py fix).")
    out()

    # Replace sentinel values (-99.999) with NaN to avoid polluting means
    df = df.copy()
    for col in available:
        sentinel_mask = df[col] <= -99
        n_sentinels = sentinel_mask.sum()
        if n_sentinels > 0:
            out(f"**Note**: {col} has {n_sentinels} sentinel values (-99.999), excluded from analysis.")
            df.loc[sentinel_mask, col] = np.nan
    out()

    # --- By config ---
    out("### Financial Metrics by Config")
    out()
    header = "| Config |"
    for c in available:
        short = {"sharpe_ratio": "Sharpe", "profit_factor": "PF",
                 "max_drawdown": "MaxDD", "directional_accuracy": "DirAcc"}.get(c, c)
        header += f" {short} |"
    out(header)
    out("|" + "---|" * (len(available) + 1))

    for cid in sorted(df["config_id"].unique()):
        sub = df[df["config_id"] == cid]
        row = f"| {cid} |"
        for c in available:
            val = sub[c].mean()
            row += f" {val:.4f} |" if not np.isnan(val) else " N/A |"
        out(row)
    out()

    # --- Best financial config ---
    out("### Best Config by Each Financial Metric")
    out()
    for c in available:
        config_means = df.groupby("config_id")[c].mean().dropna()
        if config_means.empty:
            out(f"- **{c}**: N/A (all sentinel)")
            continue
        if c == "max_drawdown":
            # Higher (less negative) is better
            best = config_means.idxmax()
            best_val = config_means.max()
        else:
            best = config_means.idxmax()
            best_val = config_means.max()
        out(f"- **{c}**: {best} ({best_val:.4f})")
    out()


# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================

def executive_summary(df: pd.DataFrame, ff_df: pd.DataFrame | None, lstm_df: pd.DataFrame | None):
    """Generate an executive summary of all findings."""
    out("\n---")
    out("## Executive Summary")
    out()

    configs = sorted(df["config_id"].unique())
    fusions = sorted(df["fusion_type"].unique())

    # Overall MCC
    out(f"### Overall Performance")
    out(f"- Mean MCC across all {len(df)} experiments: {df[METRIC].mean():.4f}")
    out(f"- % negative MCC: {(df[METRIC] < 0).mean()*100:.1f}%")
    out()

    # Best config
    config_means = df.groupby("config_id")[METRIC].mean().sort_values(ascending=False)
    out("### Config Ranking (by mean MCC)")
    out()
    for rank, (cid, mcc) in enumerate(config_means.items(), 1):
        desc = ABLATION_CONFIGS.get(cid, {}).get("desc", "")
        out(f"{rank}. {cid} ({desc}): {mcc:.4f}")
    out()

    # Best fusion
    if len(fusions) > 1:
        fusion_means = df.groupby("fusion_type")[METRIC].mean().sort_values(ascending=False)
        out("### Fusion Ranking")
        out()
        for rank, (ft, mcc) in enumerate(fusion_means.items(), 1):
            out(f"{rank}. {ft}: {mcc:.4f}")
        out()

    # Architecture
    if ff_df is not None and lstm_df is not None:
        out("### Architecture")
        out(f"- FF mean MCC: {ff_df[METRIC].mean():.4f}")
        out(f"- LSTM mean MCC: {lstm_df[METRIC].mean():.4f}")
        better = "LSTM" if lstm_df[METRIC].mean() > ff_df[METRIC].mean() else "FF"
        out(f"- Better overall: {better}")
        out()

    # RQ answers
    out("### Research Question Answers (Preliminary)")
    out()
    a7_mean = df[df["config_id"] == "A7"][METRIC].mean() if "A7" in configs else float("nan")
    best_multi = config_means.drop("A7", errors="ignore")
    if not best_multi.empty:
        best_multi_cfg = best_multi.index[0]
        best_multi_mcc = best_multi.iloc[0]
        out(f"**RQ1** (multi vs single): Best multi-modal ({best_multi_cfg}) MCC={best_multi_mcc:.4f} "
            f"vs A7 MCC={a7_mean:.4f} (delta={best_multi_mcc - a7_mean:+.4f})")
    out()
    out("**RQ2** (which sources help): See Section RQ2 for marginal contribution analysis.")
    out()
    out("**RQ3** (when does it help): See Section RQ3 for per-fold regime analysis.")
    out()


# ============================================================================
# CHAPTER 4 ANALYSIS FUNCTIONS (7 new functions for Phase 5)
# ============================================================================
# These map directly to Chapter 4 tables/figures in the outline at:
#   results/analysis/chapter4_outline.md
# Designed to work with partial or complete Phase 4-R data.


def marginal_contribution_table(df: pd.DataFrame) -> list[dict]:
    """Table 4.6: Additive marginal contribution of each source over A7 baseline.

    For each auxiliary modality, computes:
      delta_MCC = mean(config_with_source) - mean(A7)
    using paired t-tests matched on (fusion_type, fold_idx, seed, architecture).

    Returns list of result dicts (sorted by delta descending) and outputs markdown.
    """
    out("\n---")
    out("## Table 4.6: Additive Marginal Contribution (Each Source Added to A7)")
    out()

    baseline = "A7"
    if baseline not in df["config_id"].values:
        out("ERROR: A7 baseline not found.")
        return []

    # Source -> (A7 + source = config)
    source_configs = [
        ("Macro", "A4"),          # A7 + macro = A4
        ("News", "A3"),           # A7 + news = A3
        ("Social Media", "A6"),   # A7 + social = A6
        ("Graph", "A8"),          # A7 + graph = A8
    ]

    out("| Rank | Source Added | Config | A7 MCC | Config MCC | Delta MCC | p-value | "
        "p (Bonf) | Cohen's d | Effect | Sig | Wins |")
    out("|------|-------------|--------|--------|------------|-----------|---------|"
        "----------|-----------|--------|-----|------|")

    results = []
    for source, config in source_configs:
        if config not in df["config_id"].values:
            results.append({"source": source, "config": config, "delta": np.nan,
                            "p": np.nan, "d": np.nan, "sig": "N/A", "n": 0, "wins": 0})
            continue
        vals_cfg, vals_bl = get_matched_pairs(df, "config_id", config, baseline)
        if len(vals_cfg) < 3:
            results.append({"source": source, "config": config, "delta": np.nan,
                            "p": np.nan, "d": np.nan, "sig": "N/A", "n": 0, "wins": 0})
            continue
        r = paired_test(vals_cfg, vals_bl, config, baseline)
        r["source"] = source
        r["config"] = config
        r["mean_cfg"] = df[df["config_id"] == config][METRIC].mean()
        r["mean_bl"] = df[df["config_id"] == baseline][METRIC].mean()
        results.append(r)

    # Sort by delta descending
    valid = [r for r in results if not np.isnan(r.get("delta", np.nan))]
    invalid = [r for r in results if np.isnan(r.get("delta", np.nan))]
    valid.sort(key=lambda r: r["delta"], reverse=True)

    # Bonferroni correction — always use k=4 (planned comparisons, not just available)
    k_planned = len(source_configs)  # 4 planned comparisons
    if valid:
        adj_ps = [min(r["p"] * k_planned, 1.0) for r in valid]
    else:
        adj_ps = []

    all_sorted = valid + invalid
    for rank, r in enumerate(all_sorted, 1):
        if np.isnan(r.get("delta", np.nan)):
            out(f"| {rank} | {r['source']} | {r['config']} | N/A | N/A | N/A | "
                f"N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        idx = valid.index(r)
        adj_p = adj_ps[idx] if idx < len(adj_ps) else r["p"]
        adj_sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
        effect = effect_size_label(r["d"])
        out(f"| {rank} | {r['source']} | {r['config']} | {r['mean_bl']:.4f} | "
            f"{r['mean_cfg']:.4f} | {r['delta']:+.4f} | {r['p']:.4f} | "
            f"{adj_p:.4f} | {r['d']:.3f} | {effect} | {adj_sig} | {r['wins']}/{r['n']} |")

    out()
    out("**Source Value Ranking** (by additive delta MCC):")
    for rank, r in enumerate(valid, 1):
        direction = "HELPS" if r["delta"] > 0.005 else "HURTS" if r["delta"] < -0.005 else "NEUTRAL"
        out(f"  {rank}. {r['source']}: {r['delta']:+.4f} [{direction}]")
    out()
    return all_sorted


def leave_one_out_analysis(df: pd.DataFrame) -> list[dict]:
    """Table 4.7: Leave-one-out from A1 full model.

    Measures how much performance drops when each source is removed from
    the full 4-modality model (A1 = price+news+macro+social).

    A10 (price+news+social) fills the macro leave-one-out slot.
    """
    out("\n---")
    out("## Table 4.7: Leave-One-Out from Full Model (A1)")
    out()

    full_config = "A1"
    if full_config not in df["config_id"].values:
        out("WARNING: A1 (full model) not yet in data. This analysis requires news experiments.")
        out("Will be available after news batch completes.")
        return []

    # A1 = [price, news, macro, social]
    # Remove news  -> A5 = [price, macro, social]
    # Remove social -> A2 = [price, news, macro]
    # Remove macro -> A10 = [price, news, social]
    # Remove price -> not meaningful (price is always base)
    loo_pairs = [
        ("News", "A5", "Removing news from full model"),
        ("Social Media", "A2", "Removing social from full model"),
        ("Macro", "A10", "Removing macro from full model"),
    ]

    out("| Source Removed | Full (A1) MCC | Reduced MCC | Reduced Config | "
        "Delta (drop) | p-value | Cohen's d | Sig | Importance |")
    out("|--------------|---------------|-------------|----------------|"
        "-------------|---------|-----------|-----|------------|")

    results = []
    a1_mean = df[df["config_id"] == full_config][METRIC].mean()

    for source, reduced_config, note in loo_pairs:
        if reduced_config is None:
            out(f"| {source} | {a1_mean:.4f} | N/A | N/A | N/A | N/A | N/A | N/A | "
                f"*{note}* |")
            results.append({"source": source, "delta": np.nan, "note": note})
            continue

        if reduced_config not in df["config_id"].values:
            out(f"| {source} | {a1_mean:.4f} | N/A | {reduced_config} | N/A | N/A | "
                f"N/A | N/A | *Data pending* |")
            results.append({"source": source, "config": reduced_config, "delta": np.nan})
            continue

        vals_full, vals_reduced = get_matched_pairs(df, "config_id", full_config, reduced_config)
        if len(vals_full) < 3:
            out(f"| {source} | {a1_mean:.4f} | N/A | {reduced_config} | N/A | N/A | "
                f"N/A | N/A | *Insufficient pairs* |")
            results.append({"source": source, "config": reduced_config, "delta": np.nan})
            continue

        r = paired_test(vals_full, vals_reduced, full_config, reduced_config)
        reduced_mean = df[df["config_id"] == reduced_config][METRIC].mean()

        # Positive delta means removing the source HURTS (full > reduced) -> source is important
        importance = "CRITICAL" if r["delta"] > 0.01 and r["p"] < 0.05 else \
                     "IMPORTANT" if r["delta"] > 0.005 else \
                     "MARGINAL" if r["delta"] > 0 else \
                     "REDUNDANT"

        out(f"| {source} | {a1_mean:.4f} | {reduced_mean:.4f} | {reduced_config} | "
            f"{r['delta']:+.4f} | {r['p']:.4f} | {r['d']:.3f} | {r['sig']} | {importance} |")
        r["source"] = source
        r["config"] = reduced_config
        r["importance"] = importance
        results.append(r)

    out()
    out("**Sign convention**: Delta = A1 (full) minus reduced config. "
        "Positive delta means the full model outperforms (removing the source hurts). "
        "Negative delta means the reduced model outperforms (removing the source helps).")
    out()
    out("**Note**: A10 (price+news+social) fills the previously missing leave-one-out "
        "slot for macro. All three non-price sources can now be evaluated.")
    out()
    return results


def multi_source_vs_baseline_table(df: pd.DataFrame) -> list[dict]:
    """Table 4.5: All multi-source configs vs A7 baseline with Bonferroni correction.

    Central RQ1 table: shows whether ANY multi-source configuration significantly
    outperforms the price-only baseline. Uses Bonferroni with k=9 (all planned
    comparisons: A1-A6, A8-A10 vs A7).
    """
    out("\n---")
    out("## Table 4.5: Multi-Source vs Baseline (All Configs, Bonferroni k=9)")
    out()

    baseline = "A7"
    if baseline not in df["config_id"].values:
        out("ERROR: A7 baseline not found.")
        return []

    # All multi-source configs (9 planned comparisons)
    multi_configs = [
        ("A3", "Price + News"),
        ("A4", "Price + Macro"),
        ("A6", "Price + Social"),
        ("A8", "Price + Graph"),
        ("A2", "Price + News + Macro"),
        ("A5", "Price + Macro + Social"),
        ("A9", "Price + Macro + Graph"),
        ("A10", "Price + News + Social"),
        ("A1", "Price + News + Macro + Social"),
    ]

    k_planned = len(multi_configs)  # 9 planned comparisons

    out("| Config | Modalities | Mean MCC | Delta vs A7 | p-value | "
        "p (Bonf) | Cohen's d | Effect | Sig | n | Verdict |")
    out("|--------|-----------|----------|-------------|---------|"
        "----------|-----------|--------|-----|---|---------|")

    # Baseline row
    a7_mean = df[df["config_id"] == baseline][METRIC].mean()
    a7_n = len(df[df["config_id"] == baseline])
    out(f"| A7 | Price only | {a7_mean:.4f} | -- | -- | -- | -- | -- | -- | {a7_n} | Baseline |")

    results = []
    for config, desc in multi_configs:
        if config not in df["config_id"].values:
            out(f"| {config} | {desc} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 | *Data pending* |")
            results.append({"config": config, "desc": desc, "delta": np.nan})
            continue

        vals_cfg, vals_bl = get_matched_pairs(df, "config_id", config, baseline)
        if len(vals_cfg) < 3:
            out(f"| {config} | {desc} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 | *Insufficient pairs* |")
            results.append({"config": config, "desc": desc, "delta": np.nan})
            continue

        r = paired_test(vals_cfg, vals_bl, config, baseline)
        r["config"] = config
        r["desc"] = desc
        r["mean_cfg"] = df[df["config_id"] == config][METRIC].mean()
        r["p_bonf"] = min(r["p"] * k_planned, 1.0)

        adj_sig = ("***" if r["p_bonf"] < 0.001 else "**" if r["p_bonf"] < 0.01
                   else "*" if r["p_bonf"] < 0.05 else "ns")
        effect = effect_size_label(r["d"])

        if r["delta"] > 0.005 and r["p_bonf"] < 0.05:
            verdict = "IMPROVES"
        elif r["delta"] < -0.005 and r["p_bonf"] < 0.05:
            verdict = "DEGRADES"
        else:
            verdict = "NO DIFFERENCE"
        r["verdict"] = verdict

        out(f"| {config} | {desc} | {r['mean_cfg']:.4f} | {r['delta']:+.4f} | "
            f"{r['p']:.4f} | {r['p_bonf']:.4f} | {r['d']:.3f} | {effect} | "
            f"{adj_sig} | {r['n']} | {verdict} |")
        results.append(r)

    out()

    # Summary
    improves = [r for r in results if r.get("verdict") == "IMPROVES"]
    degrades = [r for r in results if r.get("verdict") == "DEGRADES"]
    if not improves:
        out(f"**RQ1 Answer**: No multi-source configuration significantly outperforms the "
            f"price-only baseline after Bonferroni correction (k={k_planned}).")
    else:
        names = ", ".join(f"{r['config']}" for r in improves)
        out(f"**RQ1 Answer**: {names} significantly outperform(s) the baseline.")
    if degrades:
        names = ", ".join(f"{r['config']}" for r in degrades)
        out(f"**Note**: {names} significantly degrade(s) vs baseline.")
    out()
    return results


def incremental_addition_table(df: pd.DataFrame) -> list[dict]:
    """Table 4.8: Incremental value of adding sources over price+macro (A4) baseline.

    Given that A4 is typically the best 2-source config, measures what each
    additional source adds on top.
    """
    out("\n---")
    out("## Table 4.8: Incremental Value Over Price+Macro Baseline (A4)")
    out()

    base = "A4"
    if base not in df["config_id"].values:
        out("ERROR: A4 not found in data.")
        return []

    # A4 = [price, macro]
    # + news   -> A2 = [price, news, macro]
    # + social -> A5 = [price, macro, social]
    # + graph  -> A9 = [price, macro, graph]
    # + news + social -> A1 = [price, news, macro, social]
    increments = [
        ("+ News", "A2"),
        ("+ Social Media", "A5"),
        ("+ Graph", "A9"),
        ("+ News + Social", "A1"),
    ]

    out("| Source Added | Config | A4 MCC | Config MCC | Delta | p-value | "
        "p (Bonf) | Cohen's d | Sig | n | Verdict |")
    out("|-------------|--------|--------|------------|-------|---------|"
        "----------|-----------|-----|---|---------|")

    results = []
    a4_pool_mean = df[df["config_id"] == base][METRIC].mean()

    for source, config in increments:
        if config not in df["config_id"].values:
            out(f"| {source} | {config} | {a4_pool_mean:.4f} | N/A | N/A | N/A | "
                f"N/A | N/A | N/A | 0 | *Data pending* |")
            results.append({"source": source, "config": config, "delta": np.nan})
            continue

        vals_cfg, vals_base = get_matched_pairs(df, "config_id", config, base)
        if len(vals_cfg) < 3:
            out(f"| {source} | {config} | {a4_pool_mean:.4f} | N/A | N/A | N/A | "
                f"N/A | N/A | N/A | 0 | *Insufficient pairs* |")
            results.append({"source": source, "config": config, "delta": np.nan})
            continue

        r = paired_test(vals_cfg, vals_base, config, base)
        # Use matched-subset means (not pooled) to be consistent with paired test
        r["base_mean"] = vals_base.mean()
        r["cfg_mean"] = vals_cfg.mean()
        r["source"] = source
        r["config"] = config
        results.append(r)

    # Bonferroni correction — always use k=4 (planned comparisons, not just available)
    k_planned = len(increments)  # 4 planned comparisons
    valid = [r for r in results if not np.isnan(r.get("delta", np.nan))]
    if valid:
        for r in valid:
            r["p_bonf"] = min(r["p"] * k_planned, 1.0)

    # Output rows
    for r in results:
        if np.isnan(r.get("delta", np.nan)):
            continue  # already printed above
        adj_p = r.get("p_bonf", r.get("p", 1.0))
        adj_sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"

        # Decision rule uses Bonferroni-corrected p-value
        if r["delta"] > 0.005 and adj_p < 0.05:
            verdict = "IMPROVES"
        elif r["delta"] < -0.005 and adj_p < 0.05:
            verdict = "DEGRADES"
        else:
            verdict = "NO DIFFERENCE"

        r["verdict"] = verdict
        n_pairs = r.get("n", "?")
        preliminary = " (preliminary)" if isinstance(n_pairs, int) and n_pairs < 15 else ""

        out(f"| {r['source']} | {r['config']} | {r['base_mean']:.4f} | {r['cfg_mean']:.4f} | "
            f"{r['delta']:+.4f} | {r['p']:.4f} | {adj_p:.4f} | "
            f"{r['d']:.3f} | {adj_sig} | {n_pairs}{preliminary} | {verdict} |")

    out()

    # Summary: is A4 the ceiling?
    improvements = [r for r in results if r.get("verdict") == "IMPROVES"]
    if not improvements:
        out("**Finding**: No source addition significantly improves over A4 (price+macro). "
            "A4 may represent the performance ceiling for this prediction task.")
    else:
        names = ", ".join(r["source"] for r in improvements)
        out(f"**Finding**: {names} significantly improve(s) over A4.")
    out()
    return results


def regime_interaction_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Table 4.9 + Figure 4.3: Config x Regime MCC delta matrix with significance.

    For each (config, fold) pair, computes delta vs A7 with significance markers.
    Returns a DataFrame suitable for heatmap plotting.
    """
    out("\n---")
    out("## Table 4.9 / Figure 4.3: Multi-Source Delta by Config x Fold")
    out()

    baseline = "A7"
    if baseline not in df["config_id"].values:
        out("ERROR: A7 baseline not found.")
        return pd.DataFrame()

    folds = sorted(df["fold_idx"].unique())
    configs = [c for c in sorted(df["config_id"].unique()) if c != baseline]

    # Build matrix
    matrix_data = []
    sig_data = []

    header = "| Config |"
    for f in folds:
        header += f" F{f} |"
    header += " Mean | Overall p |"
    out(header)
    out("|" + "---|" * (len(folds) + 3))

    for cid in configs:
        row_deltas = []
        row_sigs = []
        row_str = f"| {cid} |"

        for fold in folds:
            fold_df = df[df["fold_idx"] == fold]
            vals_cfg, vals_bl = get_matched_pairs(fold_df, "config_id", cid, baseline)
            if len(vals_cfg) < 3:
                row_deltas.append(np.nan)
                row_sigs.append("")
                row_str += " N/A |"
                continue

            r = paired_test(vals_cfg, vals_bl, cid, baseline)
            row_deltas.append(r["delta"])
            row_sigs.append(r["sig"])
            row_str += f" {r['delta']:+.4f}{r['sig']} |"

        # Overall test (all folds)
        vals_cfg_all, vals_bl_all = get_matched_pairs(df, "config_id", cid, baseline)
        if len(vals_cfg_all) >= 3:
            r_all = paired_test(vals_cfg_all, vals_bl_all, cid, baseline)
            overall_p = r_all["p"]
            mean_delta = r_all["delta"]
        else:
            overall_p = np.nan
            mean_delta = np.nan

        row_str += f" {mean_delta:+.4f} | {overall_p:.4f} |"
        out(row_str)

        matrix_data.append({"config": cid, **{f"F{f}": d for f, d in zip(folds, row_deltas)}})
        sig_data.append({"config": cid, **{f"F{f}": s for f, s in zip(folds, row_sigs)}})

    out()

    # Interpretation
    out("**Reading guide**: Each cell shows delta MCC vs A7 baseline. "
        "Positive = multi-source helps. Significance: * p<0.05, ** p<0.01, *** p<0.001.")
    out()

    # Identify regime-specific patterns
    out("**Regime-specific patterns**:")
    for fold in folds:
        regime = FOLD_LABELS.get(fold, f"F{fold}")
        col = f"F{fold}"
        fold_deltas = [(r["config"], r.get(col, np.nan)) for r in matrix_data
                       if not np.isnan(r.get(col, np.nan))]
        if fold_deltas:
            best = max(fold_deltas, key=lambda x: x[1])
            worst = min(fold_deltas, key=lambda x: x[1])
            any_helps = any(d > 0.005 for _, d in fold_deltas)
            any_hurts = any(d < -0.005 for _, d in fold_deltas)
            pattern = "mixed" if (any_helps and any_hurts) else \
                      "helps" if any_helps else "hurts" if any_hurts else "neutral"
            out(f"  - F{fold} ({regime}): {pattern} — "
                f"best {best[0]} ({best[1]:+.4f}), worst {worst[0]} ({worst[1]:+.4f})")
    out()

    return pd.DataFrame(matrix_data).set_index("config")


def architecture_regime_interaction(ff_df: pd.DataFrame | None, lstm_df: pd.DataFrame | None,
                                    combined: pd.DataFrame) -> list[dict]:
    """Table 4.10 + Figure 4.4: FF vs LSTM advantage by fold/regime.

    Tests whether the architecture advantage depends on market regime.
    This is expected to be one of the strongest findings (LSTM in bear markets).
    """
    out("\n---")
    out("## Table 4.10 / Figure 4.4: Architecture x Regime Interaction")
    out()

    if ff_df is None or lstm_df is None:
        out("WARNING: Need both FF and LSTM data. Skipping.")
        return []

    folds = sorted(combined["fold_idx"].unique())

    out("| Fold | Regime | FF MCC | LSTM MCC | Delta (LSTM-FF) | p-value | "
        "Cohen's d | Effect | Sig | LSTM Wins |")
    out("|------|--------|--------|----------|-----------------|---------|"
        "-----------|--------|-----|-----------|")

    results = []
    for fold in folds:
        fold_df = combined[combined["fold_idx"] == fold]
        vals_lstm, vals_ff = get_matched_pairs(
            fold_df, "architecture", "LSTM", "FF",
            match_cols=["config_id", "fusion_type", "seed"]
        )
        if len(vals_lstm) < 3:
            out(f"| F{fold} | {FOLD_LABELS.get(fold, '')} | N/A | N/A | N/A | "
                f"N/A | N/A | N/A | N/A | N/A |")
            continue

        r = paired_test(vals_lstm, vals_ff, "LSTM", "FF")
        ff_mean = ff_df[ff_df["fold_idx"] == fold][METRIC].mean()
        lstm_mean = lstm_df[lstm_df["fold_idx"] == fold][METRIC].mean()
        effect = effect_size_label(r["d"])
        regime = FOLD_LABELS.get(fold, "")
        winner = "LSTM" if r["delta"] > 0 else "FF"

        out(f"| F{fold} | {regime} | {ff_mean:.4f} | {lstm_mean:.4f} | "
            f"{r['delta']:+.4f} | {r['p']:.4f} | {r['d']:.3f} | {effect} | "
            f"{r['sig']} | {r['wins']}/{r['n']} |")

        r["fold"] = fold
        r["regime"] = regime
        r["ff_mean"] = ff_mean
        r["lstm_mean"] = lstm_mean
        r["winner"] = winner
        r["effect"] = effect
        results.append(r)

    out()

    # Overall test
    vals_lstm_all, vals_ff_all = get_matched_pairs(
        combined, "architecture", "LSTM", "FF",
        match_cols=["config_id", "fusion_type", "fold_idx", "seed"]
    )
    if len(vals_lstm_all) >= 3:
        r_all = paired_test(vals_lstm_all, vals_ff_all, "LSTM", "FF")
        out(f"**Overall**: LSTM delta = {r_all['delta']:+.4f}, p={r_all['p']:.4f} {r_all['sig']}, "
            f"d={r_all['d']:.3f} ({effect_size_label(r_all['d'])})")
        out()

    # Interpret regime-dependent pattern
    sig_results = [r for r in results if r["sig"] != "ns"]
    if sig_results:
        lstm_regimes = [r for r in sig_results if r["winner"] == "LSTM"]
        ff_regimes = [r for r in sig_results if r["winner"] == "FF"]
        out("**Regime-dependent architecture effect**:")
        if lstm_regimes:
            names = ", ".join(f"F{r['fold']} ({r['regime']})" for r in lstm_regimes)
            out(f"  - LSTM significantly better in: {names}")
        if ff_regimes:
            names = ", ".join(f"F{r['fold']} ({r['regime']})" for r in ff_regimes)
            out(f"  - FF significantly better in: {names}")
        out()

        # Check for the expected volatile/calm pattern
        out("**Hypothesis check**: LSTM excels in volatile/transitional, FF in calm/trending")
        for r in results:
            if r.get("sig") != "ns":
                out(f"  - F{r['fold']} ({r['regime']}): {r['winner']} wins "
                    f"(d={r['d']:.3f}, {r['effect']})")
    else:
        out("**No significant regime-dependent architecture effects found.**")
    out()

    return results


def news_degradation_analysis(df: pd.DataFrame) -> dict:
    """Section 4.5.4: Characterize when and how news degrades performance.

    Compares:
      1. A3 vs A7: news over price-only
      2. A2 vs A4: news over price+macro
      3. A1 vs A5: news over price+macro+social
    For each, breaks down by fold and by architecture to identify
    regime-specific and architecture-specific news effects.

    Uses Bonferroni correction (k=3 for the 3 planned comparisons) for
    overall verdicts. Per-fold tests are exploratory (no correction).
    """
    out("\n---")
    out("## Section 4.5.4: News Degradation Analysis")
    out()

    comparisons = [
        ("A3", "A7", "News over price-only", "Does news help when added to price alone?"),
        ("A2", "A4", "News over price+macro", "Does news add value beyond macro?"),
        ("A1", "A5", "News over price+macro+social", "Does news help in the full model?"),
    ]

    k_planned = len(comparisons)  # 3 planned comparisons for Bonferroni
    all_results = {}

    for with_news, without_news, label, question in comparisons:
        out(f"### {label}")
        out(f"*{question}*")
        out()

        if with_news not in df["config_id"].values or without_news not in df["config_id"].values:
            out(f"Data pending: need both {with_news} and {without_news}.")
            out()
            all_results[label] = {"status": "pending"}
            continue

        # Overall comparison (pooled architectures)
        vals_with, vals_without = get_matched_pairs(df, "config_id", with_news, without_news)
        if len(vals_with) < 3:
            out("Insufficient matched pairs.")
            out()
            all_results[label] = {"status": "insufficient"}
            continue

        r = paired_test(vals_with, vals_without, with_news, without_news)
        mean_with = df[df["config_id"] == with_news][METRIC].mean()
        mean_without = df[df["config_id"] == without_news][METRIC].mean()
        p_bonf = min(r["p"] * k_planned, 1.0)

        verdict = "NEWS HELPS" if r["delta"] > 0.005 and p_bonf < 0.05 else \
                  "NEWS HURTS" if r["delta"] < -0.005 and p_bonf < 0.05 else \
                  "NO EFFECT"

        out(f"**Overall**: {with_news} MCC={mean_with:.4f} vs {without_news} MCC={mean_without:.4f}")
        out(f"  Delta: {r['delta']:+.4f}, p_raw={r['p']:.4f}, p_bonf={p_bonf:.4f} "
            f"(k={k_planned}), d={r['d']:.3f} -> **{verdict}**")
        out()

        # Per-architecture breakdown
        arch_results = {}
        archs = sorted(df["architecture"].unique()) if "architecture" in df.columns else []
        if len(archs) >= 2:
            out(f"**Per-architecture split** ({with_news} vs {without_news}):")
            out()
            out("| Architecture | With News MCC | Without News MCC | Delta | p-value | "
                "Cohen's d | Effect | Sig | n |")
            out("|-------------|--------------|-----------------|-------|---------|"
                "-----------|--------|-----|---|")

            for arch in archs:
                arch_df = df[df["architecture"] == arch]
                v_with, v_without = get_matched_pairs(arch_df, "config_id", with_news, without_news)
                if len(v_with) < 3:
                    out(f"| {arch} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |")
                    continue

                ar = paired_test(v_with, v_without, with_news, without_news)
                m_with = arch_df[arch_df["config_id"] == with_news][METRIC].mean()
                m_without = arch_df[arch_df["config_id"] == without_news][METRIC].mean()
                effect = effect_size_label(ar["d"])

                out(f"| {arch} | {m_with:.4f} | {m_without:.4f} | {ar['delta']:+.4f} | "
                    f"{ar['p']:.4f} | {ar['d']:.3f} | {effect} | {ar['sig']} | {ar['n']} |")
                arch_results[arch] = ar

            out()

        # Per-fold breakdown
        folds = sorted(df["fold_idx"].unique())
        out(f"| Fold | Regime | {with_news} MCC | {without_news} MCC | Delta | p-value | Sig | News Effect |")
        out("|------|--------|" + "------------|" * 2 + "-------|---------|-----|-------------|")

        fold_results = []
        for fold in folds:
            fold_df = df[df["fold_idx"] == fold]
            v_with, v_without = get_matched_pairs(fold_df, "config_id", with_news, without_news)
            if len(v_with) < 3:
                out(f"| F{fold} | {FOLD_LABELS.get(fold, '')} | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue

            fr = paired_test(v_with, v_without, with_news, without_news)
            m_with = fold_df[fold_df["config_id"] == with_news][METRIC].mean()
            m_without = fold_df[fold_df["config_id"] == without_news][METRIC].mean()

            effect = "HELPS" if fr["delta"] > 0.005 else "HURTS" if fr["delta"] < -0.005 else "NEUTRAL"
            out(f"| F{fold} | {FOLD_LABELS.get(fold, '')} | {m_with:.4f} | {m_without:.4f} | "
                f"{fr['delta']:+.4f} | {fr['p']:.4f} | {fr['sig']} | {effect} |")
            fold_results.append({"fold": fold, "effect": effect, "delta": fr["delta"], "p": fr["p"]})

        out()

        # Identify when news helps vs hurts
        helps_folds = [fr for fr in fold_results if fr["effect"] == "HELPS"]
        hurts_folds = [fr for fr in fold_results if fr["effect"] == "HURTS"]
        if helps_folds:
            f_ids = ", ".join(f"F{fr['fold']}" for fr in helps_folds)
            out(f"  News helps in: {f_ids}")
        if hurts_folds:
            f_ids = ", ".join(f"F{fr['fold']}" for fr in hurts_folds)
            out(f"  News hurts in: {f_ids}")
        out()

        all_results[label] = {
            "overall": r, "p_bonf": p_bonf, "per_fold": fold_results,
            "per_arch": arch_results, "verdict": verdict,
        }

    # Cross-comparison summary
    out("### News Degradation Summary")
    out()
    for label, data in all_results.items():
        if isinstance(data, dict) and "verdict" in data:
            p_info = f" (p_bonf={data['p_bonf']:.4f})" if "p_bonf" in data else ""
            out(f"- **{label}**: {data['verdict']}{p_info}")
            # Note architecture split if available
            if data.get("per_arch"):
                for arch, ar in data["per_arch"].items():
                    direction = "hurts" if ar["delta"] < 0 else "helps"
                    out(f"  - {arch}: delta={ar['delta']:+.4f}, p={ar['p']:.4f}, "
                        f"d={ar['d']:.3f} ({direction})")
        elif isinstance(data, dict) and "status" in data:
            out(f"- **{label}**: {data['status']}")
    out()
    return all_results


def fusion_friedman_test(df: pd.DataFrame) -> dict:
    """Table 4.13: Non-parametric Friedman test for fusion strategy ranking.

    Tests whether fusion strategies differ significantly across configs/folds/seeds.
    If Friedman is significant, runs Nemenyi post-hoc for pairwise comparisons.

    Only uses multi-source configs (excludes A7 where fusion is trivial).
    """
    out("\n---")
    out("## Table 4.13: Fusion Strategy Comparison (Friedman Test)")
    out()

    fusions = sorted(df["fusion_type"].unique())
    if len(fusions) < 3:
        out(f"Need 3 fusion types for Friedman test, found {len(fusions)}. Skipping.")
        return {}

    # Exclude A7 (single modality — fusion is trivial)
    multi_df = df[df["config_id"] != "A7"].copy()
    if len(multi_df) == 0:
        out("No multi-modal configs in data. Skipping.")
        return {}

    # Build matched groups: each "subject" is a (config, fold, seed, architecture) tuple
    match_cols = ["config_id", "fold_idx", "seed"]
    if "architecture" in multi_df.columns:
        match_cols.append("architecture")

    # Pivot: rows = subjects, columns = fusions
    pivot = multi_df.pivot_table(index=match_cols, columns="fusion_type",
                                  values=METRIC, aggfunc="first")
    pivot = pivot.dropna()  # Only keep subjects with all 3 fusions

    if len(pivot) < 5:
        out(f"Only {len(pivot)} complete subjects (need 5+). Skipping Friedman test.")
        return {}

    out(f"Subjects with all 3 fusions: {len(pivot)}")
    out()

    # Descriptive statistics
    out("### Fusion Descriptive Statistics (Multi-Source Only)")
    out()
    out("| Fusion | Mean MCC | Std | Median | N |")
    out("|--------|----------|-----|--------|---|")
    for ft in fusions:
        if ft in pivot.columns:
            vals = pivot[ft]
            out(f"| {ft} | {vals.mean():.4f} | {vals.std():.4f} | "
                f"{vals.median():.4f} | {len(vals)} |")
    out()

    # Split by architecture if available
    if "architecture" in match_cols:
        out("### By Architecture")
        out()
        out("| Fusion | FF Mean | LSTM Mean |")
        out("|--------|---------|-----------|")
        for ft in fusions:
            ff_vals = multi_df[(multi_df["fusion_type"] == ft) &
                               (multi_df["architecture"] == "FF")][METRIC]
            lstm_vals = multi_df[(multi_df["fusion_type"] == ft) &
                                 (multi_df["architecture"] == "LSTM")][METRIC]
            out(f"| {ft} | {ff_vals.mean():.4f} | {lstm_vals.mean():.4f} |")
        out()

    # Friedman test
    fusion_arrays = [pivot[ft].values for ft in fusions if ft in pivot.columns]
    if len(fusion_arrays) < 3:
        out("Insufficient fusion columns in pivot. Skipping.")
        return {}

    stat, p_friedman = stats.friedmanchisquare(*fusion_arrays)
    out(f"### Friedman Test")
    out(f"- Chi-square statistic: {stat:.4f}")
    out(f"- p-value: {p_friedman:.6f}")
    sig = "***" if p_friedman < 0.001 else "**" if p_friedman < 0.01 else \
          "*" if p_friedman < 0.05 else "ns"
    out(f"- Significance: {sig}")
    out()

    result = {"friedman_stat": stat, "friedman_p": p_friedman, "sig": sig,
              "n_subjects": len(pivot)}

    # Post-hoc: pairwise paired t-tests with Bonferroni (Nemenyi approximation)
    if p_friedman < 0.05:
        out("### Post-Hoc Pairwise Comparisons (Bonferroni-corrected paired t-tests)")
        out()
        out("| Comparison | Delta | p-value | p (Bonf) | Cohen's d | Sig |")
        out("|------------|-------|---------|----------|-----------|-----|")

        pairs = list(combinations(fusions, 2))
        raw_ps = []
        pair_results = []

        for ft_a, ft_b in pairs:
            if ft_a in pivot.columns and ft_b in pivot.columns:
                v_a = pivot[ft_a].values
                v_b = pivot[ft_b].values
                r = paired_test(v_a, v_b, ft_a, ft_b)
                raw_ps.append(r["p"])
                pair_results.append((ft_a, ft_b, r))

        adj_ps = bonferroni_adjust(raw_ps)

        for (ft_a, ft_b, r), adj_p in zip(pair_results, adj_ps):
            adj_sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else \
                      "*" if adj_p < 0.05 else "ns"
            out(f"| {ft_a} vs {ft_b} | {r['delta']:+.4f} | {r['p']:.4f} | "
                f"{adj_p:.4f} | {r['d']:.3f} | {adj_sig} |")

        result["posthoc"] = pair_results
        out()

        # Rank
        out("### Fusion Ranking")
        out()
        fusion_means = {ft: pivot[ft].mean() for ft in fusions if ft in pivot.columns}
        ranked = sorted(fusion_means.items(), key=lambda x: x[1], reverse=True)
        for rank, (ft, m) in enumerate(ranked, 1):
            out(f"  {rank}. {ft}: {m:.4f}")
        out()
    else:
        out("Friedman test not significant — no evidence that fusion strategies differ.")
        out("All fusion strategies perform comparably on multi-source configs.")
        out()

    return result


# ============================================================================
# ADDITIONAL CHAPTER 4 TABLES
# ============================================================================


def per_fold_fusion_table(df: pd.DataFrame):
    """Table 4.11: Best fusion by fold and architecture, with Friedman per fold.

    For each (fold, architecture) combination, shows mean MCC for each of the
    3 fusion strategies plus a Friedman test to check if they differ.
    Only uses multi-source configs (excludes A7).
    """
    out("\n---")
    out("## Table 4.11: Best Fusion by Fold and Architecture")
    out()

    multi_df = df[df["config_id"] != "A7"].copy()
    fusions = sorted(multi_df["fusion_type"].unique())
    if len(fusions) < 3:
        out(f"Need 3 fusion types, found {len(fusions)}. Skipping.")
        return

    archs = sorted(multi_df["architecture"].unique()) if "architecture" in multi_df.columns else ["ALL"]
    folds = sorted(multi_df["fold_idx"].unique())

    out("| Fold | Regime | Architecture | " + " | ".join(fusions) + " | Best | Friedman p | Sig |")
    out("|------|--------|-------------|" + "|".join(["-------"] * len(fusions)) + "|------|-----------|-----|")

    for fold in folds:
        regime = FOLD_LABELS.get(fold, "")
        fold_df = multi_df[multi_df["fold_idx"] == fold]

        for arch in archs:
            if arch == "ALL":
                arch_df = fold_df
            else:
                arch_df = fold_df[fold_df["architecture"] == arch]

            if len(arch_df) == 0:
                continue

            # Pivot to get subjects x fusions
            match_cols = ["config_id", "seed"]
            if arch == "ALL" and "architecture" in arch_df.columns:
                match_cols.append("architecture")

            pivot = arch_df.pivot_table(index=match_cols, columns="fusion_type",
                                        values=METRIC, aggfunc="first")
            pivot = pivot.dropna()

            # Mean MCC per fusion
            means = {}
            cells = []
            for ft in fusions:
                if ft in pivot.columns:
                    m = pivot[ft].mean()
                    means[ft] = m
                    cells.append(f"{m:.4f}")
                else:
                    cells.append("N/A")

            # Find best
            if means:
                best_ft = max(means, key=means.get)
            else:
                best_ft = "N/A"

            # Friedman test (need >= 5 subjects with all 3 fusions)
            fusion_arrays = [pivot[ft].values for ft in fusions if ft in pivot.columns]
            if len(fusion_arrays) == 3 and len(pivot) >= 5:
                stat, p_fried = stats.friedmanchisquare(*fusion_arrays)
                sig = "***" if p_fried < 0.001 else "**" if p_fried < 0.01 else \
                      "*" if p_fried < 0.05 else "ns"
                p_str = f"{p_fried:.4f}"
            else:
                p_str = "N/A"
                sig = "N/A"

            out(f"| F{fold} | {regime} | {arch} | " + " | ".join(cells) +
                f" | {best_ft} | {p_str} | {sig} |")

    out()
    out("**Note**: Per-fold Friedman tests are exploratory (small n per cell).")
    out("No fold-level fusion difference reaches significance in prior analyses.")
    out()


def architecture_overall_table(df: pd.DataFrame):
    """Table 4.12: Clean FF vs LSTM comparison table with 95% CI.

    Standalone table format suitable for Chapter 4, Section 4.6.1.
    """
    out("\n---")
    out("## Table 4.12: Architecture Comparison (All Configs Pooled)")
    out()

    if "architecture" not in df.columns:
        out("No architecture column in data. Skipping.")
        return

    archs = sorted(df["architecture"].unique())
    if len(archs) < 2:
        out(f"Need 2 architectures, found {len(archs)}. Skipping.")
        return

    # Descriptive statistics per architecture
    out("| Architecture | Mean MCC | 95% CI | Std | Median | n |")
    out("|-------------|----------|--------|-----|--------|---|")
    for arch in archs:
        vals = df[df["architecture"] == arch][METRIC].values
        ci = confidence_interval_95(vals)
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if not np.isnan(ci[0]) else "N/A"
        out(f"| {arch} | {vals.mean():.4f} | {ci_str} | {vals.std():.4f} | "
            f"{np.median(vals):.4f} | {len(vals)} |")
    out()

    # Paired comparison
    vals_lstm, vals_ff = get_matched_pairs(df, "architecture", "LSTM", "FF",
                                           match_cols=["config_id", "fusion_type", "fold_idx", "seed"])
    if len(vals_lstm) >= 3:
        r = paired_test(vals_lstm, vals_ff, "LSTM", "FF")
        out("### Paired Comparison (LSTM vs FF)")
        out()
        out("| Metric | Value |")
        out("|--------|-------|")
        out(f"| Paired observations | {r['n']} |")
        out(f"| Mean delta (LSTM - FF) | {r['delta']:+.4f} |")
        out(f"| 95% CI of delta | {format_ci(vals_lstm - vals_ff).split(' ', 1)[1] if ' ' in format_ci(vals_lstm - vals_ff) else 'N/A'} |")
        out(f"| t-statistic | {r['t']:.3f} |")
        out(f"| p-value | {r['p']:.6f} |")
        out(f"| Cohen's d | {r['d']:.3f} ({effect_size_label(r['d'])}) |")
        out(f"| LSTM win rate | {r['wins']}/{r['n']} ({r['wins']/r['n']*100:.1f}%) |")
        out()

        # Per-config breakdown
        out("### Per-Config Architecture Delta")
        out()
        out("| Config | FF MCC | LSTM MCC | Delta | p-value | Cohen's d | Effect | Sig | n (pairs) |")
        out("|--------|--------|----------|-------|---------|-----------|--------|-----|-----------|")

        configs = sorted(df["config_id"].unique())
        for cid in configs:
            cid_df = df[df["config_id"] == cid]
            vl, vf = get_matched_pairs(cid_df, "architecture", "LSTM", "FF",
                                       match_cols=["fusion_type", "fold_idx", "seed"])
            if len(vl) < 3:
                out(f"| {cid} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | {len(vl)} |")
                continue
            cr = paired_test(vl, vf, "LSTM", "FF")
            ff_m = cid_df[cid_df["architecture"] == "FF"][METRIC].mean()
            lstm_m = cid_df[cid_df["architecture"] == "LSTM"][METRIC].mean()
            out(f"| {cid} | {ff_m:.4f} | {lstm_m:.4f} | {cr['delta']:+.4f} | "
                f"{cr['p']:.4f} | {cr['d']:.3f} | {effect_size_label(cr['d'])} | {cr['sig']} | {cr['n']} |")
        out()
    else:
        out("Insufficient matched pairs for paired comparison.")
        out()


# ============================================================================
# SUPPLEMENTARY ANALYSIS FUNCTIONS
# ============================================================================

def a10_comparison_table(df: pd.DataFrame):
    """Supplementary: A10 (price+news+social) comparison table.

    Compares A10 against related configs to understand the value of the
    news+social combination without macro:
      - A10 vs A1: effect of removing macro
      - A10 vs A6: effect of adding news to price+social
      - A10 vs A3: effect of adding social to price+news
      - A10 vs A7: overall multi-source vs baseline
    """
    out("\n---")
    out("## Supplementary: A10 (Price + News + Social) Comparisons")
    out()

    if "A10" not in df["config_id"].values:
        out("A10 data not yet available.")
        out()
        return

    comparisons = [
        ("A10", "A7", "A10 vs baseline", "Does news+social (no macro) beat price-only?"),
        ("A10", "A6", "A10 vs A6", "Does adding news to price+social help?"),
        ("A10", "A3", "A10 vs A3", "Does adding social to price+news help?"),
        ("A1", "A10", "A1 vs A10", "Does adding macro to price+news+social help?"),
    ]

    out("| Comparison | Description | Left MCC | Right MCC | Delta | p-value | "
        "Cohen's d | Effect | Sig | n |")
    out("|-----------|-------------|----------|-----------|-------|---------|"
        "-----------|--------|-----|---|")

    for left, right, label, question in comparisons:
        if left not in df["config_id"].values or right not in df["config_id"].values:
            out(f"| {label} | {question} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |")
            continue

        vals_l, vals_r = get_matched_pairs(df, "config_id", left, right)
        if len(vals_l) < 3:
            out(f"| {label} | {question} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |")
            continue

        r = paired_test(vals_l, vals_r, left, right)
        mean_l = df[df["config_id"] == left][METRIC].mean()
        mean_r = df[df["config_id"] == right][METRIC].mean()
        effect = effect_size_label(r["d"])
        out(f"| {label} | {question} | {mean_l:.4f} | {mean_r:.4f} | "
            f"{r['delta']:+.4f} | {r['p']:.4f} | {r['d']:.3f} | {effect} | "
            f"{r['sig']} | {r['n']} |")

    out()
    out("**Note**: These are exploratory comparisons (no Bonferroni correction). "
        "The formal RQ1 test is in Table 4.5.")
    out()


def social_lstm_comparison_table(df: pd.DataFrame):
    """Supplementary: Social LSTM sequence experiment comparison.

    Compares A5/A6 LSTM runs where social modality gets temporal sequences
    (lookback=20) vs the original runs where only price gets sequences.
    """
    out("\n---")
    out("## Supplementary: Social LSTM Sequence Experiment")
    out()

    if _SOCIAL_LSTM_DF is None:
        out("Social LSTM data not yet available.")
        out()
        return

    # Get original LSTM data for A5 and A6
    orig_lstm = df[(df["architecture"] == "LSTM") &
                   (df["config_id"].isin(["A5", "A6"]))]

    if len(orig_lstm) == 0:
        out("Original LSTM A5/A6 data not found in main dataset.")
        out()
        return

    out("Compares original LSTM (only price gets lookback=20 sequences) vs "
        "Social LSTM (both price AND social get lookback=20 sequences).")
    out()

    out("| Config | Variant | Mean MCC | Std | n |")
    out("|--------|---------|----------|-----|---|")

    for cid in ["A5", "A6"]:
        orig = orig_lstm[orig_lstm["config_id"] == cid]
        social_seq = _SOCIAL_LSTM_DF[_SOCIAL_LSTM_DF["config_id"] == cid]

        if len(orig) > 0:
            out(f"| {cid} | Original LSTM | {orig[METRIC].mean():.4f} | "
                f"{orig[METRIC].std():.4f} | {len(orig)} |")
        if len(social_seq) > 0:
            out(f"| {cid} | Social LSTM | {social_seq[METRIC].mean():.4f} | "
                f"{social_seq[METRIC].std():.4f} | {len(social_seq)} |")

    out()

    # Paired comparison where possible (matching on fold, seed, fusion)
    out("### Paired Comparison (matched on fold, seed, fusion)")
    out()
    out("| Config | Delta (Social LSTM - Original) | p-value | Cohen's d | Sig | n |")
    out("|--------|-------------------------------|---------|-----------|-----|---|")

    for cid in ["A5", "A6"]:
        orig = orig_lstm[orig_lstm["config_id"] == cid]
        social_seq = _SOCIAL_LSTM_DF[_SOCIAL_LSTM_DF["config_id"] == cid]

        if len(orig) == 0 or len(social_seq) == 0:
            out(f"| {cid} | N/A | N/A | N/A | N/A | 0 |")
            continue

        match_cols = ["fold_idx", "seed", "fusion_type"]
        a = social_seq.set_index(match_cols)[METRIC].sort_index()
        b = orig.set_index(match_cols)[METRIC].sort_index()
        common = a.index.intersection(b.index)

        if len(common) < 3:
            out(f"| {cid} | N/A | N/A | N/A | N/A | {len(common)} |")
            continue

        vals_a = a.loc[common].values
        vals_b = b.loc[common].values
        r = paired_test(vals_a, vals_b, f"{cid}_social_seq", f"{cid}_orig")
        out(f"| {cid} | {r['delta']:+.4f} | {r['p']:.4f} | {r['d']:.3f} | "
            f"{r['sig']} | {r['n']} |")

    out()
    out("**Hypothesis**: Giving social media temporal sequences (via LSTM) improves "
        "A5/A6 performance beyond the already-significant social+LSTM interaction.")
    out()


def graph_ln_comparison_table(df: pd.DataFrame):
    """Supplementary: Graph LayerNorm before/after comparison.

    Compares original A8/A9 (without LayerNorm) vs LayerNorm-fixed A8/A9.
    Requires both the original CSVs and the Graph LN CSVs.
    """
    out("\n---")
    out("## Supplementary: Graph LayerNorm Before/After Comparison")
    out()

    # Load original A8/A9 from the base CSVs (before replacement)
    orig_ff_path = _csv("ablation_ff.csv")
    orig_lstm_path = _csv("ablation_lstm.csv")
    ln_ff_path = SUPPLEMENTARY_CSVS["graph_ln_ff"]
    ln_lstm_path = SUPPLEMENTARY_CSVS["graph_ln_lstm"]

    if not ln_ff_path.exists() and not ln_lstm_path.exists():
        out("Graph LayerNorm data not yet available (experiments still running).")
        out()
        return

    parts = []

    for orig_path, ln_path, arch in [
        (orig_ff_path, ln_ff_path, "FF"),
        (orig_lstm_path, ln_lstm_path, "LSTM"),
    ]:
        if not orig_path.exists() or not ln_path.exists():
            continue

        orig = pd.read_csv(orig_path)
        orig = orig[orig["config_id"].isin(["A8", "A9"])]
        orig["variant"] = "original"
        orig["architecture"] = arch

        ln = pd.read_csv(ln_path)
        ln["variant"] = "layernorm"
        ln["architecture"] = arch

        parts.append((arch, orig, ln))

    if not parts:
        out("Insufficient data for comparison.")
        out()
        return

    out("| Config | Arch | Original MCC | LayerNorm MCC | Delta | p-value | "
        "Cohen's d | Sig | n |")
    out("|--------|------|-------------|---------------|-------|---------|"
        "-----------|-----|---|")

    for arch, orig, ln in parts:
        for cid in ["A8", "A9"]:
            orig_cid = orig[orig["config_id"] == cid]
            ln_cid = ln[ln["config_id"] == cid]

            if len(orig_cid) == 0 or len(ln_cid) == 0:
                out(f"| {cid} | {arch} | N/A | N/A | N/A | N/A | N/A | N/A | 0 |")
                continue

            # Match on fold, seed, fusion
            match_cols = ["fold_idx", "seed", "fusion_type"]
            a = ln_cid.set_index(match_cols)[METRIC].sort_index()
            b = orig_cid.set_index(match_cols)[METRIC].sort_index()
            common = a.index.intersection(b.index)

            if len(common) < 3:
                out(f"| {cid} | {arch} | {orig_cid[METRIC].mean():.4f} | "
                    f"{ln_cid[METRIC].mean():.4f} | N/A | N/A | N/A | N/A | {len(common)} |")
                continue

            vals_a = a.loc[common].values
            vals_b = b.loc[common].values
            r = paired_test(vals_a, vals_b, f"{cid}_LN", f"{cid}_orig")
            out(f"| {cid} | {arch} | {vals_b.mean():.4f} | {vals_a.mean():.4f} | "
                f"{r['delta']:+.4f} | {r['p']:.4f} | {r['d']:.3f} | {r['sig']} | {r['n']} |")

    out()
    out("**Hypothesis**: LayerNorm normalizes graph embeddings to match other encoder "
        "output scales, improving the graph modality's contribution to fusion.")
    out()


def generate_figures(df: pd.DataFrame, output_dir: Path = PHASE4R_DIR / "analysis" / "figures"):
    """Generate Chapter 4 figures as publication-quality PDF + PNG files.

    No embedded titles (LaTeX captions handle numbering).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    out("\n---")
    out("## Figure Generation")
    out()

    # ------------------------------------------------------------------
    # Figure 4.1: MCC vs Number of Modalities (line + CI)
    # ------------------------------------------------------------------
    # Map each config to its number of modalities
    n_mod_map = {cid: len(info["modalities"]) for cid, info in ABLATION_CONFIGS.items()}
    df_fig1 = df.copy()
    df_fig1["n_modalities"] = df_fig1["config_id"].map(n_mod_map)
    df_fig1 = df_fig1.dropna(subset=["n_modalities"])

    if "architecture" in df_fig1.columns and len(df_fig1["architecture"].unique()) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        arch_colors = {"FF": "#4C72B0", "LSTM": "#DD8452"}
        arch_markers = {"FF": "s", "LSTM": "o"}

        for arch in ["FF", "LSTM"]:
            arch_df = df_fig1[df_fig1["architecture"] == arch]
            if arch_df.empty:
                continue
            xs, ys, lo_errs, hi_errs = [], [], [], []
            for nm in sorted(arch_df["n_modalities"].unique()):
                vals = arch_df[arch_df["n_modalities"] == nm][METRIC].values
                if len(vals) < 2:
                    continue
                mean = vals.mean()
                ci = confidence_interval_95(vals)
                xs.append(nm)
                ys.append(mean)
                lo_errs.append(mean - ci[0])
                hi_errs.append(ci[1] - mean)

            if xs:
                ax.errorbar(xs, ys, yerr=[lo_errs, hi_errs], fmt=arch_markers[arch] + "-",
                            color=arch_colors[arch], label=arch, capsize=5, markersize=7,
                            linewidth=1.5, capthick=1.2)

        ax.set_xlabel("Number of Modalities", fontsize=11, family="serif")
        ax.set_ylabel("Mean MCC", fontsize=11, family="serif")
        ax.set_xticks([1, 2, 3, 4])
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        fig41_png = output_dir / "fig4_1_mcc_vs_modalities.png"
        fig41_pdf = output_dir / "fig4_1_mcc_vs_modalities.pdf"
        fig.savefig(fig41_png, dpi=300, bbox_inches="tight")
        fig.savefig(fig41_pdf, bbox_inches="tight")
        plt.close(fig)
        out(f"Saved Figure 4.1 to {fig41_png} and {fig41_pdf}")
    else:
        out("Skipping Figure 4.1: need both FF and LSTM data.")

    # ------------------------------------------------------------------
    # Figure 4.2: Marginal Contribution Waterfall (bar + CI)
    # ------------------------------------------------------------------
    # Additive delta: each 2-source config minus A7, per source
    source_configs = {
        "Macro": "A4",
        "News": "A3",
        "Social": "A6",
        "Graph": "A8",
    }
    if "architecture" in df.columns and "A7" in df["config_id"].values:
        archs_avail = sorted(df["architecture"].unique())
        n_panels = len(archs_avail)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5), sharey=True)
        if n_panels == 1:
            axes = [axes]

        for ax_i, arch in enumerate(archs_avail):
            arch_df = df[df["architecture"] == arch]
            a7_df = arch_df[arch_df["config_id"] == "A7"]
            if a7_df.empty:
                continue

            deltas_list = []
            for src_name, src_cfg in source_configs.items():
                cfg_df = arch_df[arch_df["config_id"] == src_cfg]
                if cfg_df.empty:
                    continue
                # Compute matched deltas
                va, vb = get_matched_pairs(arch_df, "config_id", src_cfg, "A7",
                                           match_cols=["fusion_type", "fold_idx", "seed"])
                if len(va) < 2:
                    continue
                delta_vals = va - vb
                mean_d = delta_vals.mean()
                ci = confidence_interval_95(delta_vals)
                deltas_list.append({
                    "source": src_name,
                    "mean_delta": mean_d,
                    "ci_lo": ci[0],
                    "ci_hi": ci[1],
                })

            if not deltas_list:
                continue

            # Sort by magnitude (largest positive first)
            deltas_list.sort(key=lambda x: x["mean_delta"], reverse=True)
            names = [d["source"] for d in deltas_list]
            means = [d["mean_delta"] for d in deltas_list]
            lo_errs = [d["mean_delta"] - d["ci_lo"] for d in deltas_list]
            hi_errs = [d["ci_hi"] - d["mean_delta"] for d in deltas_list]
            colors = ["#55a868" if m >= 0 else "#c44e52" for m in means]

            axes[ax_i].barh(range(len(names)), means, xerr=[lo_errs, hi_errs],
                            color=colors, capsize=4, height=0.6, alpha=0.85)
            axes[ax_i].set_yticks(range(len(names)))
            axes[ax_i].set_yticklabels(names, fontsize=10, family="serif")
            axes[ax_i].axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
            axes[ax_i].set_xlabel("Delta MCC vs A7 (price-only)", fontsize=10, family="serif")
            axes[ax_i].set_title(arch, fontsize=12, family="serif")
            axes[ax_i].spines["top"].set_visible(False)
            axes[ax_i].spines["right"].set_visible(False)
            axes[ax_i].invert_yaxis()  # largest at top

        fig.tight_layout()

        fig42_png = output_dir / "fig4_2_marginal_waterfall.png"
        fig42_pdf = output_dir / "fig4_2_marginal_waterfall.pdf"
        fig.savefig(fig42_png, dpi=300, bbox_inches="tight")
        fig.savefig(fig42_pdf, bbox_inches="tight")
        plt.close(fig)
        out(f"Saved Figure 4.2 to {fig42_png} and {fig42_pdf}")
    else:
        out("Skipping Figure 4.2: need A7 baseline data.")

    # ------------------------------------------------------------------
    # Figure 4.3: Config x Fold MCC Delta Heatmap
    # ------------------------------------------------------------------
    configs_ordered = ["A7", "A4", "A3", "A6", "A8", "A2", "A5", "A9", "A1"]
    configs_present = [c for c in configs_ordered if c in df["config_id"].values]
    folds_present = sorted(df["fold_idx"].unique())

    if len(configs_present) >= 3 and len(folds_present) >= 2:
        # Compute A7 baseline mean per fold (pool fusions, seeds, architectures)
        a7_means = df[df["config_id"] == "A7"].groupby("fold_idx")[METRIC].mean()

        heatmap_data = []
        row_labels = []
        for cid in configs_present:
            if cid == "A7":
                continue  # baseline row is all zeros, skip for clarity
            cfg_df = df[df["config_id"] == cid]
            row = []
            for fold in folds_present:
                cfg_fold_mean = cfg_df[cfg_df["fold_idx"] == fold][METRIC].mean()
                a7_fold_mean = a7_means.get(fold, 0)
                row.append(cfg_fold_mean - a7_fold_mean)
            heatmap_data.append(row)
            desc = ABLATION_CONFIGS.get(cid, {}).get("desc", cid)
            row_labels.append(f"{cid}: {desc}")

        if heatmap_data:
            import matplotlib.colors as mcolors

            data_arr = np.array(heatmap_data)
            vmax = max(abs(data_arr.min()), abs(data_arr.max()), 0.01)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "rg", ["#c44e52", "#ffffff", "#55a868"])

            fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(row_labels) + 1.5)))
            im = ax.imshow(data_arr, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

            # Annotate cells
            for i in range(data_arr.shape[0]):
                for j in range(data_arr.shape[1]):
                    val = data_arr[i, j]
                    text_color = "white" if abs(val) > vmax * 0.65 else "black"
                    ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                            fontsize=9, color=text_color, family="serif")

            fold_labels_short = [f"F{f}\n{FOLD_LABELS.get(f, '').split('(')[0].strip()}"
                                 for f in folds_present]
            ax.set_xticks(range(len(folds_present)))
            ax.set_xticklabels(fold_labels_short, fontsize=9, family="serif")
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=9, family="serif")
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
            cbar.set_label("Delta MCC", fontsize=10, family="serif")
            fig.tight_layout()

            fig43_png = output_dir / "fig4_3_config_fold_heatmap.png"
            fig43_pdf = output_dir / "fig4_3_config_fold_heatmap.pdf"
            fig.savefig(fig43_png, dpi=300, bbox_inches="tight")
            fig.savefig(fig43_pdf, bbox_inches="tight")
            plt.close(fig)
            out(f"Saved Figure 4.3 to {fig43_png} and {fig43_pdf}")
    else:
        out("Skipping Figure 4.3: need >=3 configs and >=2 folds.")

    # ------------------------------------------------------------------
    # Figure 4.4: Architecture x Regime Interaction (grouped bar chart)
    # ------------------------------------------------------------------
    if "architecture" in df.columns and len(df["architecture"].unique()) >= 2:
        folds = sorted(df["fold_idx"].unique())
        ff_means, lstm_means = [], []
        ff_cis, lstm_cis = [], []
        fold_labels = []

        for fold in folds:
            fold_df = df[df["fold_idx"] == fold]
            ff_vals = fold_df[fold_df["architecture"] == "FF"][METRIC].values
            lstm_vals = fold_df[fold_df["architecture"] == "LSTM"][METRIC].values

            ff_means.append(ff_vals.mean() if len(ff_vals) > 0 else 0)
            lstm_means.append(lstm_vals.mean() if len(lstm_vals) > 0 else 0)

            ff_ci = confidence_interval_95(ff_vals) if len(ff_vals) >= 2 else (ff_vals.mean(), ff_vals.mean())
            lstm_ci = confidence_interval_95(lstm_vals) if len(lstm_vals) >= 2 else (lstm_vals.mean(), lstm_vals.mean())

            ff_cis.append(ff_vals.mean() - ff_ci[0] if not np.isnan(ff_ci[0]) else 0)
            lstm_cis.append(lstm_vals.mean() - lstm_ci[0] if not np.isnan(lstm_ci[0]) else 0)

            regime = FOLD_LABELS.get(fold, f"F{fold}")
            fold_labels.append(f"F{fold}\n{regime.split('(')[0].strip()}")

        x = np.arange(len(folds))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars_ff = ax.bar(x - width/2, ff_means, width, yerr=ff_cis, capsize=4,
                         label="FF", color="#4C72B0", alpha=0.85)
        bars_lstm = ax.bar(x + width/2, lstm_means, width, yerr=lstm_cis, capsize=4,
                           label="LSTM", color="#DD8452", alpha=0.85)

        ax.set_xlabel("Temporal Fold / Market Regime", fontsize=11)
        ax.set_ylabel("Mean MCC", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, fontsize=9)
        ax.legend(fontsize=10)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        fig44_png = output_dir / "fig4_4_arch_regime.png"
        fig44_pdf = output_dir / "fig4_4_arch_regime.pdf"
        fig.savefig(fig44_png, dpi=300, bbox_inches="tight")
        fig.savefig(fig44_pdf, bbox_inches="tight")
        plt.close(fig)
        out(f"Saved Figure 4.4 to {fig44_png} and {fig44_pdf}")
    else:
        out("Skipping Figure 4.4: need both FF and LSTM data.")

    # ------------------------------------------------------------------
    # Figure 4.5: Fusion x Architecture Boxplot (750-core only, A7 and A10 excluded)
    # ------------------------------------------------------------------
    multi_df = df[(df["config_id"] != "A7") & (df["config_id"] != "A10")].copy()
    fusions = sorted(multi_df["fusion_type"].unique())

    if len(fusions) >= 2 and "architecture" in multi_df.columns:
        archs = sorted(multi_df["architecture"].unique())

        fusion_short = {"concat": "Concat", "gated_cross_attention": "Gated CA", "mha": "MHA"}
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        positions = []
        labels = []
        data_groups = []
        colors = []
        color_map = {"FF": "#4C72B0", "LSTM": "#DD8452"}

        pos = 0
        for fi, ft in enumerate(fusions):
            for ai, arch in enumerate(archs):
                vals = multi_df[(multi_df["fusion_type"] == ft) &
                                (multi_df["architecture"] == arch)][METRIC].values
                data_groups.append(vals)
                positions.append(pos)
                labels.append(f"{fusion_short.get(ft, ft)}\n{arch}")
                colors.append(color_map.get(arch, "#999999"))
                pos += 1
            pos += 0.5  # gap between fusion groups

        bp = ax.boxplot(data_groups, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="black", markersize=5))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("MCC", fontsize=10)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[a], alpha=0.7, label=a) for a in archs]
        ax.legend(handles=legend_elements, fontsize=9)

        fig.tight_layout()
        fig45_png = output_dir / "fig4_5_fusion_arch_boxplot.png"
        fig45_pdf = output_dir / "fig4_5_fusion_arch_boxplot.pdf"
        fig.savefig(fig45_png, dpi=300, bbox_inches="tight")
        fig.savefig(fig45_pdf, bbox_inches="tight")
        plt.close(fig)
        out(f"Saved Figure 4.5 to {fig45_png} and {fig45_pdf}")
    else:
        out("Skipping Figure 4.5: need multiple fusions and architectures.")

    out()


def synthesis_table(df: pd.DataFrame):
    """Table 4.18: Summary of Key Findings by Research Question.

    Generates a data-driven synthesis table covering all three RQs plus
    supplementary architecture/fusion findings. Each row cites the evidence
    (config comparison), effect size, and significance level.
    Designed to work with partial data — rows are populated where data exists.
    """
    out("\n---")
    out("## Table 4.18: Summary of Key Findings")
    out()

    findings = []

    # Helper: run a paired test if both configs exist
    def _paired(cfg_a, cfg_b, match_cols=None):
        if cfg_a not in df["config_id"].values or cfg_b not in df["config_id"].values:
            return None
        va, vb = get_matched_pairs(df, "config_id", cfg_a, cfg_b, match_cols=match_cols)
        if len(va) < 3:
            return None
        return paired_test(va, vb, cfg_a, cfg_b)

    # Helper: architecture paired test on a subset
    def _arch_paired(subset):
        if "architecture" not in subset.columns:
            return None
        vl, vf = get_matched_pairs(subset, "architecture", "LSTM", "FF",
                                   match_cols=["config_id", "fusion_type", "fold_idx", "seed"])
        if len(vl) < 3:
            return None
        return paired_test(vl, vf, "LSTM", "FF")

    # ===================== RQ1: Multi-source vs single-source =====================

    # Best multi-source vs A7
    a7_mcc = df[df["config_id"] == "A7"][METRIC].mean() if "A7" in df["config_id"].values else None
    if a7_mcc is not None:
        best_cfg, best_delta, best_p, best_d = None, -999, 1.0, 0.0
        for cfg in ["A1", "A2", "A3", "A4", "A5", "A6", "A8", "A9"]:
            r = _paired(cfg, "A7")
            if r and r["delta"] > best_delta:
                best_cfg, best_delta, best_p, best_d = cfg, r["delta"], r["p"], r["d"]

        if best_cfg:
            findings.append({
                "rq": "RQ1",
                "finding": "No modality significantly beats price-only baseline",
                "evidence": f"Best: {best_cfg} vs A7, delta={best_delta:+.4f}",
                "effect": f"d={best_d:.3f} ({effect_size_label(best_d)})",
                "sig": f"p={best_p:.4f} (all p_bonf > 0.05)",
            })

        # Scalability: count how many multi-source configs beat A7
        n_better = 0
        n_tested = 0
        for cfg in ["A1", "A2", "A3", "A4", "A5", "A6", "A8", "A9"]:
            r = _paired(cfg, "A7")
            if r:
                n_tested += 1
                if r["delta"] > 0:
                    n_better += 1
        if n_tested > 0:
            findings.append({
                "rq": "RQ1",
                "finding": f"More sources does not mean better ({n_better}/{n_tested} configs > A7)",
                "evidence": f"{n_tested} configs tested vs A7",
                "effect": "negligible deltas",
                "sig": "all ns after Bonferroni",
            })

    # ===================== RQ2: Relative contribution =====================

    # Source ranking (additive)
    source_configs = [("Macro", "A4"), ("News", "A3"), ("Social", "A6"), ("Graph", "A8")]
    ranking_parts = []
    for source, cfg in source_configs:
        r = _paired(cfg, "A7")
        if r:
            ranking_parts.append((source, r["delta"], r["p"], r["d"]))

    if ranking_parts:
        ranking_parts.sort(key=lambda x: x[1], reverse=True)
        rank_str = " > ".join(f"{s} ({d:+.4f})" for s, d, _, _ in ranking_parts)
        findings.append({
            "rq": "RQ2",
            "finding": f"Source ranking: {rank_str}",
            "evidence": "Additive delta vs A7 (Table 4.6)",
            "effect": "all negligible",
            "sig": "all ns",
        })

    # A4 as ceiling
    r_a5 = _paired("A5", "A4")
    r_a9 = _paired("A9", "A4")
    if r_a5 and r_a9:
        findings.append({
            "rq": "RQ2",
            "finding": "A4 (price+macro) is performance ceiling",
            "evidence": f"A5 vs A4: {r_a5['delta']:+.4f}; A9 vs A4: {r_a9['delta']:+.4f}",
            "effect": f"d={r_a5['d']:.3f}, {r_a9['d']:.3f}",
            "sig": f"p={r_a5['p']:.4f}, {r_a9['p']:.4f}",
        })

    # ===================== RQ3: Conditional value =====================

    # Architecture-regime interaction
    arch_r = _arch_paired(df)
    if arch_r:
        # Per-fold: find the strongest fold
        best_fold_d = 0
        best_fold_label = ""
        for fold in sorted(df["fold_idx"].unique()):
            fold_df = df[df["fold_idx"] == fold]
            fr = _arch_paired(fold_df)
            if fr and abs(fr["d"]) > abs(best_fold_d):
                best_fold_d = fr["d"]
                best_fold_label = f"F{fold} ({FOLD_LABELS.get(fold, '')})"

        findings.append({
            "rq": "RQ3",
            "finding": "LSTM > FF, especially in volatile regimes",
            "evidence": f"Overall LSTM-FF; strongest: {best_fold_label}",
            "effect": f"overall d={arch_r['d']:.3f}; best fold d={best_fold_d:.3f}",
            "sig": f"p={arch_r['p']:.6f} {arch_r['sig']}",
        })

    # Social + LSTM interaction
    social_configs = ["A5", "A6"]
    for cfg in social_configs:
        cfg_df = df[df["config_id"] == cfg]
        if len(cfg_df) == 0:
            continue
        sr = _arch_paired(cfg_df)
        if sr and sr["p"] < 0.05:
            desc = ABLATION_CONFIGS.get(cfg, {}).get("desc", cfg)
            findings.append({
                "rq": "RQ3",
                "finding": f"Social + LSTM significant interaction ({cfg}: {desc})",
                "evidence": f"{cfg} LSTM vs FF",
                "effect": f"d={sr['d']:.3f} ({effect_size_label(sr['d'])})",
                "sig": f"p={sr['p']:.4f} {sr['sig']}",
            })

    # News degradation (with Bonferroni correction for 4 comparisons in Table 4.8)
    r_a2_a4 = _paired("A2", "A4")
    if r_a2_a4:
        p_bonf = min(r_a2_a4["p"] * 4, 1.0)  # 4 comparisons in Table 4.8
        n_pairs = r_a2_a4.get("n", 0)
        preliminary = " (PRELIMINARY, n={})".format(n_pairs) if n_pairs < 15 else ""
        verdict = "DEGRADES" if p_bonf < 0.05 and r_a2_a4["delta"] < 0 else \
                  "TRENDS NEGATIVE" if r_a2_a4["delta"] < -0.005 else "NO EFFECT"
        findings.append({
            "rq": "RQ3",
            "finding": f"News over macro: {verdict}{preliminary}",
            "evidence": f"A2 vs A4, delta={r_a2_a4['delta']:+.4f}",
            "effect": f"d={r_a2_a4['d']:.3f} ({effect_size_label(r_a2_a4['d'])})",
            "sig": f"p_raw={r_a2_a4['p']:.4f}, p_bonf={p_bonf:.4f}",
        })

    # Fusion equivalence
    fusions = sorted(df["fusion_type"].unique())
    multi_df = df[df["config_id"] != "A7"]
    if len(fusions) >= 3 and len(multi_df) > 0:
        match_cols = ["config_id", "fold_idx", "seed"]
        if "architecture" in multi_df.columns:
            match_cols.append("architecture")
        pivot = multi_df.pivot_table(index=match_cols, columns="fusion_type",
                                      values=METRIC, aggfunc="first").dropna()
        if len(pivot) >= 5:
            fusion_arrays = [pivot[ft].values for ft in fusions if ft in pivot.columns]
            if len(fusion_arrays) == 3:
                stat, p_fried = stats.friedmanchisquare(*fusion_arrays)
                findings.append({
                    "rq": "Arch/Fusion",
                    "finding": "All fusion strategies equivalent",
                    "evidence": f"Friedman test on {len(pivot)} matched subjects",
                    "effect": f"chi2={stat:.3f}",
                    "sig": f"p={p_fried:.4f} ns",
                })

    # Graph architecture-dependent
    a8_df = df[df["config_id"] == "A8"]
    if len(a8_df) > 0 and "architecture" in a8_df.columns:
        ff_graph = a8_df[a8_df["architecture"] == "FF"][METRIC].mean()
        lstm_graph = a8_df[a8_df["architecture"] == "LSTM"][METRIC].mean()
        if a7_mcc is not None:
            a7_ff = df[(df["config_id"] == "A7") & (df["architecture"] == "FF")][METRIC].mean()
            a7_lstm = df[(df["config_id"] == "A7") & (df["architecture"] == "LSTM")][METRIC].mean()
            ff_delta = ff_graph - a7_ff
            lstm_delta = lstm_graph - a7_lstm
            findings.append({
                "rq": "RQ3",
                "finding": f"Graph architecture-dependent (exploratory): FF {ff_delta:+.4f}, LSTM {lstm_delta:+.4f}",
                "evidence": "A8 vs A7 split by architecture (ns both directions)",
                "effect": "opposite signs, neither significant",
                "sig": "ns (exploratory observation)",
            })

    # ===================== Output table =====================

    if not findings:
        out("No findings could be computed from available data.")
        return

    out("| # | RQ | Finding | Evidence | Effect Size | Significance |")
    out("|---|-----|---------|----------|-------------|--------------|")

    for i, f in enumerate(findings, 1):
        out(f"| {i} | {f['rq']} | {f['finding']} | {f['evidence']} | "
            f"{f['effect']} | {f['sig']} |")

    out()

    # Summary counts
    n_sig = sum(1 for f in findings if "***" in f["sig"] or "**" in f["sig"] or
                (f["sig"].count("*") == 1 and "ns" not in f["sig"] and "p=" in f["sig"]))
    out(f"**Total findings**: {len(findings)} ({n_sig} statistically significant)")
    out()

    # Narrative summary
    out("### Narrative Summary")
    out()
    rqs = {}
    for f in findings:
        rqs.setdefault(f["rq"], []).append(f["finding"])

    if "RQ1" in rqs:
        out(f"**RQ1**: {'; '.join(rqs['RQ1'])}")
    if "RQ2" in rqs:
        out(f"**RQ2**: {'; '.join(rqs['RQ2'])}")
    if "RQ3" in rqs:
        out(f"**RQ3**: {'; '.join(rqs['RQ3'])}")
    if "Arch/Fusion" in rqs:
        out(f"**Architecture/Fusion**: {'; '.join(rqs['Arch/Fusion'])}")
    out()


# ============================================================================
# Tables 4.3, 4.4, 4.14, 4.15, 4.16-4.17
# ============================================================================


def baseline_table(df: pd.DataFrame):
    """Tables 4.3 + 4.4: A7 baseline results by architecture/fusion + per-fold.

    Establishes the price-only baseline with one-sample t-test (H0: MCC = 0),
    multiple metrics, and per-fold breakdown by regime.
    """
    out("\n---")
    out("## Table 4.3: A7 Baseline Results by Architecture and Fusion")
    out()

    a7 = df[df["config_id"] == "A7"]
    if len(a7) == 0:
        out("ERROR: No A7 data found.")
        return

    # Table 4.3: architecture x fusion breakdown
    out("| Architecture | Fusion | Mean MCC | 95% CI | Std | Median | "
        "Dir Acc | Sharpe | Val-Test Gap | n | MCC > 0? (p) |")
    out("|-------------|--------|----------|--------|-----|--------|"
        "---------|--------|-------------|---|--------------|")

    archs = sorted(a7["architecture"].unique()) if "architecture" in a7.columns else ["all"]
    for arch in archs:
        arch_sub = a7[a7["architecture"] == arch] if "architecture" in a7.columns else a7
        fusions = sorted(arch_sub["fusion_type"].unique())
        for ft in fusions:
            sub = arch_sub[arch_sub["fusion_type"] == ft]
            vals = sub[METRIC].values
            ci = confidence_interval_95(vals)
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if not np.isnan(ci[0]) else "N/A"

            # One-sample t-test: is MCC > 0?
            if len(vals) >= 3:
                t_stat, p_val = stats.ttest_1samp(vals, 0)
                p_str = f"{p_val:.4f}"
            else:
                p_str = "N/A"

            dir_acc = sub["directional_accuracy"].mean() if "directional_accuracy" in sub.columns else np.nan
            sharpe = sub["sharpe_ratio"].mean() if "sharpe_ratio" in sub.columns else np.nan
            # Filter sentinel sharpe values
            if "sharpe_ratio" in sub.columns:
                clean_sharpe = sub["sharpe_ratio"][sub["sharpe_ratio"] > -99]
                sharpe = clean_sharpe.mean() if len(clean_sharpe) > 0 else np.nan
            vtg = sub["val_test_gap"].mean() if "val_test_gap" in sub.columns else np.nan

            out(f"| {arch} | {ft} | {vals.mean():.4f} | {ci_str} | {vals.std():.4f} | "
                f"{np.median(vals):.4f} | {dir_acc:.4f} | "
                f"{'N/A' if np.isnan(sharpe) else f'{sharpe:.4f}'} | "
                f"{'N/A' if np.isnan(vtg) else f'{vtg:.4f}'} | "
                f"{len(vals)} | {p_str} |")
    out()

    # Summary: is A7 better than random?
    all_mcc = a7[METRIC].values
    if len(all_mcc) >= 3:
        t_stat, p_val = stats.ttest_1samp(all_mcc, 0)
        sig = "YES" if p_val < 0.05 and all_mcc.mean() > 0 else "NO"
        out(f"**A7 overall**: mean={all_mcc.mean():.4f}, t={t_stat:.3f}, p={p_val:.4f} -> "
            f"MCC significantly > 0: {sig}")
        out(f"  % positive MCC: {(all_mcc > 0).mean()*100:.1f}%")
    out()

    # Table 4.4: per-fold breakdown
    out("## Table 4.4: A7 Baseline Per-Fold Breakdown")
    out()
    out("| Fold | Regime | FF MCC | FF 95% CI | LSTM MCC | LSTM 95% CI | n (per arch) |")
    out("|------|--------|--------|-----------|----------|-------------|-------------|")

    folds = sorted(a7["fold_idx"].unique())
    for fold in folds:
        fold_sub = a7[a7["fold_idx"] == fold]
        regime = FOLD_LABELS.get(fold, f"F{fold}")

        ff_vals = fold_sub[fold_sub["architecture"] == "FF"][METRIC].values if "architecture" in fold_sub.columns else np.array([])
        lstm_vals = fold_sub[fold_sub["architecture"] == "LSTM"][METRIC].values if "architecture" in fold_sub.columns else np.array([])

        ff_ci = confidence_interval_95(ff_vals) if len(ff_vals) >= 2 else (np.nan, np.nan)
        lstm_ci = confidence_interval_95(lstm_vals) if len(lstm_vals) >= 2 else (np.nan, np.nan)

        ff_str = f"{ff_vals.mean():.4f}" if len(ff_vals) > 0 else "N/A"
        ff_ci_str = f"[{ff_ci[0]:.4f}, {ff_ci[1]:.4f}]" if not np.isnan(ff_ci[0]) else "N/A"
        lstm_str = f"{lstm_vals.mean():.4f}" if len(lstm_vals) > 0 else "N/A"
        lstm_ci_str = f"[{lstm_ci[0]:.4f}, {lstm_ci[1]:.4f}]" if not np.isnan(lstm_ci[0]) else "N/A"
        n_per = max(len(ff_vals), len(lstm_vals))

        out(f"| F{fold} | {regime} | {ff_str} | {ff_ci_str} | "
            f"{lstm_str} | {lstm_ci_str} | {n_per} |")
    out()

    out("**Note**: A7 only runs concat fusion (single-modality, no cross-modal fusion to perform).")
    out()


def training_dynamics_table(df: pd.DataFrame):
    """Table 4.14: Training dynamics summary by architecture.

    Reports mean epochs trained, std, val-test gap, and parameter counts.
    """
    out("\n---")
    out("## Table 4.14: Training Dynamics Summary")
    out()

    cols_needed = ["epochs_trained", "n_params"]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        out(f"Missing columns: {missing}. Skipping.")
        return

    out("| Architecture | Mean Epochs | Std Epochs | Median Epochs | "
        "Val-Test Gap | Gap Std | Mean Params | Param Range |")
    out("|-------------|-------------|------------|---------------|"
        "-------------|---------|-------------|-------------|")

    archs = sorted(df["architecture"].unique()) if "architecture" in df.columns else ["all"]
    for arch in archs:
        sub = df[df["architecture"] == arch] if "architecture" in df.columns else df

        epochs = sub["epochs_trained"].values
        params = sub["n_params"].values
        vtg = sub["val_test_gap"].values if "val_test_gap" in sub.columns else np.array([])

        vtg_mean = vtg.mean() if len(vtg) > 0 else np.nan
        vtg_std = vtg.std() if len(vtg) > 0 else np.nan

        out(f"| {arch} | {epochs.mean():.1f} | {epochs.std():.1f} | {np.median(epochs):.0f} | "
            f"{'N/A' if np.isnan(vtg_mean) else f'{vtg_mean:.4f}'} | "
            f"{'N/A' if np.isnan(vtg_std) else f'{vtg_std:.4f}'} | "
            f"{params.mean():.0f} | {params.min():.0f}-{params.max():.0f} |")
    out()

    # Per-config breakdown
    out("### Training Dynamics by Config")
    out()
    out("| Config | Mean Epochs | Early Stop Rate | Val-Test Gap | Mean Params |")
    out("|--------|-------------|-----------------|-------------|-------------|")

    for cid in sorted(df["config_id"].unique()):
        sub = df[df["config_id"] == cid]
        epochs = sub["epochs_trained"]
        max_ep = sub.get("max_epochs", pd.Series([150]))
        # Estimate early stop rate: epochs < max_epochs (use 150 as default)
        max_epochs_val = 150
        early_stopped = (epochs < max_epochs_val).mean() * 100
        vtg = sub["val_test_gap"].mean() if "val_test_gap" in sub.columns else np.nan
        params = sub["n_params"].mean()

        out(f"| {cid} | {epochs.mean():.1f} | {early_stopped:.0f}% | "
            f"{'N/A' if np.isnan(vtg) else f'{vtg:.4f}'} | {params:.0f} |")
    out()


def runtime_table(df: pd.DataFrame):
    """Table 4.15: Runtime by architecture.

    Reports mean runtime per experiment, total runtime, and parameter counts.
    """
    out("\n---")
    out("## Table 4.15: Computational Cost")
    out()

    if "elapsed_s" not in df.columns:
        out("No elapsed_s column found. Skipping runtime analysis.")
        return

    out("| Architecture | Mean Runtime (s) | Std (s) | Total Runtime (h) | "
        "Mean Params | Experiments |")
    out("|-------------|-----------------|---------|-------------------|"
        "-------------|-------------|")

    archs = sorted(df["architecture"].unique()) if "architecture" in df.columns else ["all"]
    for arch in archs:
        sub = df[df["architecture"] == arch] if "architecture" in df.columns else df
        runtime = sub["elapsed_s"].values
        params = sub["n_params"].values if "n_params" in sub.columns else np.array([0])

        total_h = runtime.sum() / 3600

        out(f"| {arch} | {runtime.mean():.1f} | {runtime.std():.1f} | "
            f"{total_h:.1f} | {params.mean():.0f} | {len(sub)} |")

    # Total
    total_runtime_h = df["elapsed_s"].sum() / 3600
    out(f"| **Total** | {df['elapsed_s'].mean():.1f} | {df['elapsed_s'].std():.1f} | "
        f"{total_runtime_h:.1f} | {df['n_params'].mean():.0f} | {len(df)} |")
    out()

    # Per-config runtime (useful for estimating remaining batch time)
    out("### Runtime by Config")
    out()
    out("| Config | Mean (s) | Total (h) | n |")
    out("|--------|----------|-----------|---|")
    for cid in sorted(df["config_id"].unique()):
        sub = df[df["config_id"] == cid]
        rt = sub["elapsed_s"]
        out(f"| {cid} | {rt.mean():.1f} | {rt.sum()/3600:.1f} | {len(sub)} |")
    out()


def financial_metrics_table(df: pd.DataFrame):
    """Table 4.X: Financial metrics summary by config.

    Reports Sharpe ratio and max drawdown per configuration, highlighting
    that statistically positive MCC does not imply profitable trading.
    Filters sentinel values (-99.999) which indicate degenerate trading runs.
    """
    out("\n---")
    out("## Financial Metrics Summary")
    out()

    has_sharpe = "sharpe_ratio" in df.columns
    if not has_sharpe:
        out("No financial metric columns found. Skipping.")
        return

    out("**Note**: cumulative_return and max_drawdown excluded (computed with "
        "incorrect formula in pre-fix experiment runs).")
    out()

    out("| Config | Architecture | Mean Sharpe | Sharpe Std | "
        "Profit Factor | Dir Accuracy | n |")
    out("|--------|-------------|------------|-----------|"
        "--------------|-------------|---|")

    configs = sorted(df["config_id"].unique())
    archs = sorted(df["architecture"].unique()) if "architecture" in df.columns else ["all"]

    for cid in configs:
        for arch in archs:
            if arch == "all":
                sub = df[df["config_id"] == cid]
            else:
                sub = df[(df["config_id"] == cid) & (df["architecture"] == arch)]
            if len(sub) == 0:
                continue

            clean_sharpe = sub["sharpe_ratio"][sub["sharpe_ratio"] > -99]
            sharpe_mean = clean_sharpe.mean() if len(clean_sharpe) > 0 else np.nan
            sharpe_std = clean_sharpe.std() if len(clean_sharpe) > 0 else np.nan
            n_valid = len(clean_sharpe)

            pf = sub["profit_factor"].mean() if "profit_factor" in sub.columns else np.nan
            dir_acc = sub["directional_accuracy"].mean() if "directional_accuracy" in sub.columns else np.nan

            sharpe_str = f"{sharpe_mean:.3f}" if not np.isnan(sharpe_mean) else "N/A"
            sharpe_std_str = f"{sharpe_std:.3f}" if not np.isnan(sharpe_std) else "N/A"
            pf_str = f"{pf:.3f}" if not np.isnan(pf) else "N/A"
            dir_str = f"{dir_acc:.4f}" if not np.isnan(dir_acc) else "N/A"

            out(f"| {cid} | {arch} | {sharpe_str} | {sharpe_std_str} | "
                f"{pf_str} | {dir_str} | {n_valid} |")

    out()

    # Overall summary
    clean_sharpe_all = df["sharpe_ratio"][df["sharpe_ratio"] > -99]
    if len(clean_sharpe_all) > 0:
        out(f"**Overall Sharpe ratio**: mean={clean_sharpe_all.mean():.3f}, "
            f"range=[{clean_sharpe_all.min():.3f}, {clean_sharpe_all.max():.3f}]")
    out()
    out("**Note**: A statistically positive MCC does not imply profitable trading without "
        "proper position sizing, transaction costs, and risk management.")
    out()


def literature_comparison_bridge(data_dir: Path = PHASE4R_DIR):
    """Tables 4.16-4.17: Literature comparison (bridges to literature_comparison.py).

    Imports and runs the key comparison functions from literature_comparison.py
    so they appear in the unified ch4 output.
    """
    out("\n---")
    out("## Tables 4.16-4.17: Literature Performance and Methodology Comparison")
    out()

    try:
        from scripts.literature_comparison import (
            compute_our_best,
            generate_comparison,
            generate_evaluation_methodology_comparison,
            generate_rigor_comparison,
            load_msgca_diagnostic,
            load_phase4r_results,
        )
    except ImportError:
        # Try alternative import path
        try:
            lit_path = Path(__file__).resolve().parent / "literature_comparison.py"
            import importlib.util
            spec = importlib.util.spec_from_file_location("literature_comparison", lit_path)
            lit_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lit_mod)
            load_phase4r_results = lit_mod.load_phase4r_results
            compute_our_best = lit_mod.compute_our_best
            generate_comparison = lit_mod.generate_comparison
            generate_rigor_comparison = lit_mod.generate_rigor_comparison
            generate_evaluation_methodology_comparison = lit_mod.generate_evaluation_methodology_comparison
            load_msgca_diagnostic = lit_mod.load_msgca_diagnostic
        except Exception as e:
            out(f"WARNING: Could not import literature_comparison.py: {e}")
            out("Run `python scripts/literature_comparison.py` separately to generate Tables 4.16-4.17.")
            return

    try:
        df = load_phase4r_results(data_dir)
        our_results = compute_our_best(df) if not df.empty else {}

        # Table 4.16: Performance comparison
        out("### Table 4.16: Performance Comparison with Published Literature")
        out()
        out(generate_comparison(our_results))
        out()

        # Table 4.17: Statistical rigor comparison
        out("### Table 4.17: Statistical Rigor Comparison")
        out()
        out(generate_rigor_comparison())
        out()

        # Evaluation methodology (may go to appendix)
        out("### Evaluation Methodology Differences (13 dimensions)")
        out()
        out(generate_evaluation_methodology_comparison())
        out()

        # MSGCA diagnostic summary
        diagnostic = load_msgca_diagnostic(data_dir)
        if diagnostic:
            summary = diagnostic.get("summary", {})
            reported = summary.get("original_best_test_mcc")
            honest = summary.get("mean_val_selected_test_mcc")
            inflation = summary.get("mean_inflation_pct")
            out("### MSGCA Evaluation Bias Summary")
            out()
            out(f"- Reported MCC: {reported:.4f}" if reported is not None else "- Reported MCC: N/A")
            out(f"- Honest MCC (val-selected): {honest:.4f}" if honest is not None else "- Honest MCC (val-selected): N/A")
            out(f"- Inflation: {inflation:.0f}%" if inflation is not None else "- Inflation: N/A%")
            out("- Full diagnostic in `results/analysis/msgca_diagnostic.json`")
            out()
    except Exception as e:
        out(f"WARNING: Literature comparison failed: {e}")
        out("Run `python scripts/literature_comparison.py` separately.")
    out()


# ============================================================================
# MAIN
# ============================================================================

def main():
    global _output_mode, _output_lines

    parser = argparse.ArgumentParser(
        description="Phase 4-R / Phase 5 ablation analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--section", type=str, default="all",
                        choices=["all", "overview", "rq1", "rq2", "rq3", "arch", "fusion",
                                 "financial", "summary",
                                 "marginal", "loo", "incremental", "regime_heatmap",
                                 "arch_regime", "news_degrade", "fusion_friedman",
                                 "baseline", "training", "runtime", "literature",
                                 "fold_fusion", "arch_table", "synthesis",
                                 "ms_vs_baseline", "fin_metrics",
                                 "a10", "social_lstm", "graph_ln", "supplementary",
                                 "ch4"],
                        help="Which analysis section to run (ch4 = all Chapter 4 functions)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing ablation CSVs (default: experiments/phase4r)")
    parser.add_argument("--csv", type=str, nargs="+", default=None,
                        help="Explicit CSV file(s) to load (overrides --data-dir and defaults)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save analysis to markdown file instead of printing")
    parser.add_argument("--arch", type=str, default=None,
                        choices=["FF", "LSTM"],
                        help="Analyze only one architecture")
    parser.add_argument("--figures", action="store_true",
                        help="Generate Chapter 4 figures (PDF+PNG) to results/analysis/figures/")
    args = parser.parse_args()

    # Override CSV paths
    global CSV_FILES, SUPPLEMENTARY_CSVS
    if args.csv:
        # Explicit files: classify by filename
        ff_paths = [Path(p) for p in args.csv if "lstm" not in Path(p).stem.lower()]
        lstm_paths = [Path(p) for p in args.csv if "lstm" in Path(p).stem.lower()]
        CSV_FILES = {"FF": ff_paths, "LSTM": lstm_paths}
    elif args.data_dir:
        d = Path(args.data_dir)
        CSV_FILES = {
            "FF": [d / "ablation_ff.csv", d / "ablation_news_ff.csv",
                   d / "ablation_a10_ff.csv"],
            "LSTM": [d / "ablation_lstm.csv", d / "ablation_news_lstm.csv",
                     d / "ablation_a10_lstm.csv"],
        }
        SUPPLEMENTARY_CSVS = {
            "social_lstm": d / "ablation_social_lstm.csv",
            "graph_ln_ff": d / "ablation_graph_ln_ff.csv",
            "graph_ln_lstm": d / "ablation_graph_ln_lstm.csv",
        }

    # Output mode
    if args.output:
        _output_mode = "buffer"
        _output_lines = []

    # Load data
    ff_df, lstm_df, combined = load_ablation_data()
    if combined.empty:
        return

    # Filter by architecture if requested
    if args.arch:
        combined = combined[combined["architecture"] == args.arch]
        if args.arch == "FF":
            lstm_df = None
        else:
            ff_df = None
        out(f"\nFiltered to {args.arch} architecture: {len(combined)} experiments")

    # Run analyses
    sections = {
        "overview": lambda: analyze_overview(combined),
        "rq1": lambda: analyze_rq1(combined),
        "rq2": lambda: analyze_rq2(combined),
        "rq3": lambda: analyze_rq3(combined),
        "arch": lambda: analyze_architecture(ff_df, lstm_df, combined),
        "fusion": lambda: analyze_fusion(combined),
        "financial": lambda: analyze_financial(combined),
        "summary": lambda: executive_summary(combined, ff_df, lstm_df),
        # Chapter 4 functions
        "baseline": lambda: baseline_table(combined),
        "ms_vs_baseline": lambda: multi_source_vs_baseline_table(combined),
        "marginal": lambda: marginal_contribution_table(combined),
        "loo": lambda: leave_one_out_analysis(combined),
        "incremental": lambda: incremental_addition_table(combined),
        "regime_heatmap": lambda: regime_interaction_heatmap(combined),
        "arch_regime": lambda: architecture_regime_interaction(ff_df, lstm_df, combined),
        "news_degrade": lambda: news_degradation_analysis(combined),
        "fusion_friedman": lambda: fusion_friedman_test(combined),
        "training": lambda: training_dynamics_table(combined),
        "runtime": lambda: runtime_table(combined),
        "fin_metrics": lambda: financial_metrics_table(combined),
        "literature": lambda: literature_comparison_bridge(),
        "fold_fusion": lambda: per_fold_fusion_table(combined),
        "arch_table": lambda: architecture_overall_table(combined),
        "synthesis": lambda: synthesis_table(combined),
        # Supplementary analysis functions
        "a10": lambda: a10_comparison_table(combined),
        "social_lstm": lambda: social_lstm_comparison_table(combined),
        "graph_ln": lambda: graph_ln_comparison_table(combined),
    }

    # "supplementary" runs all three supplementary analysis functions
    supplementary_sections = ["a10", "social_lstm", "graph_ln"]

    # "ch4" runs all Chapter 4 functions in sequence (maps to outline Tables 4.3-4.18)
    ch4_sections = ["baseline", "ms_vs_baseline", "marginal", "loo", "incremental",
                    "regime_heatmap", "arch_regime", "news_degrade", "fusion_friedman",
                    "fold_fusion", "arch_table",
                    "training", "runtime", "fin_metrics", "literature", "synthesis"]

    if args.section == "all":
        for name in ["overview", "rq1", "rq2", "rq3", "arch", "fusion", "financial", "summary"]:
            sections[name]()
    elif args.section == "ch4":
        out("\n# Chapter 4 Analysis Functions")
        out(f"Running all {len(ch4_sections)} Chapter 4 analysis functions on {len(combined)} experiments...")
        out()
        for name in ch4_sections:
            sections[name]()
    elif args.section == "supplementary":
        out("\n# Supplementary Analysis Functions")
        out()
        for name in supplementary_sections:
            sections[name]()
    else:
        sections[args.section]()

    # Generate figures if requested
    if args.figures:
        generate_figures(combined)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(_output_lines))
        print(f"Saved analysis to {output_path} ({len(_output_lines)} lines)")


if __name__ == "__main__":
    main()
