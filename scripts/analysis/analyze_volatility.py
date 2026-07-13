"""Comprehensive analysis of volatility prediction experiments (750 runs).

Mirrors the direction prediction analysis framework:
- Config vs baseline (A7) with Bonferroni correction
- Architecture comparison (FF vs LSTM)
- Fusion strategy comparison (Friedman test)
- Per-fold regime breakdown
- Cross-target comparison (direction vs volatility)

Usage (run from the artifact root; reads results/volatility/):
    python scripts/analysis/analyze_volatility.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


def _vol_dir() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "results" / "volatility").is_dir():
            return p / "results" / "volatility"
    return Path("results/volatility")  # CWD-relative fallback


PHASE4R = _vol_dir()


def load_and_merge():
    """Load all 4 volatility CSVs and merge into consolidated files."""
    # Phase 1 (concat only)
    concat_ff = pd.read_csv(PHASE4R / "volatility_concat_ff.csv")
    concat_lstm = pd.read_csv(PHASE4R / "volatility_concat_lstm.csv")

    # Phase 2 (gated_cross_attention + mha)
    full_ff = pd.read_csv(PHASE4R / "volatility_full_ff.csv")
    full_lstm = pd.read_csv(PHASE4R / "volatility_full_lstm.csv")

    # Merge
    ff_all = pd.concat([concat_ff, full_ff], ignore_index=True)
    lstm_all = pd.concat([concat_lstm, full_lstm], ignore_index=True)

    # The merged files already ship in results/volatility/; do not overwrite the
    # committed evidence. (Re-derive into the CWD only if you want a fresh copy.)

    both = pd.concat([ff_all, lstm_all], ignore_index=True)
    print(f"Merged: {len(ff_all)} FF + {len(lstm_all)} LSTM = {len(both)} total")
    print(f"Configs: {sorted(both.config_id.unique())}")
    print(f"Fusions: {sorted(both.fusion_type.unique())}")
    print(f"Seeds: {sorted(both.seed.unique())}")
    print(f"Folds: {sorted(both.fold_idx.unique())}")
    print(f"Lookbacks: {sorted(both.lookback.unique())}")
    return both


def config_vs_baseline(df):
    """Paired t-tests: each config vs A7 baseline on RMSE."""
    print("\n" + "=" * 70)
    print("CONFIG VS BASELINE (A7) — Paired t-tests on RMSE")
    print("=" * 70)

    a7 = df[df.config_id == "A7"]
    configs = sorted(df.config_id.unique())
    k = len([c for c in configs if c != "A7"])  # number of comparisons for Bonferroni

    results = []
    for cid in configs:
        if cid == "A7":
            mean_rmse = a7.vol_rmse.mean()
            mean_r2 = a7.vol_r2.mean()
            results.append({
                "config": cid, "mean_rmse": mean_rmse, "mean_r2": mean_r2,
                "delta": 0, "p_raw": np.nan, "p_bonf": np.nan, "d": np.nan, "n": len(a7)
            })
            continue

        cfg = df[df.config_id == cid]
        merged = cfg.merge(a7, on=["fold_idx", "seed", "lookback", "fusion_type"],
                          suffixes=("_cfg", "_a7"))

        if len(merged) == 0:
            # A7 only has concat — merge without fusion_type for non-concat configs
            merged = cfg.merge(a7, on=["fold_idx", "seed", "lookback"],
                              suffixes=("_cfg", "_a7"))

        delta = merged["vol_rmse_cfg"] - merged["vol_rmse_a7"]
        t_stat, p_val = stats.ttest_rel(merged["vol_rmse_cfg"], merged["vol_rmse_a7"])
        d = delta.mean() / delta.std() if delta.std() > 0 else 0
        p_bonf = min(p_val * k, 1.0)

        results.append({
            "config": cid,
            "mean_rmse": cfg.vol_rmse.mean(),
            "mean_r2": cfg.vol_r2.mean(),
            "delta": delta.mean(),
            "p_raw": p_val,
            "p_bonf": p_bonf,
            "d": d,
            "n": len(merged)
        })

    results_df = pd.DataFrame(results).sort_values("mean_rmse")
    print(f"\nBaseline A7 RMSE: {a7.vol_rmse.mean():.6f}")
    print(f"Bonferroni k={k}\n")
    print(f"{'Config':<8} {'RMSE':>10} {'R2':>8} {'Delta':>10} {'p_raw':>8} {'p_bonf':>8} {'d':>8} {'n':>5}")
    print("-" * 70)
    for _, r in results_df.iterrows():
        sign = "+" if r["delta"] > 0 else ""
        print(f"{r['config']:<8} {r['mean_rmse']:>10.6f} {r['mean_r2']:>8.4f} "
              f"{sign}{r['delta']:>9.6f} {r['p_raw']:>8.4f} {r['p_bonf']:>8.4f} {r['d']:>8.3f} {r['n']:>5.0f}")

    return results_df


def architecture_comparison(df):
    """Compare FF vs LSTM on volatility prediction."""
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON — FF vs LSTM")
    print("=" * 70)

    ff = df[df.lookback == 1]
    lstm = df[df.lookback == 20]

    merged = ff.merge(lstm, on=["config_id", "fusion_type", "fold_idx", "seed"],
                     suffixes=("_ff", "_lstm"))

    delta = merged["vol_rmse_ff"] - merged["vol_rmse_lstm"]
    t_stat, p_val = stats.ttest_rel(merged["vol_rmse_ff"], merged["vol_rmse_lstm"])
    d = delta.mean() / delta.std() if delta.std() > 0 else 0

    print(f"\nFF mean RMSE:   {merged.vol_rmse_ff.mean():.6f}")
    print(f"LSTM mean RMSE: {merged.vol_rmse_lstm.mean():.6f}")
    better = "FF" if delta.mean() < 0 else "LSTM"
    print(f"Delta (FF-LSTM): {delta.mean():.6f} ({better} better)")
    print(f"p-value: {p_val:.4f}, Cohen's d: {d:.3f}, n={len(merged)}")

    # Per-config architecture comparison
    print(f"\n{'Config':<8} {'FF RMSE':>10} {'LSTM RMSE':>10} {'Delta':>10} {'p':>8}")
    print("-" * 50)
    for cid in sorted(merged.config_id_ff.unique() if "config_id_ff" in merged.columns
                      else merged.config_id.unique()):
        col = "config_id_ff" if "config_id_ff" in merged.columns else "config_id"
        sub = merged[merged[col] == cid]
        d_sub = sub["vol_rmse_ff"] - sub["vol_rmse_lstm"]
        if len(sub) > 1:
            _, p = stats.ttest_rel(sub["vol_rmse_ff"], sub["vol_rmse_lstm"])
        else:
            p = np.nan
        print(f"{cid:<8} {sub.vol_rmse_ff.mean():>10.6f} {sub.vol_rmse_lstm.mean():>10.6f} "
              f"{d_sub.mean():>10.6f} {p:>8.4f}")


def fusion_comparison(df):
    """Friedman test across fusion strategies."""
    print("\n" + "=" * 70)
    print("FUSION STRATEGY COMPARISON — Friedman test")
    print("=" * 70)

    # Only multi-source configs (A7 has only concat)
    multi = df[df.config_id != "A7"]
    fusions = sorted(multi.fusion_type.unique())

    if len(fusions) < 3:
        print(f"\nOnly {len(fusions)} fusion types found. Need 3 for Friedman test.")
        print("Phase 2 may not be complete yet.")
        return

    # Create wide format: each row = (config, fold, seed, lookback), columns = fusion RMSE
    pivot = multi.pivot_table(
        index=["config_id", "fold_idx", "seed", "lookback"],
        columns="fusion_type",
        values="vol_rmse"
    ).dropna()

    print(f"\nSubjects (matched across all 3 fusions): {len(pivot)}")
    print(f"Fusions: {list(pivot.columns)}")

    for f in pivot.columns:
        print(f"  {f}: mean RMSE = {pivot[f].mean():.6f}")

    if len(pivot) > 0 and len(pivot.columns) >= 3:
        chi2, p = stats.friedmanchisquare(*[pivot[f] for f in pivot.columns])
        print(f"\nFriedman chi2 = {chi2:.3f}, p = {p:.4f}")
        if p < 0.05:
            print("  SIGNIFICANT — fusion strategies differ for volatility")
        else:
            print("  Not significant — fusion strategies equivalent for volatility")
    else:
        print("\nInsufficient data for Friedman test")


def per_fold_breakdown(df):
    """Per-fold RMSE breakdown."""
    print("\n" + "=" * 70)
    print("PER-FOLD BREAKDOWN")
    print("=" * 70)

    regimes = {0: "Pre-COVID", 1: "Recovery", 2: "Bull-Bear", 3: "Bear", 4: "AI Rally"}

    # A7 baseline per fold
    a7 = df[df.config_id == "A7"]
    print(f"\n{'Fold':<6} {'Regime':<12} {'A7 RMSE':>10} {'A7 R2':>8} {'Best Config':>12} {'Best RMSE':>10}")
    print("-" * 62)

    for fold in sorted(df.fold_idx.unique()):
        fold_data = df[df.fold_idx == fold]
        a7_fold = a7[a7.fold_idx == fold]

        config_means = fold_data.groupby("config_id")["vol_rmse"].mean()
        best_config = config_means.idxmin()
        best_rmse = config_means.min()

        a7_rmse = a7_fold.vol_rmse.mean()
        a7_r2 = a7_fold.vol_r2.mean()

        regime = regimes.get(fold, "Unknown")
        print(f"F{fold:<5} {regime:<12} {a7_rmse:>10.6f} {a7_r2:>8.4f} {best_config:>12} {best_rmse:>10.6f}")


def overall_summary(df):
    """Print overall summary statistics."""
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    print(f"\nTotal runs: {len(df)}")
    print(f"FF: {len(df[df.lookback == 1])}, LSTM: {len(df[df.lookback == 20])}")
    print(f"Overall RMSE: {df.vol_rmse.mean():.6f} (+/- {df.vol_rmse.std():.6f})")
    print(f"Overall MAE:  {df.vol_mae.mean():.6f}")
    print(f"Overall R2:   {df.vol_r2.mean():.4f}")
    print(f"All R2 negative: {(df.vol_r2 < 0).all()}")
    print(f"Target type check: {df.target_type.unique()}")


def main():
    print("VOLATILITY PREDICTION ANALYSIS — Full 750-run ablation")
    print("=" * 70)

    df = load_and_merge()
    overall_summary(df)
    config_vs_baseline(df)
    architecture_comparison(df)
    fusion_comparison(df)
    per_fold_breakdown(df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
