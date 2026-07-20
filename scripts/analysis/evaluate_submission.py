#!/usr/bin/env python3
"""Compute the benchmark claim block for a submission (see SUBMITTING.md).

Pairs a challenger's per-(fold, seed) test MCC against the shipped tuned
price-only baseline (natcore_A7) and reports the statistics the protocol
requires: per-fold mean deltas (the honest unit, n=5), fold-block bootstrap
95% CI, p_fold, Bonferroni-corrected p at the declared family size k, and
the descriptive pooled effect size. Statistics mirror
make_reference_table_v2.py (same bootstrap: 10,000 resamples, seed 42).

Usage:
    evaluate_submission.py SUBMISSION.csv [--k 1]
        [--baseline-arch {ff,lstm,envelope}] [--restrict-folds 0,1,2,3]
        [--json OUT.json]

The default baseline is the per-(fold, seed) upper envelope of the two
shipped arms (max over ff/lstm): a conservative bar, not a runnable
model; pass --baseline-arch ff|lstm for like-for-like pairing. With
--restrict-folds (coverage rule 8), certification requires the declared
fold subset with at least four folds, and the verdict is tagged
RESTRICTED COVERAGE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
BASELINE_GLOB = "results/native_core/native_core_*.csv"
BASELINE_PREFIX = "natcore_A7_"
SEEDS = (42, 123, 456)
N_BOOT, BOOT_SEED = 10_000, 42


def load_baseline(arch: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(ROOT.glob(BASELINE_GLOB))]
    base = pd.concat(frames, ignore_index=True)
    base = base[base["experiment_name"].str.startswith(BASELINE_PREFIX)]
    # architecture is carried by price_encoder (feedforward|lstm);
    # head_architecture is the multitask head, identical across arms
    base = base.rename(columns={"price_encoder": "arch"})
    base["arch"] = base["arch"].map({"feedforward": "ff", "lstm": "lstm"})
    if arch in ("ff", "lstm"):
        base = base[base["arch"] == arch]
        return base[["fold_idx", "seed", "mcc"]]
    # conservative default: the per-(fold, seed) upper ENVELOPE of the two
    # shipped arms (max over ff/lstm) -- a test-selected bar, not a
    # runnable model
    return (base.groupby(["fold_idx", "seed"], as_index=False)["mcc"].max())


def expected_n_test() -> dict[int, int]:
    """Per-fold test-set sizes implied by the frozen folds (from the
    shipped baseline rows; identical for every conforming assembly)."""
    frames = [pd.read_csv(p) for p in sorted(ROOT.glob(BASELINE_GLOB))]
    base = pd.concat(frames, ignore_index=True)
    base = base[base["experiment_name"].str.startswith(BASELINE_PREFIX)]
    return base.groupby("fold_idx")["n_test"].agg(
        lambda s: int(s.mode().iloc[0])).to_dict()


def fold_block_ci(cells: pd.DataFrame) -> tuple[float, float]:
    """Percentile CI of the mean delta, resampling folds (blocks)."""
    rng = np.random.default_rng(BOOT_SEED)
    folds = sorted(cells["fold_idx"].unique())
    by_fold = {f: cells.loc[cells["fold_idx"] == f, "delta"].values for f in folds}
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(folds, size=len(folds), replace=True)
        means[i] = np.concatenate([by_fold[f] for f in pick]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    ap.add_argument("--k", type=int, default=None,
                    help="declared family size (every configuration compared, "
                         "rule 3); omitting it defaults to 1 with a warning")
    ap.add_argument("--baseline-arch", default="envelope",
                    choices=["ff", "lstm", "envelope", "stronger"],
                    help="ff|lstm = like-for-like; envelope (default) = "
                         "per-cell max over both shipped arms ('stronger' "
                         "is a deprecated alias for envelope)")
    ap.add_argument("--restrict-folds", default=None,
                    help="comma-separated fold_idx subset (coverage rule 8)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    if a.baseline_arch == "stronger":
        a.baseline_arch = "envelope"
    k_declared = a.k is not None
    if not k_declared:
        a.k = 1

    sub = pd.read_csv(a.submission)
    need = {"challenger", "fold_idx", "seed", "mcc"}
    if not need.issubset(sub.columns):
        sys.exit(f"submission must have columns {sorted(need)}")
    if sub.duplicated(subset=["fold_idx", "seed"]).any():
        sys.exit("submission has duplicate (fold_idx, seed) rows -- the "
                 "contract is one row per (fold, seed); see SUBMITTING.md")
    name = sub["challenger"].iloc[0]
    base = load_baseline(a.baseline_arch)

    merged = sub.merge(base, on=["fold_idx", "seed"], suffixes=("_sub", "_base"))
    seed_matched = True
    submitted_folds = {int(f) for f in sub["fold_idx"].unique()}
    if merged.empty or {int(f) for f in merged["fold_idx"]} != submitted_folds:
        # fold-level fallback: pair fold means (rule 1 prefers seeds 42/123/456)
        seed_matched = False
        s = sub.groupby("fold_idx")["mcc"].mean()
        b = base.groupby("fold_idx")["mcc"].mean()
        merged = pd.DataFrame({"fold_idx": s.index, "mcc_sub": s.values,
                               "mcc_base": b.reindex(s.index).values,
                               "seed": -1})
    merged["delta"] = merged["mcc_sub"] - merged["mcc_base"]
    if a.restrict_folds:
        keep = [int(x) for x in a.restrict_folds.split(",")]
        merged = merged[merged["fold_idx"].isin(keep)]

    fold_means = merged.groupby("fold_idx")["delta"].mean()
    n_folds = len(fold_means)
    # fold-weighted headline, consistent with p_fold and the fold-t CI
    # (identical to the cell mean on a balanced submission)
    mean_delta = float(fold_means.mean())
    t, p_fold = stats.ttest_1samp(fold_means, 0.0)
    p_bonf = min(1.0, float(p_fold) * a.k)
    lo, hi = fold_block_ci(merged)
    # Fold-mean t interval: the certifying scale. A five-block percentile
    # bootstrap undercovers at n=5 (roughly 84% actual coverage), so its
    # exclusions are descriptive; this interval is the one consistent with
    # p_fold and the verdict logic.
    if n_folds > 1:
        t_lo, t_hi = stats.t.interval(
            0.95, n_folds - 1,
            loc=float(fold_means.mean()), scale=float(fold_means.sem()))
    else:
        t_lo = t_hi = float("nan")
    sd = merged["delta"].std(ddof=1)
    d = float(mean_delta / sd) if len(merged) > 1 and sd > 0 else float("nan")

    conforms = None
    if "n_test" in sub.columns:
        exp = expected_n_test()
        # check EVERY row per fold, not just the first — a submission whose
        # non-first rows carry a wrong n_test must not pass conformance
        bad = {int(f): (sorted({int(x) for x in g}), exp.get(int(f)))
               for f, g in sub.groupby("fold_idx")["n_test"]
               if exp.get(int(f)) is not None
               and any(int(x) != exp[int(f)] for x in g)}
        conforms = not bad

    # coverage gate: the full frozen grid by default; with --restrict-folds
    # (rule 8), the declared subset with at least four folds certifies and
    # the verdict is tagged as restricted
    required = ({int(x) for x in a.restrict_folds.split(",")}
                if a.restrict_folds else set(range(5)))
    covered = {int(f) for f in fold_means.index}
    coverage_ok = covered == required and n_folds >= (4 if a.restrict_folds else 5)
    # seed-contract gate: certification requires the three contract seeds
    # (42/123/456) paired at seed level in EVERY covered fold (rule 1) --
    # a best-seed-per-fold selection must not certify through the tool
    # whose purpose is policing selection
    if seed_matched:
        seeds_by_fold = merged.groupby("fold_idx")["seed"].agg(
            lambda s: set(SEEDS).issubset({int(x) for x in s}))
        seeds_ok = bool(seeds_by_fold.all())
    else:
        seeds_ok = False
    supported = mean_delta > 0 and p_bonf < 0.05 and coverage_ok and seeds_ok
    verdict = "SUPPORTED" if supported else "WITHIN THE REFERENCE NULL"
    if a.restrict_folds:
        verdict += f"  [RESTRICTED COVERAGE -- folds {sorted(covered)} per rule 8]"
    if not seeds_ok:
        verdict += ("  [SEED CONTRACT NOT MET -- certification requires seeds "
                    "42/123/456 paired per covered fold (rule 1)]")
    if conforms is False:
        supported = False
        verdict = ("NOT COMPARABLE  [ASSEMBLY MISMATCH -- n_test disagrees "
                   "with the frozen folds]")

    arm_label = ("envelope of the ff/lstm arms -- per-cell max; a "
                 "conservative bar, not a runnable model"
                 if a.baseline_arch == "envelope" else f"{a.baseline_arch} arm")
    print(f"challenger        : {name}")
    print(f"baseline          : shipped tuned price-only "
          f"({arm_label}){'' if seed_matched else '  [fold-mean pairing: seeds not matched]'}")
    print(f"cells paired      : {len(merged)}  "
          f"(folds: {[int(f) for f in sorted(merged['fold_idx'].unique())]})")
    print("per-fold mean dMCC: "
          + "  ".join(f"F{int(f)}:{v:+.4f}" for f, v in fold_means.items()))
    print(f"dMCC (mean)       : {mean_delta:+.4f}")
    print(f"95% CI fold-block : [{lo:+.4f}, {hi:+.4f}]   (10,000 resamples; descriptive)")
    print(f"95% CI fold-t     : [{t_lo:+.4f}, {t_hi:+.4f}]   (t on {n_folds} fold means; certifying scale)")
    print(f"p_fold (n={n_folds})     : {p_fold:.3f}")
    print(f"p_bonf (k={a.k})     : {p_bonf:.3f}")
    print(f"pooled d (n={len(merged)}, descriptive): {d:+.3f}")
    if conforms is not None:
        print(f"fold conformance  : "
              f"{'OK (n_test matches the frozen folds)' if conforms else 'FAILED -- ' + str(bad)}")
    print(f"VERDICT           : {verdict}")
    if not k_declared:
        print("NOTE: family size not declared; defaulting to k=1 -- declare "
              "the full set of configurations you compared (rule 3).")
    if not seed_matched:
        print("NOTE: rule 1 expects seeds 42/123/456 matched to the baseline.")
    if a.json:
        a.json.write_text(json.dumps({
            "challenger": name, "baseline_arch": a.baseline_arch,
            "seed_matched": seed_matched, "n_cells": len(merged),
            "fold_means": {int(k): float(v) for k, v in fold_means.items()},
            "delta_mcc": mean_delta, "ci95": [lo, hi],
            "ci95_fold_t": [float(t_lo), float(t_hi)],
            "p_fold": float(p_fold), "k": a.k, "k_declared": k_declared,
            "p_bonf": p_bonf, "seed_contract_met": seeds_ok,
            "pooled_d": float(d), "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
