#!/usr/bin/env python3
"""Compute the benchmark claim block for a submission (see SUBMITTING.md).

Pairs a challenger's per-(fold, seed) test MCC against the shipped tuned
price-only baseline (natcore_A7) and reports the statistics the protocol
requires: per-fold mean deltas (the honest unit, n=5), fold-block bootstrap
95% CI, p_fold, Bonferroni-corrected p at the declared family size k, the
descriptive pooled effect size, and a descriptive read against the untuned
logistic-price anchor. Statistics mirror make_reference_table_v2.py
(same bootstrap: 10,000 resamples, seed 42).

Usage (score mode -- per-(fold, seed) MCC values you computed):
    evaluate_submission.py SUBMISSION.csv --k K --baseline-arch {ff,lstm}
        [--restrict-folds 0,1,2,3] [--json OUT.json]

Usage (predictions mode -- the evaluator recomputes MCC itself):
    evaluate_submission.py --predictions PREDS.csv --k K
        --baseline-arch {ff,lstm} [--name my_model] [--json OUT.json]
    PREDS.csv columns: seed, date (YYYY-MM-DD), ticker, y_pred (-1/+1 or 0/1).
    Scores are recomputed against the frozen labels table
    (data/processed/labels_direction.parquet); every (fold, seed) must cover
    the frozen test set exactly, so assembly conformance is intrinsic.

Fail-closed rules (a claim cannot certify without them):
  * --baseline-arch is REQUIRED. ff|lstm pair like-for-like against a
    runnable, validation-selected baseline arm. `envelope` (per-cell max
    over both arms) is available as an explicit conservative SENSITIVITY
    read only -- it is test-selected and not a runnable model, so it can
    never yield SUPPORTED.
  * --k (the declared comparison-family size, rule 3) is REQUIRED for
    certification. Without it the verdict is UNCERTIFIED.
  * Score-mode submissions must carry per-row n_test matching the frozen
    folds; without it the verdict is UNCERTIFIED (assembly unverified),
    and a mismatch is NOT COMPARABLE. Predictions mode verifies assembly
    by construction.
  * Certification also requires the full declared fold coverage and the
    three contract seeds (42/123/456) paired in every covered fold.
  * DUAL BAR: a SUPPORTED claim must clear BOTH references -- corrected
    fold-level significance against the declared tuned arm, AND a
    positive fold-level contrast against the strongest shipped classical
    anchor (untuned logistic-price). C1 exists to kill weak-baseline
    inflation; a claim can therefore never certify below a simple
    runnable price model. The anchor leg is a hard floor (its p is
    reported); the significance requirement stays on the tuned-arm
    contrast.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
BASELINE_GLOB = "results/native_core/native_core_*.csv"
BASELINE_PREFIX = "natcore_A7_"
LABELS_PARQUET = ROOT / "data" / "processed" / "labels_direction.parquet"
ANCHORS_JSON = ROOT / "results" / "analysis" / "naive_anchors.json"
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
    # explicit sensitivity option: the per-(fold, seed) upper ENVELOPE of
    # the two shipped arms (max over ff/lstm) -- test-selected, not a
    # runnable model; can never certify
    return (base.groupby(["fold_idx", "seed"], as_index=False)["mcc"].max())


def expected_n_test() -> dict[int, int]:
    """Per-fold test-set sizes implied by the frozen folds (from the
    shipped baseline rows; identical for every conforming assembly)."""
    frames = [pd.read_csv(p) for p in sorted(ROOT.glob(BASELINE_GLOB))]
    base = pd.concat(frames, ignore_index=True)
    base = base[base["experiment_name"].str.startswith(BASELINE_PREFIX)]
    return base.groupby("fold_idx")["n_test"].agg(
        lambda s: int(s.mode().iloc[0])).to_dict()


def mcc_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MCC for labels in {-1, +1} from confusion counts."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == -1) & (y_pred == -1)).sum())
    fp = int(((y_true == -1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == -1)).sum())
    denom = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def score_predictions(preds_path: Path, name: str) -> pd.DataFrame:
    """Predictions mode: recompute per-(fold, seed) MCC against the frozen
    labels; exit NOT COMPARABLE on any coverage violation."""
    if not LABELS_PARQUET.exists():
        sys.exit(f"frozen labels table missing at {LABELS_PARQUET} -- run "
                 "scripts/analysis/make_labels_table.py")
    labels = pd.read_parquet(LABELS_PARQUET)
    preds = pd.read_csv(preds_path, dtype={"date": str, "ticker": str})
    need = {"seed", "date", "ticker", "y_pred"}
    if not need.issubset(preds.columns):
        sys.exit(f"predictions file must have columns {sorted(need)}")
    # normalize 0/1 predictions to -1/+1
    vals = set(pd.unique(preds["y_pred"]))
    if vals <= {0, 1}:
        preds["y_pred"] = preds["y_pred"].map({0: -1, 1: 1})
    elif not vals <= {-1, 1}:
        sys.exit(f"y_pred must be in {{-1,+1}} or {{0,1}}; got {sorted(vals)[:6]}")
    if preds.duplicated(subset=["seed", "date", "ticker"]).any():
        sys.exit("NOT COMPARABLE: duplicate (seed, date, ticker) prediction rows")
    n_frozen = len(labels)
    rows = []
    for seed, g in preds.groupby("seed"):
        m = g.merge(labels, on=["date", "ticker"], how="inner")
        extra = len(g) - len(m)
        if extra:
            sys.exit(f"NOT COMPARABLE: seed {seed} has {extra} prediction rows "
                     "outside the frozen test set (wrong dates/tickers or "
                     "dead-zone days)")
        if len(m) != n_frozen:
            sys.exit(f"NOT COMPARABLE: seed {seed} covers {len(m)} of the "
                     f"{n_frozen} frozen test rows -- predictions mode "
                     "requires exact full coverage per seed")
        for fold, mf in m.groupby("fold_idx"):
            rows.append({
                "challenger": name, "fold_idx": int(fold), "seed": int(seed),
                "mcc": mcc_binary(mf["y_true"].to_numpy(),
                                  mf["y_pred"].to_numpy()),
                "n_test": len(mf),
            })
    out = pd.DataFrame(rows)
    print(f"predictions mode  : recomputed MCC for {out['seed'].nunique()} "
          f"seed(s) x {out['fold_idx'].nunique()} folds against the frozen "
          f"labels ({n_frozen} test rows; coverage exact)")
    return out


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
    ap.add_argument("submission", type=Path, nargs="?",
                    help="score-mode CSV (challenger, fold_idx, seed, mcc "
                         "[, n_test]); omit when using --predictions")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="per-example predictions CSV (seed, date, ticker, "
                         "y_pred); the evaluator recomputes MCC itself")
    ap.add_argument("--name", default=None,
                    help="challenger label for predictions mode "
                         "(default: the predictions file stem)")
    ap.add_argument("--k", type=int, default=None,
                    help="declared comparison-family size (rule 3); REQUIRED "
                         "for certification -- without it the verdict is "
                         "UNCERTIFIED")
    ap.add_argument("--baseline-arch", required=True,
                    choices=["ff", "lstm", "envelope", "stronger"],
                    help="REQUIRED. ff|lstm = like-for-like against a runnable "
                         "arm; envelope = explicit conservative sensitivity "
                         "read (test-selected; can never certify). 'stronger' "
                         "is a deprecated alias for envelope")
    ap.add_argument("--restrict-folds", default=None,
                    help="comma-separated fold_idx subset (coverage rule 8)")
    ap.add_argument("--social-coverage-justified", action="store_true",
                    help="attest that the restriction is the rule-8 "
                         "social-coverage tail (the submitted model consumes "
                         "the social source, whose coverage ends 2022-12-30). "
                         "REQUIRED for a --restrict-folds claim to certify -- "
                         "the tool cannot see which sources a model consumes, "
                         "so this entitlement is a submitter declaration, "
                         "audit-verified at Level 3, not machine-checked.")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    if a.baseline_arch == "stronger":
        print("NOTE: --baseline-arch 'stronger' is a deprecated alias for "
              "'envelope' (per-cell max; non-certifiable).", file=sys.stderr)
        a.baseline_arch = "envelope"
    k_declared = a.k is not None
    if k_declared and a.k < 1:
        sys.exit("--k must be a positive integer: the comparison-family size "
                 "covers every configuration compared (rule 3), so the minimum "
                 "is 1 (the submitted configuration itself). A k below 1 would "
                 "zero or invert the Bonferroni correction -- the multiplicity "
                 "gate certification depends on.")
    if not k_declared:
        a.k = 1

    if (a.predictions is None) == (a.submission is None):
        sys.exit("provide exactly one input: a score-mode CSV or --predictions")

    if a.predictions is not None:
        name = a.name or a.predictions.stem
        sub = score_predictions(a.predictions, name)
        assembly_verified = True   # coverage checked against frozen labels
    else:
        sub = pd.read_csv(a.submission)
        need = {"challenger", "fold_idx", "seed", "mcc"}
        if not need.issubset(sub.columns):
            sys.exit(f"submission must have columns {sorted(need)}")
        # MCC is mathematically bounded to [-1, 1]; a non-finite or
        # out-of-range value is malformed input, not a score. Reject it
        # at the boundary -- a NaN cell would otherwise be skipped by the
        # per-fold mean while its seed still counts as present, letting a
        # submitter poison bad seeds into a passing seed contract.
        _mcc = pd.to_numeric(sub["mcc"], errors="coerce")
        if not np.isfinite(_mcc).all() or (_mcc.abs() > 1).any():
            sys.exit("NOT COMPARABLE: every mcc must be a finite value in "
                     "[-1, 1] (MCC's range). Non-finite or out-of-range rows "
                     "are malformed -- fix or drop them; a dropped (fold, seed) "
                     "then fails the seed contract, as it should.")
        sub["mcc"] = _mcc
        if sub.duplicated(subset=["fold_idx", "seed"]).any():
            sys.exit("submission has duplicate (fold_idx, seed) rows -- the "
                     "contract is one row per (fold, seed); see SUBMITTING.md")
        name = sub["challenger"].iloc[0]
        assembly_verified = False  # resolved below via n_test conformance

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
    # the anchor leg is ALWAYS computed over the full submitted grid,
    # BEFORE any restriction -- restricting the anchor to a fold subset
    # would let a submitter drop the anchor's strongest fold and certify
    # below the full-universe logistic reference (fold selection through
    # the tool that polices selection)
    full_grid = merged.copy()
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
    if not assembly_verified and "n_test" in sub.columns:
        exp = expected_n_test()
        # check EVERY row per fold, not just the first — a submission whose
        # non-first rows carry a wrong n_test must not pass conformance
        bad = {int(f): (sorted({int(x) for x in g}), exp.get(int(f)))
               for f, g in sub.groupby("fold_idx")["n_test"]
               if exp.get(int(f)) is not None
               and any(int(x) != exp[int(f)] for x in g)}
        conforms = not bad
        assembly_verified = conforms

    # coverage gate: the full frozen grid by default; with --restrict-folds
    # (rule 8), the declared subset with at least four folds certifies and
    # the verdict is tagged as restricted
    required = ({int(x) for x in a.restrict_folds.split(",")}
                if a.restrict_folds else set(range(5)))
    covered = {int(f) for f in fold_means.index}
    # a certifying restriction is hard-gated to the ONE documented
    # coverage-rule subset {0,1,2,3} (rule 8, social-family coverage
    # tail) AND requires the submitter's explicit entitlement attestation
    # (the tool cannot see which sources a model consumes, so an
    # unattested restriction is best-k-of-5 fold selection -- a model
    # weak on the F4 stub could otherwise drop it and certify)
    restriction_certifiable = (not a.restrict_folds) or (
        required == {0, 1, 2, 3} and a.social_coverage_justified)
    coverage_ok = (covered == required
                   and n_folds >= (4 if a.restrict_folds else 5)
                   and restriction_certifiable)
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

    # dual bar, leg 2: the challenger's fold-level contrast against the
    # untuned logistic-price anchor must be positive (hard floor),
    # computed on the FULL five-fold grid regardless of --restrict-folds
    anchor_ok = None
    anchor_fold_deltas = None
    anchor_p = float("nan")
    if ANCHORS_JSON.exists():
        _anchors = json.loads(ANCHORS_JSON.read_text())
        _lp = _anchors.get("anchors", {}).get("logistic-price", {})
        _pf = _lp.get("mcc_per_fold")
        if _pf:
            _sub_means = full_grid.groupby("fold_idx")["mcc_sub"].mean()
            anchor_fold_deltas = {int(f): float(_sub_means[f] - _pf[int(f)])
                                  for f in _sub_means.index if int(f) < len(_pf)}
            _vals = list(anchor_fold_deltas.values())
            # the floor binds only when all five folds are present
            anchor_ok = (len(_vals) == 5 and float(np.mean(_vals)) > 0)
            if len(_vals) > 1:
                _, anchor_p = stats.ttest_1samp(_vals, 0.0)

    certifiable = (k_declared and assembly_verified and coverage_ok
                   and seeds_ok and a.baseline_arch != "envelope"
                   and anchor_ok is not None)
    supported = (mean_delta > 0 and p_bonf < 0.05 and certifiable
                 and bool(anchor_ok))
    verdict = "SUPPORTED" if supported else "WITHIN THE REFERENCE NULL"
    if (mean_delta > 0 and p_bonf < 0.05 and certifiable
            and anchor_ok is False):
        verdict += ("  [SIGNIFICANT VS THE TUNED ARM BUT DOES NOT CLEAR "
                    "THE CLASSICAL ANCHOR -- certification refused; C1 "
                    "forbids certifying below a simple runnable price "
                    "model]")
    if not k_declared:
        verdict = ("UNCERTIFIED -- comparison family undeclared (pass --k "
                   "covering every configuration you compared, rule 3)")
    elif not assembly_verified and conforms is None:
        verdict = ("UNCERTIFIED -- assembly unverified (score mode without "
                   "n_test; add per-row n_test or use --predictions)")
    if a.baseline_arch == "envelope":
        verdict += ("  [ENVELOPE REFERENCE -- test-selected sensitivity "
                    "read; not certifiable]")
    if a.restrict_folds:
        _fg = float(full_grid.groupby("fold_idx")["delta"].mean().mean())
        if restriction_certifiable:
            verdict += (f"  [RESTRICTED COVERAGE -- folds {sorted(covered)}; "
                        "submitter-declared rule-8 social-coverage entitlement, "
                        "audit-verified at Level 3, not machine-checked; "
                        f"full five-fold mean delta {_fg:+.4f}]")
        elif required == {0, 1, 2, 3}:
            verdict += (f"  [NON-CERTIFYING RESTRICTION -- folds {sorted(covered)} "
                        "is the rule-8 subset but its entitlement was not "
                        "declared; pass --social-coverage-justified only if the "
                        "model consumes the social source (audited at Level 3); "
                        f"full five-fold mean delta {_fg:+.4f}]")
        else:
            verdict += (f"  [NON-CERTIFYING RESTRICTION -- folds {sorted(covered)} "
                        "is not the documented rule-8 subset {0,1,2,3}; "
                        f"analysis only; full five-fold mean delta {_fg:+.4f}]")
    if k_declared and assembly_verified and not seeds_ok:
        verdict += ("  [SEED CONTRACT NOT MET -- certification requires seeds "
                    "42/123/456 paired per covered fold (rule 1)]")
    if conforms is False:
        supported = False
        verdict = ("NOT COMPARABLE  [ASSEMBLY MISMATCH -- n_test disagrees "
                   "with the frozen folds]")
    # fabrication check: recomputing the metric verifies ASSEMBLY, not
    # provenance -- a submitter who echoes the shipped frozen labels as
    # predictions scores a perfect MCC. Anything far above the task's
    # realistic ceiling is flagged for the mandatory Level-3 code audit
    # (the paper's stated defense), not silently certified.
    if float(full_grid["mcc_sub"].abs().max()) > 0.5:
        verdict += ("  [FABRICATION CHECK -- a per-fold MCC exceeds 0.5, far "
                    "above this task's realistic ceiling (reference grid near "
                    "0.01); recompute verifies assembly, not that predictions "
                    "came from a model. Requires the Level-3 code audit.]")

    arm_label = ("envelope of the ff/lstm arms -- per-cell max; a "
                 "test-selected sensitivity bar, not a runnable model"
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
    # dual bar, leg 2 readout: the untuned logistic-price anchor (the
    # significance test lives on the tuned-arm contrast; the anchor is the
    # hard floor a SUPPORTED claim must also clear)
    if anchor_fold_deltas:
        mean_anchor = float(np.mean(list(anchor_fold_deltas.values())))
        print("vs logistic anchor: "
              + "  ".join(f"F{f}:{v:+.4f}" for f, v in anchor_fold_deltas.items())
              + f"   mean {mean_anchor:+.4f}  p={anchor_p:.3f}"
              + f"   (hard floor for certification: "
              + f"{'cleared' if anchor_ok else 'NOT cleared'})")
    print(f"VERDICT           : {verdict}")
    if not k_declared:
        print("NOTE: family size not declared; statistics shown at k=1 are "
              "NOT a certification (rule 3).")
    if not seed_matched:
        print("NOTE: rule 1 expects seeds 42/123/456 matched to the baseline.")
    if a.json:
        a.json.write_text(json.dumps({
            "challenger": name, "baseline_arch": a.baseline_arch,
            "mode": "predictions" if a.predictions is not None else "scores",
            "seed_matched": seed_matched, "n_cells": len(merged),
            "fold_means": {int(k): float(v) for k, v in fold_means.items()},
            "delta_mcc": mean_delta, "ci95": [lo, hi],
            "ci95_fold_t": [float(t_lo), float(t_hi)],
            "p_fold": float(p_fold), "k": a.k, "k_declared": k_declared,
            "p_bonf": p_bonf, "seed_contract_met": seeds_ok,
            "assembly_verified": bool(assembly_verified),
            "anchor_deltas": anchor_fold_deltas,
            "anchor_cleared": anchor_ok,
            "anchor_p": None if math.isnan(anchor_p) else float(anchor_p),
            "pooled_d": float(d), "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
