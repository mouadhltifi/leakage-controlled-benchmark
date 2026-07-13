"""Equivalence-gate comparison between a v2 sweep CSV and the archived v1 A7 FF baseline.

Spec reference: the design spec

The gate check is deliberately narrow: it compares *only* the A7
price-only FF configuration, aggregated across five folds and three
seeds (15 total runs), against the archived reference in
``results/analysis/phase5_full_750core.md``:

    mean MCC        : 0.0071    (tolerance +/- 0.005)
    per-fold MCCs   : [0.0263, 0.0459, -0.0085, -0.0093, -0.0191]
                      (tolerance +/- 0.015 per fold)

Two failure modes are accommodated with informative notes:

* ``n_v2_runs < 15`` — sweep is incomplete (likely a mid-run Ctrl-C or
  a compute failure). ``passed=False``; callers should investigate
  before rerunning.
* One or more per-fold means drift beyond tolerance — the ``notes``
  field names the failing folds and by how much.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference values (spec Section 6.3; sourced from phase5_full_750core.md).
# ---------------------------------------------------------------------------

#: Archived v1 A7 FF mean MCC across 5 folds x 3 seeds (concat fusion,
#: parallel multi-task head, price-only).
REFERENCE_MEAN_MCC: float = 0.0071

#: Archived v1 A7 FF per-fold MCCs in fold-index order 0..4.
#: Fold regime labels (F0..F4): Pre-COVID, Recovery, Bull-to-Bear,
#: Bear Market, AI Rally.
REFERENCE_PER_FOLD_MCC: tuple[float, ...] = (
    0.0263,
    0.0459,
    -0.0085,
    -0.0093,
    -0.0191,
)

#: Default gate tolerances (spec Section 6.3).
DEFAULT_MEAN_TOLERANCE: float = 0.005
DEFAULT_PER_FOLD_TOLERANCE: float = 0.015

#: Number of runs expected in a complete equivalence sweep
#: (5 folds x 3 seeds).
EXPECTED_N_RUNS: int = 15


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class EquivalenceResult:
    """Structured output of :func:`compare_to_reference`.

    Attributes
    ----------
    passed
        ``True`` iff *both* the mean-MCC check *and* every per-fold check
        are within their tolerances *and* the sweep has the expected
        number of rows.
    mean_mcc_v2
        Mean MCC across every row of the v2 CSV.
    mean_mcc_v1
        Archived v1 reference mean (spec Section 6.3).
    mean_diff
        ``mean_mcc_v2 - mean_mcc_v1``.
    mean_tolerance
        Allowed ``|mean_diff|``.
    per_fold_v2
        v2 per-fold mean MCCs in fold-index order ``0..4``. NaN is used
        when a fold is missing from the CSV.
    per_fold_v1
        v1 archived per-fold MCCs in fold-index order ``0..4``.
    per_fold_diffs
        Element-wise ``per_fold_v2 - per_fold_v1``.
    per_fold_tolerance
        Allowed ``|per_fold_diffs[i]|`` for every ``i``.
    n_v2_runs
        Number of rows in the v2 CSV.
    notes
        Human-readable breadcrumbs — one entry per failure mode, plus an
        "all checks passed" note when the gate is green.
    """

    passed: bool
    mean_mcc_v2: float
    mean_mcc_v1: float
    mean_diff: float
    mean_tolerance: float
    per_fold_v2: list[float]
    per_fold_v1: list[float]
    per_fold_diffs: list[float]
    per_fold_tolerance: float
    n_v2_runs: int
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_to_reference(
    v2_csv: str | Path,
    reference_mean: float = REFERENCE_MEAN_MCC,
    reference_per_fold: list[float] | None = None,
    mean_tolerance: float = DEFAULT_MEAN_TOLERANCE,
    per_fold_tolerance: float = DEFAULT_PER_FOLD_TOLERANCE,
) -> EquivalenceResult:
    """Compare an equivalence-sweep CSV to the archived v1 reference.

    Parameters
    ----------
    v2_csv
        Path to the CSV written by
        :func:`mmfp.experiments.sweep.run_sweep` for the A7 sweep.
    reference_mean
        Archived mean MCC. Defaults to :data:`REFERENCE_MEAN_MCC`.
    reference_per_fold
        Archived per-fold MCCs in fold-index order ``0..4``. Defaults
        to :data:`REFERENCE_PER_FOLD_MCC`.
    mean_tolerance
        Allowed absolute difference between v2 mean and v1 mean.
    per_fold_tolerance
        Allowed absolute difference per fold.

    Returns
    -------
    EquivalenceResult
        Structured outcome; see the dataclass docstring.

    Raises
    ------
    FileNotFoundError
        If ``v2_csv`` does not exist.
    """
    csv_path = Path(v2_csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"compare_to_reference: v2 CSV not found at {csv_path}"
        )

    if reference_per_fold is None:
        reference_per_fold = list(REFERENCE_PER_FOLD_MCC)
    else:
        reference_per_fold = list(reference_per_fold)

    df = pd.read_csv(csv_path)
    notes: list[str] = []

    # ----- Dataset shape --------------------------------------------
    if "mcc" not in df.columns:
        raise ValueError(
            f"compare_to_reference: expected 'mcc' column in {csv_path}, "
            f"got {sorted(df.columns)}"
        )
    if "fold_idx" not in df.columns:
        raise ValueError(
            f"compare_to_reference: expected 'fold_idx' column in {csv_path}, "
            f"got {sorted(df.columns)}"
        )

    # Coerce MCC to numeric (empty strings become NaN) and drop any
    # rows without a recorded MCC so a resumed sweep with zeroed-out
    # placeholders cannot poison the mean.
    df = df.copy()
    df["mcc"] = pd.to_numeric(df["mcc"], errors="coerce")
    df["fold_idx"] = pd.to_numeric(df["fold_idx"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["mcc", "fold_idx"])
    n_v2_runs = int(len(df))

    # ----- Incomplete-sweep guard -----------------------------------
    if n_v2_runs < EXPECTED_N_RUNS:
        notes.append(
            f"incomplete: found {n_v2_runs} rows with valid (mcc, fold_idx) in "
            f"{csv_path}; expected {EXPECTED_N_RUNS} (5 folds x 3 seeds)."
        )

    # ----- Mean MCC -------------------------------------------------
    if n_v2_runs == 0:
        mean_mcc_v2 = float("nan")
    else:
        mean_mcc_v2 = float(df["mcc"].mean())
    mean_diff = mean_mcc_v2 - reference_mean

    mean_pass = (
        n_v2_runs > 0
        and abs(mean_diff) <= mean_tolerance
    )
    if n_v2_runs > 0 and not mean_pass:
        notes.append(
            f"mean MCC drift: v2={mean_mcc_v2:+.4f}, v1={reference_mean:+.4f}, "
            f"diff={mean_diff:+.4f}, tolerance=+/-{mean_tolerance:.4f}."
        )

    # ----- Per-fold -------------------------------------------------
    per_fold_v2: list[float] = []
    per_fold_diffs: list[float] = []
    per_fold_failures: list[str] = []

    fold_means = (
        df.groupby("fold_idx")["mcc"].mean().to_dict() if n_v2_runs > 0 else {}
    )

    for fold in range(len(reference_per_fold)):
        mean_ref = reference_per_fold[fold]
        mean_v2 = float(fold_means.get(fold, float("nan")))
        per_fold_v2.append(mean_v2)
        if pd.isna(mean_v2):
            per_fold_diffs.append(float("nan"))
            per_fold_failures.append(
                f"F{fold}: missing in v2 (no rows with fold_idx={fold})."
            )
            continue
        diff = mean_v2 - mean_ref
        per_fold_diffs.append(diff)
        if abs(diff) > per_fold_tolerance:
            per_fold_failures.append(
                f"F{fold}: v2={mean_v2:+.4f}, v1={mean_ref:+.4f}, "
                f"diff={diff:+.4f}, tolerance=+/-{per_fold_tolerance:.4f}."
            )

    if per_fold_failures:
        notes.append("per-fold drift:")
        notes.extend(f"  {line}" for line in per_fold_failures)

    per_fold_pass = not per_fold_failures

    # ----- Verdict --------------------------------------------------
    passed = bool(
        n_v2_runs >= EXPECTED_N_RUNS
        and mean_pass
        and per_fold_pass
    )
    if passed:
        notes.insert(0, "equivalence gate PASSED: all checks within tolerance.")
    else:
        notes.insert(0, "equivalence gate FAILED: see breakdown below.")

    return EquivalenceResult(
        passed=passed,
        mean_mcc_v2=mean_mcc_v2,
        mean_mcc_v1=float(reference_mean),
        mean_diff=mean_diff,
        mean_tolerance=float(mean_tolerance),
        per_fold_v2=per_fold_v2,
        per_fold_v1=list(reference_per_fold),
        per_fold_diffs=per_fold_diffs,
        per_fold_tolerance=float(per_fold_tolerance),
        n_v2_runs=n_v2_runs,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(result: EquivalenceResult) -> str:
    """Format an :class:`EquivalenceResult` as a human-readable markdown string.

    Intended for both on-disk reports and log output. Lines are stable so
    that the student can diff reports between sweep attempts.
    """
    headline = "PASSED" if result.passed else "FAILED"
    lines: list[str] = [
        "# Equivalence Gate Report — A7 Price-Only",
        "",
        f"**Result**: {headline}",
        "",
        "## Summary",
        "",
        f"- Runs in v2 CSV: {result.n_v2_runs} "
        f"(expected {EXPECTED_N_RUNS}: 5 folds x 3 seeds)",
        f"- Mean tolerance:     +/- {result.mean_tolerance:.4f}",
        f"- Per-fold tolerance: +/- {result.per_fold_tolerance:.4f}",
        "",
        "## Mean MCC",
        "",
        "| Source | Mean MCC |",
        "| --- | --- |",
        f"| v1 archived | {result.mean_mcc_v1:+.4f} |",
        f"| v2 (this sweep) | {result.mean_mcc_v2:+.4f} |",
        f"| diff (v2 - v1) | {result.mean_diff:+.4f} |",
        "",
        "## Per-Fold MCC",
        "",
        "| Fold | v1 archived | v2 this sweep | diff | within +/- tol? |",
        "| --- | --- | --- | --- | --- |",
    ]
    tol = result.per_fold_tolerance
    for i, (v2, v1, diff) in enumerate(
        zip(result.per_fold_v2, result.per_fold_v1, result.per_fold_diffs)
    ):
        if pd.isna(v2):
            v2_str = "NaN"
            diff_str = "NaN"
            within = "NO (missing)"
        else:
            v2_str = f"{v2:+.4f}"
            diff_str = f"{diff:+.4f}"
            within = "yes" if abs(diff) <= tol else "NO"
        lines.append(
            f"| F{i} | {v1:+.4f} | {v2_str} | {diff_str} | {within} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MEAN_TOLERANCE",
    "DEFAULT_PER_FOLD_TOLERANCE",
    "EXPECTED_N_RUNS",
    "EquivalenceResult",
    "REFERENCE_MEAN_MCC",
    "REFERENCE_PER_FOLD_MCC",
    "compare_to_reference",
    "format_report",
]
