"""Tests for :mod:`mmfp.experiments.equivalence`.

These tests exercise the comparison logic against synthetic CSVs.
The actual gate execution against the trained model lives in
``scripts/run_equivalence_gate.py`` and is not part of the unit test
suite (runtime too large).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pytest

from mmfp.experiments.equivalence import (
    DEFAULT_MEAN_TOLERANCE,
    DEFAULT_PER_FOLD_TOLERANCE,
    EXPECTED_N_RUNS,
    REFERENCE_MEAN_MCC,
    REFERENCE_PER_FOLD_MCC,
    compare_to_reference,
    format_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_synthetic_csv(
    path: Path, rows: Iterable[dict[str, object]]
) -> Path:
    """Write a minimal CSV with ``mcc`` and ``fold_idx`` columns plus
    whatever other keys the rows contain. Used to emulate a sweep output
    without invoking the trainer."""
    rows = list(rows)
    if not rows:
        # Write an empty-but-headered CSV so pandas can still read it.
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["mcc", "fold_idx"])
            writer.writeheader()
        return path
    fieldnames = sorted({k for row in rows for k in row.keys()})
    if "mcc" not in fieldnames:
        fieldnames.append("mcc")
    if "fold_idx" not in fieldnames:
        fieldnames.append("fold_idx")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _reference_rows() -> list[dict[str, object]]:
    """15 rows matching the archived per-fold means exactly.

    Three seeds per fold, each row carrying the fold's reference MCC so
    the per-fold mean and the overall mean both equal the references.
    """
    rows: list[dict[str, object]] = []
    for fold, mcc in enumerate(REFERENCE_PER_FOLD_MCC):
        for seed in (42, 123, 2024):
            rows.append(
                {
                    "experiment_name": "A7_price_only",
                    "fold_idx": fold,
                    "seed": seed,
                    "mcc": mcc,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Happy path: exact reference reproduced.
# ---------------------------------------------------------------------------


def test_compare_to_reference_passes_on_exact_match(tmp_path: Path) -> None:
    csv_path = tmp_path / "sweep.csv"
    _write_synthetic_csv(csv_path, _reference_rows())

    result = compare_to_reference(csv_path)

    assert result.passed is True
    assert result.n_v2_runs == EXPECTED_N_RUNS
    assert result.mean_mcc_v1 == pytest.approx(REFERENCE_MEAN_MCC)
    # Per-fold v2 and v1 should agree to within floating noise.
    for v2, v1 in zip(result.per_fold_v2, result.per_fold_v1):
        assert v2 == pytest.approx(v1, abs=1e-9)
    # The first note names the PASSED verdict explicitly.
    assert any("PASSED" in note for note in result.notes)


def test_compare_to_reference_passes_within_tolerance(tmp_path: Path) -> None:
    """Small per-fold perturbation still well inside +/-0.015 passes."""
    rows = _reference_rows()
    # Nudge each MCC by +0.002 — well under per-fold +-0.015 and
    # mean +-0.005.
    for row in rows:
        row["mcc"] = float(row["mcc"]) + 0.002
    csv_path = tmp_path / "sweep.csv"
    _write_synthetic_csv(csv_path, rows)

    result = compare_to_reference(csv_path)

    assert result.passed is True
    assert abs(result.mean_diff) < DEFAULT_MEAN_TOLERANCE
    for diff in result.per_fold_diffs:
        assert abs(diff) < DEFAULT_PER_FOLD_TOLERANCE


# ---------------------------------------------------------------------------
# Mean drift.
# ---------------------------------------------------------------------------


def test_compare_to_reference_fails_on_mean_drift(tmp_path: Path) -> None:
    """Mean off by 0.01 must fail."""
    rows = _reference_rows()
    # Add a constant so the mean drifts by ~0.01 but per-fold drifts are
    # all below +/-0.015 tolerance (so only the mean check fires).
    bump = 0.01
    for row in rows:
        row["mcc"] = float(row["mcc"]) + bump
    csv_path = tmp_path / "sweep.csv"
    _write_synthetic_csv(csv_path, rows)

    result = compare_to_reference(csv_path)

    assert result.passed is False
    # Note: the reference mean (0.0071) is a rounded literal, whereas the
    # perturbed v2 mean is (sum(ref_per_fold)/5 + bump). The diff is
    # therefore ~bump, up to the literal's rounding (~1e-4).
    assert result.mean_diff == pytest.approx(bump, abs=1e-3)
    # Mean tolerance is 0.005; a 0.01 bump must trigger the mean check.
    assert abs(result.mean_diff) > DEFAULT_MEAN_TOLERANCE
    joined = "\n".join(result.notes)
    assert "FAILED" in joined
    assert "mean" in joined.lower() or "per-fold" in joined.lower()


def test_compare_to_reference_fails_when_one_fold_drifts(tmp_path: Path) -> None:
    """A single fold off by 0.02 fails with a fold-identified note."""
    rows = _reference_rows()
    # Perturb only fold 2 by a large amount.
    offending_fold = 2
    for row in rows:
        if row["fold_idx"] == offending_fold:
            row["mcc"] = float(row["mcc"]) + 0.02
    csv_path = tmp_path / "sweep.csv"
    _write_synthetic_csv(csv_path, rows)

    result = compare_to_reference(csv_path)

    assert result.passed is False
    # Per-fold diff for the offending fold exceeds tolerance.
    assert abs(result.per_fold_diffs[offending_fold]) > DEFAULT_PER_FOLD_TOLERANCE
    # Other folds are still within tolerance.
    for fold, diff in enumerate(result.per_fold_diffs):
        if fold == offending_fold:
            continue
        assert abs(diff) <= DEFAULT_PER_FOLD_TOLERANCE, (
            f"Unexpected drift at F{fold}: {diff:+.4f}"
        )
    # Note names the failing fold by index (F2 in this case).
    assert any(f"F{offending_fold}" in note for note in result.notes), (
        f"Expected a note mentioning F{offending_fold}; got notes={result.notes!r}"
    )


# ---------------------------------------------------------------------------
# Missing CSV.
# ---------------------------------------------------------------------------


def test_compare_to_reference_missing_csv_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.csv"
    with pytest.raises(FileNotFoundError):
        compare_to_reference(missing)


# ---------------------------------------------------------------------------
# Partial sweep.
# ---------------------------------------------------------------------------


def test_compare_to_reference_partial_sweep_marked_incomplete(tmp_path: Path) -> None:
    """Fewer than 15 rows must fail with an 'incomplete' note."""
    # Keep only the first 10 rows of the reference.
    rows = _reference_rows()[:10]
    csv_path = tmp_path / "sweep.csv"
    _write_synthetic_csv(csv_path, rows)

    result = compare_to_reference(csv_path)

    assert result.passed is False
    assert result.n_v2_runs == 10
    joined = "\n".join(result.notes).lower()
    assert "incomplete" in joined, (
        f"Expected 'incomplete' in notes, got {result.notes!r}"
    )


def test_compare_to_reference_empty_csv_marked_incomplete(tmp_path: Path) -> None:
    """A header-only CSV with no rows must fail with incomplete notes and NaN stats."""
    csv_path = tmp_path / "sweep.csv"
    _write_synthetic_csv(csv_path, [])

    result = compare_to_reference(csv_path)

    assert result.passed is False
    assert result.n_v2_runs == 0
    joined = "\n".join(result.notes).lower()
    assert "incomplete" in joined


# ---------------------------------------------------------------------------
# format_report produces plausible markdown.
# ---------------------------------------------------------------------------


def test_format_report_passed_contains_headline(tmp_path: Path) -> None:
    csv_path = tmp_path / "sweep.csv"
    _write_synthetic_csv(csv_path, _reference_rows())

    result = compare_to_reference(csv_path)
    report = format_report(result)

    assert "# Equivalence Gate Report" in report
    assert "**Result**: PASSED" in report
    # One table row per fold plus header.
    assert "| F0 |" in report and "| F4 |" in report


def test_format_report_failed_contains_headline(tmp_path: Path) -> None:
    rows = _reference_rows()
    # Large bump so it definitely fails.
    for row in rows:
        row["mcc"] = float(row["mcc"]) + 0.1
    csv_path = tmp_path / "sweep.csv"
    _write_synthetic_csv(csv_path, rows)

    result = compare_to_reference(csv_path)
    report = format_report(result)

    assert "**Result**: FAILED" in report
    # Diff column should show the bump.
    assert "+0.100" in report
