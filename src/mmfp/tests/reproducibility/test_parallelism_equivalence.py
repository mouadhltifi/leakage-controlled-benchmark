"""Milestone 9 make-or-break: parallel sweeps must match serial sweeps exactly.

Spec reference: milestone-9 revision of Q14 (the design spec).

Claim under test
----------------

Running a sweep with ``parallelism=1`` and ``parallelism>=2`` on the CPU
path produces **bit-identical** :class:`ResultRecord` rows once the
results are aligned by ``(fold_idx, seed, experiment_name)``. If this
claim does not hold, the multiprocess implementation is wrong and the
research platform must stay serial.

Strategy
--------

1. Build a 4-entry override grid over A7 price-only (2 seeds x 2 folds,
   3 epochs each) so the whole test completes inside ~2 minutes.
2. Run the grid through :func:`run_sweep` once with ``parallelism=1``
   writing to ``serial.csv``.
3. Run the same grid again with ``parallelism=2`` writing to
   ``parallel.csv``.
4. Sort both CSVs by ``(fold_idx, seed)`` and assert cell-by-cell
   equality on every float column of :class:`ResultRecord`, plus all
   axis-fingerprint string columns and the integer meta columns that
   must match exactly.

Meta columns that legitimately differ between runs (``elapsed_seconds``
and ``git_sha``) are excluded from the equality assertion.

Sensitivity
-----------

A positive test (seeds differ -> metrics differ) is covered by the
existing CPU determinism suite; here we only assert the parallel/serial
invariant.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.load import dump_config
from mmfp.config.schema import ExperimentConfig
from mmfp.config.validate import validate_experiment_config
from mmfp.data.paths import FEATURES_DIR
from mmfp.experiments.result_schema import ResultRecord
from mmfp.experiments.sweep import run_sweep


# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------


#: Columns that MUST match bit-identically between a serial and a parallel
#: sweep of the same config. Float metrics are compared as exact equality
#: because CPU determinism guarantees bit-identity; epoch counts and sample
#: counts are integers that should match exactly as well.
_EXACT_MATCH_COLUMNS: tuple[str, ...] = (
    # identification
    "experiment_name", "config_hash", "seed", "fold_idx",
    # axis fingerprint
    "news_encoder", "news_aggregation", "fusion_strategy",
    "head_architecture", "targets", "graph_source",
    "lookback", "price_encoder",
    # direction metrics
    "mcc", "accuracy", "f1",
    # regression metrics
    "r2", "rmse", "sharpe_ratio",
    # volatility metrics (None/NaN both sides)
    "vol_rmse", "vol_r2",
    # integer meta
    "n_train", "n_val", "n_test", "n_params", "epochs_trained",
    # float meta that depends on training only (not wall-clock)
    "best_val_metric",
    # platform identifiers
    "platform_version",
)


#: Columns that ARE allowed to differ between runs:
#: ``elapsed_seconds`` is wall-clock and definitely changes across
#: serial vs parallel scheduling. ``git_sha`` is captured at run start
#: and is stable within one pytest invocation but still excluded for
#: symmetry with the wall-clock column.
_COLUMNS_ALLOWED_TO_DIFFER: tuple[str, ...] = (
    "elapsed_seconds",
    "git_sha",
)


def _require_price_parquet() -> None:
    """Skip the test if the required cached parquet is missing."""
    p = FEATURES_DIR / "price_features.parquet"
    if not p.exists():
        pytest.skip(
            f"Parallelism equivalence test needs cached price features at {p}. "
            "Run the v1 price-feature build first."
        )


# ---------------------------------------------------------------------------
# Config factory + grid
# ---------------------------------------------------------------------------


def _base_cpu_cfg(max_epochs: int = 3) -> ExperimentConfig:
    """Build the A7 FF price-only base config used by the test sweep."""
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "A7_parallelism_test"
    d["seed"] = 42
    d["device"] = "cpu"
    d["data"]["fold_idx"] = 0
    d["data"]["lookback"] = 1
    d["model"]["price_encoder"] = "feedforward"
    d["training"]["max_epochs"] = max_epochs
    d["training"]["min_epochs"] = 0
    d["training"]["patience"] = max_epochs  # full budget used
    d["training"]["batch_size"] = 128
    d["training"]["scheduler"] = "none"
    d["logging"]["per_epoch_log_every"] = 1
    cfg = ExperimentConfig.model_validate(d)
    return validate_experiment_config(cfg)


def _write_base_toml(tmp_path: Path) -> Path:
    """Persist the base config as TOML for :func:`run_sweep` to consume."""
    cfg = _base_cpu_cfg()
    path = tmp_path / "base.toml"
    dump_config(cfg, path)
    return path


def _small_grid() -> list[dict[str, Any]]:
    """2 seeds x 2 folds = 4 runs. Small enough to fit in the test budget."""
    grid: list[dict[str, Any]] = []
    for seed in (42, 43):
        for fold in (0, 1):
            grid.append({"seed": seed, "data.fold_idx": fold})
    return grid


# ---------------------------------------------------------------------------
# Row-alignment + comparison helpers
# ---------------------------------------------------------------------------


def _sort_for_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by (fold_idx, seed) so serial and parallel line up."""
    return (
        df.sort_values(["fold_idx", "seed"], kind="mergesort")
        .reset_index(drop=True)
    )


def _values_equal_exact(lhs: object, rhs: object) -> bool:
    """Cell-wise equality respecting NaN-eq-NaN and empty-string semantics.

    The CSV round-trip turns ``None`` metrics into empty strings; pandas
    may then read them back as NaN or as empty strings depending on the
    column dtype. Both sides of the comparison go through the same
    pipeline so we mostly need NaN-eq-NaN, but we also tolerate the
    ``"" == NaN`` case that crops up on string columns.
    """
    # Treat NaN-NaN as equal (pandas default behaviour is to return False).
    if _is_nan(lhs) and _is_nan(rhs):
        return True
    # Empty string from one side vs NaN from the other (CSV round-trip).
    if (lhs == "" and _is_nan(rhs)) or (rhs == "" and _is_nan(lhs)):
        return True
    return lhs == rhs


def _is_nan(x: object) -> bool:
    """NaN detector that also rejects non-floats (safe to call on strings)."""
    return isinstance(x, float) and math.isnan(x)


def _assert_rows_bit_identical(
    serial: pd.DataFrame, parallel: pd.DataFrame,
) -> None:
    """Assert that aligned serial + parallel rows are equal on every tracked column.

    Uses ``_EXACT_MATCH_COLUMNS`` to pick the columns that must match;
    columns in ``_COLUMNS_ALLOWED_TO_DIFFER`` are skipped.
    """
    assert len(serial) == len(parallel), (
        f"Serial and parallel runs produced different row counts: "
        f"{len(serial)} vs {len(parallel)}"
    )
    serial = _sort_for_alignment(serial)
    parallel = _sort_for_alignment(parallel)

    # Headers must be identical (same ResultRecord schema on both sides).
    assert list(serial.columns) == list(parallel.columns), (
        f"Serial CSV has columns {list(serial.columns)}; "
        f"parallel CSV has {list(parallel.columns)}"
    )

    diffs: list[str] = []
    for col in _EXACT_MATCH_COLUMNS:
        for idx in range(len(serial)):
            a = serial.iloc[idx][col]
            b = parallel.iloc[idx][col]
            if not _values_equal_exact(a, b):
                diffs.append(
                    f"row {idx} col {col!r}: serial={a!r} parallel={b!r}"
                )
    if diffs:
        head = "\n  ".join(diffs[:10])
        total = len(diffs)
        raise AssertionError(
            f"Parallel sweep differs from serial sweep on {total} cell(s). "
            f"First {min(total, 10)}:\n  {head}"
        )


# ---------------------------------------------------------------------------
# The critical deliverable
# ---------------------------------------------------------------------------


def test_parallelism_bit_identical_cpu(tmp_path: Path) -> None:
    """4-entry CPU sweep: p=1 vs p=2 must be bit-identical on every metric.

    This is the make-or-break test for Milestone 9. If it ever fails,
    the parallel path has introduced a nondeterminism and the sweep
    runner must revert to serial-only until the regression is fixed.
    """
    _require_price_parquet()

    base_toml = _write_base_toml(tmp_path)
    grid = _small_grid()

    serial_csv = tmp_path / "serial.csv"
    parallel_csv = tmp_path / "parallel.csv"

    # -- serial reference --
    t0 = time.perf_counter()
    serial_df = run_sweep(
        base_cfg_path=base_toml,
        overrides=grid,
        output_csv=serial_csv,
        resume=True,
        parallelism=1,
        device_strategy="cpu",
    )
    serial_wall = time.perf_counter() - t0

    # -- parallel candidate --
    t0 = time.perf_counter()
    parallel_df = run_sweep(
        base_cfg_path=base_toml,
        overrides=grid,
        output_csv=parallel_csv,
        resume=True,
        parallelism=2,
        device_strategy="cpu",
    )
    parallel_wall = time.perf_counter() - t0

    # Row count sanity.
    assert len(serial_df) == len(grid), (
        f"Serial sweep expected {len(grid)} rows, got {len(serial_df)}"
    )
    assert len(parallel_df) == len(grid), (
        f"Parallel sweep expected {len(grid)} rows, got {len(parallel_df)}"
    )

    # Bit-identity check — the whole point of the test.
    _assert_rows_bit_identical(serial_df, parallel_df)

    # Wall-clock sanity: not a hard assertion (CI noise) but helpful to
    # see in failure output. Just ensure both finished.
    assert serial_wall > 0 and parallel_wall > 0


def test_parallelism_resume_skips_completed(tmp_path: Path) -> None:
    """Restart scenario: parallel sweep resumes from a partially-written CSV.

    We simulate a crash after 2 of 4 runs by:

    1. Running a 2-entry sub-grid to completion (writes 2 rows).
    2. Starting the full 4-entry sweep with ``resume=True``; only the
       two missing overrides should actually execute.
    3. Asserting the final CSV has exactly 4 rows, no duplicates, and
       the first 2 rows (identified by fingerprint) were never re-run.
    """
    _require_price_parquet()

    base_toml = _write_base_toml(tmp_path)
    full_grid = _small_grid()
    partial_grid = full_grid[:2]  # first two entries only

    out_csv = tmp_path / "out.csv"

    # Phase 1: "crashed" run that only completed 2 rows.
    phase1_df = run_sweep(
        base_cfg_path=base_toml,
        overrides=partial_grid,
        output_csv=out_csv,
        resume=True,
        parallelism=2,
        device_strategy="cpu",
    )
    assert len(phase1_df) == 2

    # Capture the fingerprints of the already-done rows; resume must not
    # overwrite them.
    phase1_rows = phase1_df.copy()

    # Phase 2: full grid with resume — should complete the remaining 2.
    phase2_df = run_sweep(
        base_cfg_path=base_toml,
        overrides=full_grid,
        output_csv=out_csv,
        resume=True,
        parallelism=2,
        device_strategy="cpu",
    )
    # Total of 4 rows, no duplicates by (seed, fold_idx).
    assert len(phase2_df) == 4
    dup_mask = phase2_df.duplicated(subset=["seed", "fold_idx"])
    assert not dup_mask.any(), (
        f"Resume sweep produced duplicate (seed, fold_idx) rows:\n"
        f"{phase2_df[dup_mask]}"
    )

    # The two already-completed fingerprints must match phase1 verbatim
    # on metric columns (resume must not re-run them).
    merged = phase2_df.merge(
        phase1_rows,
        on=["experiment_name", "config_hash", "seed", "fold_idx"],
        suffixes=("_p2", "_p1"),
    )
    assert len(merged) == 2
    for col in ("mcc", "accuracy", "f1", "r2", "rmse"):
        for idx in range(2):
            a = merged.iloc[idx][f"{col}_p1"]
            b = merged.iloc[idx][f"{col}_p2"]
            if not _values_equal_exact(a, b):
                raise AssertionError(
                    f"Resume re-ran an already-completed row: "
                    f"{col} phase1={a!r} phase2={b!r}"
                )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this machine",
)
def test_parallelism_mps_tolerance(tmp_path: Path) -> None:
    """MPS parallel path matches MPS serial within 1e-4 on FF (non-LSTM).

    The base config is A7 feedforward (no LSTM), so MPS determinism is
    stronger than it would be with an LSTM path. We still compare with
    a 1e-4 tolerance on float metrics — matching the existing MPS
    reproducibility contract — and we only compare the handful of
    metric columns (MPS can re-order numerically-equivalent reductions
    between runs).

    This test is skipped on non-Apple hardware.
    """
    _require_price_parquet()

    base_toml = _write_base_toml(tmp_path)
    grid = _small_grid()

    serial_csv = tmp_path / "serial_mps.csv"
    parallel_csv = tmp_path / "parallel_mps.csv"

    serial_df = run_sweep(
        base_cfg_path=base_toml,
        overrides=grid,
        output_csv=serial_csv,
        resume=True,
        parallelism=1,
        device_strategy="mps",
    )
    parallel_df = run_sweep(
        base_cfg_path=base_toml,
        overrides=grid,
        output_csv=parallel_csv,
        resume=True,
        parallelism=2,
        device_strategy="mps",
    )

    serial_df = _sort_for_alignment(serial_df)
    parallel_df = _sort_for_alignment(parallel_df)

    assert len(serial_df) == len(parallel_df) == len(grid)

    # Structural columns must still match exactly.
    for col in (
        "experiment_name", "config_hash", "seed", "fold_idx",
        "news_encoder", "targets", "graph_source", "lookback",
        "price_encoder", "n_train", "n_val", "n_test", "n_params",
    ):
        for idx in range(len(serial_df)):
            a = serial_df.iloc[idx][col]
            b = parallel_df.iloc[idx][col]
            assert _values_equal_exact(a, b), (
                f"MPS parallel structural column {col!r} differs at row {idx}: "
                f"serial={a!r} parallel={b!r}"
            )

    # Metrics within 1e-4 tolerance per the MPS contract.
    tol = 1e-4
    for col in ("mcc", "accuracy", "f1", "r2", "rmse",
                "sharpe_ratio", "best_val_metric"):
        for idx in range(len(serial_df)):
            a = serial_df.iloc[idx][col]
            b = parallel_df.iloc[idx][col]
            # None/empty round-trips
            if _is_nan(a) and _is_nan(b):
                continue
            if (a == "" and _is_nan(b)) or (b == "" and _is_nan(a)):
                continue
            delta = abs(float(a) - float(b))
            assert delta <= tol, (
                f"MPS metric {col!r} row {idx} differs by {delta} > {tol}: "
                f"serial={a} parallel={b}"
            )


# ---------------------------------------------------------------------------
# Schema-level self-check: ensure the column list above stays in sync with
# ResultRecord. If a new field is added to ResultRecord without classifying
# it as "match exact" or "allowed to differ", this test fails at collection.
# ---------------------------------------------------------------------------


def test_result_record_columns_are_partitioned() -> None:
    """Every :class:`ResultRecord` field is either tracked or explicitly skipped.

    This catches silent schema drift: if someone adds a new metric to
    ``ResultRecord``, the equivalence test must decide whether it should
    match exactly or be allowed to drift across parallel/serial runs.
    """
    known = set(_EXACT_MATCH_COLUMNS) | set(_COLUMNS_ALLOWED_TO_DIFFER)
    actual = set(ResultRecord.column_names())
    missing = actual - known
    extra = known - actual
    assert not missing, (
        f"ResultRecord has untracked fields {sorted(missing)}; "
        "add them to _EXACT_MATCH_COLUMNS or _COLUMNS_ALLOWED_TO_DIFFER."
    )
    assert not extra, (
        f"Tracked column list references fields not on ResultRecord: "
        f"{sorted(extra)}"
    )
