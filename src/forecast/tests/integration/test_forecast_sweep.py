"""Integration test: :func:`forecast.experiments.sweep.run_forecast_sweep`.

Drives a 4-run sweep (2 seeds × 2 folds, parallelism=1) end-to-end and
verifies that the resulting CSV has:

* 4 rows (no fingerprint collisions at those axes).
* Headers that include the two v3-native columns
  (``quantile_coverage_80``, ``quantile_interval_width``).
* Every row has a populated coverage + width value (not empty string).

``assemble_fold`` is monkeypatched to emit synthetic data (same pattern
as :mod:`test_run_one_forecast`). Parallelism=1 because multi-process
safety is v2's responsibility and already tested upstream.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import pytest

import forecast.experiments.runner as runner_module
from forecast.experiments.sweep import run_forecast_sweep
from mmfp.data.assemble import FoldArtifacts


def _install_synthetic_assemble(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifacts_factory: Callable[..., FoldArtifacts],
) -> None:
    """Monkeypatch assemble_fold to emit a tiny synthetic fold."""

    def _stub_assemble(cfg):
        return synthetic_artifacts_factory(
            n_tickers=3,
            n_days=200,
            lookback=20,
            seed=cfg.seed,
        )

    monkeypatch.setattr(runner_module, "assemble_fold", _stub_assemble)


def _write_cfg_toml(path: Path) -> None:
    """Drop a minimal v3 TOML config at ``path``.

    The sweep driver will load the v3 defaults, then merge this TOML on
    top, then apply the per-override dict. Sizes are tuned for a fast
    CPU run (lookback=20, hidden_dim=16, max_epochs=2).
    """
    path.write_text(
        """
name = "v3_sweep_test"
seed = 42
device = "cpu"

[forecast]
lookback = 20
hidden_dim = 16
warmup_epochs = 0

[data]
lookback = 20

[training]
batch_size = 16
max_epochs = 2
min_epochs = 0
patience = 1000
learning_rate = 3e-3
scheduler = "none"

[head]
targets = ["return"]
""".lstrip()
    )


@pytest.mark.slow
def test_run_forecast_sweep_writes_4_rows_with_new_columns(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifacts_factory,
    tmp_path: Path,
):
    """2 seeds × 2 folds = 4 runs; CSV must have 4 rows with both new columns."""
    _install_synthetic_assemble(monkeypatch, synthetic_artifacts_factory)

    cfg_toml = tmp_path / "cfg.toml"
    _write_cfg_toml(cfg_toml)

    out_csv = tmp_path / "sweep.csv"

    overrides: list[dict[str, Any]] = [
        {"seed": 42, "data.fold_idx": 0},
        {"seed": 43, "data.fold_idx": 0},
        {"seed": 42, "data.fold_idx": 1},
        {"seed": 43, "data.fold_idx": 1},
    ]

    df = run_forecast_sweep(
        base_cfg_path=cfg_toml,
        overrides=overrides,
        output_csv=out_csv,
        resume=True,
        parallelism=1,
        device_strategy="cpu",
    )

    # Dataframe should have exactly 4 rows.
    assert len(df) == 4, f"expected 4 rows, got {len(df)}"

    # Header check: read the raw CSV to catch missing columns.
    assert out_csv.exists()
    with out_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        rows = list(reader)

    assert "quantile_coverage_80" in headers, (
        f"quantile_coverage_80 missing from CSV header: {headers}"
    )
    assert "quantile_interval_width" in headers, (
        f"quantile_interval_width missing from CSV header: {headers}"
    )
    assert len(rows) == 4

    # Every row has both new columns populated.
    for row in rows:
        cov = row["quantile_coverage_80"]
        width = row["quantile_interval_width"]
        assert cov != "", f"coverage empty in row: {row}"
        assert width != "", f"width empty in row: {row}"
        # Numeric round-trip.
        float(cov)
        float(width)

    # Fingerprint uniqueness: names, config_hash, seed, fold_idx must
    # combine to 4 distinct fingerprints (no dedup skipped anything).
    fingerprints = {
        "|".join(
            [row["experiment_name"], row["config_hash"], row["seed"], row["fold_idx"]]
        )
        for row in rows
    }
    assert len(fingerprints) == 4, (
        f"fingerprints collided: {fingerprints}"
    )


@pytest.mark.slow
def test_run_forecast_sweep_resume_skips_existing(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifacts_factory,
    tmp_path: Path,
):
    """Running the same sweep twice with resume=True does not duplicate rows."""
    _install_synthetic_assemble(monkeypatch, synthetic_artifacts_factory)

    cfg_toml = tmp_path / "cfg.toml"
    _write_cfg_toml(cfg_toml)

    out_csv = tmp_path / "sweep.csv"
    overrides = [
        {"seed": 42, "data.fold_idx": 0},
        {"seed": 43, "data.fold_idx": 0},
    ]

    # First pass.
    df1 = run_forecast_sweep(
        base_cfg_path=cfg_toml,
        overrides=overrides,
        output_csv=out_csv,
        resume=True,
        parallelism=1,
        device_strategy="cpu",
    )
    assert len(df1) == 2

    # Second pass with the same overrides -- resume should skip both.
    df2 = run_forecast_sweep(
        base_cfg_path=cfg_toml,
        overrides=overrides,
        output_csv=out_csv,
        resume=True,
        parallelism=1,
        device_strategy="cpu",
    )
    assert len(df2) == 2, f"resume failed; CSV grew to {len(df2)} rows"
