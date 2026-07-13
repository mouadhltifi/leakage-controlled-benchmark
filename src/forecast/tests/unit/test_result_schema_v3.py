"""Unit tests for :class:`forecast.experiments.result_schema.ForecastResultRecord`.

Coverage map (M5 the architecture spec):

* The v3 record has all v2 columns plus ``quantile_coverage_80`` and
  ``quantile_interval_width`` in the declared field order.
* :func:`append_result` round-trips a :class:`ForecastResultRecord` and
  :func:`load_results` recovers every field (headers include the two
  new columns).
* :func:`record_fingerprint` on a v3 record matches
  :func:`fingerprint_from_row` on the CSV-loaded row — extending the
  schema does not break dedup.
* :func:`config_hash` is stable across v3 configs (inherited behaviour
  from v2, re-checked to confirm the re-export chain works).
* The two new columns survive ``None`` round-trips (loaded as empty
  string — matching v2's ``to_row`` semantics for None).
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

from forecast.config.schema import V3ExperimentConfig
from forecast.experiments.result_schema import (
    ForecastResultRecord,
    ResultRecord,
    append_result,
    config_hash,
    fingerprint_from_row,
    load_results,
    record_fingerprint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    name: str = "v3_test",
    seed: int = 42,
    fold_idx: int = 0,
    coverage: float | None = 0.81,
    width: float | None = 1.23,
) -> ForecastResultRecord:
    """Construct a populated ForecastResultRecord for round-trip tests."""
    return ForecastResultRecord(
        experiment_name=name,
        config_hash="deadbeef" * 8,
        seed=seed,
        fold_idx=fold_idx,
        news_encoder="none",
        news_aggregation="none",
        fusion_strategy="tft",
        head_architecture="quantile",
        targets="return,direction,volatility",
        graph_source="none",
        lookback=60,
        price_encoder="tft",
        mcc=0.011,
        accuracy=0.52,
        f1=0.49,
        r2=0.0,
        rmse=0.018,
        sharpe_ratio=0.33,
        vol_rmse=0.015,
        vol_r2=0.0,
        n_train=10_000,
        n_val=2_000,
        n_test=2_000,
        n_params=500_000,
        epochs_trained=20,
        best_val_metric=-0.123,
        elapsed_seconds=180.0,
        platform_version="v3-3.0.0-dev",
        git_sha="abc1234",
        quantile_coverage_80=coverage,
        quantile_interval_width=width,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_v3_record_extends_v2_fields_and_adds_two_new():
    """Every v2 ResultRecord column must be present on ForecastResultRecord,
    and the v3 record adds exactly ``quantile_coverage_80`` and
    ``quantile_interval_width``.
    """
    v2_field_names = {f.name for f in dataclass_fields(ResultRecord)}
    v3_field_names = {f.name for f in dataclass_fields(ForecastResultRecord)}

    # v3 is a superset of v2.
    assert v2_field_names.issubset(v3_field_names), (
        f"v3 missing v2 fields: {v2_field_names - v3_field_names}"
    )

    # The two new fields are present.
    new_fields = v3_field_names - v2_field_names
    assert new_fields == {"quantile_coverage_80", "quantile_interval_width"}


def test_v3_record_is_a_resultrecord_instance():
    """Subclassing must preserve isinstance so v2-consumer code that
    accepts ``ResultRecord`` accepts a v3 record."""
    rec = _make_record()
    assert isinstance(rec, ResultRecord)
    assert isinstance(rec, ForecastResultRecord)


def test_v3_record_csv_roundtrip_preserves_new_columns(tmp_path: Path):
    """append_result -> load_results must recover both new columns."""
    csv = tmp_path / "results.csv"
    rec = _make_record(coverage=0.79, width=1.44)
    append_result(csv, rec)

    rows = load_results(csv)
    assert len(rows) == 1
    row = rows[0]

    # The two new columns appear in the CSV header -> dict round-trip.
    assert "quantile_coverage_80" in row
    assert "quantile_interval_width" in row
    # Values round-trip as strings (csv writer coerces floats to str).
    assert float(row["quantile_coverage_80"]) == 0.79
    assert float(row["quantile_interval_width"]) == 1.44


def test_v3_record_csv_preserves_none_as_empty_string(tmp_path: Path):
    """``None`` for the two new fields must round-trip as empty string
    (matching v2's to_row convention for optional metrics).
    """
    csv = tmp_path / "results.csv"
    rec = _make_record(coverage=None, width=None)
    append_result(csv, rec)

    rows = load_results(csv)
    assert len(rows) == 1
    row = rows[0]
    assert row["quantile_coverage_80"] == ""
    assert row["quantile_interval_width"] == ""


def test_fingerprint_is_unique_per_name_hash_seed_fold(tmp_path: Path):
    """Schema extension must not break dedup. record_fingerprint and
    fingerprint_from_row must agree on the same record -> CSV row."""
    csv = tmp_path / "results.csv"
    rec = _make_record(name="uA", seed=7, fold_idx=2)
    append_result(csv, rec)

    rows = load_results(csv)
    assert len(rows) == 1

    fp_from_record = record_fingerprint(rec)
    fp_from_row = fingerprint_from_row(rows[0])
    assert fp_from_record == fp_from_row

    # Two different seeds -> different fingerprints.
    rec2 = _make_record(name="uA", seed=8, fold_idx=2)
    assert record_fingerprint(rec) != record_fingerprint(rec2)

    # Two different folds -> different fingerprints.
    rec3 = _make_record(name="uA", seed=7, fold_idx=3)
    assert record_fingerprint(rec) != record_fingerprint(rec3)

    # Two different names -> different fingerprints.
    rec4 = _make_record(name="uB", seed=7, fold_idx=2)
    assert record_fingerprint(rec) != record_fingerprint(rec4)


def test_config_hash_is_stable_for_v3_cfg(minimal_cfg_dict: dict[str, Any]):
    """The re-exported config_hash runs on a v3 config without error and
    is deterministic across repeated calls."""
    cfg1 = V3ExperimentConfig.model_validate(minimal_cfg_dict)
    cfg2 = V3ExperimentConfig.model_validate(minimal_cfg_dict)
    assert config_hash(cfg1) == config_hash(cfg2)
    # A changed field should change the hash.
    mutated = dict(minimal_cfg_dict)
    mutated["seed"] = 999
    cfg3 = V3ExperimentConfig.model_validate(mutated)
    assert config_hash(cfg1) != config_hash(cfg3)
