"""Tests for :mod:`mmfp.experiments.result_schema`."""

from __future__ import annotations

import copy
import csv
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.experiments.result_schema import (
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


def _make_cfg(**overrides: Any) -> ExperimentConfig:
    """Build a validated ExperimentConfig from defaults + flat overrides."""
    d = copy.deepcopy(DEFAULT_CONFIG)
    for dotted, value in overrides.items():
        cursor = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = value
    return ExperimentConfig.model_validate(d)


def _make_record(**overrides: Any) -> ResultRecord:
    """Build a ResultRecord populated with canonical values + overrides."""
    base: dict[str, Any] = {
        "experiment_name": "A7_price_only",
        "config_hash": "0" * 64,
        "seed": 42,
        "fold_idx": 0,
        "news_encoder": "none",
        "news_aggregation": "none",
        "fusion_strategy": "concat",
        "head_architecture": "parallel_multitask",
        "targets": "direction,return",
        "graph_source": "none",
        "lookback": 1,
        "price_encoder": "feedforward",
        "mcc": 0.01,
        "accuracy": 0.51,
        "f1": 0.55,
        "r2": -0.001,
        "rmse": 0.012,
        "sharpe_ratio": 0.5,
        "vol_rmse": None,
        "vol_r2": None,
        "n_train": 1000,
        "n_val": 200,
        "n_test": 300,
        "n_params": 12345,
        "epochs_trained": 42,
        "best_val_metric": 0.02,
        "elapsed_seconds": 12.5,
        "platform_version": "2.0.0-dev",
        "git_sha": "abcdef012345",
    }
    base.update(overrides)
    return ResultRecord(**base)


# ---------------------------------------------------------------------------
# ResultRecord contract
# ---------------------------------------------------------------------------


class TestResultRecordShape:
    def test_all_spec_fields_present(self) -> None:
        """Every field named in spec Section 3.12 is declared on the dataclass."""
        expected = {
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
            # volatility metrics
            "vol_rmse", "vol_r2",
            # meta
            "n_train", "n_val", "n_test", "n_params", "epochs_trained",
            "best_val_metric", "elapsed_seconds",
            "platform_version", "git_sha",
        }
        got = {f.name for f in fields(ResultRecord)}
        assert got == expected, f"mismatch: only in spec={expected - got}, only in dc={got - expected}"

    def test_none_metrics_round_trip_empty_string(self, tmp_path: Path) -> None:
        """``None`` metric values serialise to empty strings in the CSV."""
        rec = _make_record(vol_rmse=None, vol_r2=None)
        out = tmp_path / "r.csv"
        append_result(out, rec)
        rows = load_results(out)
        assert len(rows) == 1
        assert rows[0]["vol_rmse"] == ""
        assert rows[0]["vol_r2"] == ""
        # Non-None metrics serialise to their repr.
        assert rows[0]["mcc"] == "0.01"

    def test_column_names_matches_fields(self) -> None:
        got = ResultRecord.column_names()
        dc_order = [f.name for f in fields(ResultRecord)]
        assert got == dc_order


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------


class TestConfigHash:
    def test_stable_across_reorderings(self) -> None:
        """Reordering sibling dict keys must not change the hash.

        ``model_dump(mode="json")`` + ``sort_keys=True`` guarantees a
        canonical serialisation. We spot-check by building two dict
        copies with different insertion orders and asserting their
        hashes match after pydantic construction.
        """
        d1 = copy.deepcopy(DEFAULT_CONFIG)
        d2 = copy.deepcopy(DEFAULT_CONFIG)

        # Shuffle top-level key order in one of them.
        reordered = {}
        for k in reversed(list(d1.keys())):
            reordered[k] = d1[k]
        d1 = reordered

        cfg1 = ExperimentConfig.model_validate(d1)
        cfg2 = ExperimentConfig.model_validate(d2)
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_different_values_different_hash(self) -> None:
        cfg1 = _make_cfg(**{"seed": 42})
        cfg2 = _make_cfg(**{"seed": 43})
        assert config_hash(cfg1) != config_hash(cfg2)

    def test_hex_digest_length(self) -> None:
        """sha256 hex is always 64 chars."""
        cfg = _make_cfg()
        h = config_hash(cfg)
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_nested_field_changes_hash(self) -> None:
        """Changing a nested field (e.g. training.batch_size) must flip the hash."""
        cfg1 = _make_cfg()
        cfg2 = _make_cfg(**{"training.batch_size": 128})
        assert config_hash(cfg1) != config_hash(cfg2)


# ---------------------------------------------------------------------------
# record_fingerprint
# ---------------------------------------------------------------------------


class TestRecordFingerprint:
    def test_unique_per_identification_tuple(self) -> None:
        """Distinct (name, hash, seed, fold) combos yield distinct fingerprints."""
        r1 = _make_record()
        r2 = _make_record(seed=43)
        r3 = _make_record(fold_idx=1)
        r4 = _make_record(config_hash="1" * 64)
        r5 = _make_record(experiment_name="A4_macro")
        fps = {
            record_fingerprint(r1),
            record_fingerprint(r2),
            record_fingerprint(r3),
            record_fingerprint(r4),
            record_fingerprint(r5),
        }
        assert len(fps) == 5

    def test_stable_across_repeat_calls(self) -> None:
        r = _make_record()
        assert record_fingerprint(r) == record_fingerprint(r)

    def test_fingerprint_does_not_depend_on_metrics(self) -> None:
        """Two records identical in (name, hash, seed, fold) but different metrics
        must share a fingerprint — the fingerprint is for dedup, not content."""
        r1 = _make_record(mcc=0.01)
        r2 = _make_record(mcc=0.05)
        assert record_fingerprint(r1) == record_fingerprint(r2)


# ---------------------------------------------------------------------------
# fingerprint_from_row
# ---------------------------------------------------------------------------


class TestFingerprintFromRow:
    def test_agrees_with_record_fingerprint(self, tmp_path: Path) -> None:
        """Roundtrip: write a record, reload the dict, fingerprints must match."""
        rec = _make_record()
        out = tmp_path / "r.csv"
        append_result(out, rec)
        rows = load_results(out)
        assert len(rows) == 1
        fp_from_row = fingerprint_from_row(rows[0])
        assert fp_from_row == record_fingerprint(rec)

    def test_missing_key_raises(self) -> None:
        bad_row = {"experiment_name": "x"}
        with pytest.raises(ValueError, match="missing"):
            fingerprint_from_row(bad_row)

    def test_non_integer_seed_raises(self) -> None:
        bad_row = {
            "experiment_name": "x",
            "config_hash": "0" * 64,
            "seed": "not-a-number",
            "fold_idx": "0",
        }
        with pytest.raises(ValueError, match="missing|integer"):
            fingerprint_from_row(bad_row)


# ---------------------------------------------------------------------------
# append_result
# ---------------------------------------------------------------------------


class TestAppendResult:
    def test_header_then_row(self, tmp_path: Path) -> None:
        out = tmp_path / "r.csv"
        rec = _make_record()
        append_result(out, rec)

        with out.open() as fh:
            reader = csv.reader(fh)
            header = next(reader)
            data_row = next(reader)

        assert header == ResultRecord.column_names()
        # Smoke-check a few values against the record's to_row output.
        row_dict = _make_record().to_row()
        for i, key in enumerate(header):
            value = "" if row_dict[key] is None else str(row_dict[key])
            assert data_row[i] == value

    def test_multiple_rows_preserved(self, tmp_path: Path) -> None:
        out = tmp_path / "r.csv"
        r1 = _make_record(seed=42)
        r2 = _make_record(seed=43)
        append_result(out, r1)
        append_result(out, r2)

        rows = load_results(out)
        assert len(rows) == 2
        assert rows[0]["seed"] == "42"
        assert rows[1]["seed"] == "43"

    def test_incompatible_header_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "r.csv"
        append_result(out, _make_record())
        # Inject an incompatible row through the low-level writer.
        # ``atomic_csv_append`` rejects schema mismatch; because every
        # ResultRecord has the same field set we can't trigger that from
        # our side, but we can assert the contract is in place via the
        # shim call.
        from mmfp.utils.io import atomic_csv_append
        with pytest.raises(ValueError, match="Schemas must agree"):
            atomic_csv_append(out, {"bogus": 1})


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------


class TestLoadResults:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "does-not-exist.csv"
        assert load_results(out) == []

    def test_round_trip(self, tmp_path: Path) -> None:
        out = tmp_path / "r.csv"
        rec = _make_record()
        append_result(out, rec)
        rows = load_results(out)
        assert len(rows) == 1
        assert rows[0]["experiment_name"] == rec.experiment_name
        assert rows[0]["seed"] == str(rec.seed)
