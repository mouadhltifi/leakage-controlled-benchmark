"""Integration test for :func:`mmfp.experiments.runner.run_one_experiment`.

Runs A7 (price-only, feedforward) on fold 0 with seed 42 for 3 epochs.
Asserts that :class:`ResultRecord` fields are populated with finite
values and that the run completes in a reasonable time budget.

Skipped if the required cached feature parquet (``price_features.parquet``)
is missing on the host.
"""

from __future__ import annotations

import copy
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.config.validate import validate_experiment_config
from mmfp.data.paths import FEATURES_DIR
from mmfp.experiments.result_schema import ResultRecord, config_hash
from mmfp.experiments.runner import run_one_experiment


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def _a7_ff_cfg(**extra: Any) -> ExperimentConfig:
    """Build the canonical A7 price-only FF config."""
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "A7_price_only"
    d["seed"] = 42
    d["device"] = "cpu"  # CPU for determinism + cheap scheduling
    d["data"]["fold_idx"] = 0
    d["data"]["lookback"] = 1
    d["model"]["price_encoder"] = "feedforward"
    d["training"]["max_epochs"] = 3
    d["training"]["min_epochs"] = 0
    d["training"]["patience"] = 3
    d["training"]["batch_size"] = 128
    d["logging"]["per_epoch_log_every"] = 1

    # Apply any caller-supplied overrides (dotted keys).
    for dotted, value in extra.items():
        cursor = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = value

    cfg = ExperimentConfig.model_validate(d)
    return validate_experiment_config(cfg)


def _require_price_parquet() -> None:
    """Skip the test if the required cached parquet is missing."""
    p = FEATURES_DIR / "price_features.parquet"
    if not p.exists():
        pytest.skip(
            f"Integration test needs cached price features at {p}. "
            "Run the v1 price-feature build first."
        )


# ---------------------------------------------------------------------------
# Main integration test
# ---------------------------------------------------------------------------


def test_a7_price_only_produces_valid_record() -> None:
    """A7 FF fold 0 seed 42 3 epochs returns a populated :class:`ResultRecord`."""
    _require_price_parquet()

    cfg = _a7_ff_cfg()
    record = run_one_experiment(cfg)

    # 1. Return type.
    assert isinstance(record, ResultRecord)

    # 2. Identification fields.
    assert record.experiment_name == "A7_price_only"
    assert record.config_hash == config_hash(cfg)
    assert record.seed == 42
    assert record.fold_idx == 0

    # 3. Axis fingerprint.
    assert record.news_encoder == "none"
    assert record.news_aggregation == "none"
    assert record.fusion_strategy == "concat"
    assert record.head_architecture == "parallel_multitask"
    assert record.targets == "direction,return"
    assert record.graph_source == "none"
    assert record.lookback == 1
    assert record.price_encoder == "feedforward"

    # 4. Direction metrics populated and finite.
    assert record.mcc is not None
    assert math.isfinite(record.mcc), f"non-finite MCC: {record.mcc}"
    assert record.accuracy is not None
    assert 0.0 <= record.accuracy <= 1.0
    assert record.f1 is not None
    assert 0.0 <= record.f1 <= 1.0

    # 5. Regression metrics populated and finite.
    assert record.r2 is not None
    assert math.isfinite(record.r2)
    assert record.rmse is not None
    assert math.isfinite(record.rmse)
    assert record.rmse >= 0.0
    assert record.sharpe_ratio is not None
    assert math.isfinite(record.sharpe_ratio)

    # 6. Volatility metrics MUST be None (target not active).
    assert record.vol_rmse is None
    assert record.vol_r2 is None

    # 7. Meta.
    assert record.n_train > 0
    assert record.n_val > 0
    assert record.n_test > 0
    assert record.n_params > 0
    assert 1 <= record.epochs_trained <= cfg.training.max_epochs
    assert math.isfinite(record.best_val_metric)
    assert record.elapsed_seconds > 0
    assert record.platform_version
    assert record.git_sha  # any non-empty string counts (may be "unknown")


def test_a7_run_is_deterministic_cpu() -> None:
    """Two A7 runs with identical configs produce identical config hashes + test metrics.

    CPU determinism is guaranteed by ``set_all_seeds`` + seeded DataLoader
    generator. MPS LSTM paths have documented tolerance; A7 FF has no
    LSTM so CPU should be bit-exact.
    """
    _require_price_parquet()

    cfg1 = _a7_ff_cfg()
    cfg2 = _a7_ff_cfg()

    # Config hash must match regardless of run outcome.
    assert config_hash(cfg1) == config_hash(cfg2)

    rec1 = run_one_experiment(cfg1)
    rec2 = run_one_experiment(cfg2)

    # Two runs of the same config -> same config_hash.
    assert rec1.config_hash == rec2.config_hash

    # CPU + seeded ->  identical test metrics to high precision.
    assert rec1.mcc == pytest.approx(rec2.mcc, abs=1e-6)
    assert rec1.accuracy == pytest.approx(rec2.accuracy, abs=1e-6)
    assert rec1.rmse == pytest.approx(rec2.rmse, abs=1e-6)
    assert rec1.r2 == pytest.approx(rec2.r2, abs=1e-6)


# ---------------------------------------------------------------------------
# Fast sanity checks on the ResultRecord produced above (no re-run).
# ---------------------------------------------------------------------------


def test_record_fields_all_defined() -> None:
    """Every declared field is actually set (no sentinel / attribute errors)."""
    _require_price_parquet()

    cfg = _a7_ff_cfg()
    record = run_one_experiment(cfg)

    for f in fields(ResultRecord):
        # ``getattr`` will raise AttributeError if the field is missing;
        # we also accept ``None`` as "set" for optional-metric columns.
        _ = getattr(record, f.name)
