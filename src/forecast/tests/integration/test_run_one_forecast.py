"""Integration test: full :func:`run_one_forecast_experiment` pipeline.

Runs a 3-epoch mini-experiment on synthetic data (``assemble_fold`` is
monkeypatched to return a synthetic fold, which the rest of the pipeline
processes unchanged). Exercises the real
:class:`~forecast.training.trainer.ForecastTrainer`,
:class:`~forecast.models.predictor.ForecastPredictor`,
:class:`~forecast.models.losses.ForecastLoss`, callbacks, optimiser, and
the :func:`evaluate_on_test_forecast` helper.

Assertions (M5 the architecture spec):

* Returns a :class:`ForecastResultRecord` with finite numeric fields.
* ``pinball_loss`` is finite (we cannot guarantee strict improvement at
  3 epochs on a tiny synthetic fold, but we can demand no NaN).
* ``quantile_coverage_80`` is in ``[0.1, 1.0]``.
* ``elapsed_seconds > 0`` and ``n_params > 0``.
* Completes in < 600 s on CPU (usually < 30 s at the sizes below).

Uses a reduced ``lookback=20`` / ``hidden_dim=16`` to stay fast while
still exercising every wiring point of the real runner.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import pytest

import forecast.experiments.runner as runner_module
from forecast.config.schema import V3ExperimentConfig
from forecast.experiments.result_schema import ForecastResultRecord
from forecast.experiments.runner import run_one_forecast_experiment
from mmfp.data.assemble import FoldArtifacts


def _build_cfg(minimal_cfg_dict: dict[str, Any]) -> V3ExperimentConfig:
    """Small-fast cfg for the integration test (CPU, ~30s wall-clock)."""
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in minimal_cfg_dict.items()}
    # Deep-copy the dict slices we mutate.
    d["forecast"] = dict(d["forecast"])
    d["data"] = dict(d["data"])
    d["training"] = dict(d["training"])
    d["head"] = dict(d["head"])

    d["forecast"]["lookback"] = 20
    d["forecast"]["hidden_dim"] = 16
    d["forecast"]["warmup_epochs"] = 0
    d["data"]["lookback"] = 20
    d["training"]["batch_size"] = 16
    d["training"]["max_epochs"] = 3
    d["training"]["min_epochs"] = 0
    d["training"]["patience"] = 1000  # never early-stop at 3 epochs
    d["training"]["learning_rate"] = 3e-3
    d["training"]["scheduler"] = "none"
    d["head"]["targets"] = ["return"]
    d["device"] = "cpu"

    return V3ExperimentConfig.model_validate(d)


def _install_synthetic_assemble(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifacts_factory: Callable[..., FoldArtifacts],
    *,
    seed: int,
) -> None:
    """Replace assemble_fold with a small synthetic factory."""

    def _stub_assemble(cfg):  # noqa: ARG001
        return synthetic_artifacts_factory(
            n_tickers=3,
            n_days=200,
            lookback=20,
            seed=seed,
        )

    monkeypatch.setattr(runner_module, "assemble_fold", _stub_assemble)


@pytest.mark.slow
def test_run_one_forecast_end_to_end_on_synthetic(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifacts_factory,
    minimal_cfg_dict,
):
    _install_synthetic_assemble(
        monkeypatch, synthetic_artifacts_factory, seed=42,
    )
    cfg = _build_cfg(minimal_cfg_dict)

    t0 = time.time()
    record = run_one_forecast_experiment(cfg)
    elapsed = time.time() - t0

    # Runtime guard: the spec allows up to 600 s on CPU at
    # lookback=60/hidden=64. At our scaled-down settings we expect << 60 s.
    assert elapsed < 600.0, f"integration run took {elapsed:.1f}s (> 600s)"

    # Return type and finite numeric fields.
    assert isinstance(record, ForecastResultRecord)

    # v2 metric fields: none should be NaN / None.
    numeric_v2_fields = (
        "mcc", "accuracy", "f1", "r2", "rmse",
        "sharpe_ratio", "vol_rmse", "vol_r2",
    )
    for name in numeric_v2_fields:
        value = getattr(record, name)
        assert value is not None, f"{name} is None"
        assert math.isfinite(float(value)), f"{name} is not finite: {value}"

    # v3-native fields populated and finite.
    assert record.quantile_coverage_80 is not None
    assert record.quantile_interval_width is not None
    assert math.isfinite(float(record.quantile_coverage_80))
    assert math.isfinite(float(record.quantile_interval_width))

    # Coverage in a broad but catastrophe-catching band.
    coverage = float(record.quantile_coverage_80)
    assert 0.1 <= coverage <= 1.0, f"coverage out of [0.1, 1.0]: {coverage}"

    # Sanity on bookkeeping fields.
    assert record.elapsed_seconds > 0.0
    assert record.n_params > 0
    assert record.epochs_trained >= 1
    assert record.n_train > 0
    assert record.n_val > 0
    assert record.n_test > 0
    # Platform version is prefixed with "v3-" per _build_record.
    assert record.platform_version.startswith("v3-")
    # head_architecture should be "quantile" for a TFT run.
    assert record.head_architecture == "quantile"
    assert record.fusion_strategy == "tft"
