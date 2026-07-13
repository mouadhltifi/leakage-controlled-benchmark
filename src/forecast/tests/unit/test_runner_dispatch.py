"""Unit tests for :func:`forecast.experiments.runner.run_one_forecast_experiment`.

These tests verify that the M5 runner wires the **v3** components (not
v2's). We mock the heavy calls (``assemble_fold`` -> synthetic
FoldArtifacts, ``Trainer.fit`` -> zero-epoch stub) so each test finishes
in milliseconds.

Coverage (M5 the architecture spec):

* The runner constructs :class:`ForecastPredictor` (not v2's
  :class:`mmfp.models.predictor.Predictor`).
* The runner constructs :class:`ForecastTrainer`.
* The loss is produced by :func:`forecast.models.losses.build_loss`
  (i.e. a v3 :class:`ForecastLoss`, not v2's MultiTaskLoss /
  VolatilityLoss).
* Callbacks monitor ``val_pinball_loss`` in ``min`` mode -- not
  ``val_mcc`` (v2 default).
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

import forecast.experiments.runner as runner_module
from forecast.config.schema import V3ExperimentConfig
from forecast.experiments.runner import run_one_forecast_experiment
from forecast.models.losses import ForecastLoss
from forecast.models.predictor import ForecastPredictor
from forecast.training.trainer import ForecastTrainer
from mmfp.data.assemble import FoldArtifacts
from mmfp.models.predictor import Predictor as V2Predictor
from mmfp.training.callbacks import (
    BestModelCheckpoint,
    EarlyStopping,
    LRLogger,
)


# ---------------------------------------------------------------------------
# Capture harness: a monkeypatched Trainer that records its arguments and
# returns a minimal History without actually training.
# ---------------------------------------------------------------------------


class _CapturedArgs:
    """Tiny holder recording what Trainer.__init__ and Trainer.fit saw."""

    def __init__(self) -> None:
        self.trainer_cls: type | None = None
        self.model: Any = None
        self.loss_fn: Any = None
        self.callbacks: list[Any] = []
        self.evaluate_called: bool = False


def _install_stub_trainer_and_evaluate(
    monkeypatch: pytest.MonkeyPatch, captured: _CapturedArgs,
) -> None:
    """Replace ForecastTrainer and evaluate_on_test_forecast with stubs.

    The stub trainer captures its construction args and returns a minimal
    :class:`~mmfp.training.trainer.History` from ``fit`` without executing
    any epochs.
    """
    import mmfp.training.trainer as v2_trainer_mod

    original_trainer_init = ForecastTrainer.__init__

    def _capturing_init(self, *args, **kwargs):  # type: ignore[no-redef]
        original_trainer_init(self, *args, **kwargs)
        captured.trainer_cls = type(self)
        captured.model = self.model
        captured.loss_fn = self.loss_fn
        captured.callbacks = list(self.callbacks)

    def _stub_fit(self, train_loader, val_loader):  # noqa: ARG001
        return v2_trainer_mod.History(
            train_loss=[0.1],
            val_loss=[0.1],
            val_metrics={"pinball_loss": [0.1]},
            best_epoch=0,
            best_metric_value=0.1,
            primary_metric="val_pinball_loss",
            monitor_mode="min",
        )

    def _stub_evaluate(*, model, test_loader, cfg, device):  # noqa: ARG001
        captured.evaluate_called = True
        return {
            "mcc": 0.0,
            "accuracy": 0.5,
            "f1": 0.0,
            "r2": 0.0,
            "rmse": 0.01,
            "sharpe_ratio": 0.0,
            "vol_rmse": 0.01,
            "vol_r2": 0.0,
            "quantile_coverage_80": 0.8,
            "quantile_interval_width": 1.0,
            "pinball_loss": 0.1,
        }

    monkeypatch.setattr(ForecastTrainer, "__init__", _capturing_init)
    monkeypatch.setattr(ForecastTrainer, "fit", _stub_fit)
    monkeypatch.setattr(
        runner_module, "evaluate_on_test_forecast", _stub_evaluate,
    )


def _install_stub_assemble(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifacts_factory: Callable[..., FoldArtifacts],
) -> None:
    """Replace ``assemble_fold`` with a factory that emits synthetic artifacts."""

    def _stub_assemble(cfg):  # noqa: ARG001
        # Use a small-fast synthetic fold sized for lookback=20.
        return synthetic_artifacts_factory(
            n_tickers=3, n_days=200, lookback=20, seed=cfg.seed,
        )

    monkeypatch.setattr(runner_module, "assemble_fold", _stub_assemble)


def _make_dispatch_cfg(minimal_cfg_dict: dict[str, Any]) -> V3ExperimentConfig:
    """Cheap cfg for the dispatch tests: lookback=20, hidden=16, cpu."""
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in minimal_cfg_dict.items()}
    d["forecast"] = dict(d["forecast"])
    d["forecast"]["lookback"] = 20
    d["forecast"]["hidden_dim"] = 16
    d["data"] = dict(d["data"])
    d["data"]["lookback"] = 20
    d["training"] = dict(d["training"])
    d["training"]["batch_size"] = 16
    d["training"]["max_epochs"] = 1
    d["training"]["min_epochs"] = 0
    d["training"]["patience"] = 1
    d["training"]["scheduler"] = "none"
    d["head"] = dict(d["head"])
    d["head"]["targets"] = ["return"]
    d["device"] = "cpu"
    return V3ExperimentConfig.model_validate(d)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_runner_builds_forecast_predictor_not_v2_predictor(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifacts_factory,
    minimal_cfg_dict,
):
    captured = _CapturedArgs()
    _install_stub_assemble(monkeypatch, synthetic_artifacts_factory)
    _install_stub_trainer_and_evaluate(monkeypatch, captured)

    cfg = _make_dispatch_cfg(minimal_cfg_dict)
    rec = run_one_forecast_experiment(cfg)

    assert isinstance(captured.model, ForecastPredictor), (
        f"Expected ForecastPredictor, got {type(captured.model).__name__}"
    )
    # And importantly NOT the v2 Predictor.
    assert not isinstance(captured.model, V2Predictor)
    # Result still populated.
    assert rec.quantile_coverage_80 is not None
    assert rec.quantile_interval_width is not None


def test_runner_uses_forecast_trainer(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifacts_factory,
    minimal_cfg_dict,
):
    captured = _CapturedArgs()
    _install_stub_assemble(monkeypatch, synthetic_artifacts_factory)
    _install_stub_trainer_and_evaluate(monkeypatch, captured)

    cfg = _make_dispatch_cfg(minimal_cfg_dict)
    run_one_forecast_experiment(cfg)

    assert captured.trainer_cls is ForecastTrainer


def test_runner_builds_v3_forecast_loss(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifacts_factory,
    minimal_cfg_dict,
):
    captured = _CapturedArgs()
    _install_stub_assemble(monkeypatch, synthetic_artifacts_factory)
    _install_stub_trainer_and_evaluate(monkeypatch, captured)

    cfg = _make_dispatch_cfg(minimal_cfg_dict)
    run_one_forecast_experiment(cfg)

    assert isinstance(captured.loss_fn, ForecastLoss), (
        f"Expected ForecastLoss, got {type(captured.loss_fn).__name__}"
    )


def test_runner_callbacks_monitor_val_pinball_loss_min(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifacts_factory,
    minimal_cfg_dict,
):
    captured = _CapturedArgs()
    _install_stub_assemble(monkeypatch, synthetic_artifacts_factory)
    _install_stub_trainer_and_evaluate(monkeypatch, captured)

    cfg = _make_dispatch_cfg(minimal_cfg_dict)
    run_one_forecast_experiment(cfg)

    # We expect exactly EarlyStopping + BestModelCheckpoint + LRLogger.
    kinds = {type(cb) for cb in captured.callbacks}
    assert EarlyStopping in kinds
    assert BestModelCheckpoint in kinds
    assert LRLogger in kinds

    es = next(cb for cb in captured.callbacks if isinstance(cb, EarlyStopping))
    assert es.monitor == "val_pinball_loss"
    assert es.mode == "min"

    bmc = next(
        cb for cb in captured.callbacks if isinstance(cb, BestModelCheckpoint)
    )
    assert bmc.monitor == "val_pinball_loss"
    assert bmc.mode == "min"
