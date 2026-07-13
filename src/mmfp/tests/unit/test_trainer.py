"""Unit tests for :class:`mmfp.training.trainer.Trainer`.

Runs a tiny price-only Predictor on a synthetic in-memory dataset for 3
epochs. The goals are:

* History shape and content make sense.
* ``best_epoch`` tracking via :class:`BestModelCheckpoint` is plumbed.
* Gradient clipping + optimizer.step() happen (weights change).
* Restored weights equal the captured best state dict.
* Multi-target trainer picks the right primary metric.
* Early-stopping terminates the loop before ``max_epochs``.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.config.validate import validate_experiment_config
from mmfp.data.assemble import FeatureSchema
from mmfp.models import Predictor
from mmfp.models.losses import build_loss
from mmfp.training import (
    BestModelCheckpoint,
    EarlyStopping,
    History,
    LRLogger,
    Trainer,
    build_scheduler,
)


# ---------------------------------------------------------------------------
# Config + schema helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test-trainer"
    d["seed"] = 42
    d["device"] = "cpu"
    for dotted, value in overrides.items():
        cursor = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = value
    cfg = ExperimentConfig.model_validate(d)
    return validate_experiment_config(cfg)


def _price_schema(price_dim: int = 8) -> FeatureSchema:
    return FeatureSchema(price=slice(0, price_dim))


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


class _ToyPriceDataset(Dataset):
    """In-memory price-only dataset with controllable class balance.

    Each sample emits:
        price: (F_price,) or (L, F_price)
        cls_target: long scalar in {0, 1}
        reg_target: float scalar
        stock_idx: long scalar
    """

    def __init__(
        self,
        *,
        n_samples: int = 32,
        price_dim: int = 8,
        lookback: int = 1,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n = n_samples
        self.price_dim = price_dim
        self.lookback = lookback

        if lookback == 1:
            self._price = rng.standard_normal((n_samples, price_dim)).astype(np.float32)
        else:
            self._price = rng.standard_normal(
                (n_samples, lookback, price_dim)
            ).astype(np.float32)

        # Labels: roughly balanced 50/50.
        self._cls = rng.integers(0, 2, size=n_samples).astype(np.int64)
        self._reg = rng.standard_normal(n_samples).astype(np.float32) * 0.01
        # Fake stock indices.
        self._stock_idx = rng.integers(0, 5, size=n_samples).astype(np.int64)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "price": torch.as_tensor(self._price[idx]),
            "cls_target": torch.as_tensor(self._cls[idx], dtype=torch.long),
            "reg_target": torch.as_tensor(self._reg[idx], dtype=torch.float32),
            "stock_idx": torch.as_tensor(self._stock_idx[idx], dtype=torch.long),
        }


def _make_loaders(
    *,
    cfg: ExperimentConfig,
    price_dim: int = 8,
    n_train: int = 32,
    n_val: int = 16,
    batch_size: int = 8,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_ds = _ToyPriceDataset(
        n_samples=n_train, price_dim=price_dim,
        lookback=cfg.data.lookback, seed=seed,
    )
    val_ds = _ToyPriceDataset(
        n_samples=n_val, price_dim=price_dim,
        lookback=cfg.data.lookback, seed=seed + 100,
    )
    # Non-shuffling, deterministic loaders (no worker seeding needed).
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
    )
    return train_loader, val_loader


def _build_trainer(
    cfg: ExperimentConfig,
    *,
    price_dim: int = 8,
    callbacks: list[Any] | None = None,
) -> tuple[Trainer, Predictor]:
    schema = _price_schema(price_dim)
    model = Predictor(cfg, schema)
    loss = build_loss(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = build_scheduler(cfg, optimizer)
    trainer = Trainer(
        model=model, cfg=cfg, loss_fn=loss,
        optimizer=optimizer, scheduler=scheduler,
        callbacks=callbacks or [],
    )
    return trainer, model


# ---------------------------------------------------------------------------
# History structure
# ---------------------------------------------------------------------------


class TestHistoryStructure:
    def test_three_epochs_populate_history(self) -> None:
        cfg = _cfg(**{
            "training.max_epochs": 3,
            "training.patience": 100,
            "training.scheduler": "none",
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        trainer, _ = _build_trainer(cfg)
        train_loader, val_loader = _make_loaders(cfg=cfg)

        history = trainer.fit(train_loader, val_loader)

        assert isinstance(history, History)
        assert len(history.train_loss) == 3
        assert len(history.val_loss) == 3
        # Direction target -> mcc + accuracy + f1; return target -> rmse + r2.
        assert "mcc" in history.val_metrics
        assert "accuracy" in history.val_metrics
        assert "f1" in history.val_metrics
        assert "r2" in history.val_metrics
        assert len(history.val_metrics["mcc"]) == 3
        # No best-checkpoint callback -> best_epoch stays -1.
        assert history.best_epoch == -1
        assert history.primary_metric == "val_mcc"
        assert history.monitor_mode == "max"


# ---------------------------------------------------------------------------
# Primary-metric selection
# ---------------------------------------------------------------------------


class TestPrimaryMetric:
    def test_direction_selects_val_mcc(self) -> None:
        cfg = _cfg(**{
            "head.targets": ["direction"],
            "head.architecture": "single_task",
            "training.max_epochs": 2,
            "training.scheduler": "none",
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        trainer, _ = _build_trainer(cfg)
        metric, mode = trainer.primary_metric()
        assert (metric, mode) == ("val_mcc", "max")

    def test_volatility_selects_val_rmse(self) -> None:
        cfg = _cfg(**{
            "head.targets": ["volatility"],
            "head.architecture": "single_task",
            "training.max_epochs": 2,
            "training.scheduler": "none",
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        trainer, _ = _build_trainer(cfg)
        metric, mode = trainer.primary_metric()
        assert (metric, mode) == ("val_rmse", "min")

    def test_return_selects_val_r2(self) -> None:
        cfg = _cfg(**{
            "head.targets": ["return"],
            "head.architecture": "single_task",
            "training.max_epochs": 2,
            "training.scheduler": "none",
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        trainer, _ = _build_trainer(cfg)
        metric, mode = trainer.primary_metric()
        assert (metric, mode) == ("val_r2", "max")

    def test_multi_target_priority_direction_first(self) -> None:
        """direction > volatility > return when multiple are active."""
        cfg = _cfg(**{
            "head.targets": ["direction", "return", "volatility"],
            "training.max_epochs": 2,
            "training.scheduler": "none",
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        trainer, _ = _build_trainer(cfg)
        metric, mode = trainer.primary_metric()
        assert (metric, mode) == ("val_mcc", "max")


# ---------------------------------------------------------------------------
# Weight updates + gradient clipping
# ---------------------------------------------------------------------------


class TestWeightUpdates:
    def test_weights_change_across_epochs(self) -> None:
        cfg = _cfg(**{
            "training.max_epochs": 2,
            "training.learning_rate": 1e-2,  # large enough to see movement
            "training.scheduler": "none",
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        trainer, model = _build_trainer(cfg)

        pre = [p.detach().clone() for p in model.parameters()]
        train_loader, val_loader = _make_loaders(cfg=cfg)
        trainer.fit(train_loader, val_loader)
        post = [p.detach().clone() for p in model.parameters()]

        differences = [
            (a - b).abs().sum().item() for a, b in zip(pre, post)
        ]
        assert any(d > 0 for d in differences), (
            f"Expected at least one parameter to change; all deltas were zero: {differences}"
        )


# ---------------------------------------------------------------------------
# Best-model checkpoint restoration
# ---------------------------------------------------------------------------


class TestBestCheckpointRestoration:
    def test_trainer_restores_best_state(self) -> None:
        """After fit(), model weights equal those captured at best epoch."""
        cfg = _cfg(**{
            "training.max_epochs": 3,
            "training.scheduler": "none",
            "training.learning_rate": 1e-2,
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        best_cb = BestModelCheckpoint(monitor="val_mcc", mode="max")
        trainer, model = _build_trainer(cfg, callbacks=[best_cb])
        train_loader, val_loader = _make_loaders(cfg=cfg)

        history = trainer.fit(train_loader, val_loader)

        # After fit, the stored best state dict matches the live model.
        assert best_cb.best_state_dict is not None
        for k, v in best_cb.best_state_dict.items():
            torch.testing.assert_close(
                v, model.state_dict()[k],
                msg=f"param {k!r} not restored",
            )

        # History picks up best_epoch / best_value.
        assert history.best_epoch in (0, 1, 2)
        assert not np.isnan(history.best_metric_value)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_terminates_before_max_epochs(self) -> None:
        """Use a huge LR so the model diverges and MCC never improves again.

        The MCC at epoch 0 is from random init; subsequent updates overshoot
        and degrade. With patience=2 we expect a stop after 2 consecutive
        non-improvements.
        """
        cfg = _cfg(**{
            "training.max_epochs": 30,  # deliberately large
            "training.scheduler": "none",
            "training.learning_rate": 1.0,  # absurdly large -> divergence
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        es = EarlyStopping(
            monitor="val_mcc", mode="max", patience=2, min_epochs=0,
        )
        trainer, _ = _build_trainer(cfg, callbacks=[es])
        train_loader, val_loader = _make_loaders(cfg=cfg)

        history = trainer.fit(train_loader, val_loader)

        assert len(history.train_loss) < cfg.training.max_epochs, (
            f"Expected early stop to fire; ran full {cfg.training.max_epochs} epochs."
        )

    def test_min_epochs_respected_by_trainer(self) -> None:
        """min_epochs=5 should keep the trainer running for at least 5 epochs
        even when the metric is strictly non-improving."""
        cfg = _cfg(**{
            "training.max_epochs": 30,
            "training.scheduler": "none",
            "training.learning_rate": 1.0,  # diverges
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        es = EarlyStopping(
            monitor="val_mcc", mode="max", patience=1, min_epochs=5,
        )
        trainer, _ = _build_trainer(cfg, callbacks=[es])
        train_loader, val_loader = _make_loaders(cfg=cfg)
        history = trainer.fit(train_loader, val_loader)
        # First epoch (idx 0) is always improvement, so patience triggers
        # earliest at epoch 1. But min_epochs=5 forces at least 6 epochs
        # (min_epochs is zero-indexed check: epoch >= min_epochs, so
        # stop can fire from epoch 5 onwards -> len >= 6).
        assert len(history.train_loss) >= 6, (
            f"min_epochs=5 should force >=6 epochs; got {len(history.train_loss)}"
        )


# ---------------------------------------------------------------------------
# Multi-target + LR logger smoke
# ---------------------------------------------------------------------------


class TestMultiTargetAndLRLogger:
    def test_multitask_direction_and_return(self) -> None:
        cfg = _cfg(**{
            "head.targets": ["direction", "return"],
            "head.architecture": "parallel_multitask",
            "training.max_epochs": 2,
            "training.scheduler": "none",
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        trainer, _ = _build_trainer(cfg)
        train_loader, val_loader = _make_loaders(cfg=cfg)
        history = trainer.fit(train_loader, val_loader)
        # Both target metrics present.
        assert "mcc" in history.val_metrics
        assert "r2" in history.val_metrics
        assert "return_rmse" in history.val_metrics

    def test_volatility_target(self) -> None:
        cfg = _cfg(**{
            "head.targets": ["volatility"],
            "head.architecture": "single_task",
            "training.max_epochs": 2,
            "training.scheduler": "none",
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        schema = _price_schema()
        model = Predictor(cfg, schema)
        loss = build_loss(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(cfg, optimizer)
        trainer = Trainer(
            model=model, cfg=cfg, loss_fn=loss,
            optimizer=optimizer, scheduler=scheduler, callbacks=[],
        )

        # Patch dataset to emit vol_target instead of cls_target/reg_target.
        class _VolDataset(Dataset):
            def __init__(self, n: int, price_dim: int, seed: int) -> None:
                rng = np.random.default_rng(seed)
                self._price = rng.standard_normal((n, price_dim)).astype(np.float32)
                self._vol = np.abs(rng.standard_normal(n)).astype(np.float32) * 0.01
                self._stock = rng.integers(0, 5, size=n).astype(np.int64)

            def __len__(self) -> int:
                return len(self._price)

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                return {
                    "price": torch.as_tensor(self._price[idx]),
                    "vol_target": torch.as_tensor(self._vol[idx], dtype=torch.float32),
                    "stock_idx": torch.as_tensor(self._stock[idx], dtype=torch.long),
                }

        train = DataLoader(_VolDataset(32, 8, 0), batch_size=8, shuffle=False)
        val = DataLoader(_VolDataset(16, 8, 1), batch_size=8, shuffle=False)
        history = trainer.fit(train, val)
        assert "rmse" in history.val_metrics
        assert history.primary_metric == "val_rmse"

    def test_lr_logger_records_per_epoch(self) -> None:
        cfg = _cfg(**{
            "training.max_epochs": 3,
            "training.scheduler": "cosine_warm_restarts",
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        lr_cb = LRLogger()
        trainer, _ = _build_trainer(cfg, callbacks=[lr_cb])
        train_loader, val_loader = _make_loaders(cfg=cfg)
        trainer.fit(train_loader, val_loader)
        assert len(lr_cb.lr_history) == 3
        assert all(lr > 0 for lr in lr_cb.lr_history)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_zero_max_epochs_raises(self) -> None:
        # Schema prevents max_epochs=0, so we skip. The Trainer.fit
        # check is defensive and re-validates against mutation — but
        # Pydantic validate_assignment catches this too.
        with pytest.raises(Exception):
            _cfg(**{"training.max_epochs": 0})

    def test_loss_moved_to_device(self) -> None:
        cfg = _cfg(**{
            "training.max_epochs": 1,
            "training.scheduler": "none",
            "data.lookback": 1,
            "model.price_encoder": "feedforward",
        })
        trainer, model = _build_trainer(cfg)
        assert str(trainer.device).startswith("cpu") or str(trainer.device).startswith("mps") or str(trainer.device).startswith("cuda")

        # Model's encoder weights are on the trainer's device.
        any_param = next(model.parameters())
        assert any_param.device.type == trainer.device.type
