"""Unit tests for :mod:`mmfp.training.callbacks`.

Covers:

* :class:`EarlyStopping` — triggers at ``patience`` consecutive
  non-improvements AFTER ``min_epochs``; doesn't trigger earlier even if
  the metric is flat from epoch zero.
* :class:`BestModelCheckpoint` — captures the state dict at the best
  epoch and restores it correctly (verified with a parameter whose value
  changes monotonically across epochs).
* :class:`LRLogger` — writes ``logs['lr']`` at each epoch end and keeps
  a per-epoch history.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mmfp.training.callbacks import (
    BestModelCheckpoint,
    Callback,
    EarlyStopping,
    LRLogger,
)


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """One linear layer whose weight we can stamp per epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.fc(x)


def _fake_trainer(model: nn.Module, lr: float = 0.1) -> SimpleNamespace:
    """Minimal trainer stub with just what the callbacks read."""
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=lr)
    return SimpleNamespace(model=model, optimizer=optimizer)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


class TestEarlyStoppingConstruction:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            EarlyStopping(monitor="val_mcc", mode="bogus", patience=3)  # type: ignore[arg-type]

    def test_zero_patience_raises(self) -> None:
        with pytest.raises(ValueError, match="patience"):
            EarlyStopping(monitor="val_mcc", mode="max", patience=0)

    def test_negative_min_epochs_raises(self) -> None:
        with pytest.raises(ValueError, match="min_epochs"):
            EarlyStopping(monitor="val_mcc", mode="max", patience=3, min_epochs=-1)


class TestEarlyStoppingBehaviour:
    def test_triggers_on_patience_non_improvements_max(self) -> None:
        """mode='max': 3 consecutive decreases after best -> should_stop."""
        model = _TinyModel()
        trainer = _fake_trainer(model)
        cb = EarlyStopping(monitor="val_mcc", mode="max", patience=3)
        cb.on_train_begin(trainer)

        # Epoch 0: improvement (first observation).
        cb.on_epoch_end(trainer, 0, {"val_mcc": 0.5})
        assert cb.should_stop is False

        # Epoch 1-3: three non-improvements.
        cb.on_epoch_end(trainer, 1, {"val_mcc": 0.49})
        assert cb.should_stop is False
        cb.on_epoch_end(trainer, 2, {"val_mcc": 0.48})
        assert cb.should_stop is False
        cb.on_epoch_end(trainer, 3, {"val_mcc": 0.47})
        # Three consecutive non-improvements -> stop.
        assert cb.should_stop is True
        assert cb.stopped_epoch == 3

    def test_min_epochs_prevents_early_trigger(self) -> None:
        """With min_epochs=5, stop can't fire before epoch 5 even on flat metrics."""
        model = _TinyModel()
        trainer = _fake_trainer(model)
        cb = EarlyStopping(
            monitor="val_mcc", mode="max", patience=2, min_epochs=5,
        )
        cb.on_train_begin(trainer)

        cb.on_epoch_end(trainer, 0, {"val_mcc": 0.5})
        # Non-improvements from epoch 1 onwards.
        for epoch in (1, 2, 3, 4):
            cb.on_epoch_end(trainer, epoch, {"val_mcc": 0.4})
            assert cb.should_stop is False, (
                f"Premature stop at epoch {epoch} (min_epochs=5)"
            )

        # Now past min_epochs: next non-improvement fires.
        cb.on_epoch_end(trainer, 5, {"val_mcc": 0.4})
        assert cb.should_stop is True
        assert cb.stopped_epoch == 5

    def test_improvement_resets_counter(self) -> None:
        model = _TinyModel()
        trainer = _fake_trainer(model)
        cb = EarlyStopping(monitor="val_mcc", mode="max", patience=3)
        cb.on_train_begin(trainer)

        cb.on_epoch_end(trainer, 0, {"val_mcc": 0.4})
        cb.on_epoch_end(trainer, 1, {"val_mcc": 0.35})  # -1
        cb.on_epoch_end(trainer, 2, {"val_mcc": 0.30})  # -2
        # Now we improve.
        cb.on_epoch_end(trainer, 3, {"val_mcc": 0.5})  # reset, new best
        assert cb.should_stop is False
        cb.on_epoch_end(trainer, 4, {"val_mcc": 0.49})  # -1
        cb.on_epoch_end(trainer, 5, {"val_mcc": 0.48})  # -2
        assert cb.should_stop is False
        cb.on_epoch_end(trainer, 6, {"val_mcc": 0.47})  # -3: stop
        assert cb.should_stop is True

    def test_mode_min_stops_on_three_increases(self) -> None:
        """mode='min': for loss-like metrics, increases count as non-improvement."""
        model = _TinyModel()
        trainer = _fake_trainer(model)
        cb = EarlyStopping(monitor="val_loss", mode="min", patience=3)
        cb.on_train_begin(trainer)

        cb.on_epoch_end(trainer, 0, {"val_loss": 1.0})
        cb.on_epoch_end(trainer, 1, {"val_loss": 1.1})  # -1
        cb.on_epoch_end(trainer, 2, {"val_loss": 1.2})  # -2
        cb.on_epoch_end(trainer, 3, {"val_loss": 1.3})  # -3
        assert cb.should_stop is True

    def test_missing_monitor_key_raises(self) -> None:
        model = _TinyModel()
        trainer = _fake_trainer(model)
        cb = EarlyStopping(monitor="val_mcc", mode="max", patience=3)
        cb.on_train_begin(trainer)
        with pytest.raises(KeyError, match="val_mcc"):
            cb.on_epoch_end(trainer, 0, {"val_loss": 0.1})


# ---------------------------------------------------------------------------
# BestModelCheckpoint
# ---------------------------------------------------------------------------


class TestBestModelCheckpoint:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            BestModelCheckpoint(monitor="val_mcc", mode="bogus")  # type: ignore[arg-type]

    def test_captures_and_restores_best_epoch_weights(self) -> None:
        """Stamp a parameter that monotonically changes across epochs;
        best-checkpoint should capture and restore the epoch with best metric."""
        model = _TinyModel()
        trainer = _fake_trainer(model)
        cb = BestModelCheckpoint(monitor="val_mcc", mode="max")
        cb.on_train_begin(trainer)

        # Simulate 5 epochs; best metric at epoch 2 (index 2).
        scores = [0.1, 0.3, 0.8, 0.4, 0.2]
        for epoch, score in enumerate(scores):
            # Stamp the weight so we can tell epochs apart.
            with torch.no_grad():
                model.fc.weight.fill_(float(epoch))
            cb.on_epoch_end(trainer, epoch, {"val_mcc": score})

        assert cb.best_epoch == 2
        assert cb.best_value == pytest.approx(0.8)

        # Weights at post-training are from the final epoch (4).
        assert model.fc.weight.mean().item() == pytest.approx(4.0)

        # Restore best: should return to epoch-2 stamp.
        cb.restore_best(model)
        assert model.fc.weight.mean().item() == pytest.approx(2.0)

    def test_state_dict_deep_copied(self) -> None:
        """Mutating model weights after a checkpoint must not poison the stored copy."""
        model = _TinyModel()
        trainer = _fake_trainer(model)
        cb = BestModelCheckpoint(monitor="val_mcc", mode="max")
        cb.on_train_begin(trainer)

        with torch.no_grad():
            model.fc.weight.fill_(5.0)
        cb.on_epoch_end(trainer, 0, {"val_mcc": 0.5})

        # Mutate live weights; checkpoint should be unaffected.
        with torch.no_grad():
            model.fc.weight.fill_(9.0)

        cb.restore_best(model)
        assert model.fc.weight.mean().item() == pytest.approx(5.0)

    def test_restore_before_any_update_is_noop(self) -> None:
        """If no epoch ever observed an improvement, restore is a no-op."""
        model = _TinyModel()
        with torch.no_grad():
            model.fc.weight.fill_(3.14)
        cb = BestModelCheckpoint(monitor="val_mcc", mode="max")
        cb.restore_best(model)
        assert model.fc.weight.mean().item() == pytest.approx(3.14)

    def test_mode_min_prefers_smaller(self) -> None:
        model = _TinyModel()
        trainer = _fake_trainer(model)
        cb = BestModelCheckpoint(monitor="val_loss", mode="min")
        cb.on_train_begin(trainer)

        losses = [1.0, 0.7, 0.4, 0.9]  # best at epoch 2
        for epoch, loss in enumerate(losses):
            with torch.no_grad():
                model.fc.weight.fill_(float(epoch))
            cb.on_epoch_end(trainer, epoch, {"val_loss": loss})

        assert cb.best_epoch == 2
        assert cb.best_value == pytest.approx(0.4)

    def test_best_value_nan_before_any_epoch(self) -> None:
        cb = BestModelCheckpoint(monitor="val_mcc", mode="max")
        # Before any epoch, best_value is nan and best_epoch == -1.
        import math

        assert math.isnan(cb.best_value)
        assert cb.best_epoch == -1

    def test_missing_monitor_key_raises(self) -> None:
        model = _TinyModel()
        trainer = _fake_trainer(model)
        cb = BestModelCheckpoint(monitor="val_mcc", mode="max")
        cb.on_train_begin(trainer)
        with pytest.raises(KeyError, match="val_mcc"):
            cb.on_epoch_end(trainer, 0, {"val_loss": 0.1})


# ---------------------------------------------------------------------------
# LRLogger
# ---------------------------------------------------------------------------


class TestLRLogger:
    def test_writes_lr_into_logs(self) -> None:
        model = _TinyModel()
        trainer = _fake_trainer(model, lr=0.05)
        cb = LRLogger()
        cb.on_train_begin(trainer)

        logs: dict[str, float] = {"val_loss": 1.0}
        cb.on_epoch_end(trainer, 0, logs)
        assert "lr" in logs
        assert logs["lr"] == pytest.approx(0.05)

    def test_history_accumulates(self) -> None:
        model = _TinyModel()
        trainer = _fake_trainer(model, lr=0.05)
        cb = LRLogger()
        cb.on_train_begin(trainer)

        # First epoch.
        cb.on_epoch_end(trainer, 0, {})
        # Mutate LR externally between epochs.
        for group in trainer.optimizer.param_groups:
            group["lr"] = 0.025
        cb.on_epoch_end(trainer, 1, {})
        # One more change.
        for group in trainer.optimizer.param_groups:
            group["lr"] = 0.01
        cb.on_epoch_end(trainer, 2, {})

        assert cb.lr_history == pytest.approx([0.05, 0.025, 0.01])

    def test_on_train_begin_resets_history(self) -> None:
        model = _TinyModel()
        trainer = _fake_trainer(model, lr=0.05)
        cb = LRLogger()
        cb.on_train_begin(trainer)
        cb.on_epoch_end(trainer, 0, {})
        assert len(cb.lr_history) == 1

        # Re-begin -> history cleared.
        cb.on_train_begin(trainer)
        assert cb.lr_history == []


# ---------------------------------------------------------------------------
# Base class contract
# ---------------------------------------------------------------------------


class TestCallbackBase:
    def test_should_stop_default_false(self) -> None:
        assert Callback().should_stop is False

    def test_default_hooks_are_noops(self) -> None:
        # No exceptions regardless of args — the trainer may pass any
        # non-None logs dict and epoch integer.
        cb = Callback()
        dummy = SimpleNamespace(model=None, optimizer=None)
        cb.on_train_begin(dummy)
        cb.on_epoch_begin(dummy, 0)
        cb.on_epoch_end(dummy, 0, {"val_loss": 0.0})
        cb.on_train_end(dummy)
