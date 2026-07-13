"""Training callbacks: early stopping, best-model checkpoint, LR logging.

A :class:`Callback` is any object implementing the lifecycle hooks below.
The :class:`~mmfp.training.trainer.Trainer` iterates over its callback
list at each epoch boundary, reads ``should_stop`` before starting the
next epoch, and restores the best-checkpoint state at the end of
:meth:`Trainer.fit` when a :class:`BestModelCheckpoint` is registered.

Design rationale
----------------

* **In-memory checkpoints only.** :class:`BestModelCheckpoint` holds a
  *copy* of the model ``state_dict`` in RAM; it does not write to disk.
  The experiment runner (M7) may add a persistent checkpointer as a
  separate callback later.
* **Single monitor.** Each stopping/checkpoint callback watches exactly
  one metric. Supporting composite metrics would blur responsibility.
* **Explicit mode.** ``mode='max'`` for higher-is-better metrics (MCC,
  R²), ``mode='min'`` for lower-is-better (RMSE, loss).

See the design spec for the authoritative
specification of callback roles.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from mmfp.training.trainer import Trainer

log = logging.getLogger(__name__)

#: Literal alias used across the callback classes.
Mode = Literal["min", "max"]


class Callback:
    """Base class for training callbacks.

    Subclasses override any hook they care about. The default
    implementations are no-ops, so subclasses only implement the hooks
    that matter to them.

    Notes
    -----
    The ``should_stop`` property is how callbacks signal the trainer to
    halt. Multiple callbacks may each set ``should_stop``; the trainer
    stops if *any* returns ``True``.
    """

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called once at the start of :meth:`Trainer.fit`."""

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        logs: dict[str, float],
    ) -> None:
        """Called at the end of each epoch.

        Parameters
        ----------
        trainer
            The trainer that owns this callback. Callbacks may read
            ``trainer.model`` to capture state.
        epoch
            Zero-based epoch index.
        logs
            Mutable dict of scalar metrics. Contains at least ``train_loss``
            and ``val_loss`` plus any target-specific metrics
            (``val_mcc``, ``val_rmse``, ``val_r2``, ...). Callbacks may
            add to it — :class:`LRLogger` stores the current learning
            rate under ``logs['lr']`` for downstream loggers.
        """

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called once after the last epoch completes."""

    @property
    def should_stop(self) -> bool:
        """Return ``True`` to request early termination."""
        return False


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


class EarlyStopping(Callback):
    """Halt training after ``patience`` consecutive non-improvements.

    Parameters
    ----------
    monitor
        Key in the epoch ``logs`` dict to watch (e.g. ``"val_mcc"``).
    mode
        Either ``"min"`` (lower is better) or ``"max"`` (higher is
        better). Determines the direction of "improvement".
    patience
        Number of consecutive epochs with no improvement before we
        request a stop. Matches v1 semantics: after ``patience``
        non-improving epochs the next ``on_epoch_end`` sets
        ``should_stop = True``.
    min_epochs
        Earliest epoch (zero-based) after which stopping is allowed.
        Before this, non-improvement is counted but no stop is
        triggered. Matches v1's ``epoch >= self.min_epochs`` gate.

    Notes
    -----
    The first epoch always registers as an improvement (there's no
    baseline yet), so ``patience`` effectively counts "consecutive
    non-improvements after the best epoch" just like in v1's inline
    implementation.
    """

    def __init__(
        self,
        monitor: str,
        mode: Mode,
        patience: int,
        min_epochs: int = 0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(
                f"EarlyStopping mode must be 'min' or 'max'; got {mode!r}"
            )
        if patience < 1:
            raise ValueError(
                f"EarlyStopping patience must be >= 1; got {patience}"
            )
        if min_epochs < 0:
            raise ValueError(
                f"EarlyStopping min_epochs must be >= 0; got {min_epochs}"
            )

        self.monitor = monitor
        self.mode: Mode = mode
        self.patience = patience
        self.min_epochs = min_epochs

        self._best_value: float | None = None
        self._epochs_since_improvement = 0
        self._should_stop = False
        self._stopped_epoch: int | None = None

    # ------------------------------------------------------------------
    # Callback API
    # ------------------------------------------------------------------

    def on_train_begin(self, trainer: "Trainer") -> None:  # noqa: ARG002
        self._best_value = None
        self._epochs_since_improvement = 0
        self._should_stop = False
        self._stopped_epoch = None

    def on_epoch_end(
        self,
        trainer: "Trainer",  # noqa: ARG002
        epoch: int,
        logs: dict[str, float],
    ) -> None:
        if self.monitor not in logs:
            raise KeyError(
                f"EarlyStopping monitor {self.monitor!r} not in logs "
                f"(available keys: {sorted(logs.keys())})"
            )

        value = float(logs[self.monitor])
        improved = self._is_improvement(value)

        if improved:
            self._best_value = value
            self._epochs_since_improvement = 0
        else:
            self._epochs_since_improvement += 1

        # v1 condition: stop if non-improve >= patience AND past min_epochs.
        if (
            self._epochs_since_improvement >= self.patience
            and epoch >= self.min_epochs
        ):
            self._should_stop = True
            self._stopped_epoch = epoch
            log.info(
                "EarlyStopping: no improvement in %d epochs on %s; "
                "stopping at epoch %d.",
                self.patience, self.monitor, epoch + 1,
            )

    # ------------------------------------------------------------------

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def stopped_epoch(self) -> int | None:
        """Zero-based index of the epoch that triggered the stop (or ``None``)."""
        return self._stopped_epoch

    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self.mode == "max":
            return value > self._best_value
        return value < self._best_value


# ---------------------------------------------------------------------------
# BestModelCheckpoint
# ---------------------------------------------------------------------------


class BestModelCheckpoint(Callback):
    """Capture the model ``state_dict`` at the best monitored epoch.

    Parameters
    ----------
    monitor
        Key in the epoch ``logs`` dict to watch.
    mode
        ``"min"`` (lower is better) or ``"max"`` (higher is better).

    Notes
    -----
    * State dicts are ``.clone()`` 'd tensor-by-tensor so subsequent
      in-place weight updates don't mutate the stored copy.
    * The trainer calls :meth:`restore_best` at the end of :meth:`fit`.
    """

    def __init__(self, monitor: str, mode: Mode) -> None:
        if mode not in ("min", "max"):
            raise ValueError(
                f"BestModelCheckpoint mode must be 'min' or 'max'; got {mode!r}"
            )
        self.monitor = monitor
        self.mode: Mode = mode

        self._best_state_dict: dict[str, torch.Tensor] | None = None
        self._best_value: float | None = None
        self._best_epoch: int = -1

    # ------------------------------------------------------------------
    # Callback API
    # ------------------------------------------------------------------

    def on_train_begin(self, trainer: "Trainer") -> None:  # noqa: ARG002
        self._best_state_dict = None
        self._best_value = None
        self._best_epoch = -1

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        logs: dict[str, float],
    ) -> None:
        if self.monitor not in logs:
            raise KeyError(
                f"BestModelCheckpoint monitor {self.monitor!r} not in logs "
                f"(available keys: {sorted(logs.keys())})"
            )
        value = float(logs[self.monitor])

        if self._is_improvement(value):
            self._best_value = value
            self._best_epoch = epoch
            # Deep-copy tensors so later in-place updates don't poison us.
            self._best_state_dict = {
                k: v.detach().clone()
                for k, v in trainer.model.state_dict().items()
            }

    # ------------------------------------------------------------------

    @property
    def best_state_dict(self) -> dict[str, torch.Tensor] | None:
        return self._best_state_dict

    @property
    def best_epoch(self) -> int:
        """Zero-based index of the best epoch, or ``-1`` if never set."""
        return self._best_epoch

    @property
    def best_value(self) -> float:
        """Best monitored metric value; ``nan`` if never observed."""
        if self._best_value is None:
            return float("nan")
        return self._best_value

    # ------------------------------------------------------------------

    def restore_best(self, model: nn.Module) -> None:
        """Load the captured state dict into ``model`` in place.

        No-op when no epoch was ever marked as an improvement (e.g. the
        monitored metric was NaN throughout, or fit ran zero epochs).
        """
        if self._best_state_dict is None:
            log.warning(
                "BestModelCheckpoint.restore_best: no best state captured; "
                "model weights left as-is."
            )
            return
        model.load_state_dict(self._best_state_dict)

    def _is_improvement(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self.mode == "max":
            return value > self._best_value
        return value < self._best_value


# ---------------------------------------------------------------------------
# LRLogger
# ---------------------------------------------------------------------------


class LRLogger(Callback):
    """Record the current learning rate at each epoch end.

    At end of epoch the callback reads ``trainer.optimizer.param_groups[0]['lr']``
    and writes it into ``logs['lr']`` so downstream consumers (the trainer's
    History, a future CSV logger, etc.) see it like any other metric.

    The full history is also stored on the callback as
    :attr:`lr_history` for tests and ad-hoc analysis.
    """

    def __init__(self) -> None:
        self.lr_history: list[float] = []

    def on_train_begin(self, trainer: "Trainer") -> None:  # noqa: ARG002
        self.lr_history = []

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,  # noqa: ARG002
        logs: dict[str, float],
    ) -> None:
        if not trainer.optimizer.param_groups:
            return
        lr = float(trainer.optimizer.param_groups[0]["lr"])
        logs["lr"] = lr
        self.lr_history.append(lr)


__all__ = [
    "BestModelCheckpoint",
    "Callback",
    "EarlyStopping",
    "LRLogger",
    "Mode",
]
