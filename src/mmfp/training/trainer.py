"""Training loop for :class:`~mmfp.models.predictor.Predictor`.

Device-aware, deterministic, and metric-agnostic. The active validation
metric is inferred from ``cfg.head.targets`` (see
:func:`_resolve_primary_metric`) so the trainer supports direction,
return, and volatility targets without special casing at call sites.

The trainer is a thin orchestrator; actual early-stopping and
best-weight-restoration logic lives in the callbacks
(:mod:`mmfp.training.callbacks`). That keeps the Trainer class readable
and lets the experiment runner (M7) compose its own callback stack
without subclassing.

Key responsibilities
--------------------

1. Iterate train / validation epochs, calling
   :func:`mmfp.utils.seeding.set_all_seeds` is **not** done here — it's
   the experiment runner's job, done once per run.
2. Move batches to device.
3. Apply gradient clipping (``cfg.training.grad_clip``).
4. Step the LR scheduler (plateau schedulers get the validation loss;
   others step purely on epoch index).
5. Compute validation metrics that match ``cfg.head.targets``.
6. Notify callbacks at epoch and train lifecycle boundaries.
7. Restore the best-weights state dict at the end of :meth:`fit` when a
   :class:`BestModelCheckpoint` is registered.

Every ``print()`` is intentionally avoided; logs flow through the
module-level :class:`logging.Logger`.

Reproducibility
---------------

For a bit-identical CPU run the caller must:

* Call :func:`set_all_seeds(cfg.seed)` **before** constructing the
  model / optimizer / data loaders.
* Construct DataLoaders with ``generator=make_dataloader_generator(cfg.seed)``
  and ``worker_init_fn=make_worker_init_fn(cfg.seed)``. Shuffling is
  deterministic given a fixed generator.
* Run on CPU (``cfg.device == "cpu"``). MPS LSTM remains
  non-deterministic — see the design spec
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from mmfp.config.schema import ExperimentConfig
from mmfp.models.predictor import Predictor
from mmfp.utils.device import resolve_device

from .callbacks import BestModelCheckpoint, Callback
from .scheduler import is_plateau_scheduler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public History dataclass
# ---------------------------------------------------------------------------


@dataclass
class History:
    """Per-epoch training / validation history.

    Attributes
    ----------
    train_loss
        Mean training loss per epoch (one float per epoch actually run).
    val_loss
        Mean validation loss per epoch.
    val_metrics
        Mapping metric name - list of per-epoch values. Keys include the
        primary metric (e.g. ``"mcc"`` for direction) and the auxiliary
        metrics computed alongside it (``"accuracy"``, ``"f1"`` when
        ``direction`` is active; ``"r2"`` alongside ``"rmse"`` for
        volatility).
    best_epoch
        Zero-based epoch of the best monitored metric; ``-1`` if no
        :class:`BestModelCheckpoint` was configured or no improvement
        was ever observed.
    best_metric_value
        Value of the primary monitor at :attr:`best_epoch`; ``nan`` when
        :attr:`best_epoch` is ``-1``.
    primary_metric
        Name of the primary validation metric (one of ``"val_mcc"``,
        ``"val_rmse"``, ``"val_r2"``). Stored so downstream logs can
        pretty-print the right key without re-deriving it.
    monitor_mode
        ``"min"`` or ``"max"`` — direction of improvement for the
        primary metric.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metrics: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = -1
    best_metric_value: float = float("nan")
    primary_metric: str = ""
    monitor_mode: Literal["min", "max"] = "max"


# ---------------------------------------------------------------------------
# Target-name mappings
# ---------------------------------------------------------------------------


#: Dataset target keys -> predictor / loss target keys. Dataset emits
#: cls/reg/vol_target; predictor outputs direction/return/volatility.
_DATASET_TO_LOSS_TARGET: dict[str, str] = {
    "cls_target": "direction",
    "reg_target": "return",
    "vol_target": "volatility",
}

#: Inverse mapping: predictor/loss target -> dataset key.
_LOSS_TO_DATASET_TARGET: dict[str, str] = {
    v: k for k, v in _DATASET_TO_LOSS_TARGET.items()
}

#: Which batch keys are tensors (to be moved to device). Anything else in
#: the batch dict is passed through untouched (e.g. bookkeeping scalars).
_BATCH_TENSOR_KEYS: tuple[str, ...] = (
    "price", "macro", "news", "social",
    "graph_features", "edge_index", "edge_index_static", "edge_index_dynamic",
    "stock_idx", "graph_stock_idx", "_batch_size",
    "cls_target", "reg_target", "vol_target",
)


# ---------------------------------------------------------------------------
# Primary-metric resolution
# ---------------------------------------------------------------------------


def _resolve_primary_metric(
    cfg: ExperimentConfig,
) -> tuple[str, Literal["min", "max"]]:
    """Return ``(metric_key, mode)`` for the monitor.

    * ``"direction"`` -> ``("val_mcc", "max")``.
    * else ``"volatility"`` -> ``("val_rmse", "min")``.
    * else (return only) -> ``("val_r2", "max")``.

    For multi-target heads the first target in ``cfg.head.targets`` wins
    — with a log message so users see which metric drives early stopping.
    """
    targets = list(cfg.head.targets)
    if not targets:
        raise ValueError(
            "Trainer: head.targets must contain at least one target."
        )

    # The spec says: "If 'direction' in targets -> val_mcc." Then fall-through.
    if "direction" in targets:
        primary = "direction"
    elif "volatility" in targets:
        primary = "volatility"
    else:
        primary = "return"

    if len(targets) > 1:
        log.info(
            "Trainer: multi-target head %s; monitoring %r by priority rule "
            "(direction > volatility > return).",
            targets, primary,
        )

    if primary == "direction":
        return "val_mcc", "max"
    if primary == "volatility":
        return "val_rmse", "min"
    return "val_r2", "max"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _compute_mcc(preds: np.ndarray, targets: np.ndarray) -> float:
    """Matthews Correlation Coefficient for binary classification."""
    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())

    num = tp * tn - fp * fn
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq <= 0:
        return 0.0
    return float(num / (denom_sq ** 0.5))


def _compute_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) == 0:
        return float("nan")
    return float((preds == targets).mean())


def _compute_f1(preds: np.ndarray, targets: np.ndarray) -> float:
    """Binary F1 (positive class = 1). Returns 0 when there are no positives."""
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _compute_rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def _compute_r2(preds: np.ndarray, targets: np.ndarray) -> float:
    """Coefficient of determination. Returns 0 for a zero-variance target."""
    if len(preds) == 0:
        return float("nan")
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Orchestrate the train / validation loop with callbacks.

    Parameters
    ----------
    model
        The fully-assembled :class:`Predictor` to train.
    cfg
        Validated experiment configuration. ``cfg.training``, ``cfg.head``
        and ``cfg.device`` are the hot paths.
    loss_fn
        The loss module (from
        :func:`mmfp.models.losses.factory.build_loss`).
    optimizer
        A PyTorch optimizer already wrapping ``model.parameters()``.
    scheduler
        Optional LR scheduler (from
        :func:`mmfp.training.scheduler.build_scheduler`). Pass ``None`` to
        disable.
    callbacks
        List of :class:`Callback` instances. The trainer will not register
        any callbacks implicitly — the caller is responsible for
        constructing and passing e.g. :class:`EarlyStopping` and
        :class:`BestModelCheckpoint`.

    Attributes
    ----------
    model, cfg, loss_fn, optimizer, scheduler, callbacks
        Construction-time arguments preserved for external inspection
        (tests mostly).
    device
        The resolved :class:`torch.device` the model lives on.
    """

    def __init__(
        self,
        model: Predictor,
        cfg: ExperimentConfig,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: Any | None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks: list[Callback] = list(callbacks) if callbacks else []

        self.device = resolve_device(cfg)
        self.model.to(self.device)
        # Loss modules that hold buffers (e.g. class_weights) must also
        # move to the active device so the CrossEntropy weight lookup
        # stays on the right device without per-step device transfers.
        self.loss_fn.to(self.device)

        self._targets: list[str] = list(cfg.head.targets)

    # ------------------------------------------------------------------
    # Primary metric resolution (public so callers can mirror it)
    # ------------------------------------------------------------------

    def primary_metric(self) -> tuple[str, Literal["min", "max"]]:
        """Return ``(metric_key, mode)`` selected by :func:`_resolve_primary_metric`."""
        return _resolve_primary_metric(self.cfg)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> History:
        """Train the model and return a :class:`History`.

        The returned history is always non-empty provided
        ``cfg.training.max_epochs >= 1``: at least one epoch runs before
        any callback can request a stop.
        """
        if self.cfg.training.max_epochs < 1:
            raise ValueError(
                f"Trainer.fit: max_epochs must be >= 1; got {self.cfg.training.max_epochs}"
            )

        primary_metric, mode = self.primary_metric()
        history = History(primary_metric=primary_metric, monitor_mode=mode)

        log.info(
            "Trainer.fit: device=%s, max_epochs=%d, monitor=%s (%s).",
            self.device, self.cfg.training.max_epochs, primary_metric, mode,
        )

        for cb in self.callbacks:
            cb.on_train_begin(self)

        start_time = time.time()
        max_epochs = self.cfg.training.max_epochs

        for epoch in range(max_epochs):
            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate_epoch(val_loader)

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            for name, value in val_metrics.items():
                history.val_metrics.setdefault(name, []).append(value)

            logs: dict[str, float] = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            self._step_scheduler(val_loss=val_loss, primary=logs.get(primary_metric))

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)

            if (epoch + 1) % self.cfg.logging.per_epoch_log_every == 0:
                log.info(
                    "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | %s=%.6f",
                    epoch + 1, max_epochs, train_loss, val_loss,
                    primary_metric, logs.get(primary_metric, float("nan")),
                )

            if any(cb.should_stop for cb in self.callbacks):
                log.info(
                    "Trainer.fit: early stop requested at epoch %d.",
                    epoch + 1,
                )
                break

        for cb in self.callbacks:
            cb.on_train_end(self)

        self._restore_best_checkpoint(history)

        elapsed = time.time() - start_time
        log.info("Trainer.fit: done in %.1fs (%d epochs).", elapsed, len(history.train_loss))
        return history

    # ------------------------------------------------------------------
    # Train / validation loops
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training pass and return the mean batch loss."""
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        grad_clip = self.cfg.training.grad_clip

        for batch in loader:
            batch = self._move_batch(batch)
            preds = self.model(batch)
            targets = self._extract_targets(batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss, _ = self.loss_fn(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            self.optimizer.step()

            running_loss += float(loss.detach().cpu().item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError(
                "Trainer._train_epoch: the training loader produced zero batches. "
                "Check dataset length and batch size."
            )
        return running_loss / n_batches

    @torch.no_grad()
    def _validate_epoch(
        self, loader: DataLoader,
    ) -> tuple[float, dict[str, float]]:
        """Run one validation pass and return ``(mean_loss, metrics_dict)``.

        The metrics dict has keys *without* the ``val_`` prefix (the
        :meth:`fit` caller re-prefixes them before emitting ``logs``).
        That keeps this helper focused on compute.
        """
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        # Collect per-target predictions / targets for metric computation.
        # Classification: argmax; Regression: raw scalar.
        collected_cls_preds: list[Tensor] = []
        collected_cls_targets: list[Tensor] = []
        collected_return_preds: list[Tensor] = []
        collected_return_targets: list[Tensor] = []
        collected_vol_preds: list[Tensor] = []
        collected_vol_targets: list[Tensor] = []

        do_direction = "direction" in self._targets
        do_return = "return" in self._targets
        do_volatility = "volatility" in self._targets

        for batch in loader:
            batch = self._move_batch(batch)
            preds = self.model(batch)
            targets = self._extract_targets(batch)

            loss, _ = self.loss_fn(preds, targets)
            running_loss += float(loss.detach().cpu().item())
            n_batches += 1

            if do_direction:
                cls_logits = preds["direction"]
                collected_cls_preds.append(cls_logits.argmax(dim=-1).detach().cpu())
                collected_cls_targets.append(targets["direction"].detach().cpu())
            if do_return:
                collected_return_preds.append(preds["return"].squeeze(-1).detach().cpu())
                collected_return_targets.append(targets["return"].detach().cpu())
            if do_volatility:
                collected_vol_preds.append(preds["volatility"].squeeze(-1).detach().cpu())
                collected_vol_targets.append(targets["volatility"].detach().cpu())

        if n_batches == 0:
            raise RuntimeError(
                "Trainer._validate_epoch: the validation loader produced zero "
                "batches. Check dataset length and batch size."
            )

        mean_loss = running_loss / n_batches
        metrics: dict[str, float] = {}

        if do_direction:
            p = torch.cat(collected_cls_preds).numpy().astype(np.int64)
            t = torch.cat(collected_cls_targets).numpy().astype(np.int64)
            metrics["mcc"] = _compute_mcc(p, t)
            metrics["accuracy"] = _compute_accuracy(p, t)
            metrics["f1"] = _compute_f1(p, t)
        if do_return:
            p = torch.cat(collected_return_preds).numpy().astype(np.float64)
            t = torch.cat(collected_return_targets).numpy().astype(np.float64)
            metrics["return_rmse"] = _compute_rmse(p, t)
            metrics["r2"] = _compute_r2(p, t)
        if do_volatility:
            p = torch.cat(collected_vol_preds).numpy().astype(np.float64)
            t = torch.cat(collected_vol_targets).numpy().astype(np.float64)
            metrics["rmse"] = _compute_rmse(p, t)
            metrics["vol_r2"] = _compute_r2(p, t)

        return mean_loss, metrics

    # ------------------------------------------------------------------
    # Scheduler step
    # ------------------------------------------------------------------

    def _step_scheduler(
        self,
        *,
        val_loss: float,
        primary: float | None,  # noqa: ARG002 - primary metric reserved for future plateau-on-metric mode
    ) -> None:
        """Advance the LR scheduler by one epoch.

        Plateau schedulers are stepped with ``val_loss`` — lower-is-better
        matches the ``mode='min'`` default set in
        :func:`~mmfp.training.scheduler.build_scheduler`. Step-based
        schedulers (cosine warm restarts) ignore the argument.
        """
        if self.scheduler is None:
            return
        if is_plateau_scheduler(self.scheduler):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensor entries of ``batch`` to :attr:`device`."""
        out: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and key in _BATCH_TENSOR_KEYS:
                out[key] = value.to(self.device, non_blocking=False)
            else:
                out[key] = value
        return out

    def _extract_targets(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Pull active-target tensors out of the batch keyed by loss-target name."""
        targets: dict[str, Tensor] = {}
        for target in self._targets:
            dataset_key = _LOSS_TO_DATASET_TARGET[target]
            if dataset_key not in batch:
                raise KeyError(
                    f"Trainer: active target {target!r} expects batch key "
                    f"{dataset_key!r} (got {sorted(batch.keys())})"
                )
            targets[target] = batch[dataset_key]
        return targets

    # ------------------------------------------------------------------
    # Checkpoint restoration
    # ------------------------------------------------------------------

    def _restore_best_checkpoint(self, history: History) -> None:
        """Load the best state dict from any :class:`BestModelCheckpoint` callback.

        If multiple best-checkpoint callbacks are registered the first in
        ``callbacks`` wins (tests don't need more than one).
        """
        for cb in self.callbacks:
            if isinstance(cb, BestModelCheckpoint):
                cb.restore_best(self.model)
                history.best_epoch = cb.best_epoch
                history.best_metric_value = cb.best_value
                return


__all__ = ["History", "Trainer"]
