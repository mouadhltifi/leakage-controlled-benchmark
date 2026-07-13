"""v3 trainer (architecture-spec §M4).

A thin subclass of v2's :class:`mmfp.training.trainer.Trainer` that
bridges the two v3-specific wiring points:

1. **Batch key set**. v2's :data:`_BATCH_TENSOR_KEYS` enumerates the
   tensor keys the trainer moves to device. The v3 windowed dataset
   emits different keys (``price_seq``, ``news_seq``, ``macro_seq``,
   ``social_seq``, ``graph_seq``, ``static_categorical``, and the
   ``y_*`` targets). The subclass extends :meth:`_move_batch` to move
   every tensor in the batch dict, regardless of key.

2. **Loss signature**. v3's :class:`ForecastLoss` takes ``(q_pred,
   y_return, y_direction, y_volatility)`` — not v2's
   ``(preds_dict, targets_dict)``. The subclass overrides
   :meth:`_train_epoch` and :meth:`_validate_epoch` to call the loss
   with the correct signature and to compute quantile-specific
   validation metrics (pinball loss, MCC from ``sign(q_median)``, 80%
   coverage, interval width, RMSE / R² on point estimate).

3. **Primary metric**. v3 monitors ``val_pinball_loss`` (mode=min)
   instead of ``val_mcc`` (mode=max). :meth:`primary_metric` returns
   the v3 pair so ``EarlyStopping`` and ``BestModelCheckpoint``
   dispatch correctly.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from forecast.config.schema import V3ExperimentConfig
from mmfp.training.callbacks import Callback
from mmfp.training.trainer import Trainer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers (v3-specific; v2's are reused through the parent class)
# ---------------------------------------------------------------------------


def _compute_pinball(
    q_pred: np.ndarray,
    y_true: np.ndarray,
    quantiles: tuple[float, ...],
) -> float:
    """Mean pinball loss over a ``(N, K)`` prediction vs ``(N,)`` target."""
    if q_pred.size == 0:
        return float("nan")
    y = y_true.reshape(-1, 1)
    err = y - q_pred
    alphas = np.asarray(quantiles, dtype=np.float64).reshape(1, -1)
    upper = alphas * err
    lower = (alphas - 1.0) * err
    loss = np.maximum(upper, lower)
    return float(loss.mean())


def _compute_coverage(
    q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray,
) -> float:
    """Fraction of ``y`` inside ``[q_lo, q_hi]``. Returns nan on empty."""
    if y.size == 0:
        return float("nan")
    inside = (y >= q_lo) & (y <= q_hi)
    return float(inside.mean())


def _compute_interval_width(
    q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray,
) -> float:
    """Mean interval width, normalised by ``std(y)``.

    Returns the raw mean width when ``std(y) <= 0`` so we don't divide
    by zero.
    """
    if y.size == 0:
        return float("nan")
    width = q_hi - q_lo
    std = float(np.std(y))
    if std <= 0.0:
        return float(width.mean())
    return float(width.mean() / std)


def _compute_mcc(preds: np.ndarray, targets: np.ndarray) -> float:
    """Matthews correlation coefficient on ``{0, 1}`` arrays.

    ``preds`` and ``targets`` must have identical shapes.
    """
    if preds.size == 0:
        return 0.0
    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    num = tp * tn - fp * fn
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq <= 0:
        return 0.0
    return float(num / (denom_sq ** 0.5))


def _compute_rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    if preds.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def _compute_r2(preds: np.ndarray, targets: np.ndarray) -> float:
    if preds.size == 0:
        return float("nan")
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# ForecastTrainer
# ---------------------------------------------------------------------------


class ForecastTrainer(Trainer):
    """v3 trainer specialised for the TFT quantile-loss pipeline.

    Inherits lifecycle, callback dispatch, scheduler-step logic, and
    best-checkpoint restoration from :class:`mmfp.training.trainer.Trainer`.
    Overrides the three hot paths where v3 differs from v2: batch-key
    set (move-to-device), loss-call signature, and validation metric
    computation.

    Parameters
    ----------
    model
        :class:`~forecast.models.predictor.ForecastPredictor`.
    cfg
        :class:`V3ExperimentConfig` — ``cfg.forecast.quantiles`` drives
        pinball / coverage metrics; ``cfg.training`` drives optimiser /
        scheduler hyperparams (inherited).
    loss_fn
        :class:`~forecast.models.losses.ForecastLoss` (or compatible).
    optimizer
        PyTorch optimiser already wrapping ``model.parameters()``.
    scheduler
        Optional scheduler (inherited semantics).
    callbacks
        List of :class:`Callback` instances.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: V3ExperimentConfig,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: Any | None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        # Intentionally skip the parent's dataset-target list plumbing
        # (``self._targets``); v3's loss is driven by quantile_loss on
        # y_return plus two optional auxiliaries, not by v2's
        # head.targets dispatch. We still call super().__init__ so
        # device resolution and loss-to-device transfer run.
        super().__init__(
            model=model,
            cfg=cfg,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
        )
        # Freeze a tuple of quantiles for metric helpers (cfg stores a
        # tuple; we re-tuple for safety across mutation).
        self._quantiles: tuple[float, ...] = tuple(
            float(q) for q in cfg.forecast.quantiles
        )
        self._median_idx: int = self._quantiles.index(0.5)
        # 80%-band indices. Prefer the outermost available pair.
        lo_pref, hi_pref = 0.1, 0.9
        if lo_pref in self._quantiles and hi_pref in self._quantiles:
            self._lo_idx = self._quantiles.index(lo_pref)
            self._hi_idx = self._quantiles.index(hi_pref)
        else:
            self._lo_idx = 0
            self._hi_idx = len(self._quantiles) - 1

    # ------------------------------------------------------------------
    # Primary metric (override)
    # ------------------------------------------------------------------

    def primary_metric(self) -> tuple[str, Literal["min", "max"]]:
        """Return ``("val_pinball_loss", "min")`` for the v3 monitor."""
        return "val_pinball_loss", "min"

    # ------------------------------------------------------------------
    # Batch movement (override)
    # ------------------------------------------------------------------

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move every tensor in ``batch`` to the trainer device.

        v2 enumerated a fixed key set for device transfer; v3's dataset
        emits a different set of keys, so we drop the gate and move any
        tensor that happens to be in the batch. Non-tensor entries
        (there shouldn't be any) pass through untouched.
        """
        out: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(self.device, non_blocking=False)
            else:
                out[key] = value
        return out

    # ------------------------------------------------------------------
    # Train epoch (override)
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        """One training pass; returns mean batch loss.

        Overrides the parent's body because v3's loss has a different
        signature: ``(q_pred, y_return, y_direction, y_volatility)``.
        """
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        grad_clip = self.cfg.training.grad_clip

        for batch in loader:
            batch = self._move_batch(batch)
            preds = self.model(batch)

            q_pred = preds["return"]  # (B, n_q)
            y_return = batch["y_return"]
            y_direction = batch.get("y_direction")
            y_volatility = batch.get("y_volatility")

            self.optimizer.zero_grad(set_to_none=True)
            loss, _ = self.loss_fn(
                q_pred,
                y_return,
                y_direction,
                y_volatility,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=grad_clip
            )
            self.optimizer.step()

            running_loss += float(loss.detach().cpu().item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError(
                "ForecastTrainer._train_epoch: training loader produced "
                "zero batches. Check dataset length and batch size."
            )
        return running_loss / n_batches

    # ------------------------------------------------------------------
    # Validation (override)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate_epoch(
        self, loader: DataLoader,
    ) -> tuple[float, dict[str, float]]:
        """Run one validation pass and return ``(mean_loss, metrics)``.

        Metrics:

        * ``pinball_loss``  — primary, mean quantile loss.
        * ``mcc``            — from ``sign(q_median)`` vs ``sign(y)``.
        * ``accuracy``       — directional accuracy on sign of q_median.
        * ``rmse``           — point-estimate rmse (q_median vs y).
        * ``r2``             — point-estimate R² (q_median vs y).
        * ``quantile_coverage_80`` — fraction in ``[q_lo, q_hi]``.
        * ``quantile_interval_width`` — mean width / std(y).
        """
        self.model.train(False)  # inference mode
        running_loss = 0.0
        n_batches = 0

        q_preds: list[Tensor] = []
        y_returns: list[Tensor] = []

        for batch in loader:
            batch = self._move_batch(batch)
            preds = self.model(batch)

            q_pred = preds["return"]
            y_return = batch["y_return"]
            y_direction = batch.get("y_direction")
            y_volatility = batch.get("y_volatility")

            loss, _ = self.loss_fn(
                q_pred,
                y_return,
                y_direction,
                y_volatility,
            )
            running_loss += float(loss.detach().cpu().item())
            n_batches += 1

            q_preds.append(q_pred.detach().cpu())
            y_returns.append(y_return.detach().cpu())

        if n_batches == 0:
            raise RuntimeError(
                "ForecastTrainer._validate_epoch: validation loader "
                "produced zero batches."
            )

        mean_loss = running_loss / n_batches

        q_all = torch.cat(q_preds, dim=0).numpy().astype(np.float64)
        y_all = torch.cat(y_returns, dim=0).numpy().astype(np.float64)

        q_median = q_all[:, self._median_idx]
        q_lo = q_all[:, self._lo_idx]
        q_hi = q_all[:, self._hi_idx]

        # Binary direction from sign; treat non-positive as down (0).
        sign_pred = (q_median > 0).astype(np.int64)
        sign_true = (y_all > 0).astype(np.int64)

        metrics: dict[str, float] = {}
        metrics["pinball_loss"] = _compute_pinball(
            q_all, y_all, self._quantiles,
        )
        metrics["mcc"] = _compute_mcc(sign_pred, sign_true)
        if sign_true.size > 0:
            metrics["accuracy"] = float((sign_pred == sign_true).mean())
        else:
            metrics["accuracy"] = float("nan")
        metrics["rmse"] = _compute_rmse(q_median, y_all)
        metrics["r2"] = _compute_r2(q_median, y_all)
        metrics["quantile_coverage_80"] = _compute_coverage(q_lo, q_hi, y_all)
        metrics["quantile_interval_width"] = _compute_interval_width(
            q_lo, q_hi, y_all,
        )

        return mean_loss, metrics


__all__ = ["ForecastTrainer"]
