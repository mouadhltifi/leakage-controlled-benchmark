"""Quantile + composite forecast loss (the architecture spec).

Two concrete classes:

* :class:`QuantileLoss` — pinball / quantile loss, mean over (batch,
  quantiles). The TFT paper's primary training objective.

* :class:`ForecastLoss` — composite that wraps ``QuantileLoss`` with
  optional weighted cross-entropy on derived direction and optional
  weighted MSE on derived volatility. Weights default to 0 so the
  composite degenerates exactly to the quantile loss on its own.

* :func:`build_loss` — config-dispatched factory that returns the loss
  module matching ``cfg.forecast.architecture``. Today only ``"tft"``
  is implemented (:class:`ForecastLoss`); the signature is prepared for
  future backbone extensions.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from forecast.config.schema import ForecastConfig, V3ExperimentConfig
from forecast.models.heads import derive_direction, derive_volatility


class QuantileLoss(nn.Module):
    """Pinball loss mean-reduced over batch and quantile axes.

    For quantile level :math:`\\alpha_k` with prediction :math:`q_k`
    and target :math:`y`:

    .. math::

       L_k = \\max\\big( \\alpha_k (y - q_k),\\ (\\alpha_k - 1)(y - q_k) \\big)

    Final loss is ``mean`` over ``(b, k)``.

    Parameters
    ----------
    quantiles
        Strictly-increasing tuple of quantile levels in (0, 1).
    """

    def __init__(self, quantiles: tuple[float, ...]) -> None:
        super().__init__()
        if len(quantiles) < 1:
            raise ValueError(
                f"quantiles must have at least 1 value; got {quantiles}"
            )
        for q in quantiles:
            if not 0.0 < q < 1.0:
                raise ValueError(
                    f"each quantile must lie in (0, 1); got {q}"
                )

        self.quantiles = tuple(float(q) for q in quantiles)
        # Stored as a buffer so it moves with the model to device.
        alphas = torch.tensor(self.quantiles, dtype=torch.float32)
        self.register_buffer("alphas", alphas, persistent=False)

    def forward(self, q_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute the mean pinball loss.

        Parameters
        ----------
        q_pred
            Predicted quantiles, shape ``(B, n_quantiles)``.
        y_true
            True scalar targets, shape ``(B,)``.

        Returns
        -------
        Tensor
            Scalar loss tensor.
        """
        if q_pred.dim() != 2:
            raise ValueError(
                f"q_pred must be (B, n_quantiles); got {tuple(q_pred.shape)}"
            )
        if y_true.dim() != 1:
            raise ValueError(
                f"y_true must be (B,); got {tuple(y_true.shape)}"
            )
        if q_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"batch mismatch: q_pred has {q_pred.shape[0]}, "
                f"y_true has {y_true.shape[0]}."
            )
        if q_pred.shape[1] != self.alphas.shape[0]:
            raise ValueError(
                f"q_pred has {q_pred.shape[1]} quantiles; loss configured "
                f"for {self.alphas.shape[0]}."
            )

        # Broadcast y to match (B, n_q).
        y_broadcast = y_true.unsqueeze(-1)  # (B, 1)
        err = y_broadcast - q_pred  # (B, n_q)

        # max(alpha * err, (alpha - 1) * err) = elementwise hinge.
        # Cast alphas to match dtype/device of err.
        alphas = self.alphas.to(err.dtype)
        upper = alphas * err
        lower = (alphas - 1.0) * err
        loss = torch.maximum(upper, lower)

        return loss.mean()


class ForecastLoss(nn.Module):
    """Composite forecast loss: quantile + optional aux CE + optional aux MSE.

    Parameters
    ----------
    cfg
        v3 :class:`~forecast.config.schema.ForecastConfig`. Consulted
        for ``quantiles``, ``direction_aux_weight``, ``volatility_aux_weight``.
    direction_class_weights
        Optional ``(2,)`` per-class weights for the CE auxiliary.
        Ignored when ``direction_aux_weight == 0``.
    """

    def __init__(
        self,
        cfg: ForecastConfig,
        direction_class_weights: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.quantile_loss = QuantileLoss(cfg.quantiles)
        self.direction_weight = float(cfg.direction_aux_weight)
        self.volatility_weight = float(cfg.volatility_aux_weight)

        if self.direction_weight > 0 and direction_class_weights is not None:
            if direction_class_weights.shape != (2,):
                raise ValueError(
                    f"direction_class_weights must be shape (2,); got "
                    f"{tuple(direction_class_weights.shape)}"
                )
            # Register as a buffer so it moves with the model.
            self.register_buffer(
                "direction_class_weights",
                direction_class_weights.to(torch.float32),
                persistent=False,
            )
        else:
            self.direction_class_weights = None  # type: ignore[assignment]

        # Cache the median index (must exist per schema validator).
        self.median_idx = cfg.quantiles.index(0.5)

        # Low/high indices for volatility derivation: first and last
        # quantile in the tuple (widest band).
        self.lo_idx = 0
        self.hi_idx = len(cfg.quantiles) - 1
        # Coverage implied by the (lo, hi) pair: hi_q - lo_q, clamped
        # to keep erfinv well-defined.
        implied = float(cfg.quantiles[self.hi_idx] - cfg.quantiles[self.lo_idx])
        self.volatility_coverage = max(0.01, min(0.99, implied))

    def forward(
        self,
        q_pred: Tensor,
        y_return: Tensor,
        y_direction: Tensor | None = None,
        y_volatility: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the composite loss.

        Parameters
        ----------
        q_pred
            Predicted quantile vector, shape ``(B, n_quantiles)``.
        y_return
            True returns, shape ``(B,)``.
        y_direction
            Optional integer target ``(B,)`` with class labels in
            ``{0, 1}``. Values equal to ``-1`` (v2 deadzone marker) are
            ignored via ``ignore_index``. Required when
            ``direction_aux_weight > 0``.
        y_volatility
            Optional float target ``(B,)`` with realised absolute
            returns. ``NaN`` values are masked out. Required when
            ``volatility_aux_weight > 0``.

        Returns
        -------
        total
            Scalar loss (sum of weighted components). Gradient flows
            through every active component.
        components
            Dict of float scalars: always includes ``'quantile'``;
            includes ``'direction'`` when that auxiliary is active and
            ``'volatility'`` when that auxiliary is active. Values are
            detached floats for logging.
        """
        components: dict[str, float] = {}

        # Primary: pinball.
        quantile_val = self.quantile_loss(q_pred, y_return)
        components["quantile"] = float(quantile_val.detach().item())
        total = quantile_val

        # Direction CE auxiliary.
        if self.direction_weight > 0:
            if y_direction is None:
                raise ValueError(
                    "direction_aux_weight > 0 but y_direction is None."
                )
            if y_direction.dtype not in (torch.int64, torch.long):
                y_direction = y_direction.to(torch.long)
            logits = derive_direction(q_pred, self.median_idx)  # (B, 2)
            weight = (
                self.direction_class_weights
                if self.direction_class_weights is not None
                else None
            )
            ce = F.cross_entropy(
                logits,
                y_direction,
                weight=weight,
                ignore_index=-1,  # v2 deadzone marker
            )
            components["direction"] = float(ce.detach().item())
            total = total + self.direction_weight * ce

        # Volatility MSE auxiliary.
        if self.volatility_weight > 0:
            if y_volatility is None:
                raise ValueError(
                    "volatility_aux_weight > 0 but y_volatility is None."
                )
            sigma = derive_volatility(
                q_pred,
                lo_idx=self.lo_idx,
                hi_idx=self.hi_idx,
                coverage=self.volatility_coverage,
            )
            mask = ~torch.isnan(y_volatility)
            if mask.any():
                mse = F.mse_loss(sigma[mask], y_volatility[mask])
            else:
                # All targets masked — return zero to avoid gradient
                # contribution but still report.
                mse = torch.zeros((), device=q_pred.device, dtype=q_pred.dtype)
            components["volatility"] = float(mse.detach().item())
            total = total + self.volatility_weight * mse

        return total, components


def build_loss(
    cfg: V3ExperimentConfig,
    class_weights: Tensor | None = None,
) -> nn.Module:
    """Factory that returns the loss module matching ``cfg.forecast.architecture``.

    Today only ``architecture == "tft"`` is implemented; the factory
    exists so downstream callers (the reused v2 trainer) never
    construct :class:`ForecastLoss` directly, keeping the loss choice
    a property of the config rather than of the call site.

    Parameters
    ----------
    cfg
        Full :class:`V3ExperimentConfig`. Only ``cfg.forecast`` is
        consulted today.
    class_weights
        Optional ``(2,)`` per-class weights forwarded to
        :class:`ForecastLoss` as ``direction_class_weights``. Pass
        ``None`` to disable class weighting. Typically computed by
        ``mmfp.evaluation.metrics.compute_class_weights`` on the train
        split.

    Returns
    -------
    nn.Module
        A loss module with the
        ``forward(q_pred, y_return, y_direction=None, y_volatility=None)
        -> (total, components)`` signature.

    Raises
    ------
    ValueError
        If ``cfg.forecast.architecture`` is not recognised.
    """
    architecture = cfg.forecast.architecture
    if architecture == "tft":
        return ForecastLoss(
            cfg=cfg.forecast,
            direction_class_weights=class_weights,
        )
    raise ValueError(
        f"build_loss: unsupported architecture {architecture!r}. "
        "Only 'tft' is implemented in v3."
    )


__all__ = ["QuantileLoss", "ForecastLoss", "build_loss"]
