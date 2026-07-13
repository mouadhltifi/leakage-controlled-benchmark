"""Multi-task loss with ``ignore_index=-1`` guard on classification.

Combines cross-entropy (direction) with MSE (return, volatility)
weighted by per-target coefficients from :class:`HeadConfig`. Weights
are normalised to sum to 1 across the active targets so dropping a
target doesn't rescale the remaining losses (spec Section 3.9).

Weight derivation
-----------------

For the three possible target sets:

* ``[direction, return]`` — ``alpha = cfg.head.mtl_alpha``,
  ``1 - alpha`` on reg. (v1 default 0.5 each.)
* ``[direction, return, volatility]`` — uses ``mtl_alpha`` and
  ``mtl_beta``; if ``mtl_beta`` is ``None`` weights default to
  ``[alpha, (1-alpha)/2, (1-alpha)/2]``.
* ``[direction, volatility]`` — same split as direction+return but
  MSE is evaluated on vol.
* ``[return]`` — single MSE, weight 1.
* ``[volatility]`` — single MSE, weight 1. (In practice use
  :class:`~mmfp.models.losses.volatility.VolatilityLoss` via the
  factory.)

Any target active at head level must be present in ``preds`` and
``targets`` at forward time.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mmfp.config.schema import HeadConfig

_VALID_TARGETS: tuple[str, ...] = ("direction", "return", "volatility")


def _resolve_weights(cfg: HeadConfig) -> dict[str, float]:
    """Return the per-target raw weights (pre-normalization).

    Follows the v1 convention used by :class:`~mmfp.models.losses.multitask.MultiTaskLoss`:

    * When exactly two targets are active the split is ``(alpha, 1 - alpha)``.
    * When three targets are active the split is ``(alpha, beta, 1 - alpha - beta)``.
    * When only one target is active the single weight is 1.
    * When neither ``direction`` nor ``return``/``volatility`` are in the
      active target set the result is undefined — callers must ensure
      at least one target is provided.
    """
    targets = list(cfg.targets)
    alpha = cfg.mtl_alpha

    if len(targets) == 1:
        return {targets[0]: 1.0}

    if len(targets) == 2:
        first, second = targets
        return {first: alpha, second: max(0.0, 1.0 - alpha)}

    # len(targets) == 3
    beta = cfg.mtl_beta if cfg.mtl_beta is not None else (1.0 - alpha) / 2.0
    gamma = max(0.0, 1.0 - alpha - beta)
    if len(targets) != 3:
        raise ValueError(
            f"Unexpected targets count: {targets}"
        )
    # Maintain the order in cfg.targets so logs read uniformly.
    weights: dict[str, float] = {}
    # First target gets alpha; second beta; third gamma.
    weights[targets[0]] = alpha
    weights[targets[1]] = beta
    weights[targets[2]] = gamma
    return weights


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    """Project the raw weights onto the probability simplex.

    Negative raws are clamped to zero first. If everything is zero the
    weights collapse to uniform to avoid division-by-zero.
    """
    clipped = {k: max(0.0, v) for k, v in weights.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / total for k, v in clipped.items()}


class MultiTaskLoss(nn.Module):
    """Weighted sum of per-target losses.

    Parameters
    ----------
    cfg
        :class:`~mmfp.config.schema.HeadConfig` providing the target
        list and mixing coefficients.
    class_weights
        Optional ``(2,)`` float tensor for the classification
        ``CrossEntropyLoss``. Pre-computed at assembly time from train
        labels (see :mod:`mmfp.evaluation.metrics`, to be added in
        Milestone 6/7; for now the caller supplies the tensor).

    Notes
    -----
    Every ``CrossEntropyLoss`` instance is constructed with
    ``ignore_index=-1`` so that a defensive caller feeding deadzone
    rows (label ``-1``) still gets a well-defined loss (resolves audit
    B.2).
    """

    def __init__(
        self,
        cfg: HeadConfig,
        class_weights: Tensor | None = None,
    ) -> None:
        super().__init__()

        if not cfg.targets:
            raise ValueError("MultiTaskLoss requires at least one target.")
        for target in cfg.targets:
            if target not in _VALID_TARGETS:
                raise ValueError(
                    f"Unknown target {target!r}. Expected one of {_VALID_TARGETS}."
                )

        self._targets = list(cfg.targets)
        self._raw_weights = _resolve_weights(cfg)
        self._weights = _normalise(self._raw_weights)

        # Register class_weights as a buffer so it moves with the module
        # across devices. ``None`` is allowed (sklearn-style class weights
        # not supplied).
        if class_weights is not None:
            if class_weights.dim() != 1 or class_weights.size(0) != 2:
                raise ValueError(
                    f"class_weights must be a (2,) tensor; got shape "
                    f"{tuple(class_weights.shape)}."
                )
            self.register_buffer("class_weights", class_weights.to(torch.float32))
        else:
            self.class_weights = None  # type: ignore[assignment]

        self.cls_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=-1,
        )
        self.reg_loss = nn.MSELoss()

    # ------------------------------------------------------------------
    # Introspection helpers (used by tests and the trainer's logging).
    # ------------------------------------------------------------------

    @property
    def weights(self) -> dict[str, float]:
        """Return the normalised weights dict (sums to 1)."""
        return dict(self._weights)

    @property
    def targets(self) -> list[str]:
        """Ordered list of active targets."""
        return list(self._targets)

    # ------------------------------------------------------------------
    # Forward pass.
    # ------------------------------------------------------------------

    def forward(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the weighted multi-task loss.

        Parameters
        ----------
        preds
            Mapping from target name to prediction tensor. Shapes:

            * ``direction`` → ``(B, 2)`` class logits.
            * ``return`` / ``volatility`` → ``(B, 1)`` or ``(B,)``.
        targets
            Mapping from target name to ground-truth tensor. Shapes:

            * ``direction`` → ``(B,)`` int64 in ``{-1, 0, 1}`` (``-1``
              ignored via ``ignore_index``).
            * ``return`` / ``volatility`` → ``(B,)`` float.

        Returns
        -------
        (Tensor, dict[str, float])
            Total weighted loss (scalar) and a dict of per-target scalar
            loss values plus ``"total"`` for logging.
        """
        per_target_scalars: dict[str, float] = {}
        total_loss: Tensor | None = None

        for target in self._targets:
            if target not in preds:
                raise KeyError(
                    f"MultiTaskLoss.forward: active target {target!r} missing "
                    f"from preds (got {list(preds.keys())})."
                )
            if target not in targets:
                raise KeyError(
                    f"MultiTaskLoss.forward: active target {target!r} missing "
                    f"from targets (got {list(targets.keys())})."
                )

            if target == "direction":
                loss_t = self.cls_loss(preds[target], targets[target].long())
            elif target == "return":
                loss_t = self.reg_loss(
                    preds[target].squeeze(-1).float(), targets[target].float()
                )
            elif target == "volatility":
                loss_t = self.reg_loss(
                    preds[target].squeeze(-1).float(), targets[target].float()
                )
            else:
                raise ValueError(f"Unknown target {target!r}")

            per_target_scalars[target] = float(loss_t.detach().cpu().item())
            weighted = self._weights[target] * loss_t
            total_loss = weighted if total_loss is None else total_loss + weighted

        assert total_loss is not None, "MultiTaskLoss produced no contributions"
        per_target_scalars["total"] = float(total_loss.detach().cpu().item())
        return total_loss, per_target_scalars


__all__ = ["MultiTaskLoss"]
