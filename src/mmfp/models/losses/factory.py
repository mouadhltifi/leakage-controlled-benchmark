"""Loss-factory utilities.

Picks between :class:`~mmfp.models.losses.multitask.MultiTaskLoss` and
:class:`~mmfp.models.losses.volatility.VolatilityLoss` based on the
active targets.

The factory also exposes :func:`compute_class_weights` for the two
supported class-weight schemes (``inverse_frequency`` and
``balanced``). See spec Section 3.9 for the discussion of why both are
available.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from mmfp.config.schema import ExperimentConfig

from .multitask import MultiTaskLoss
from .volatility import VolatilityLoss


def compute_class_weights(
    labels: np.ndarray,
    *,
    scheme: str = "inverse_frequency",
) -> Tensor:
    """Compute per-class weights from an array of integer labels.

    Parameters
    ----------
    labels
        1-D integer array of class labels. Deadzone (``-1``) values are
        ignored.
    scheme
        One of:

        * ``"inverse_frequency"`` — ``1 / count_c``, rescaled so the
          minimum weight equals 1. Matches v1 convention.
        * ``"balanced"`` — sklearn-style
          ``n_samples / (n_classes * count_c)``.

    Returns
    -------
    torch.Tensor
        ``(n_classes,)`` float32 tensor. ``n_classes`` is inferred as
        ``max(labels) + 1`` after filtering deadzone.
    """
    if scheme not in ("inverse_frequency", "balanced"):
        raise ValueError(
            f"Unknown class_weights scheme {scheme!r}. "
            "Expected 'inverse_frequency' or 'balanced'."
        )

    valid = labels[labels >= 0]
    if valid.size == 0:
        raise ValueError(
            "compute_class_weights received no non-deadzone labels."
        )

    n_classes = int(valid.max()) + 1
    counts = np.bincount(valid, minlength=n_classes).astype(np.float64)
    # Guard against zero-count classes (should not happen for a
    # well-formed training set, but protects against degenerate splits).
    counts = np.where(counts == 0, 1.0, counts)

    if scheme == "inverse_frequency":
        weights = 1.0 / counts
        weights = weights / weights.min()
    else:  # balanced
        n_samples = float(valid.size)
        weights = n_samples / (n_classes * counts)

    return torch.as_tensor(weights, dtype=torch.float32)


def build_loss(
    cfg: ExperimentConfig,
    class_weights: Tensor | None = None,
) -> nn.Module:
    """Instantiate the loss module matching ``cfg.head.targets``.

    Parameters
    ----------
    cfg
        Validated experiment config.
    class_weights
        Optional ``(2,)`` float tensor of per-class weights for the
        classification branch. Callers compute this from the train
        split (see :func:`compute_class_weights`) and pass it here.

    Returns
    -------
    nn.Module
        Either a :class:`MultiTaskLoss` or a :class:`VolatilityLoss`.

    Raises
    ------
    ValueError
        If the target list is empty (caught by the schema) or if an
        unknown target name sneaks in.
    """
    targets = cfg.head.targets
    if not targets:
        raise ValueError("head.targets must be non-empty.")

    # Single volatility target gets the dedicated class; everything else
    # flows through MultiTaskLoss (which also handles the single-target
    # path for 'direction' or 'return' cleanly).
    if targets == ["volatility"]:
        return VolatilityLoss()

    return MultiTaskLoss(cfg.head, class_weights=class_weights)


__all__ = ["build_loss", "compute_class_weights"]
