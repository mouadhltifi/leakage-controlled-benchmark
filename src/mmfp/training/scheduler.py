"""Learning-rate scheduler factory.

Returns a scheduler instance selected by ``cfg.training.scheduler``.

Supported choices
-----------------

* ``"cosine_warm_restarts"`` — :class:`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`
  with ``T_0=10``, ``T_mult=2``. Matches the v1 default.
* ``"reduce_on_plateau"`` — :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`
  with ``mode='min'``, ``patience=10``, ``factor=0.5``. The trainer steps
  this with the *validation loss* at epoch end — it's the only scheduler
  that needs a metric argument.
* ``"none"`` — returns ``None``. Trainer simply skips ``scheduler.step()``.

The per-scheduler hyperparameters above are hard-coded by spec Section
3.11 so individual experiments don't need to bikeshed them.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from mmfp.config.schema import ExperimentConfig

log = logging.getLogger(__name__)


def build_scheduler(
    cfg: ExperimentConfig,
    optimizer: Optimizer,
) -> Any | None:
    """Instantiate the scheduler selected by ``cfg.training.scheduler``.

    Parameters
    ----------
    cfg
        Validated experiment configuration. Only ``cfg.training.scheduler``
        is consulted.
    optimizer
        The optimizer whose learning rate the scheduler will adjust.

    Returns
    -------
    Any | None
        The concrete scheduler instance, or ``None`` when
        ``cfg.training.scheduler == "none"``.

        The return type is broad because PyTorch's LR schedulers don't
        share a public ABC that both ``_LRScheduler`` and
        ``ReduceLROnPlateau`` inherit from — the latter lives outside the
        ``_LRScheduler`` hierarchy until PyTorch 2.0+.
    """
    choice = cfg.training.scheduler
    if choice == "cosine_warm_restarts":
        log.debug("build_scheduler: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)")
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    if choice == "reduce_on_plateau":
        log.debug("build_scheduler: ReduceLROnPlateau(mode='min', patience=10, factor=0.5)")
        return ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5,
        )
    if choice == "none":
        log.debug("build_scheduler: none (scheduler disabled)")
        return None
    raise ValueError(
        f"Unknown scheduler {choice!r}. "
        "Expected 'cosine_warm_restarts', 'reduce_on_plateau' or 'none'."
    )


def is_plateau_scheduler(scheduler: Any | None) -> bool:
    """Return ``True`` if the scheduler requires a metric arg to ``step()``.

    Plateau schedulers read the *validation loss* at each epoch end;
    others advance purely by the epoch index.
    """
    return scheduler is not None and isinstance(scheduler, ReduceLROnPlateau)


__all__ = ["build_scheduler", "is_plateau_scheduler"]
