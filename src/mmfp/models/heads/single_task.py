"""Single-task head: one active target, one small MLP.

Used for the single-task leg of axis 1. The config validator enforces
``len(cfg.head.targets) == 1`` for this architecture so the head can
safely index ``targets[0]``.
"""

from __future__ import annotations

from torch import Tensor

from mmfp.config.schema import ExperimentConfig

from .base import PredictionHead
from .parallel_multitask import _TARGET_OUTPUT_DIMS, _build_target_mlp


class SingleTaskHead(PredictionHead):
    """Single-target MLP head.

    Parameters
    ----------
    cfg
        Full experiment config. ``cfg.model.hidden_dim``,
        ``cfg.model.dropout`` and ``cfg.head.targets`` are consulted.
        ``cfg.head.targets`` must have length 1 (enforced by the cross-
        field validator).
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()

        targets = cfg.head.targets
        if len(targets) != 1:
            raise ValueError(
                "SingleTaskHead requires exactly one target; got "
                f"{targets}. Config validator should have caught this."
            )

        target = targets[0]
        if target not in _TARGET_OUTPUT_DIMS:
            raise ValueError(
                f"Unknown target {target!r}. "
                f"Expected one of {list(_TARGET_OUTPUT_DIMS.keys())}."
            )

        self._target = target
        self.head = _build_target_mlp(
            hidden_dim=cfg.model.hidden_dim,
            out_dim=_TARGET_OUTPUT_DIMS[target],
            dropout=cfg.model.dropout,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return a single-key dict for the active target."""
        return {self._target: self.head(x)}


__all__ = ["SingleTaskHead"]
