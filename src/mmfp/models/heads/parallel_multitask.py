"""Parallel multi-task head.

Each active target gets its own small MLP reading the same shared fused
representation. Matches v1's ``MultiTaskHead`` semantics but:

* Parameterised by the active targets list from ``cfg.head.targets`` so
  the head cleanly supports single-target, dual-target or triple-target
  ablation without dead branches.
* ``mid_dim = max(32, hidden_dim // 2)`` per audit fix (v1 had a
  hardcoded mid_dim path for the 64-dim hidden case).
* Dropout from ``cfg.model.dropout``; no hardcoded 0.3.

Per-target output dims
----------------------

* ``direction``: 2 logits (binary classification up/down).
* ``return``: 1 scalar (next-day log return).
* ``volatility``: 1 scalar (next-day absolute log return).
"""

from __future__ import annotations

from torch import Tensor, nn

from mmfp.config.schema import ExperimentConfig

from .base import PredictionHead

#: Output dimensions per target. Not a config knob — dictated by task.
_TARGET_OUTPUT_DIMS: dict[str, int] = {
    "direction": 2,
    "return": 1,
    "volatility": 1,
}


def _mid_dim(hidden_dim: int) -> int:
    """``max(32, H // 2)`` — audit-fix formula shared across heads."""
    return max(32, hidden_dim // 2)


def _build_target_mlp(
    hidden_dim: int, out_dim: int, dropout: float,
) -> nn.Sequential:
    """Single-target MLP used by parallel, sequential and single-task heads."""
    mid = _mid_dim(hidden_dim)
    return nn.Sequential(
        nn.Linear(hidden_dim, mid),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mid, out_dim),
    )


class ParallelMultiTaskHead(PredictionHead):
    """Independent per-target MLPs over the shared fused representation.

    Parameters
    ----------
    cfg
        Full experiment config. ``cfg.model.hidden_dim``,
        ``cfg.model.dropout`` and ``cfg.head.targets`` are consulted.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()

        hidden_dim = cfg.model.hidden_dim
        dropout = cfg.model.dropout
        targets = cfg.head.targets

        if not targets:
            raise ValueError("ParallelMultiTaskHead requires at least one target.")

        self._targets: list[str] = list(targets)
        self.heads = nn.ModuleDict()
        for target in self._targets:
            if target not in _TARGET_OUTPUT_DIMS:
                raise ValueError(
                    f"Unknown target {target!r}. "
                    f"Expected one of {list(_TARGET_OUTPUT_DIMS.keys())}."
                )
            self.heads[target] = _build_target_mlp(
                hidden_dim=hidden_dim,
                out_dim=_TARGET_OUTPUT_DIMS[target],
                dropout=dropout,
            )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Run each active target head on the fused representation."""
        return {target: self.heads[target](x) for target in self._targets}


__all__ = ["ParallelMultiTaskHead", "_build_target_mlp", "_mid_dim", "_TARGET_OUTPUT_DIMS"]
