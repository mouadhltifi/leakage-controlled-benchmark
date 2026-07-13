"""Sequential cascade head: forecast first, then classify from the forecast.

Implements the second architecture candidate of axis 1 (spec Section
3.8). The regression branch (``return`` or ``volatility`` per
``cfg.head.cascade_reg_target``) produces a scalar prediction that is
concatenated with the fused representation to condition the
classification branch.

Gradient behaviour
------------------

By default (``cfg.head.detach_cascade == False``) the reg prediction is
**not detached** before being fed into the cls branch. This means the
classification loss backpropagates through the reg branch as well,
yielding joint optimization of both tasks. This is the v2 default per
spec Q7.

Setting ``cfg.head.detach_cascade = True`` calls ``.detach()`` on
``reg_pred`` before concatenation, giving the v1 "cascade-as-feature"
interpretation in which only the reg loss trains the reg branch.

Targets
-------

The config validator (``_rule_sequential_cascade_targets``) enforces:

* ``"direction"`` is in ``cfg.head.targets``.
* ``cfg.head.cascade_reg_target`` is in ``cfg.head.targets``.

So this head never needs to build a target that isn't active.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mmfp.config.schema import ExperimentConfig

from .base import PredictionHead
from .parallel_multitask import _TARGET_OUTPUT_DIMS, _build_target_mlp, _mid_dim


class SequentialCascadeHead(PredictionHead):
    """Forecast-then-classify cascade head.

    Parameters
    ----------
    cfg
        Full experiment config. ``cfg.model.hidden_dim``,
        ``cfg.model.dropout``, ``cfg.head.targets``,
        ``cfg.head.cascade_reg_target`` and ``cfg.head.detach_cascade``
        are consulted.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()

        hidden_dim = cfg.model.hidden_dim
        dropout = cfg.model.dropout
        targets = cfg.head.targets
        reg_target = cfg.head.cascade_reg_target

        if "direction" not in targets:
            raise ValueError(
                "SequentialCascadeHead requires 'direction' in head.targets; "
                f"got {targets}. Config validator should have caught this."
            )
        if reg_target not in targets:
            raise ValueError(
                f"SequentialCascadeHead requires cascade_reg_target "
                f"({reg_target!r}) to be in head.targets {targets}. "
                "Config validator should have caught this."
            )
        if reg_target not in ("return", "volatility"):
            raise ValueError(
                f"cascade_reg_target must be 'return' or 'volatility'; got "
                f"{reg_target!r}."
            )

        self._reg_target = reg_target
        self._detach_cascade = cfg.head.detach_cascade
        self._targets: list[str] = list(targets)

        # Regression branch: same structure as the parallel head.
        self.reg_head = _build_target_mlp(
            hidden_dim=hidden_dim,
            out_dim=_TARGET_OUTPUT_DIMS[reg_target],
            dropout=dropout,
        )

        # Classification branch reads [fused ; reg_pred].
        mid = _mid_dim(hidden_dim)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim + _TARGET_OUTPUT_DIMS[reg_target], mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, _TARGET_OUTPUT_DIMS["direction"]),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Cascade regression then classification.

        Notes
        -----
        When ``detach_cascade=False`` (default), the cls loss's gradient
        flows into ``reg_head.weight`` through ``reg_pred``. The unit
        test ``test_sequential_cascade_gradient_flow`` verifies this
        explicitly.
        """
        reg_pred = self.reg_head(x)  # (B, out_dim_reg)

        cls_input_reg = reg_pred.detach() if self._detach_cascade else reg_pred
        cls_input = torch.cat([x, cls_input_reg], dim=-1)
        cls_logits = self.cls_head(cls_input)

        out: dict[str, Tensor] = {
            "direction": cls_logits,
            self._reg_target: reg_pred,
        }
        return out


__all__ = ["SequentialCascadeHead"]
