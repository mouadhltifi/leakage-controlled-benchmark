"""Volatility-only MSE loss.

Used when ``cfg.head.targets == ["volatility"]``. Keeps the factory
clean — the :class:`~mmfp.models.losses.multitask.MultiTaskLoss` would
work for the single-volatility case too, but a dedicated class removes
ambiguity in the trainer's logging and matches v1's ``VolatilityLoss``
output shape.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VolatilityLoss(nn.Module):
    """MSE regression loss on ``|log_return|``.

    Parameters
    ----------
    None.

    Notes
    -----
    :meth:`forward` accepts either a prediction dict (from the head) or
    a raw tensor so the factory can route single-task-volatility and
    multi-task-volatility cases with equal ease.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reg_loss = nn.MSELoss()

    def forward(
        self,
        preds: dict[str, Tensor] | Tensor,
        targets: dict[str, Tensor] | Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute MSE between ``preds`` and ``targets``.

        Accepts dicts to stay compatible with the trainer's uniform
        ``(preds_dict, targets_dict)`` calling convention. When a dict
        is passed the ``"volatility"`` key is required.
        """
        pred_tensor = (
            preds["volatility"] if isinstance(preds, dict) else preds
        )
        target_tensor = (
            targets["volatility"] if isinstance(targets, dict) else targets
        )
        loss = self.reg_loss(
            pred_tensor.squeeze(-1).float(), target_tensor.float()
        )
        scalars = {
            "volatility": float(loss.detach().cpu().item()),
            "total": float(loss.detach().cpu().item()),
        }
        return loss, scalars


__all__ = ["VolatilityLoss"]
