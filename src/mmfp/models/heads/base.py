"""Abstract base class for prediction heads.

A head turns the fused representation into per-target predictions. The
interface is a dict so the trainer and losses can iterate over active
targets uniformly regardless of whether one or three are emitted (spec
Section 3.8).

Contract
--------

* ``forward(x)`` takes ``(B, H)`` and returns ``dict[str, Tensor]``
  whose keys are a subset of ``{"direction", "return", "volatility"}``.
* Shapes per key:

  - ``direction`` → ``(B, 2)`` class logits.
  - ``return`` → ``(B, 1)`` regression output.
  - ``volatility`` → ``(B, 1)`` regression output.

* Heads do **not** apply softmax / sigmoid; that stays in the loss or
  evaluation layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class PredictionHead(nn.Module, ABC):
    """Abstract base for prediction heads."""

    @abstractmethod
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Project the fused representation to per-target predictions.

        Parameters
        ----------
        x
            Fused representation, shape ``(B, H)``.

        Returns
        -------
        dict[str, Tensor]
            Mapping from target name to its prediction tensor.
        """
        ...


__all__ = ["PredictionHead"]
