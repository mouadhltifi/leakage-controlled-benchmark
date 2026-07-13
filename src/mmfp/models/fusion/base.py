"""Abstract base class for modality fusion strategies.

Fusion takes a dict of per-modality ``(B, H)`` encodings and produces a
single ``(B, H)`` fused representation. The dict-in / tensor-out
interface avoids depending on modality order and makes single-modality
fallback handling explicit — each strategy knows how to behave when
only one key is present, so the :class:`~mmfp.models.predictor.Predictor`
does not need to branch on fusion type (spec Section 3.7).

Contract
--------

* ``forward(encodings)`` accepts ``dict[str, torch.Tensor]`` mapping
  modality name to ``(B, H)`` tensor. Keys are a subset of
  ``{"price", "macro", "news", "social", "graph"}``.
* Returns ``(B, H)`` float tensor.
* Single-modality input (``len(encodings) == 1``) returns the sole
  encoding **unchanged**. This is a spec-level requirement so that
  ablation configs with a single active modality never go through a
  parametric projection the experiment didn't intend.
* Every strategy pulls dropout and head count from the model config — no
  hardcoded values (fixes v1 audit findings in ``fusion.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class FusionStrategy(nn.Module, ABC):
    """Abstract base for all fusion strategies.

    Subclasses must implement :meth:`forward` taking the encodings
    dict. The base class factors out the single-modality pass-through
    check so concrete strategies can focus on the multi-modality path.
    """

    @abstractmethod
    def forward(self, encodings: dict[str, Tensor]) -> Tensor:
        """Fuse ``{modality: (B, H)}`` into ``(B, H)``."""
        ...

    @staticmethod
    def _single_modality_passthrough(
        encodings: dict[str, Tensor],
    ) -> Tensor | None:
        """Return the sole encoding if exactly one modality is present.

        Helper used by concrete strategies to implement the spec
        requirement that single-modality input returns unchanged. Returns
        ``None`` when the caller should proceed with the multi-modality
        path.
        """
        if len(encodings) == 0:
            raise ValueError(
                "Fusion received an empty encodings dict; at least one "
                "modality must be active."
            )
        if len(encodings) == 1:
            return next(iter(encodings.values()))
        return None


__all__ = ["FusionStrategy"]
