"""Concatenation baseline fusion.

Flattens modality encodings into ``(B, n_modalities * H)`` and projects
back to ``(B, H)`` through ``Linear + ReLU + Dropout + LayerNorm``.

Differences from v1's ``ConcatFusion``:

* Dropout comes from ``cfg.model.dropout`` (v1 hardcoded 0.3).
* Final ``LayerNorm(H)`` ensures the fused vector enters the head at
  the same scale as single-modality encoder outputs.
* Single-modality fallback returns the sole encoding unchanged (spec
  requirement).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mmfp.config.schema import ExperimentConfig

from .base import FusionStrategy


class ConcatFusion(FusionStrategy):
    """Concatenation fusion with a single linear projection back to ``H``.

    Parameters
    ----------
    cfg
        Full experiment config. ``cfg.model.hidden_dim`` and
        ``cfg.model.dropout`` are consulted.
    n_modalities
        Number of active modalities. The projection's input dimension is
        ``n_modalities * cfg.model.hidden_dim``.

    Notes
    -----
    The concatenation order at runtime matches Python's insertion order
    on the encodings dict, so long as the :class:`~mmfp.models.predictor.Predictor`
    always inserts modality encodings in the same canonical order the
    fusion module was built for.
    """

    def __init__(self, cfg: ExperimentConfig, n_modalities: int) -> None:
        super().__init__()

        if n_modalities < 1:
            raise ValueError(
                f"ConcatFusion requires n_modalities >= 1; got {n_modalities}"
            )

        hidden_dim = cfg.model.hidden_dim
        self._hidden_dim = hidden_dim
        self._n_modalities = n_modalities

        if n_modalities == 1:
            # No parameters needed — single-modality input flows through
            # the passthrough branch in ``forward``.
            self.projection: nn.Module | None = None
            self.norm: nn.Module | None = None
        else:
            self.projection = nn.Sequential(
                nn.Linear(n_modalities * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.model.dropout),
            )
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, encodings: dict[str, Tensor]) -> Tensor:
        """Concatenate and project the per-modality encodings."""
        single = self._single_modality_passthrough(encodings)
        if single is not None:
            return single

        concatenated = torch.cat(list(encodings.values()), dim=-1)
        assert self.projection is not None and self.norm is not None
        return self.norm(self.projection(concatenated))


__all__ = ["ConcatFusion"]
