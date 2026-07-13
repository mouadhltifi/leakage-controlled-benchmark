"""Fusion strategies and factory.

Three strategies ported from v1 with hardcoded values removed and
single-modality pass-through made explicit:

* :class:`ConcatFusion` — concatenation + projection baseline.
* :class:`GatedCrossAttentionFusion` — MSGCA-style cross-attention
  with sigmoid gating and a configurable primary modality.
* :class:`MultiheadAttentionFusion` — self-attention over the modality
  sequence, no privileged modality.

Use :func:`build_fusion` to instantiate the one named in
``cfg.fusion.strategy``.
"""

from __future__ import annotations

from mmfp.config.schema import ExperimentConfig

from .base import FusionStrategy
from .concat import ConcatFusion
from .gated_cross_attention import GatedCrossAttentionFusion
from .multihead_attention import MultiheadAttentionFusion


def build_fusion(cfg: ExperimentConfig, n_modalities: int) -> FusionStrategy:
    """Instantiate the fusion module selected by ``cfg.fusion.strategy``.

    Parameters
    ----------
    cfg
        Validated experiment config.
    n_modalities
        Number of active modality encoders (used by
        :class:`ConcatFusion` to size its projection).

    Returns
    -------
    FusionStrategy
        A concrete fusion module ready to be called on an encodings
        dict.

    Raises
    ------
    ValueError
        If ``cfg.fusion.strategy`` is not one of the three supported
        values (should not happen after config validation).
    """
    strategy = cfg.fusion.strategy
    if strategy == "concat":
        return ConcatFusion(cfg, n_modalities=n_modalities)
    if strategy == "gated_cross_attention":
        return GatedCrossAttentionFusion(cfg)
    if strategy == "multihead_attention":
        return MultiheadAttentionFusion(cfg)
    raise ValueError(f"Unknown fusion strategy: {strategy!r}")


__all__ = [
    "ConcatFusion",
    "FusionStrategy",
    "GatedCrossAttentionFusion",
    "MultiheadAttentionFusion",
    "build_fusion",
]
