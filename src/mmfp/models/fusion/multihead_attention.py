"""Multi-head self-attention fusion (DASF-Net-style).

Stacks modality encodings as a sequence of tokens and applies a
transformer-style self-attention + residual + average-pool across the
modality dimension. Unlike :class:`GatedCrossAttentionFusion`, no
modality is privileged; attention discovers the cross-modal routing
from scratch.

Ported from v1 ``src/models/fusion.py::MultiHeadAttentionFusion`` with
the usual fixes: dropout from ``cfg.model.dropout``, heads from
``cfg.model.num_heads``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mmfp.config.schema import ExperimentConfig

from .base import FusionStrategy


class MultiheadAttentionFusion(FusionStrategy):
    """Self-attention over modality tokens with residual + mean pool.

    Parameters
    ----------
    cfg
        Full experiment config. ``cfg.model.hidden_dim``,
        ``cfg.model.num_heads`` and ``cfg.model.dropout`` are consulted.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()

        hidden_dim = cfg.model.hidden_dim
        num_heads = cfg.model.num_heads
        dropout = cfg.model.dropout

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"model.hidden_dim ({hidden_dim}) must be divisible by "
                f"model.num_heads ({num_heads}) for multi-head attention."
            )

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, encodings: dict[str, Tensor]) -> Tensor:
        """Encode modalities as a sequence and mean-pool the result."""
        single = self._single_modality_passthrough(encodings)
        if single is not None:
            return single

        # Stack modalities into (B, n_mod, H).
        tokens = torch.stack(list(encodings.values()), dim=1)

        attn_out, _ = self.mha(tokens, tokens, tokens)
        # Residual + LN over the modality sequence.
        post_attn = self.layer_norm(attn_out + tokens)

        # Mean pool across the modality axis -> (B, H).
        return post_attn.mean(dim=1)


__all__ = ["MultiheadAttentionFusion"]
