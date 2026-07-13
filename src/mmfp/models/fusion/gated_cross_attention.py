"""Gated cross-attention fusion (MSGCA-style).

One modality acts as the primary query; the rest form the key/value
sequence. A sigmoid gate modulates how much of the cross-attention
output flows into the residual sum with the primary encoding.

Ported from v1 ``src/models/fusion.py::GatedCrossAttention`` with the
following fixes (audit B.2 / v1 hardcoded values):

* ``num_heads`` comes from ``cfg.model.num_heads`` (v1 hardcoded 4 and
  8 in separate modules).
* Dropouts come from ``cfg.model.dropout`` (v1 hardcoded 0.1 on the
  MHA dropout and 0.2 on the gate-output dropout).
* The primary modality name comes from
  :attr:`~mmfp.config.schema.FusionConfig.primary_modality`; fail loud
  when absent from the encodings dict rather than silently re-routing.
* Single-modality input passes through unchanged (spec requirement).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mmfp.config.schema import ExperimentConfig

from .base import FusionStrategy


class GatedCrossAttentionFusion(FusionStrategy):
    """Primary-modality-as-query cross-attention with a sigmoid gate.

    Parameters
    ----------
    cfg
        Full experiment config. ``cfg.model.hidden_dim``,
        ``cfg.model.num_heads``, ``cfg.model.dropout`` and
        ``cfg.fusion.primary_modality`` are consulted.
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

        self._primary_modality = cfg.fusion.primary_modality
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encodings: dict[str, Tensor]) -> Tensor:
        """Primary attends over auxiliaries; sigmoid-gate the output."""
        single = self._single_modality_passthrough(encodings)
        if single is not None:
            return single

        if self._primary_modality not in encodings:
            raise ValueError(
                f"GatedCrossAttentionFusion: primary modality "
                f"{self._primary_modality!r} not in encodings "
                f"(got {list(encodings.keys())}). Either enable the "
                "primary modality or switch fusion strategy."
            )

        primary = encodings[self._primary_modality]
        auxiliaries = [
            v for k, v in encodings.items() if k != self._primary_modality
        ]

        # Primary as single-token query: (B, 1, H).
        query = primary.unsqueeze(1)
        # Stack auxiliaries as key/value sequence: (B, n_aux, H).
        aux_stack = torch.stack(auxiliaries, dim=1)

        attn_out, _ = self.mha(query, aux_stack, aux_stack)
        attn_out = attn_out.squeeze(1)  # (B, H)

        gate_input = torch.cat([primary, attn_out], dim=-1)
        gate_weight = self.gate(gate_input)

        fused = primary + gate_weight * self.dropout(attn_out)
        return self.layer_norm(fused)


__all__ = ["GatedCrossAttentionFusion"]
