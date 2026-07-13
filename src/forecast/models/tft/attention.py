"""Interpretable Multi-Head Attention (Lim et al. 2021, §4.5).

The "interpretable" qualifier reflects the paper's key modification to
standard multi-head attention: **the value projection is shared across
heads**. Per-head attention weights therefore operate on the same value
tensor, so the mean over heads of the attention weights has a direct
interpretation as "which keys contributed most to this query" — a
reading that is not available in standard MHA (where each head has its
own, incomparable value projection).

Formula (Lim et al. 2021, Eq. 14-16):

    H_h = softmax(Q_h K_h^T / sqrt(d_attn)) V
    # Note: V has no per-head index — it is the SAME value tensor.
    InterpretableMultiHead(Q, K, V) = W_h * mean_h(H_h)

With ``n_heads == 1`` this reduces to standard single-head scaled
dot-product attention (our v3 default per the architecture spec).

Determinism note
----------------

Per the architecture spec / §11.2 and the milestone brief, we
implement the attention math with explicit ``matmul`` + ``softmax`` and
**do not** use :func:`torch.nn.functional.scaled_dot_product_attention`
(the fused SDPA kernel is not bit-identical across devices / PyTorch
versions). A runtime assertion in ``test_attention.py`` pins this.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class InterpretableMultiHeadAttention(nn.Module):
    """Masked multi-head attention with shared value projection.

    Parameters
    ----------
    hidden_dim
        Width of the input and output embeddings ``H``. Must be
        divisible by ``n_heads`` so each head gets an equal-sized
        ``d_attn = H / n_heads`` slice.
    n_heads
        Number of attention heads. TFT paper default = 1 (preserves
        interpretability); >1 is supported but averages heads at the
        output.
    dropout
        Dropout probability applied to the post-softmax attention
        weights (standard attention dropout).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}")
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {n_heads}")
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads "
                f"({n_heads}); got remainder {hidden_dim % n_heads}."
            )

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.d_attn = hidden_dim // n_heads

        # Per-head Q and K projections — independent weights per head.
        # Implemented as a single Linear(H -> H) whose output is split.
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # SHARED value projection — the interpretable modification.
        # Single projection of width d_attn (not H), reused by every head.
        self.v_proj = nn.Linear(hidden_dim, self.d_attn, bias=True)

        # Output projection after averaging heads.
        self.out_proj = nn.Linear(self.d_attn, hidden_dim, bias=True)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        causal_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply self-attention.

        Parameters
        ----------
        x
            Input tensor of shape ``(B, L, H)``. Used as queries, keys,
            and values (self-attention).
        causal_mask
            Optional boolean tensor of shape ``(L, L)`` where True
            indicates positions to MASK OUT (i.e. `-inf` pre-softmax).
            If ``None``, no masking is applied (bidirectional attention).

        Returns
        -------
        attention_out
            Tensor of shape ``(B, L, H)``.
        attention_weights
            Tensor of shape ``(B, L, L)``. For ``n_heads > 1`` this is
            the mean across heads. Sums to 1 along the last axis
            (subject to the causal mask).
        """
        if x.dim() != 3:
            raise ValueError(
                f"Attention input must be (B, L, H); got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Attention input hidden dim {x.shape[-1]} != "
                f"self.hidden_dim {self.hidden_dim}."
            )

        B, L, _ = x.shape

        # Per-head Q, K projections; shape (B, n_heads, L, d_attn).
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_attn).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_attn).transpose(1, 2)

        # Shared V projection; shape (B, L, d_attn), broadcast across heads.
        v_shared = self.v_proj(x)  # (B, L, d_attn)
        # Expand to (B, n_heads, L, d_attn) by broadcasting across the
        # new head axis.
        v = v_shared.unsqueeze(1).expand(B, self.n_heads, L, self.d_attn)

        # Scaled dot-product attention, manual implementation (no SDPA).
        # scores: (B, n_heads, L, L).
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_attn)

        if causal_mask is not None:
            if causal_mask.shape != (L, L):
                raise ValueError(
                    f"causal_mask shape {tuple(causal_mask.shape)} != "
                    f"expected ({L}, {L})."
                )
            # Broadcast mask to (1, 1, L, L) for per-head masking.
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Softmax over keys.
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum: (B, n_heads, L, d_attn).
        head_out = torch.matmul(attn, v)

        # Mean across heads (interpretability-preserving aggregation).
        # Shape (B, L, d_attn).
        pooled = head_out.mean(dim=1)

        # Final linear projection back to hidden_dim.
        out = self.out_proj(pooled)

        # Mean attention weights across heads (interpretability signal).
        # Shape (B, L, L).
        attn_weights_mean = attn.mean(dim=1)

        return out, attn_weights_mean

    @staticmethod
    def build_causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
        """Build a boolean causal mask of shape ``(L, L)``.

        Returns ``True`` above the main diagonal (i.e. positions to
        mask out — future timesteps should not attend).
        """
        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask


__all__ = ["InterpretableMultiHeadAttention"]
