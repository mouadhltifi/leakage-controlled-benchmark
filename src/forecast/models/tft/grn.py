"""Gated Residual Network (Lim et al. 2021, §4.1).

The GRN is the workhorse of the Temporal Fusion Transformer: every
non-linear mapping in the model is a GRN (used ~15 times in the full
body, including inside every VSN, in the static-covariate encoders, in
static enrichment, and in the position-wise feed-forward block after
the interpretable multi-head attention).

Equations (Lim et al. 2021, Eq. 2-4), with optional context ``c``:

    eta_2 = ELU(W_2 a + W_3 c + b_2)
    eta_1 = W_1 eta_2 + b_1
    GRN(a, c) = LayerNorm( a_skip + GLU(eta_1) )
    GLU(x)   = sigmoid(W_g x + b_g) * (W_v x + b_v)

where ``a_skip`` is either ``a`` itself (when ``input_dim == output_dim``)
or a learned projection ``W_s a`` (when dims differ). Dropout is applied
on ``eta_1`` before the GLU, matching the paper's appendix description
and the reference implementations.

The GRN is the only place where context is injected: caller decides
whether to pass a per-timestep context ``c`` (e.g. static covariate
context broadcast to the sequence length) or ``None``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _GatedLinearUnit(nn.Module):
    """Gated Linear Unit used inside the GRN.

    Computes ``sigmoid(W_g x + b_g) * (W_v x + b_v)`` via a single
    Linear projection of width ``2 * hidden_dim`` followed by a split.
    Matches the reference implementation shape (no separate Linear pair)
    for parameter efficiency; the math is identical.

    Parameters
    ----------
    hidden_dim
        Input and output dimensionality (GLU preserves dim).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply gated linear unit.

        Parameters
        ----------
        x
            ``(..., hidden_dim)`` tensor.

        Returns
        -------
        Tensor
            ``(..., hidden_dim)`` tensor.
        """
        projected = self.fc(x)
        gate, value = projected.chunk(2, dim=-1)
        return torch.sigmoid(gate) * value


class GRN(nn.Module):
    """Gated Residual Network (Lim et al. 2021, §4.1).

    Performs a non-linear transformation ``a -> output`` with an
    optional context vector ``c`` added after the first linear. Uses a
    GLU gate on the residual branch and a LayerNorm over the skip sum.

    Flow:

    1.  Optional residual projection if ``input_dim != output_dim``.
    2.  ``eta_2 = ELU(W_2 a + W_3 c)`` (context term skipped if ``c``
        is ``None``).
    3.  ``eta_1 = W_1 eta_2`` (the ``fc2`` layer).
    4.  Dropout on ``eta_1``.
    5.  ``GLU(eta_1)``.
    6.  ``LayerNorm(skip + GLU_out)``.

    Parameters
    ----------
    input_dim
        Dimensionality of ``a``.
    hidden_dim
        Hidden width of the intermediate activations.
    output_dim
        Output dimensionality. If ``None``, defaults to ``input_dim``.
    context_dim
        Dimensionality of the optional context vector. If ``None``, the
        context path is disabled; passing a non-``None`` ``c`` to
        :meth:`forward` then raises ``ValueError``.
    dropout
        Dropout probability applied to ``eta_1`` before the GLU.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        context_dim: int | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive; got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.context_dim = context_dim

        # Primary path.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            # Context is added to fc1 output before ELU; use bias=False
            # so the effective bias is shared with fc1.
            self.context_proj: nn.Linear | None = nn.Linear(
                context_dim, hidden_dim, bias=False
            )
        else:
            self.context_proj = None

        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = _GatedLinearUnit(self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)

        # Residual projection only if dims differ.
        if input_dim != self.output_dim:
            self.skip_proj: nn.Linear | None = nn.Linear(
                input_dim, self.output_dim
            )
        else:
            self.skip_proj = None

    def forward(self, a: Tensor, c: Tensor | None = None) -> Tensor:
        """Apply GRN.

        Parameters
        ----------
        a
            Input tensor of shape ``(..., input_dim)``.
        c
            Optional context tensor. Must be broadcastable onto ``a``'s
            leading dimensions with trailing dim ``context_dim``. The
            caller is responsible for the broadcast (e.g. unsqueezing
            a ``(B, context_dim)`` context to ``(B, 1, context_dim)``
            and letting broadcast handle the sequence axis).

        Returns
        -------
        Tensor
            Tensor of shape ``(..., output_dim)``.

        Raises
        ------
        ValueError
            If ``c`` is passed but the module was constructed without
            ``context_dim``, or vice versa.
        """
        if c is not None and self.context_proj is None:
            raise ValueError(
                "GRN was constructed without context_dim but a context "
                "tensor was passed to forward()."
            )
        if c is None and self.context_proj is not None:
            raise ValueError(
                "GRN was constructed with context_dim but no context "
                "tensor was passed to forward()."
            )

        # Residual branch (projected to output_dim if needed).
        skip = a if self.skip_proj is None else self.skip_proj(a)

        x = self.fc1(a)
        if c is not None:
            # Context is a per-sample (or per-sample-per-time) vector;
            # rely on broadcasting for shape alignment.
            assert self.context_proj is not None  # for mypy
            x = x + self.context_proj(c)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.layer_norm(skip + x)


__all__ = ["GRN"]
