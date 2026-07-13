"""Variable Selection Network (Lim et al. 2021, §4.2).

Implements a soft selection over a set of input variables. Each
variable gets its own GRN (per-variable non-linearity). A second GRN
consumes the flattened concatenation of all variables (optionally
conditioned on a static context) and produces a softmax over the
variable dimension — the "selection weights". The output is the
weighted sum of the per-variable GRN outputs.

Equations (Lim et al. 2021, Eq. 5-7):

    v_chi      = softmax( GRN_v( flatten(xi), c ) )      # (..., n_vars)
    tilde_xi_j = GRN_xi_j( xi_j )                        # (..., input_dim)
    output     = sum_j v_chi[j] * tilde_xi_j             # (..., input_dim)

We follow the reference implementation in one detail: when
``n_variables == 1`` there is nothing to select over, so we bypass the
softmax-GRN and return a placeholder weight tensor of shape
``(..., 1)`` equal to 1. This simplifies callers that always request a
weight tensor for interpretability logging.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from forecast.models.tft.grn import GRN


class VariableSelectionNetwork(nn.Module):
    """Soft selection over a fixed set of equally-shaped input variables.

    Each variable is a tensor of shape ``(..., input_dim)``; the module
    processes ``n_variables`` of them. Used in TFT for (a) static
    covariates, (b) past-observed covariates across timesteps, and
    optionally (c) future-known covariates.

    Parameters
    ----------
    n_variables
        Number of input variables to select over (>= 1).
    input_dim
        Per-variable hidden width ``H``. All variables must share this.
    hidden_dim
        GRN hidden-layer width (typically equal to ``input_dim``).
    context_dim
        Optional static context dim. When not ``None``, the selection
        GRN consumes the static context alongside the flattened inputs.
    dropout
        Dropout probability inside the GRNs.
    """

    def __init__(
        self,
        n_variables: int,
        input_dim: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if n_variables < 1:
            raise ValueError(
                f"n_variables must be >= 1; got {n_variables}"
            )
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive; got {input_dim}")

        self.n_variables = n_variables
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # Per-variable GRNs (independent weights per variable).
        self.per_var_grns = nn.ModuleList(
            [
                GRN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=input_dim,
                    context_dim=None,
                    dropout=dropout,
                )
                for _ in range(n_variables)
            ]
        )

        # Selection GRN (only meaningful when n_variables > 1).
        if n_variables > 1:
            self.selection_grn: GRN | None = GRN(
                input_dim=n_variables * input_dim,
                hidden_dim=hidden_dim,
                output_dim=n_variables,
                context_dim=context_dim,
                dropout=dropout,
            )
        else:
            self.selection_grn = None

    def _stack_inputs(
        self, inputs: list[Tensor] | dict[str, Tensor]
    ) -> Tensor:
        """Normalise the ``inputs`` argument to a list and stack them.

        Returns a tensor of shape ``(..., n_variables, input_dim)``.
        """
        if isinstance(inputs, dict):
            # Stable key order: sort so forward is deterministic.
            keys = sorted(inputs.keys())
            tensors = [inputs[k] for k in keys]
        else:
            tensors = list(inputs)

        if len(tensors) != self.n_variables:
            raise ValueError(
                f"VSN expected {self.n_variables} input tensors; "
                f"got {len(tensors)}."
            )

        # Validate shape consistency.
        reference_shape = tensors[0].shape
        for i, t in enumerate(tensors):
            if t.shape != reference_shape:
                raise ValueError(
                    f"VSN input {i} shape {tuple(t.shape)} != reference "
                    f"shape {tuple(reference_shape)}."
                )
            if t.shape[-1] != self.input_dim:
                raise ValueError(
                    f"VSN input {i} last-dim {t.shape[-1]} != input_dim "
                    f"{self.input_dim}."
                )

        # Stack along a new second-to-last axis for shape
        # (..., n_variables, input_dim).
        return torch.stack(tensors, dim=-2)

    def forward(
        self,
        inputs: list[Tensor] | dict[str, Tensor],
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply variable selection.

        Parameters
        ----------
        inputs
            Either a list of ``n_variables`` tensors (each
            ``(..., input_dim)``) or a dict keyed by variable name.
            All tensors must share the same leading shape.
        context
            Optional static context tensor broadcastable onto the
            selection GRN's input. Must be ``None`` when the module
            was constructed with ``context_dim=None``.

        Returns
        -------
        selected
            Tensor of shape ``(..., input_dim)`` — the weighted sum
            of per-variable GRN outputs.
        weights
            Tensor of shape ``(..., n_variables)`` — softmax weights
            over the variable axis (sum to 1 along the last axis).
            When ``n_variables == 1`` returns a placeholder
            ``(..., 1)`` tensor of ones.
        """
        stacked = self._stack_inputs(inputs)
        # stacked: (..., n_variables, input_dim).

        # Per-variable GRN outputs.
        per_var_outputs = []
        for j in range(self.n_variables):
            # Select variable j along the -2 axis.
            # Shape: (..., input_dim).
            xi_j = stacked[..., j, :]
            per_var_outputs.append(self.per_var_grns[j](xi_j))
        # Stack: (..., n_variables, input_dim).
        tilde = torch.stack(per_var_outputs, dim=-2)

        if self.n_variables == 1 or self.selection_grn is None:
            # Trivial selection — single variable, weight is 1 by
            # construction. Return the single GRN output.
            leading_shape = stacked.shape[:-2]
            weights = torch.ones(
                (*leading_shape, 1),
                dtype=stacked.dtype,
                device=stacked.device,
            )
            selected = tilde[..., 0, :]
            return selected, weights

        # Flatten the variable+feature dims for the selection GRN:
        # (..., n_variables * input_dim).
        flat = stacked.reshape(
            *stacked.shape[:-2], self.n_variables * self.input_dim
        )

        # Selection logits -> softmax across variables.
        logits = self.selection_grn(flat, context)  # (..., n_variables)
        weights = torch.softmax(logits, dim=-1)

        # Broadcast multiply and sum across variable axis.
        # weights: (..., n_variables) -> (..., n_variables, 1).
        # tilde:   (..., n_variables, input_dim).
        selected = (weights.unsqueeze(-1) * tilde).sum(dim=-2)

        return selected, weights


__all__ = ["VariableSelectionNetwork"]
