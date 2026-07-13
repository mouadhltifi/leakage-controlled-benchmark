"""Per-modality light projections (the architecture spec).

Applies a ``Linear(F_mod, H) + LayerNorm(H)`` per active modality so
that all streams enter the TFT body at the same hidden width. No ReLU
— the downstream VSN's GRN provides the non-linearity (adding a ReLU
here would be redundant and would slightly distort the VSN's signal).

Parameter budget: ``5 modalities x 64 x avg(F_mod) ~= 15 k``.
"""

from __future__ import annotations

from torch import Tensor, nn


class ModalityProjections(nn.Module):
    """Per-modality ``Linear + LayerNorm`` projection to hidden_dim.

    Parameters
    ----------
    feature_dims
        Mapping from modality name to its per-timestep feature
        dimension ``F_mod``. Must be non-empty. Example:
        ``{"price": 13, "news": 129, "macro": 9, "social": 7, "graph": 64}``.
    hidden_dim
        Target output dim ``H`` (default 64).

    Notes
    -----
    The module stores the set of active modalities as
    ``self.active_modalities`` in sorted order, so iteration is
    deterministic for downstream consumers (VSN input order, etc).
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError("feature_dims must be non-empty.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}")
        for name, dim in feature_dims.items():
            if dim <= 0:
                raise ValueError(
                    f"feature_dims[{name!r}] must be positive; got {dim}"
                )

        self.hidden_dim = hidden_dim
        self.active_modalities: tuple[str, ...] = tuple(sorted(feature_dims))

        # Parallel ModuleDict keyed by modality name.
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, hidden_dim) for name, dim in feature_dims.items()}
        )
        self.layer_norms = nn.ModuleDict(
            {name: nn.LayerNorm(hidden_dim) for name in feature_dims}
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Project each modality tensor to ``(B, L, H)``.

        Parameters
        ----------
        batch
            Dict from modality name to tensor of shape ``(B, L, F_mod)``.
            Every key must be in :attr:`active_modalities`; missing
            keys raise ``KeyError`` to catch plumbing bugs early.

        Returns
        -------
        dict
            Dict from modality name to tensor of shape ``(B, L, H)``.
            Keys match ``batch.keys()``; iteration order matches sorted
            :attr:`active_modalities`.
        """
        out: dict[str, Tensor] = {}
        # Iterate sorted so output order is stable regardless of caller.
        for name in self.active_modalities:
            if name not in batch:
                raise KeyError(
                    f"ModalityProjections is configured for modality "
                    f"{name!r} but it is missing from the input batch."
                )
            x = batch[name]
            if x.dim() != 3:
                raise ValueError(
                    f"{name!r} tensor must be (B, L, F_mod); got "
                    f"{tuple(x.shape)}"
                )
            projected = self.projections[name](x)
            out[name] = self.layer_norms[name](projected)
        # Caller bugs (extra keys passed in but not configured) are
        # tolerated silently — we only projected what we own. The
        # per-modality KeyError above catches the opposite direction.
        return out


__all__ = ["ModalityProjections"]
