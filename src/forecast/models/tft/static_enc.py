"""Static Covariate Encoders (Lim et al. 2021, §4.2 end).

Takes static categorical features (ticker id, sector id), embeds them,
concatenates, and produces four separate context vectors via four
parallel GRNs:

* ``c_s`` — static selection context, fed to the past-VSN.
* ``c_c`` — LSTM cell-state initialisation.
* ``c_h`` — LSTM hidden-state initialisation.
* ``c_e`` — enrichment context, added to the LSTM output per-timestep.

Each context is produced by an independent GRN so the four downstream
consumers can learn complementary projections of the same static
information.

Per the architecture spec defaults:

* ticker (55 categories) → 16-dim embedding
* sector (11 categories) → 8-dim embedding
* concatenated flat vector is 24-dim.

The GRNs project 24-dim → hidden_dim separately for each context.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from forecast.models.tft.grn import GRN


def _embed_dim_for_cardinality(cardinality: int) -> int:
    """Heuristic embedding dimensionality for a categorical variable.

    Follows the spec's hard-coded defaults (55 → 16, 11 → 8) and
    falls back to a log2-style rule for other cardinalities. The
    explicit defaults keep the total static-vector width stable
    regardless of the exact cardinality count.
    """
    if cardinality == 55:
        return 16
    if cardinality == 11:
        return 8
    # Fallback for unexpected cardinalities: clamp to [4, 32].
    return max(4, min(32, int(round(cardinality**0.5)) * 2))


class StaticCovariateEncoders(nn.Module):
    """Produce four static contexts from static categorical inputs.

    Parameters
    ----------
    static_cardinalities
        List of per-category cardinalities, e.g. ``[55, 11]`` for
        (ticker, sector). The order here determines the order in
        which the forward pass consumes its integer input tensor.
    hidden_dim
        Output dimension of each context vector (TFT's ``H``).
    dropout
        Dropout probability inside the GRNs.
    """

    #: Order in which the four static contexts are returned.
    CONTEXT_KEYS: tuple[str, ...] = ("c_s", "c_c", "c_h", "c_e")

    def __init__(
        self,
        static_cardinalities: list[int],
        hidden_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not static_cardinalities:
            raise ValueError("static_cardinalities must be non-empty.")
        if any(c <= 0 for c in static_cardinalities):
            raise ValueError(
                f"All cardinalities must be >= 1; got {static_cardinalities}"
            )

        self.static_cardinalities = list(static_cardinalities)
        self.hidden_dim = hidden_dim

        # Embedding per static categorical.
        embed_dims = [_embed_dim_for_cardinality(c) for c in static_cardinalities]
        self.embed_dims = embed_dims
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=c, embedding_dim=d)
                for c, d in zip(static_cardinalities, embed_dims)
            ]
        )
        self.concat_dim = sum(embed_dims)

        # Four parallel GRNs: one per context.
        self.context_grns = nn.ModuleDict(
            {
                key: GRN(
                    input_dim=self.concat_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    context_dim=None,
                    dropout=dropout,
                )
                for key in self.CONTEXT_KEYS
            }
        )

    def forward(self, static_cat: Tensor) -> dict[str, Tensor]:
        """Produce the four static contexts.

        Parameters
        ----------
        static_cat
            Integer tensor of shape ``(B, n_static)`` where column ``i``
            contains values in ``[0, static_cardinalities[i])``.

        Returns
        -------
        dict
            Keys ``'c_s'``, ``'c_c'``, ``'c_h'``, ``'c_e'``; each value
            is a float tensor of shape ``(B, hidden_dim)``.
        """
        if static_cat.dim() != 2:
            raise ValueError(
                f"static_cat must have shape (B, n_static); got "
                f"{tuple(static_cat.shape)}"
            )
        if static_cat.shape[1] != len(self.static_cardinalities):
            raise ValueError(
                f"static_cat has {static_cat.shape[1]} columns; expected "
                f"{len(self.static_cardinalities)}"
            )

        # Embed each column and concatenate.
        embedded = []
        for i, emb in enumerate(self.embeddings):
            embedded.append(emb(static_cat[:, i]))
        flat = torch.cat(embedded, dim=-1)  # (B, concat_dim)

        return {key: self.context_grns[key](flat) for key in self.CONTEXT_KEYS}


__all__ = ["StaticCovariateEncoders"]
