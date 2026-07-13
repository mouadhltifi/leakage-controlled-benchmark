"""Graph attention encoder for inter-stock relations.

Wraps a 2-layer (or deeper) Graph Attention Network from
``torch_geometric``. Inputs are flat node features ``(N_batch, F_node)``
and a single batched ``edge_index`` of shape ``(2, E_batch)`` where
multiple graphs are stacked PyG-style (collate offsets node indices by
``sample_pos * N_stocks``).

Design choices
--------------

* **LayerNorm between conv layers.** v1 shipped without LN between GAT
  conv layers and had an asymmetric training behaviour the audit flagged
  (ID-01, mid-experiment fix). v2 applies ``nn.LayerNorm(H)`` after each
  GAT conv so both FF and LSTM downstream paths see a stable scale.
* **Configurable depth and heads.** ``cfg.model.graph_gat_layers`` and
  ``cfg.model.graph_gat_heads`` control architecture. Default is 2
  layers with 4 heads (v1 parity).
* **Static + dynamic averaging.** The static-plus-dynamic source uses
  two independent encoders (one per adjacency) whose outputs are
  averaged before the final LayerNorm. Averaging preserves the
  ``(N, H)`` shape without a concat+project step, so fusion inputs stay
  at width ``H`` regardless of the graph source (spec Q12).

Per-layer hidden dimension
--------------------------

A GAT conv with ``heads=k`` concatenates per-head outputs, so the
effective output width is ``heads * out_channels``. To keep the
encoder's overall width at ``H`` we use:

* Intermediate conv layers: ``out_channels = H // heads``, ``heads=k``,
  ``concat=True``. Concatenation gives ``H`` features per node.
* Final conv layer: ``heads=1``, ``concat=False``, ``out_channels=H``.
  Matches v1 final-layer convention and avoids a dangling projection.

This means ``H`` must be divisible by ``heads`` for intermediate layers;
the constructor checks.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv

from mmfp.config.schema import ExperimentConfig

from .base import EncoderBase


class _SingleGraphGATStack(nn.Module):
    """A single GAT stack operating on one edge-index source.

    Separated from :class:`GraphGATEncoder` so the
    ``static_plus_dynamic`` case can instantiate two independent stacks
    without re-implementing the layer construction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(
                f"GraphGATEncoder requires num_layers >= 1; got {num_layers}"
            )
        if heads < 1:
            raise ValueError(
                f"GraphGATEncoder requires heads >= 1; got {heads}"
            )
        if num_layers >= 2 and hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads "
                f"({heads}) for multi-layer GAT stacks."
            )

        self.num_layers = num_layers

        convs: list[GATConv] = []
        norms: list[nn.LayerNorm] = []

        if num_layers == 1:
            # Single layer: direct input_dim -> hidden_dim with 1 head.
            convs.append(
                GATConv(
                    input_dim,
                    hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
            )
            norms.append(nn.LayerNorm(hidden_dim))
        else:
            # First layer: input_dim -> (hidden_dim // heads) * heads == hidden_dim.
            per_head = hidden_dim // heads
            convs.append(
                GATConv(
                    input_dim,
                    per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            norms.append(nn.LayerNorm(hidden_dim))

            # Middle layers: hidden_dim -> hidden_dim with heads concatenated.
            for _ in range(num_layers - 2):
                convs.append(
                    GATConv(
                        hidden_dim,
                        per_head,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                    )
                )
                norms.append(nn.LayerNorm(hidden_dim))

            # Final layer: hidden_dim -> hidden_dim with one averaged head.
            convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
            )
            norms.append(nn.LayerNorm(hidden_dim))

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Run the stack and return post-final-LN node representations."""
        h = x
        last = len(self.convs) - 1
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h, edge_index)
            h = norm(h)
            if i != last:
                # ELU matches v1 intermediate activation; no activation on
                # the final conv so the caller can shape-mix freely.
                h = nn.functional.elu(h)
        return h


class GraphGATEncoder(EncoderBase):
    """Graph attention encoder with an optional static+dynamic averaging head.

    Parameters
    ----------
    cfg
        Full experiment config. ``cfg.model.hidden_dim``,
        ``cfg.model.dropout``, ``cfg.model.graph_gat_layers``,
        ``cfg.model.graph_gat_heads`` and ``cfg.graph.source`` are
        consulted.
    input_dim
        Number of node-feature columns (``F_node``). For the study this
        is the price feature dimension since node features are
        per-stock-per-day scaled price vectors.

    Forward signature
    -----------------

    ``forward`` accepts either the standard ``(x, edge_index)`` pair for
    single-source graphs or ``(x, edge_index_static, edge_index_dynamic)``
    for the static+dynamic combination. The caller (usually the
    :class:`~mmfp.models.predictor.Predictor`) routes the correct keys
    from the collated batch.

    Returns ``(N, H)`` where ``N == len(x)`` (flat node layout —
    :class:`~mmfp.datasets.collate.make_collate_fn` already batches nodes
    PyG-style by offsetting ``edge_index``).
    """

    def __init__(self, cfg: ExperimentConfig, input_dim: int) -> None:
        super().__init__(output_dim=cfg.model.hidden_dim)

        if input_dim < 1:
            raise ValueError(
                f"GraphGATEncoder requires input_dim >= 1; got {input_dim}"
            )

        hidden_dim = self.output_dim
        num_layers = cfg.model.graph_gat_layers
        heads = cfg.model.graph_gat_heads
        dropout = cfg.model.dropout

        self._graph_source = cfg.graph.source

        self.stack_primary = _SingleGraphGATStack(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

        # Two-stack mode: one extra stack for the second adjacency source.
        if self._graph_source == "static_plus_dynamic":
            self.stack_secondary: nn.Module = _SingleGraphGATStack(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
            )
        else:
            # Register ``None`` rather than deleting the attribute so
            # downstream introspection (e.g. ``hasattr``) is stable.
            self.stack_secondary = nn.Identity()

        # Shared final LN after averaging (or single-stack pass-through).
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        *,
        edge_index_static: Tensor | None = None,
        edge_index_dynamic: Tensor | None = None,
    ) -> Tensor:
        """Encode node features.

        Parameters
        ----------
        x
            ``(N, F_node)`` node features. For batched inputs, ``N ==
            B * N_stocks`` and nodes are PyG-stacked.
        edge_index
            ``(2, E)`` — required for single-source modes
            (``static_gics``, ``dynamic_corr``).
        edge_index_static
            ``(2, E_s)`` — required for ``static_plus_dynamic``.
        edge_index_dynamic
            ``(2, E_d)`` — required for ``static_plus_dynamic``.

        Returns
        -------
        Tensor
            ``(N, H)`` node-level representations, layer-normalised.
        """
        if x.dim() != 2:
            raise ValueError(
                f"GraphGATEncoder expects (N, F_node); got shape {tuple(x.shape)}"
            )

        if self._graph_source == "static_plus_dynamic":
            if edge_index_static is None or edge_index_dynamic is None:
                raise ValueError(
                    "GraphGATEncoder: static_plus_dynamic mode requires "
                    "both edge_index_static and edge_index_dynamic."
                )
            h_static = self.stack_primary(x, edge_index_static)
            h_dynamic = self.stack_secondary(x, edge_index_dynamic)
            h = 0.5 * (h_static + h_dynamic)
        else:
            if edge_index is None:
                raise ValueError(
                    f"GraphGATEncoder: edge_index is required for source "
                    f"{self._graph_source!r}."
                )
            h = self.stack_primary(x, edge_index)

        return self.final_norm(h)


__all__ = ["GraphGATEncoder"]
