"""Full multimodal predictor: encoders + fusion + head.

Assembles the encoder stack, the fusion strategy and the prediction
head from a validated :class:`~mmfp.config.schema.ExperimentConfig` and
the fold's :class:`~mmfp.data.assemble.FeatureSchema` (which carries
per-modality input dims).

Assembly rules (spec Section 3.10)
----------------------------------

* Price: :class:`PriceLSTMEncoder` when ``cfg.model.price_encoder ==
  "lstm"`` AND ``cfg.data.lookback > 1``, else
  :class:`PriceFFEncoder`.
* Macro, news, social: :class:`TabularFFEncoder`.
* Graph: :class:`GraphGATEncoder`. The encoder internally handles
  single vs static+dynamic sources.
* Fusion: built via :func:`~mmfp.models.fusion.build_fusion`.
* Head: built via :func:`~mmfp.models.heads.build_head`.

Forward pass (batch dict -> predictions dict)
---------------------------------------------

The caller supplies a batch dict in the exact shape that
:func:`mmfp.datasets.collate.make_collate_fn` produces:

* ``"price"`` — either ``(B, F_price)`` or ``(B, L, F_price)``.
* ``"macro"``, ``"news"``, ``"social"`` — ``(B, F_mod)`` per active
  modality.
* ``"graph_features"`` — ``(B * N, F_price)`` (collated PyG-style).
* ``"edge_index"`` or ``"edge_index_static"`` + ``"edge_index_dynamic"``.
* ``"graph_stock_idx"`` — ``(B,)`` long, in ``[0, B*N)``.
* ``"_batch_size"`` — ``(,)`` scalar tensor with ``B`` (bookkeeping
  from the collate fn).

The predictor returns a dict ``{target: tensor}`` for each active
target in ``cfg.head.targets``.

Encoder insertion order
-----------------------

Fusion expects a stable key order so :class:`ConcatFusion`'s
concatenation is deterministic run-to-run. We use the canonical
``("price", "macro", "news", "social", "graph")`` tuple; dict
insertion order (Python 3.7+) preserves this at iteration time.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mmfp.config.schema import ExperimentConfig
from mmfp.data.assemble import FeatureSchema
from mmfp.data.universe import N_STOCKS

from .encoders import (
    GraphGATEncoder,
    PriceFFEncoder,
    PriceLSTMEncoder,
    TabularFFEncoder,
)
from .fusion import build_fusion
from .heads import build_head

#: Canonical encoder iteration order. Dictates concat layout and
#: modality token order in attention fusion.
_MODALITY_ORDER: tuple[str, ...] = ("price", "macro", "news", "social", "graph")

#: Non-graph tabular modalities that resolve to ``TabularFFEncoder``.
_TABULAR_MODALITIES: tuple[str, ...] = ("macro", "news", "social")


class Predictor(nn.Module):
    """Compose encoders, fusion and head into a single forward-passable module.

    Parameters
    ----------
    cfg
        Validated :class:`ExperimentConfig`. The cross-field validator
        must have run so target/fusion/graph combinations are already
        verified.
    feature_schema
        Output of assembly — provides per-modality input dimensions and
        the price feature width used as the graph node-feature dim.

    Attributes
    ----------
    cfg
        The config the predictor was built against (stored for
        downstream logging / reproducibility).
    encoders
        ``nn.ModuleDict`` keyed by modality name. Insertion order
        matches :data:`_MODALITY_ORDER` filtered by active modalities.
    fusion
        The concrete fusion module.
    head
        The concrete prediction head.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        feature_schema: FeatureSchema,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self._feature_schema = feature_schema

        # ---- Build encoders in canonical order. ----------------------
        self.encoders = nn.ModuleDict()

        # Price (always present).
        price_dim = self._slice_width(feature_schema, "price")
        if price_dim < 1:
            raise ValueError(
                "FeatureSchema has zero-width price slice; price is always on."
            )
        if cfg.model.price_encoder == "lstm" and cfg.data.lookback > 1:
            self.encoders["price"] = PriceLSTMEncoder(cfg, input_dim=price_dim)
        else:
            self.encoders["price"] = PriceFFEncoder(cfg, input_dim=price_dim)

        # Tabular modalities.
        if cfg.macro.enabled:
            dim = self._slice_width(feature_schema, "macro")
            self.encoders["macro"] = TabularFFEncoder(cfg, input_dim=dim)
        if cfg.news.enabled:
            dim = self._slice_width(feature_schema, "news")
            self.encoders["news"] = TabularFFEncoder(cfg, input_dim=dim)
        if cfg.social.enabled:
            dim = self._slice_width(feature_schema, "social")
            self.encoders["social"] = TabularFFEncoder(cfg, input_dim=dim)

        # Graph modality — node features use the price width.
        self._graph_enabled = cfg.graph.enabled
        if self._graph_enabled:
            self.encoders["graph"] = GraphGATEncoder(cfg, input_dim=price_dim)

        # ---- Fusion and head. ----------------------------------------
        self.fusion = build_fusion(cfg, n_modalities=len(self.encoders))
        self.head = build_head(cfg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_width(schema: FeatureSchema, modality: str) -> int:
        """Return the ``schema[modality]`` slice width (``stop - start``)."""
        sl = schema.range_for(modality)
        return sl.stop - sl.start

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run encoders, fusion and head on a collated batch.

        Parameters
        ----------
        batch
            Collated batch dict produced by
            :func:`~mmfp.datasets.collate.make_collate_fn`.

        Returns
        -------
        dict[str, Tensor]
            Mapping from target name to prediction tensor. Keys match
            ``cfg.head.targets`` (possibly with additional cascade
            side-products in the sequential case).
        """
        encodings: dict[str, Tensor] = {}
        batch_size = self._resolve_batch_size(batch)

        for modality in _MODALITY_ORDER:
            if modality not in self.encoders:
                continue

            if modality == "price":
                encodings["price"] = self.encoders["price"](batch["price"])
            elif modality in _TABULAR_MODALITIES:
                encodings[modality] = self.encoders[modality](batch[modality])
            elif modality == "graph":
                encodings["graph"] = self._forward_graph(batch, batch_size)
            else:  # pragma: no cover - unreachable with current _MODALITY_ORDER
                raise RuntimeError(f"Unknown modality in encoder dict: {modality!r}")

        fused = self.fusion(encodings)
        return self.head(fused)

    def _resolve_batch_size(self, batch: dict[str, Tensor]) -> int:
        """Return the batch size in a device-agnostic way.

        The collate function emits ``_batch_size`` when graph mode is
        active. Otherwise we fall back to ``price.shape[0]``. This keeps
        the predictor usable for non-graph tests that omit the collate
        bookkeeping tensor.
        """
        if "_batch_size" in batch:
            return int(batch["_batch_size"].item())
        return batch["price"].shape[0]

    def _forward_graph(self, batch: dict[str, Tensor], batch_size: int) -> Tensor:
        """Run the GAT encoder and extract the target-stock node embedding.

        Parameters
        ----------
        batch
            Collated batch containing ``graph_features`` ``(B*N, F)``,
            one or two edge-index tensors, and ``graph_stock_idx``
            ``(B,)``.
        batch_size
            ``B`` — resolved by the caller.

        Returns
        -------
        Tensor
            ``(B, H)`` — per-sample graph representation obtained by
            selecting the target stock's node embedding from the GAT
            output.
        """
        if "graph_features" not in batch:
            raise KeyError(
                "Predictor: graph enabled but batch missing 'graph_features'."
            )
        if "graph_stock_idx" not in batch:
            raise KeyError(
                "Predictor: graph enabled but batch missing 'graph_stock_idx'."
            )

        graph_feats = batch["graph_features"]  # (B*N, F_price)
        stock_idx = batch["graph_stock_idx"]  # (B,) in [0, B*N)
        encoder = self.encoders["graph"]
        source = self.cfg.graph.source

        if source == "static_plus_dynamic":
            if "edge_index_static" not in batch or "edge_index_dynamic" not in batch:
                raise KeyError(
                    "Predictor: static_plus_dynamic requires both "
                    "'edge_index_static' and 'edge_index_dynamic' in the batch."
                )
            node_out = encoder(
                graph_feats,
                edge_index_static=batch["edge_index_static"],
                edge_index_dynamic=batch["edge_index_dynamic"],
            )
        else:
            if "edge_index" not in batch:
                raise KeyError(
                    "Predictor: graph source requires 'edge_index' in the batch."
                )
            node_out = encoder(graph_feats, batch["edge_index"])

        # Select the per-sample target node embedding: (B, H).
        # ``stock_idx`` is already a global index into the mega-graph.
        expected_n_nodes = batch_size * N_STOCKS
        if node_out.shape[0] != expected_n_nodes:
            raise ValueError(
                f"Predictor: expected {expected_n_nodes} nodes in graph "
                f"output ({batch_size} batches x {N_STOCKS} stocks); got "
                f"{node_out.shape[0]}."
            )

        return node_out[stock_idx]


__all__ = ["Predictor"]
