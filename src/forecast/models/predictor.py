"""Config-driven assembly of the v3 forecast stack (the architecture spec).

:class:`ForecastPredictor` is the public entry point that wires:

* :class:`~forecast.models.projections.ModalityProjections` — per-modality
  ``Linear + LayerNorm`` from ``F_mod`` to ``H``.
* :class:`~forecast.models.tft.body.TFTBody` — the Temporal Fusion
  Transformer body (VSN, LSTM local processor, interpretable MHA, gated
  skips).
* :class:`~forecast.models.heads.QuantileHead` — ``Linear(H, n_quantiles)``
  output projection.

Interface mirrors v2's :class:`mmfp.models.predictor.Predictor` so the
reused :class:`mmfp.training.trainer.Trainer` drops in once M4 adapts
the loss signature.

Active-modality resolution
--------------------------

Price is always active. ``news`` / ``macro`` / ``social`` / ``graph`` are
included iff the corresponding ``cfg.<modality>.enabled`` flag is true.
Modality order is fixed by the alphabetically-sorted iteration that
:class:`ModalityProjections` enforces internally; the predictor passes
that same sorted dict through the TFT body.

Feature dimensions
------------------

Flat modalities (``price``, ``macro``, ``news``, ``social``) read their
per-timestep feature dimension from the fold's
:class:`~mmfp.data.assemble.FeatureSchema`. ``graph`` is the exception:
its per-timestep width is the pre-computed GAT embedding width
(``graph_node_dim``, default 64), because graph features are not part of
the flat feature matrix — they are cached separately per fold (spec
§2.1).

Quantile-index caching
----------------------

The median index (used by direction derivation) and the lo/hi band
indices (used by volatility derivation) are resolved once at
construction time. The schema validator guarantees the median ``0.5``
is present; the lo/hi defaults are the outermost available quantiles,
which for the default ``(0.1, 0.25, 0.5, 0.75, 0.9)`` tuple gives the
80%-coverage pair.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from forecast.config.schema import V3ExperimentConfig
from forecast.models.heads import QuantileHead, derive_direction, derive_volatility
from forecast.models.projections import ModalityProjections
from forecast.models.tft.body import TFTBody
from mmfp.data.assemble import FeatureSchema

#: Canonical active-modality order used to build the projections dict.
#: ``ModalityProjections`` alphabetises internally, so this constant is
#: only load-bearing for the human-readable construction contract.
_MODALITY_ORDER: tuple[str, ...] = ("price", "news", "macro", "social", "graph")

#: Mapping from modality → per-sample batch key in the input dict.
#: Mirrors the key names emitted by the v3 windowed dataset (M4).
_MODALITY_BATCH_KEY: dict[str, str] = {
    "price": "price_seq",
    "news": "news_seq",
    "macro": "macro_seq",
    "social": "social_seq",
    "graph": "graph_seq",
}


class ForecastPredictor(nn.Module):
    """Assembles :class:`ModalityProjections` → :class:`TFTBody` →
    :class:`QuantileHead` from a :class:`V3ExperimentConfig`.

    The forward pass consumes a batch dict (per-modality sequences +
    static categorical indices) and returns a dict with the predicted
    quantile vector, derived direction logits, derived Gaussian sigma,
    and the two interpretability signals (VSN weights, attention
    weights).

    Parameters
    ----------
    cfg
        Validated :class:`V3ExperimentConfig`. The predictor reads
        :attr:`~V3ExperimentConfig.forecast` for architectural knobs
        (hidden dim, quantile set, static cardinalities, graph node
        dim) and the modality ``enabled`` flags
        (``cfg.news.enabled``, …) to determine which projections to
        instantiate. Note: ``cfg.price.enabled`` is unconditionally
        honoured — price is always present by v3 contract.
    feature_schema
        Fold-level :class:`FeatureSchema` produced by
        :func:`mmfp.data.assemble.assemble_fold`. Provides the
        per-timestep feature dimension for each flat modality. The
        schema must carry a non-None slice for every modality that
        ``cfg.<modality>.enabled`` requests, or :class:`ValueError` is
        raised at construction.
    graph_node_dim
        Pre-computed GAT node-embedding width. Defaults to
        ``cfg.forecast.graph_node_dim`` (64) when left at its default.

    Attributes
    ----------
    cfg
        The config the predictor was built against.
    active_modalities
        Tuple of modality names the predictor is wired for, in sorted
        order (matches :attr:`projections.active_modalities`).
    projections
        :class:`ModalityProjections` handling the ``F_m → H`` linears.
    body
        :class:`TFTBody` wrapping VSN + LSTM + attention + gated skips.
    head
        :class:`QuantileHead` final projection to the quantile vector.
    median_idx, lo_idx, hi_idx
        Cached quantile indices used by direction / volatility
        derivations.
    volatility_coverage
        Implied coverage of the ``[lo_idx, hi_idx]`` band; fed to
        :func:`derive_volatility`.
    """

    def __init__(
        self,
        cfg: V3ExperimentConfig,
        feature_schema: FeatureSchema,
        graph_node_dim: int = 64,
    ) -> None:
        super().__init__()

        if graph_node_dim <= 0:
            raise ValueError(
                f"graph_node_dim must be positive; got {graph_node_dim}"
            )

        self.cfg = cfg
        self._feature_schema = feature_schema
        self.graph_node_dim = graph_node_dim

        hidden_dim = cfg.forecast.hidden_dim

        # ---- Resolve active modalities + per-modality F_mod. -----------
        feature_dims: dict[str, int] = {}

        # Price is always active.
        feature_dims["price"] = self._flat_width(feature_schema, "price")

        if cfg.news.enabled:
            feature_dims["news"] = self._flat_width(feature_schema, "news")
        if cfg.macro.enabled:
            feature_dims["macro"] = self._flat_width(feature_schema, "macro")
        if cfg.social.enabled:
            feature_dims["social"] = self._flat_width(feature_schema, "social")
        if cfg.graph.enabled:
            # Graph node embeddings are pre-computed (the architecture spec);
            # they are NOT part of the flat feature schema.
            feature_dims["graph"] = graph_node_dim

        # Store in sorted order for deterministic iteration (matches
        # ModalityProjections' internal behaviour).
        self.active_modalities: tuple[str, ...] = tuple(sorted(feature_dims))

        # ---- Build projections. --------------------------------------
        self.projections = ModalityProjections(
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
        )

        # ---- Build TFT body. -----------------------------------------
        self.body = TFTBody(
            cfg=cfg.forecast,
            n_past_modalities=len(self.active_modalities),
            static_cardinalities=[cfg.forecast.n_tickers, cfg.forecast.n_sectors],
        )

        # ---- Build quantile head. ------------------------------------
        self.head = QuantileHead(
            hidden_dim=hidden_dim,
            quantiles=cfg.forecast.quantiles,
        )

        # ---- Resolve quantile indices (schema guarantees 0.5 exists). --
        quantiles = tuple(float(q) for q in cfg.forecast.quantiles)
        if 0.5 not in quantiles:  # belt-and-braces against a pathological cfg
            raise ValueError(
                f"quantiles must include the median 0.5; got {quantiles}"
            )
        self.median_idx: int = quantiles.index(0.5)

        # Lo/hi pair for volatility: prefer (0.1, 0.9) if present, else
        # outermost available.
        self.lo_idx: int
        self.hi_idx: int
        if 0.1 in quantiles and 0.9 in quantiles:
            self.lo_idx = quantiles.index(0.1)
            self.hi_idx = quantiles.index(0.9)
        else:
            self.lo_idx = 0
            self.hi_idx = len(quantiles) - 1
        implied = float(quantiles[self.hi_idx] - quantiles[self.lo_idx])
        # Clamp to the open interval (0, 1) so the inverse-Normal quantile
        # used in derive_volatility stays finite.
        self.volatility_coverage: float = max(0.01, min(0.99, implied))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flat_width(schema: FeatureSchema, modality: str) -> int:
        """Return the ``schema[modality]`` slice width.

        Raises :class:`KeyError` via :meth:`FeatureSchema.range_for` if
        the modality is absent from the schema (caller bug: config
        enables modality but fold didn't materialise it).
        """
        sl = schema.range_for(modality)
        width = sl.stop - sl.start
        if width <= 0:
            raise ValueError(
                f"FeatureSchema has non-positive width for modality "
                f"{modality!r}: slice={sl!r}"
            )
        return width

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters across the whole stack."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the full v3 forward pass.

        Parameters
        ----------
        batch
            Dict with per-modality sequence tensors and the static
            categorical tensor. Expected keys:

            * ``'price_seq'`` — ``(B, L, F_price)`` float32.
            * ``'news_seq'`` — ``(B, L, F_news)``, only when
              ``cfg.news.enabled``.
            * ``'macro_seq'`` — ``(B, L, F_macro)``, only when
              ``cfg.macro.enabled``.
            * ``'social_seq'`` — ``(B, L, F_social)``, only when
              ``cfg.social.enabled``.
            * ``'graph_seq'`` — ``(B, L, graph_node_dim)``, only when
              ``cfg.graph.enabled``.
            * ``'static_categorical'`` — ``(B, 2)`` long tensor
              holding ``[ticker_id, sector_id]``.

            Extra keys are silently ignored so the v2 collate dict
            (which carries ``_batch_size``, per-fold graph edges, …)
            can be passed through unchanged once M4 bridges the
            datasets.

        Returns
        -------
        dict[str, Tensor]
            Keys:

            * ``'return'`` — ``(B, n_quantiles)`` predicted quantile
              vector.
            * ``'direction'`` — ``(B, 2)`` derived-direction logits.
            * ``'volatility'`` — ``(B,)`` Gaussian-implied sigma
              (strictly positive when quantiles are sorted).
            * ``'vsn_weights'`` — ``(B, L, n_past_modalities)``
              per-timestep modality importance (sums to 1 across the
              modality axis).
            * ``'attention_weights'`` — ``(B, L, L)`` causal attention
              map.
        """
        if "static_categorical" not in batch:
            raise KeyError(
                "ForecastPredictor: batch missing 'static_categorical' "
                "(expected (B, 2) long tensor of [ticker_id, sector_id])."
            )

        # Gather per-modality sequences in sorted order.
        modality_batch: dict[str, Tensor] = {}
        for modality in self.active_modalities:
            key = _MODALITY_BATCH_KEY[modality]
            if key not in batch:
                raise KeyError(
                    f"ForecastPredictor: modality {modality!r} is active "
                    f"but batch is missing key {key!r}."
                )
            modality_batch[modality] = batch[key]

        # Project each modality to hidden_dim.
        projected = self.projections(modality_batch)  # {m: (B, L, H)}

        # TFT body.
        static_cat = batch["static_categorical"]
        body_out = self.body(projected, static_cat)  # dict
        last_hidden = body_out["last_hidden"]  # (B, H)
        vsn_weights = body_out["vsn_weights_past"]  # (B, L, n_mod)
        attention_weights = body_out["attention_weights"]  # (B, L, L)

        # Quantile projection.
        q = self.head(last_hidden)  # (B, n_quantiles)

        # Derivations.
        direction_logits = derive_direction(q, self.median_idx)  # (B, 2)
        sigma = derive_volatility(
            q,
            lo_idx=self.lo_idx,
            hi_idx=self.hi_idx,
            coverage=self.volatility_coverage,
        )  # (B,)

        return {
            "return": q,
            "direction": direction_logits,
            "volatility": sigma,
            "vsn_weights": vsn_weights,
            "attention_weights": attention_weights,
        }


__all__ = ["ForecastPredictor"]
