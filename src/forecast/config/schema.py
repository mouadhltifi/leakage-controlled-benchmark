"""Pydantic v2 schema for v3's ``ForecastConfig`` and ``V3ExperimentConfig``.

v3 reuses v2's modality sub-configs (``DataConfig``, ``NewsConfig``, ...)
verbatim from :mod:`mmfp.config.schema`. The only net-new schema is

* :class:`ForecastConfig` — TFT-specific architectural and loss knobs
  (the architecture spec).
* :class:`V3ExperimentConfig` — the root config, composed of v2's
  ``ExperimentConfig`` plus a ``forecast`` field.

Cross-field validation (architecture × lookback, news-encoder lock,
graph-refresh divisibility, etc.) lives in :mod:`forecast.config.validate`.

v2's ``model``, ``fusion`` and ``head`` fields remain on
``V3ExperimentConfig`` for the reused trainer / runner dispatch, but are
**unused when ``forecast.architecture == "tft"``** (the TFT body subsumes
fusion via VSN + attention, and the quantile head replaces v2's MLP head).
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

# v2 sub-configs imported verbatim — v3 adds no fields to any of these.
from mmfp.config.schema import (
    DataConfig,
    ExperimentConfig,
    FusionConfig,
    GraphConfig,
    HeadConfig,
    IntradayConfig,
    LoggingConfig,
    MacroConfig,
    ModelConfig,
    NewsConfig,
    PriceConfig,
    SocialConfig,
    StandardizationConfig,
    TrainingConfig,
    _StrictModel,
)

# ---------------------------------------------------------------------------
# New: ForecastConfig (the architecture spec).
# ---------------------------------------------------------------------------


class ForecastConfig(_StrictModel):
    """TFT-specific knobs: architecture choice, hidden size, quantile set,
    auxiliary loss weights, graph-precompute policy, static cardinalities.

    Every field has a default calibrated to the architecture spec
    ``extra="forbid"`` is inherited from :class:`mmfp.config.schema._StrictModel`
    to catch typos (e.g. ``hiddem_dim``).

    Parameters
    ----------
    architecture
        Forecasting backbone. Only ``"tft"`` is implemented in v3; the
        ``Literal`` is kept open for future extensions (e.g., PatchTST).
    hidden_dim
        TFT hidden width ``H`` used throughout the body (projections,
        GRN, LSTM, attention). Constrained to 16..512.
    n_heads
        Attention heads in the interpretable multi-head attention module.
        ``1`` is the paper default (preserves interpretability via shared
        value heads). ``hidden_dim`` must be divisible by ``n_heads``;
        enforced cross-field in :mod:`forecast.config.validate`.
    n_lstm_layers
        Stacked LSTM layers in the local-processing block. ``1`` is the
        default; >1 adds ~30 k params per extra layer.
    dropout
        Global dropout rate, applied inside every GRN, on LSTM output,
        on attention output, and on the quantile-head input. Input
        dropout is intentionally 0.
    lookback
        Sequence length seen by TFT. Paired with ``DataConfig.lookback``;
        validators warn if they disagree or drop below 30.
    quantiles
        Strictly-sorted tuple of quantiles in (0, 1) to predict. Must
        include the median (0.5). Default matches Lim et al. 2021.
    direction_aux_weight
        Weight on the optional cross-entropy auxiliary over the derived
        direction. Default 0 leaves pinball as the sole objective.
    volatility_aux_weight
        Weight on the optional MSE auxiliary over the derived sigma.
    graph_precompute
        If True, the GAT-node embeddings are computed once per fold and
        cached on disk (see the architecture spec). If False (future
        work), the GAT sits inside the per-sample computation graph.
    graph_node_dim
        Dimensionality of per-node graph embeddings when graph is enabled.
    n_tickers
        Static-categorical cardinality for ``ticker_id`` embedding.
    n_sectors
        Static-categorical cardinality for ``sector_id`` embedding.
    warmup_epochs
        Number of linear-warmup epochs applied before the cosine
        scheduler takes over. 0 disables warmup.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    architecture: Literal["tft"] = "tft"
    hidden_dim: int = Field(default=64, ge=16, le=512)
    n_heads: int = Field(default=1, ge=1, le=8)
    n_lstm_layers: int = Field(default=1, ge=1, le=4)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    lookback: int = Field(default=60, ge=1, le=252)
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    direction_aux_weight: float = Field(default=0.0, ge=0.0)
    volatility_aux_weight: float = Field(default=0.0, ge=0.0)
    graph_precompute: bool = True
    graph_node_dim: int = Field(default=64, ge=16, le=512)
    n_tickers: int = Field(default=55, ge=1)
    n_sectors: int = Field(default=11, ge=1)
    warmup_epochs: int = Field(default=5, ge=0, le=20)

    @field_validator("quantiles")
    @classmethod
    def _validate_quantiles(
        cls, v: tuple[float, ...] | list[float]
    ) -> tuple[float, ...]:
        """Enforce strict monotone increase, bounds in (0, 1), median present.

        TOML round-trips tuples as lists, so the input type is widened to
        ``list`` and coerced back to ``tuple`` for immutability.
        """
        if not isinstance(v, (tuple, list)):
            raise ValueError(
                f"quantiles must be a tuple/list of floats, got {type(v).__name__}"
            )
        if len(v) < 2:
            raise ValueError(
                f"quantiles must contain at least 2 values (point + at least one band), got {v!r}"
            )
        # Convert to plain floats so equality works across numpy/torch inputs.
        quantiles = tuple(float(q) for q in v)
        # Strict bounds (0, 1) — endpoints 0 and 1 are degenerate under pinball loss.
        for q in quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError(
                    f"each quantile must lie strictly in (0, 1); got {q}"
                )
        # Strictly increasing.
        for a, b in zip(quantiles, quantiles[1:]):
            if not (a < b):
                raise ValueError(
                    f"quantiles must be strictly increasing; got {quantiles}"
                )
        # Median must be present (used by direction derivation + point forecast).
        if 0.5 not in quantiles:
            raise ValueError(
                f"quantiles must include the median 0.5; got {quantiles}"
            )
        return quantiles


# ---------------------------------------------------------------------------
# New: V3ExperimentConfig — root config extending v2's ExperimentConfig.
# ---------------------------------------------------------------------------


class V3ExperimentConfig(ExperimentConfig):
    """Root v3 config: all v2 fields verbatim plus a ``forecast`` sub-config.

    Inheriting from :class:`mmfp.config.schema.ExperimentConfig` preserves
    every v2 field and validator (including ``extra="forbid"``), then adds
    :attr:`forecast` as the TFT-specific subconfig.

    When ``forecast.architecture == "tft"`` (the only v3-supported value),
    the inherited :attr:`model`, :attr:`fusion`, and :attr:`head` fields
    are **ignored by the v3 predictor**: the TFT body subsumes fusion and
    the quantile head replaces v2's MLP head. They remain on the schema so
    that the reused ``mmfp.experiments.runner`` dispatcher can select v2
    vs v3 at build time without schema-level divergence.

    Cross-field validation (see :mod:`forecast.config.validate`):

    * ``news.enabled`` locks ``news.encoder == "qwen3_embedding"`` and
      ``news.pca_dims == 128`` (v3 scope lock).
    * ``data.lookback < 30`` warns when architecture is TFT.
    * ``graph.source == "dynamic_corr"`` + ``forecast.graph_precompute``
      requires ``graph.dynamic_refresh_every`` divides the lookback.
    * ``forecast.hidden_dim`` must be divisible by ``forecast.n_heads``.
    * ``training.optimizer == "adam"`` with TFT emits an AdamW-recommended
      warning.
    """

    forecast: ForecastConfig = Field(default_factory=ForecastConfig)


# ---------------------------------------------------------------------------
# Re-exports: convenience so downstream callers can say
# ``from forecast.config.schema import DataConfig`` without reaching
# across to ``mmfp.config``.
# ---------------------------------------------------------------------------


__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ForecastConfig",
    "FusionConfig",
    "GraphConfig",
    "HeadConfig",
    "IntradayConfig",
    "LoggingConfig",
    "MacroConfig",
    "ModelConfig",
    "NewsConfig",
    "PriceConfig",
    "SocialConfig",
    "StandardizationConfig",
    "TrainingConfig",
    "V3ExperimentConfig",
]
