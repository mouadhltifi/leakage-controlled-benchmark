"""Cross-field validation rules for :class:`V3ExperimentConfig`.

v2 rules from :mod:`mmfp.config.validate` are applied first via
:func:`validate_experiment_config` (inherited behaviour on the v2
subconfigs — gated cross-attention, single-task one-target, intraday-
not-implemented, etc.). v3 then layers on rules that depend on
:class:`~forecast.config.schema.ForecastConfig` fields.

Rules added in v3 (in execution order):

1. **Lookback coherence** — if ``forecast.architecture == "tft"`` and
   ``data.lookback < 30``, emit a ``UserWarning`` (TFT expects ≥ 30).
2. **News encoder scope lock** — if ``news.enabled`` is True, enforce
   ``news.encoder == "qwen3_embedding"`` *and* ``news.pca_dims == 128``.
   v3 does not re-run the encoder sweep; this is locked by scope.
3. **Graph precompute cache alignment** — if
   ``forecast.graph_precompute`` is True and ``graph.source == "dynamic_corr"``,
   ``graph.dynamic_refresh_every`` must divide ``forecast.lookback``
   without remainder. Otherwise the per-day cache would be misaligned
   with rolling snapshot windows.
4. **Optimizer recommendation** — if ``training.optimizer == "adam"``
   under ``forecast.architecture == "tft"``, emit a ``UserWarning``
   recommending AdamW (decoupled weight decay is load-bearing at v3's
   ~500 k param scale).
5. **MHA head divisibility** — ``forecast.hidden_dim % forecast.n_heads``
   must be 0 for :class:`torch.nn.MultiheadAttention` to partition the
   hidden dim into equal-sized per-head projections.

All error paths raise :class:`ConfigValidationError` with an actionable
message naming the offending field(s). All warning paths emit
:class:`UserWarning` with ``stacklevel=2`` so pytest attributes them to
the caller under ``warnings.catch_warnings``.
"""

from __future__ import annotations

import logging
import warnings

from mmfp.config.validate import (
    ConfigValidationError,
    validate_experiment_config,
)

from forecast.config.schema import V3ExperimentConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual rule functions. Each takes a V3ExperimentConfig, raises
# ConfigValidationError on failure, or emits a warning on soft issues.
# ---------------------------------------------------------------------------


def _rule_lookback_coherent_with_tft(cfg: V3ExperimentConfig) -> None:
    """Warn when TFT architecture is paired with a too-short lookback.

    TFT's interpretable attention needs enough past timesteps to show
    meaningful temporal patterns. Empirically < 30 trading days is too
    short (see the architecture spec).
    """
    if cfg.forecast.architecture != "tft":
        return
    if cfg.data.lookback < 30:
        warnings.warn(
            f"forecast.architecture='tft' expects data.lookback >= 30; "
            f"got lookback={cfg.data.lookback}. TFT attention may not "
            f"see enough history to produce meaningful weights.",
            UserWarning,
            stacklevel=2,
        )


def _rule_news_encoder_scope_lock(cfg: V3ExperimentConfig) -> None:
    """Enforce the v3 Qwen3 + PCA128 scope lock.

    Defined in the v3 scope doc (the 5-encoder sweep was a v2
    exercise; v3 re-uses precomputed Qwen3 embeddings only).
    """
    if not cfg.news.enabled:
        return
    if cfg.news.encoder != "qwen3_embedding":
        raise ConfigValidationError(
            f"v3 locks news.encoder='qwen3_embedding' per the v3 scope doc; "
            f"got news.encoder={cfg.news.encoder!r}."
        )
    if cfg.news.pca_dims != 128:
        raise ConfigValidationError(
            f"v3 locks news.pca_dims=128 per the v3 scope doc (Qwen3 embeddings "
            f"are 1024-dim; PCA128 is the calibrated target); "
            f"got news.pca_dims={cfg.news.pca_dims}."
        )


def _rule_graph_precompute_cache_alignment(cfg: V3ExperimentConfig) -> None:
    """Enforce dynamic-graph refresh cadence divides TFT lookback.

    With ``graph_precompute=True``, the per-fold GAT cache stores one
    node-embedding per (date, ticker). For dynamic correlation graphs,
    the underlying adjacency is only refreshed every
    ``graph.dynamic_refresh_every`` trading days. If this refresh cadence
    doesn't divide the ``forecast.lookback`` window, then windowed views
    into the cache don't align on refresh boundaries (a subtle source of
    leakage-like inconsistency).

    Guard rail: refuse the combination.
    """
    if not cfg.forecast.graph_precompute:
        return
    if cfg.graph.source != "dynamic_corr":
        return
    refresh = cfg.graph.dynamic_refresh_every
    lookback = cfg.forecast.lookback
    if refresh <= 0 or lookback % refresh != 0:
        raise ConfigValidationError(
            f"graph.source='dynamic_corr' with forecast.graph_precompute=True "
            f"requires graph.dynamic_refresh_every ({refresh}) to divide "
            f"forecast.lookback ({lookback}) with remainder 0 so the cached "
            f"per-date node embeddings align with adjacency refresh boundaries."
        )


def _rule_optimizer_recommendation(cfg: V3ExperimentConfig) -> None:
    """Recommend AdamW over Adam under TFT architecture.

    Per Loshchilov & Hutter (2019), Adam with L2 weight decay applies
    decay to the *adaptive* gradients rather than the parameters; at
    v3's larger scale (weight_decay=1e-4, ~500k params) this distorts
    effective decay. AdamW decouples correctly.
    """
    if cfg.forecast.architecture != "tft":
        return
    if cfg.training.optimizer == "adam":
        warnings.warn(
            "training.optimizer='adam' with forecast.architecture='tft': "
            "AdamW is recommended at v3 scale (decoupled weight decay; "
            "see Loshchilov & Hutter 2019). Consider training.optimizer='adamw'.",
            UserWarning,
            stacklevel=2,
        )


def _rule_hidden_dim_divisible_by_heads(cfg: V3ExperimentConfig) -> None:
    """Enforce ``hidden_dim % n_heads == 0`` for MHA partitioning.

    :class:`torch.nn.MultiheadAttention` requires the embedding dimension
    to be divisible by the number of heads so each head gets an equal
    slice of the hidden vector.
    """
    h = cfg.forecast.hidden_dim
    n = cfg.forecast.n_heads
    if n <= 0 or h % n != 0:
        raise ConfigValidationError(
            f"forecast.hidden_dim ({h}) must be divisible by "
            f"forecast.n_heads ({n}) for MultiheadAttention; "
            f"got remainder {h % n if n > 0 else 'n/a'}."
        )


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------


#: v3 rules applied *after* v2's ``validate_experiment_config`` pass.
#: Order matters only for error legibility when multiple rules would fail;
#: each rule is independent.
_V3_RULES: list = [
    _rule_lookback_coherent_with_tft,
    _rule_news_encoder_scope_lock,
    _rule_graph_precompute_cache_alignment,
    _rule_optimizer_recommendation,
    _rule_hidden_dim_divisible_by_heads,
]


def validate_forecast_config(cfg: V3ExperimentConfig) -> V3ExperimentConfig:
    """Run v2 cross-field rules then v3-specific rules.

    Parameters
    ----------
    cfg
        A constructed :class:`V3ExperimentConfig`. Single-field validation
        is already complete at this point (pydantic ran at construction).

    Returns
    -------
    V3ExperimentConfig
        The same config, unchanged, on success. Enables fluent-style
        ``validate_forecast_config(load_config(...))``.

    Raises
    ------
    ConfigValidationError
        On any hard v2 or v3 rule violation.
    NotImplementedError
        Propagated from v2's intraday-not-implemented rule.
    """
    # v2 rules first (intraday, gated-cross-attention, single-task, etc.).
    validate_experiment_config(cfg)
    for rule in _V3_RULES:
        rule(cfg)
    return cfg


__all__ = [
    "ConfigValidationError",
    "validate_forecast_config",
]
