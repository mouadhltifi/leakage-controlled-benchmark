"""Feature engineering package.

Public entry points
-------------------

* :class:`~mmfp.features.scalers.FittedScaler` — audit-critical
  fit/transform discipline for per-fold standardization (Milestone 2).
* :mod:`mmfp.features.news_aggregate` — five per-stock-day embedding
  aggregation strategies (axis 3; Milestone 3).
* :mod:`mmfp.features.news_stats` — the 11-dim statistical news
  path with the ``log1p`` + ``empty_day_policy`` audit fixes (Milestone 3).
* :mod:`mmfp.features.news_encode` — HF-model-agnostic article encoder
  (Milestone 3; used for regenerating cached parquets).
* :mod:`mmfp.features.price_ta` — technical-indicator features
  + rolling-z-score (Milestone 3).
* :mod:`mmfp.features.macro_events` — FOMC window flags + macro
  rolling-z-score (Milestone 3).
* :mod:`mmfp.features.social_features` — StockTwits daily aggregates
  (Milestone 3).
"""

from mmfp.features.news_aggregate import (
    AGGREGATION_REGISTRY,
    AggregationStrategy,
    ArithmeticMean,
    AttentionWeighted,
    MaxSentiment,
    RecencyWeighted,
    SphericalMean,
    aggregate_daily,
    build_strategy,
)
from mmfp.features.news_stats import (
    NEWS_STATS_FEATURES,
    build_news_stats_11dim,
)
from mmfp.features.scalers import FittedScaler

__all__ = [
    "AGGREGATION_REGISTRY",
    "AggregationStrategy",
    "ArithmeticMean",
    "AttentionWeighted",
    "FittedScaler",
    "MaxSentiment",
    "NEWS_STATS_FEATURES",
    "RecencyWeighted",
    "SphericalMean",
    "aggregate_daily",
    "build_news_stats_11dim",
    "build_strategy",
]
