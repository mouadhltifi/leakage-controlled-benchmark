"""Per-stock-day news-embedding aggregation strategies (Axis 3).

This module is the headline Milestone 3 deliverable. It replaces v1's
ad-hoc ``_aggregate_768_embeddings`` with five principled strategies
sharing a common protocol and registry.

Strategies
----------

=================== ==========================================================
``arithmetic_mean`` ``v_out = (1/N) sum v_i``. The MSGCA/v1 baseline.
``spherical_mean``  ``v_mean = (1/N) sum v_i``; ``v_out = v_mean /
                    max(||v_mean||, eps)``. Decouples direction from
                    magnitude; the scalar ``||v_mean||`` is a
                    consensus-strength signal (dispersion feature).
``attention_weighted`` ``w_i = softmax(|sent_i| / tau)``; sum ``w_i v_i``.
                    Saliency-weighted by sentiment magnitude.
``recency_weighted`` ``w_i proportional to decay^(N-1-i)``; normalised.
                    Emphasises the latest article of the day.
``max_sentiment``   ``v_out = v_{argmax |sent_i|}``. Single-article pick.
                    Stable tiebreak: first occurrence of the maximum.
=================== ==========================================================

All strategies emit a ``news_volume`` (raw article count) and a
``has_news`` binary flag alongside the embedding. The ``has_news``
flag is intended to be carried through ``FittedScaler(passthrough_cols=...)``
downstream.

Invariants
----------

* Spherical mean: zero-vector input yields zero output (no ``NaN`` or
  ``Inf``). See ``mmfp.tests.numerical.test_spherical_mean_zero_vector``.
* Attention: zero-valued sentiment reduces to uniform weights, i.e. the
  arithmetic mean.
* Recency: ``decay=1.0`` reduces to uniform weights, i.e. the arithmetic
  mean.
* Max sentiment: first-occurrence tiebreak.
* Single-article day: every strategy returns the one input vector.

The module provides:

* :class:`AggregationStrategy` — structural :class:`typing.Protocol`.
* Five concrete strategies (classes with an ``aggregate`` method).
* :data:`AGGREGATION_REGISTRY` — name to class lookup.
* :func:`aggregate_daily` — per-stock-per-day roll-up over a
  per-article ``DataFrame``.

See the design spec for the authoritative
specification.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Floor used by spherical mean to avoid divide-by-zero on a zero-norm mean.
_SPHERICAL_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


class AggregationStrategy(Protocol):
    """Structural protocol for a single-stock-single-day aggregator.

    Implementations collapse an ``(n_articles, D)`` matrix of embeddings
    into a ``(D,)`` vector plus a ``dict`` of scalar "extras" (e.g. the
    spherical-mean dispersion).
    """

    def aggregate(
        self,
        embeddings: np.ndarray,
        sent_scores: np.ndarray,
        article_idx: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Return ``(aggregated (D,), extras: dict[str, float])``.

        Parameters
        ----------
        embeddings
            ``(n_articles, D)`` ``float`` array of per-article vectors.
        sent_scores
            ``(n_articles,)`` sentiment score per article in ``[-1, 1]``
            (typically ``p_pos - p_neg``). Only used by strategies that
            depend on sentiment (``attention_weighted``, ``max_sentiment``).
        article_idx
            ``(n_articles,)`` integer index defining intra-day ordering,
            with the latest article having the highest value. Only used
            by ``recency_weighted``. Implementations MUST respect this
            ordering, not the row order of ``embeddings``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, float]]
            ``(out, extras)``. ``out`` has shape ``(D,)``. ``extras``
            may be empty; every key must map to a finite Python
            ``float`` scalar.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_shapes(
    embeddings: np.ndarray,
    sent_scores: np.ndarray,
    article_idx: np.ndarray,
) -> None:
    """Shared pre-aggregation shape/type validation.

    Raises
    ------
    ValueError
        If shapes are inconsistent or the arrays are empty.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D (n_articles, D); got shape {embeddings.shape}"
        )
    n, _ = embeddings.shape
    if n == 0:
        raise ValueError("aggregate() called with zero articles")
    if sent_scores.shape != (n,):
        raise ValueError(
            f"sent_scores shape {sent_scores.shape} does not match n_articles={n}"
        )
    if article_idx.shape != (n,):
        raise ValueError(
            f"article_idx shape {article_idx.shape} does not match n_articles={n}"
        )


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class ArithmeticMean:
    """Plain mean pool.

    ``v_out = (1/N) sum v_i``.

    Matches the v1/MSGCA default and acts as the reference baseline for
    every other strategy. Emits no extras.
    """

    def aggregate(
        self,
        embeddings: np.ndarray,
        sent_scores: np.ndarray,
        article_idx: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        _validate_shapes(embeddings, sent_scores, article_idx)
        out = embeddings.mean(axis=0)
        return out.astype(np.float32, copy=False), {}


class SphericalMean:
    """L2-projected mean with dispersion signal.

    ``v_mean = (1/N) sum v_i``; ``v_out = v_mean / max(||v_mean||, eps)``.

    The direction of the daily consensus is kept; its *magnitude*
    (``||v_mean||``) is returned separately in ``extras["dispersion"]``
    so downstream code can expose it as a scalar feature. Zero-norm
    input returns the zero vector (no ``NaN``).
    """

    def aggregate(
        self,
        embeddings: np.ndarray,
        sent_scores: np.ndarray,
        article_idx: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        _validate_shapes(embeddings, sent_scores, article_idx)
        v_mean = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(v_mean))
        if norm < _SPHERICAL_EPS:
            # Zero-vector input: return zero vector and zero dispersion.
            # NEVER divide by zero. This is the critical robustness case.
            out = np.zeros_like(v_mean)
            return out.astype(np.float32, copy=False), {"dispersion": 0.0}
        out = v_mean / norm
        return out.astype(np.float32, copy=False), {"dispersion": norm}


class AttentionWeighted:
    """Sentiment-magnitude attention.

    ``w_i = softmax(|sent_i| / tau)``; ``v_out = sum w_i v_i``.

    Gives more weight to articles with a strong (positive or negative)
    sentiment signal. When every article is sentiment-neutral (all
    ``|sent_i| == 0``) the softmax produces uniform weights, reducing
    the strategy to :class:`ArithmeticMean`.

    Parameters
    ----------
    tau
        Softmax temperature. Larger ``tau`` -> softer weighting (more
        uniform). Must be positive.
    """

    def __init__(self, tau: float = 1.0) -> None:
        if tau <= 0:
            raise ValueError(f"AttentionWeighted tau must be > 0; got {tau}")
        self.tau = float(tau)

    def aggregate(
        self,
        embeddings: np.ndarray,
        sent_scores: np.ndarray,
        article_idx: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        _validate_shapes(embeddings, sent_scores, article_idx)
        logits = np.abs(sent_scores.astype(np.float64)) / self.tau
        # Numerically stable softmax.
        logits = logits - logits.max()
        weights = np.exp(logits)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            # Degenerate case — fall back to uniform. Should not happen
            # after the max-subtract normalisation above, but belt-and-
            # braces for safety when embeddings contain NaN.
            weights = np.ones_like(weights) / weights.size
        else:
            weights = weights / total
        out = (embeddings * weights[:, None]).sum(axis=0)
        return out.astype(np.float32, copy=False), {}


class RecencyWeighted:
    """Exponential-decay recency weighting.

    With ``N`` articles sorted by ``article_idx`` ascending, the weight
    of the ``i``-th article is ``decay^(N-1-i)`` (so the latest article
    has weight ``1``) normalised to sum to one. ``decay=1.0`` reduces
    to uniform weights, i.e. :class:`ArithmeticMean`.

    Parameters
    ----------
    decay
        Geometric decay rate in ``(0, 1]``. Values near ``1`` keep
        older articles salient; near ``0`` concentrates mass on the
        latest article.
    """

    def __init__(self, decay: float = 0.9) -> None:
        if not (0.0 < decay <= 1.0):
            raise ValueError(
                f"RecencyWeighted decay must be in (0, 1]; got {decay}"
            )
        self.decay = float(decay)

    def aggregate(
        self,
        embeddings: np.ndarray,
        sent_scores: np.ndarray,
        article_idx: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        _validate_shapes(embeddings, sent_scores, article_idx)
        n = embeddings.shape[0]
        # Rank articles 0..N-1 by article_idx ascending (stable sort).
        order = np.argsort(article_idx, kind="stable")
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n)
        # Weight for rank r is decay^((N-1)-r). Rank N-1 = latest -> 1.0.
        # ranks are 0..N-1 so (N-1)-ranks is also 0..N-1.
        exponents = (n - 1) - ranks
        weights = np.power(self.decay, exponents.astype(np.float64))
        total = float(weights.sum())
        if total <= 0.0:
            weights = np.ones(n, dtype=np.float64) / n
        else:
            weights = weights / total
        out = (embeddings * weights[:, None]).sum(axis=0)
        return out.astype(np.float32, copy=False), {}


class MaxSentiment:
    """Single-article pick by sentiment magnitude.

    ``v_out = v_{argmax |sent_i|}``. Ties are broken by first
    occurrence (``numpy.argmax`` semantics). Emits no extras.
    """

    def aggregate(
        self,
        embeddings: np.ndarray,
        sent_scores: np.ndarray,
        article_idx: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        _validate_shapes(embeddings, sent_scores, article_idx)
        scores = np.abs(sent_scores.astype(np.float64))
        # numpy.argmax returns the first occurrence on ties — the stable
        # tiebreak required by the spec.
        best = int(np.argmax(scores))
        out = embeddings[best].copy()
        return out.astype(np.float32, copy=False), {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


#: Name-to-class map, one entry per axis-3 candidate. Names match the
#: ``news.aggregation`` Literal in ``NewsConfig``.
AGGREGATION_REGISTRY: dict[str, type] = {
    "arithmetic_mean": ArithmeticMean,
    "spherical_mean": SphericalMean,
    "attention_weighted": AttentionWeighted,
    "recency_weighted": RecencyWeighted,
    "max_sentiment": MaxSentiment,
}


def build_strategy(
    name: str,
    *,
    decay: float = 0.9,
    attention_tau: float = 1.0,
) -> AggregationStrategy:
    """Instantiate an aggregation strategy by name.

    Parameters
    ----------
    name
        One of the keys in :data:`AGGREGATION_REGISTRY`.
    decay
        Decay rate passed to :class:`RecencyWeighted`. Ignored by
        other strategies.
    attention_tau
        Softmax temperature for :class:`AttentionWeighted`. Ignored
        by other strategies.
    """
    if name not in AGGREGATION_REGISTRY:
        raise ValueError(
            f"Unknown aggregation strategy {name!r}. "
            f"Expected one of {sorted(AGGREGATION_REGISTRY)}."
        )
    cls = AGGREGATION_REGISTRY[name]
    if name == "recency_weighted":
        return cls(decay=decay)
    if name == "attention_weighted":
        return cls(tau=attention_tau)
    return cls()


# ---------------------------------------------------------------------------
# Dataframe-level driver
# ---------------------------------------------------------------------------


def _infer_embedding_columns(df: pd.DataFrame) -> list[str]:
    """Extract the ``emb_0, emb_1, ..., emb_{D-1}`` columns from ``df``.

    Columns are sorted numerically by the suffix so that ``emb_10`` does
    not come before ``emb_2``. Raises if no embedding columns are found.
    """
    emb_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("emb_")]
    if not emb_cols:
        raise ValueError(
            "Per-article DataFrame has no columns named 'emb_*'. "
            "Expected schema: Date, Ticker, emb_0..emb_{D-1}, sent_score."
        )

    def _suffix(c: str) -> int:
        try:
            return int(c.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return sorted(emb_cols, key=_suffix)


def aggregate_daily(
    per_article_embeddings: pd.DataFrame,
    strategy: str,
    *,
    dispersion_feature: bool = False,
    decay: float = 0.9,
    attention_tau: float = 1.0,
) -> pd.DataFrame:
    """Roll per-article embeddings up to per-(Ticker, Date) vectors.

    Parameters
    ----------
    per_article_embeddings
        Tidy ``DataFrame`` with at least the columns:

        * ``Date`` — ``datetime64`` or coercible.
        * ``Ticker`` — canonical study symbol.
        * ``emb_0, emb_1, ..., emb_{D-1}`` — numeric.
        * ``sent_score`` — ``float``, typically ``p_pos - p_neg``.

        May optionally include ``article_idx`` or ``article_order``.
        If neither is present the row order within each ``(Ticker, Date)``
        group is used as the intra-day ordering (oldest first).

    strategy
        One of :data:`AGGREGATION_REGISTRY`'s keys.

    dispersion_feature
        If ``True`` and ``strategy=="spherical_mean"``, append a
        ``dispersion`` column (the pre-normalisation L2 norm of the
        mean). For other strategies ``dispersion_feature`` is ignored
        (no such signal exists).

    decay
        Forwarded to :class:`RecencyWeighted`. Ignored by other strategies.

    attention_tau
        Forwarded to :class:`AttentionWeighted`. Ignored otherwise.

    Returns
    -------
    pandas.DataFrame
        One row per (Ticker, Date), columns in order:

        * ``Ticker`` (``str``),
        * ``Date`` (``datetime64[ns]``, tz-naive, midnight),
        * ``emb_0, ..., emb_{D-1}`` (``float32``),
        * ``dispersion`` (``float32``, only when
          ``dispersion_feature=True`` and spherical),
        * ``news_volume`` (``int64``, raw article count),
        * ``has_news`` (``int8``, always ``1`` — every output row has
          at least one article).

        Sorted by ``(Ticker, Date)`` ascending. Output order within a
        ticker matches input date order.

    Notes
    -----
    This function is intentionally a pure aggregator: it does NOT emit
    rows for days with zero articles. The assembly step
    (:mod:`mmfp.data.assemble`, Milestone 4) will merge these per-day
    rows against the full trading calendar and apply the configured
    ``empty_day_policy``.
    """
    if strategy not in AGGREGATION_REGISTRY:
        raise ValueError(
            f"Unknown aggregation strategy {strategy!r}. "
            f"Expected one of {sorted(AGGREGATION_REGISTRY)}."
        )
    for required in ("Date", "Ticker", "sent_score"):
        if required not in per_article_embeddings.columns:
            raise ValueError(
                f"Per-article DataFrame missing required column {required!r}. "
                f"Got: {list(per_article_embeddings.columns)}"
            )

    emb_cols = _infer_embedding_columns(per_article_embeddings)
    d_model = len(emb_cols)

    aggregator = build_strategy(
        strategy, decay=decay, attention_tau=attention_tau,
    )

    df = per_article_embeddings.copy()
    # Normalise Date to tz-naive midnight for stable grouping. Upstream
    # caches use both tz-aware UTC (news_per_article_768) and tz-naive
    # (price_features); we coerce here so aggregate_daily is robust.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if getattr(df["Date"].dtype, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["Date"] = df["Date"].dt.normalize()

    if df["Date"].isna().any():
        n_bad = int(df["Date"].isna().sum())
        log.warning("aggregate_daily: dropping %d rows with NaT Date", n_bad)
        df = df.dropna(subset=["Date"])

    # Determine the intra-day ordering column.
    if "article_idx" in df.columns:
        order_col = "article_idx"
    elif "article_order" in df.columns:
        order_col = "article_order"
    else:
        # Fallback: row order within group.
        df = df.reset_index(drop=True)
        df["__row_order__"] = np.arange(len(df), dtype=np.int64)
        order_col = "__row_order__"

    # Sort so every (Ticker, Date) group is contiguous and ordered by
    # article position — keeps recency weighting deterministic.
    df = df.sort_values(["Ticker", "Date", order_col], kind="mergesort")

    # Collect per-group aggregated rows.
    rows: list[dict[str, object]] = []
    include_dispersion = bool(
        dispersion_feature and strategy == "spherical_mean"
    )

    # ``groupby`` with sort=False preserves the (Ticker, Date) ordering
    # we just built.
    grouped = df.groupby(["Ticker", "Date"], sort=False, observed=True)
    for (ticker, date), group in grouped:
        emb_mat = group[emb_cols].to_numpy(dtype=np.float32, copy=False)
        sent = group["sent_score"].to_numpy(dtype=np.float32, copy=False)
        idx = group[order_col].to_numpy(dtype=np.int64, copy=False)
        vec, extras = aggregator.aggregate(emb_mat, sent, idx)

        row: dict[str, object] = {"Ticker": ticker, "Date": date}
        for col, val in zip(emb_cols, vec):
            row[col] = float(val)
        if include_dispersion:
            row["dispersion"] = float(extras.get("dispersion", 0.0))
        row["news_volume"] = int(len(group))
        row["has_news"] = np.int8(1)
        rows.append(row)

    if not rows:
        # Empty input: return a schema-preserving empty DataFrame.
        cols = (
            ["Ticker", "Date"]
            + emb_cols
            + (["dispersion"] if include_dispersion else [])
            + ["news_volume", "has_news"]
        )
        empty = pd.DataFrame({c: pd.Series(dtype="float32") for c in cols})
        empty["Ticker"] = empty["Ticker"].astype("string")
        empty["Date"] = pd.to_datetime(empty["Date"])
        empty["news_volume"] = empty["news_volume"].astype("int64")
        empty["has_news"] = empty["has_news"].astype("int8")
        return empty

    out = pd.DataFrame(rows)
    # Stable final ordering for downstream merge operations.
    out = out.sort_values(["Ticker", "Date"], kind="mergesort").reset_index(
        drop=True
    )

    # Coerce dtypes.
    for c in emb_cols:
        out[c] = out[c].astype(np.float32)
    if include_dispersion:
        out["dispersion"] = out["dispersion"].astype(np.float32)
    out["news_volume"] = out["news_volume"].astype(np.int64)
    out["has_news"] = out["has_news"].astype(np.int8)

    # Keep column ordering predictable:
    #   Ticker, Date, emb_*, [dispersion], news_volume, has_news
    tail = (
        (["dispersion"] if include_dispersion else [])
        + ["news_volume", "has_news"]
    )
    out = out[["Ticker", "Date", *emb_cols, *tail]]

    log.debug(
        "aggregate_daily: strategy=%s rows=%d tickers=%d D=%d",
        strategy,
        len(out),
        out["Ticker"].nunique(),
        d_model,
    )
    return out


__all__ = [
    "AGGREGATION_REGISTRY",
    "AggregationStrategy",
    "ArithmeticMean",
    "AttentionWeighted",
    "MaxSentiment",
    "RecencyWeighted",
    "SphericalMean",
    "aggregate_daily",
    "build_strategy",
]
