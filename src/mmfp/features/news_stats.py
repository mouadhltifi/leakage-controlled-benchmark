"""11-dim statistical news features with the audit-critical fixes.

Ports the sentiment-aggregation logic from the earlier feature pipeline with the
following corrections:

1. ``log1p(news_volume)`` is always available behind a flag (default
   on). v1 exposed raw counts only, which dominated the scale during
   standardisation.
2. Empty-day policy is an explicit flag, not an implicit
   ``asfreq("B").fillna(0)`` side effect:

   * ``"zero_fill_has_news_flag"`` (v1-compatible) — zero-fill every
     business-day row with no articles and flip ``has_news`` off. This
     is what v1 effectively did but now it's named and documented.
   * ``"sentinel_vector"`` — leave the sentiment columns as ``NaN``
     (the assembly step can then decide whether to drop those rows or
     mask them). Still emits ``has_news=0`` for empty days.

3. ``has_news`` is a first-class passthrough binary feature, independent
   of which aggregation strategy was used for the embedding path.

The eleven feature columns (in output order) are:

* ``news_sent_mean``, ``news_sent_std``, ``news_sent_min``,
  ``news_sent_max`` — summary stats of ``sent_score`` across a day.
* ``news_pos_mean``, ``news_neg_mean`` — mean of FinBERT ``p_pos`` and
  ``p_neg`` probabilities.
* ``news_volume`` — article count (log1p-transformed when
  ``log1p_volume=True``).
* ``news_sent_mean_w{W}``, ``news_sent_std_w{W}`` — rolling mean of the
  per-day mean/std over ``rolling_window`` trading days.
* ``news_volume_w{W}`` — rolling sum of ``news_volume`` (log1p'd too
  when ``log1p_volume=True``).
* ``has_news`` — 0/1 flag.

This adds up to 11 columns; the v1 ``news_feat_cols`` in
``assemble_dataset.py`` expected exactly these names. The platform's
``FoldArtifacts`` (Milestone 4) declares ``log1p_cols =
{"news_volume", "news_volume_w3"}`` and
``passthrough_cols = {"has_news"}`` on the :class:`FittedScaler`.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: The 11 feature columns emitted by :func:`build_news_stats_11dim`,
#: ordered for deterministic downstream use.
NEWS_STATS_FEATURES: list[str] = [
    "news_sent_mean",
    "news_sent_std",
    "news_sent_min",
    "news_sent_max",
    "news_pos_mean",
    "news_neg_mean",
    "news_volume",
    "news_sent_mean_w{W}",
    "news_sent_std_w{W}",
    "news_volume_w{W}",
    "has_news",
]

#: Columns that :class:`~mmfp.features.scalers.FittedScaler` should
#: ``log1p``-transform by default when standardising the news block.
LOG1P_COLS: list[str] = ["news_volume", "news_volume_w{W}"]

#: Columns that should pass through the scaler unchanged.
PASSTHROUGH_COLS: list[str] = ["has_news"]


EmptyDayPolicy = Literal["zero_fill_has_news_flag", "sentinel_vector"]


def _feature_name(template: str, window: int) -> str:
    """Substitute the ``{W}`` placeholder for the rolling window size."""
    return template.format(W=window)


def _daily_aggregate(
    per_article_sentiments: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse per-article rows into per-``(Ticker, Date)`` stats.

    Takes the raw per-article frame with columns ``Date``, ``Ticker``,
    ``p_pos``, ``p_neg`` (``p_neu`` optional) and returns one row per
    unique ``(Ticker, Date)`` with the seven point-in-time features.
    """
    df = per_article_sentiments.copy()
    if "sent_score" not in df.columns:
        if not {"p_pos", "p_neg"}.issubset(df.columns):
            raise ValueError(
                "Per-article DataFrame must provide either 'sent_score' or "
                "both 'p_pos' and 'p_neg' columns."
            )
        df["sent_score"] = (
            df["p_pos"].astype(np.float64) - df["p_neg"].astype(np.float64)
        )

    # Some upstream parquets named these columns differently; accept
    # both. p_neu is optional (never used by the 11-dim features).
    df.rename(
        columns={
            "sent_pos": "p_pos",
            "sent_neg": "p_neg",
            "sent_neu": "p_neu",
        },
        inplace=True,
    )

    for required in ("Date", "Ticker", "p_pos", "p_neg"):
        if required not in df.columns:
            raise ValueError(
                f"Per-article DataFrame missing required column {required!r}. "
                f"Got: {list(df.columns)}"
            )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if getattr(df["Date"].dtype, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["Date"] = df["Date"].dt.normalize()

    grouped = df.groupby(["Ticker", "Date"], sort=True, observed=True)
    # ``agg`` with named outputs keeps the column layout stable.
    daily = grouped.agg(
        news_sent_mean=("sent_score", "mean"),
        news_sent_std=("sent_score", "std"),
        news_sent_min=("sent_score", "min"),
        news_sent_max=("sent_score", "max"),
        news_pos_mean=("p_pos", "mean"),
        news_neg_mean=("p_neg", "mean"),
        news_volume=("sent_score", "count"),
    ).reset_index()
    # groupby's std returns NaN for one-article days (ddof=1); the
    # study interpretation is "no spread within the day", so zero-fill.
    daily["news_sent_std"] = daily["news_sent_std"].fillna(0.0)
    daily["news_volume"] = daily["news_volume"].astype(np.int64)
    return daily


def _reindex_business_days(
    per_ticker_daily: pd.DataFrame,
    *,
    window: int,
    policy: EmptyDayPolicy,
) -> pd.DataFrame:
    """Expand to the full business-day grid per ticker and apply rolling.

    The function operates on a single-ticker frame; the public
    :func:`build_news_stats_11dim` applies it group-by-group.
    """
    ticker = per_ticker_daily["Ticker"].iloc[0]
    frame = (
        per_ticker_daily.drop(columns=["Ticker"])
        .set_index("Date")
        .sort_index()
    )
    # Business-day reindex over the contiguous range. Upstream assembly
    # aligns to the trading calendar later so any extra B-day rows are
    # harmless (they'll be merged away in Milestone 4).
    frame = frame.asfreq("B")

    # Before computing rolling stats we need to honour the empty-day
    # policy. Both policies yield the same rolling behaviour when the
    # intermediate NaNs are first filled with zeros (rolling treats
    # NaN as missing); we do the fill here so the rolling columns are
    # never NaN regardless of policy.
    filled_for_rolling = frame.fillna(
        {
            "news_sent_mean": 0.0,
            "news_sent_std": 0.0,
            "news_volume": 0,
        }
    )

    roll_mean_col = _feature_name("news_sent_mean_w{W}", window)
    roll_std_col = _feature_name("news_sent_std_w{W}", window)
    roll_vol_col = _feature_name("news_volume_w{W}", window)

    frame[roll_mean_col] = (
        filled_for_rolling["news_sent_mean"]
        .rolling(window, min_periods=1)
        .mean()
    )
    frame[roll_std_col] = (
        filled_for_rolling["news_sent_std"]
        .rolling(window, min_periods=1)
        .mean()
    )
    frame[roll_vol_col] = (
        filled_for_rolling["news_volume"]
        .rolling(window, min_periods=1)
        .sum()
    )

    # has_news uses the *raw* news_volume (pre zero-fill) to tell apart
    # "real article" from "business day with no coverage".
    has_news = (frame["news_volume"].fillna(0) > 0).astype(np.int8)
    frame["has_news"] = has_news

    if policy == "zero_fill_has_news_flag":
        for col in (
            "news_sent_mean",
            "news_sent_std",
            "news_sent_min",
            "news_sent_max",
            "news_pos_mean",
            "news_neg_mean",
            "news_volume",
        ):
            frame[col] = frame[col].fillna(0.0)
    elif policy == "sentinel_vector":
        # Keep sentiment columns as NaN so the assembly step can decide
        # what to do. Volume itself is interpretable: 0 = no articles.
        frame["news_volume"] = frame["news_volume"].fillna(0)
    else:
        raise ValueError(f"Unknown empty_day_policy: {policy!r}")

    frame["Ticker"] = ticker
    return frame.reset_index()


def build_news_stats_11dim(
    per_article_sentiments: pd.DataFrame,
    *,
    log1p_volume: bool = True,
    empty_day_policy: EmptyDayPolicy = "zero_fill_has_news_flag",
    rolling_window: int = 3,
) -> pd.DataFrame:
    """Construct the 11-dim statistical news feature table.

    Parameters
    ----------
    per_article_sentiments
        Tidy ``DataFrame`` with at least ``Date``, ``Ticker``, ``p_pos``,
        ``p_neg`` columns (``p_neu`` is accepted but unused). A
        ``sent_score`` column overrides the ``p_pos - p_neg`` calculation
        if present.
    log1p_volume
        When ``True`` (default), apply ``numpy.log1p`` to both
        ``news_volume`` and ``news_volume_w{W}``. This is the
        audit-critical fix: volume was previously an integer count that
        dominated standardisation.
    empty_day_policy
        ``"zero_fill_has_news_flag"`` to zero-fill sentiment columns on
        days without articles; ``"sentinel_vector"`` to leave them
        ``NaN`` for the assembly step to handle.
    rolling_window
        Trailing-window length for the ``_w{W}`` features. The study
        uses 3 (DASF-Net optimum) by default.

    Returns
    -------
    pandas.DataFrame
        Business-day grid with exactly 11 feature columns per
        ``(Ticker, Date)`` pair, plus the ``Date`` and ``Ticker``
        identifiers. ``has_news`` is ``int8``, numeric columns are
        ``float32``. Sorted ``(Ticker, Date)`` ascending.

    Raises
    ------
    ValueError
        If ``rolling_window < 1``, required columns are missing, or
        ``empty_day_policy`` is unknown.
    """
    if rolling_window < 1:
        raise ValueError(
            f"rolling_window must be >= 1; got {rolling_window}"
        )
    if empty_day_policy not in (
        "zero_fill_has_news_flag",
        "sentinel_vector",
    ):
        raise ValueError(f"Unknown empty_day_policy: {empty_day_policy!r}")

    if per_article_sentiments.empty:
        # Build a schema-preserving empty frame.
        out = pd.DataFrame(
            {name: pd.Series(dtype="float32") for name in _resolve_feature_names(rolling_window)}
        )
        out.insert(0, "Ticker", pd.Series(dtype="string"))
        out.insert(1, "Date", pd.to_datetime(pd.Series(dtype="object")))
        out["has_news"] = out["has_news"].astype("int8")
        return out

    daily_per_ticker = _daily_aggregate(per_article_sentiments)

    out_chunks: list[pd.DataFrame] = []
    for _ticker, grp in daily_per_ticker.groupby("Ticker", sort=True):
        expanded = _reindex_business_days(
            grp, window=rolling_window, policy=empty_day_policy,
        )
        out_chunks.append(expanded)

    out = pd.concat(out_chunks, ignore_index=True)

    # Apply the log1p transform to the volume columns after the rolling
    # sum was computed on raw counts. Doing it here (rather than inside
    # ``_reindex_business_days``) keeps the rolling semantics intact.
    if log1p_volume:
        # log1p on NaN is NaN; with zero-fill policy there are no NaNs
        # to worry about at this point; with sentinel policy the
        # sentinel values live in the sentiment columns, not volume.
        out["news_volume"] = np.log1p(out["news_volume"].astype(np.float64))
        roll_vol_col = _feature_name("news_volume_w{W}", rolling_window)
        out[roll_vol_col] = np.log1p(out[roll_vol_col].astype(np.float64))

    # Finalise dtypes.
    numeric_cols = [
        "news_sent_mean",
        "news_sent_std",
        "news_sent_min",
        "news_sent_max",
        "news_pos_mean",
        "news_neg_mean",
        "news_volume",
        _feature_name("news_sent_mean_w{W}", rolling_window),
        _feature_name("news_sent_std_w{W}", rolling_window),
        _feature_name("news_volume_w{W}", rolling_window),
    ]
    for c in numeric_cols:
        out[c] = out[c].astype(np.float32)
    out["has_news"] = out["has_news"].astype(np.int8)

    # Column ordering for determinism.
    ordered = (
        ["Ticker", "Date"]
        + _resolve_feature_names(rolling_window)
    )
    out = out[ordered]
    out = out.sort_values(["Ticker", "Date"], kind="mergesort").reset_index(
        drop=True
    )

    log.debug(
        "build_news_stats_11dim: rows=%d tickers=%d window=%d policy=%s log1p=%s",
        len(out),
        out["Ticker"].nunique(),
        rolling_window,
        empty_day_policy,
        log1p_volume,
    )
    return out


def _resolve_feature_names(window: int) -> list[str]:
    """Concrete column names in output order for the given ``window``."""
    return [_feature_name(name, window) for name in NEWS_STATS_FEATURES]


__all__ = [
    "EmptyDayPolicy",
    "LOG1P_COLS",
    "NEWS_STATS_FEATURES",
    "PASSTHROUGH_COLS",
    "build_news_stats_11dim",
]
