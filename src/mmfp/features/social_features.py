"""StockTwits daily aggregate features.

Port of the logic in ``work/scripts/build_stocktwits_features.py`` into
a library function. The input is the per-message parquet produced by
:func:`mmfp.data.loaders.load_stocktwits`; the output is a tidy
per-(Ticker, Date) frame matching the canonical
``social_features.parquet`` layout.

Per-stock-day features (in order):

* ``social_sent_mean`` — net sentiment across labelled messages
  (mapped into ``[-1, +1]``).
* ``social_sent_std`` — placeholder (StockTwits has no continuous
  scores). Always ``0.0`` — kept for v1-compatible schema.
* ``social_sent_min``, ``social_sent_max`` — extreme label seen on
  the day (or ``0`` if unlabelled).
* ``social_pos_mean``, ``social_neg_mean`` — share of bullish / bearish
  over ALL messages (labelled + unlabelled).
* ``social_volume`` — total messages (labelled + unlabelled).
* ``social_sent_mean_w{W}``, ``social_sent_std_w{W}`` — rolling means.
* ``social_volume_w{W}`` — rolling sum.
* ``social_bullish_ratio``, ``social_net_sentiment``,
  ``social_sentiment_intensity`` — StockTwits-specific.
* ``social_labeled_volume``, ``social_log_volume`` — volume helpers.
* ``social_bullish_ratio_w{W}``, ``social_net_sentiment_w{W}`` — rolling.
* ``has_social`` — 0/1 flag (new; audit-critical for empty-day handling).

Every column except ``has_social`` and ``social_sent_std`` is returned
as ``float32``.

``log1p_volume`` re-uses ``FittedScaler(log1p_cols=...)`` downstream
(see :data:`LOG1P_COLS`) rather than baking the log into the parquet,
so the raw counts remain inspectable. The v1 ``social_log_volume``
column is always emitted for inspection/parity, regardless of the flag.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mmfp.data.paths import FEATURES_DIR
from mmfp.data.universe import ALL_TICKERS

log = logging.getLogger(__name__)

#: Default trailing-window length for the ``_w{W}`` features.
DEFAULT_WINDOW: int = 3

#: Columns that should be ``log1p``-transformed by
#: :class:`~mmfp.features.scalers.FittedScaler` when ``log1p_volume=True``.
LOG1P_COLS: list[str] = [
    "social_volume",
    "social_volume_w{W}",
    "social_labeled_volume",
]

#: Binary flags that pass through the scaler unchanged.
PASSTHROUGH_COLS: list[str] = ["has_social"]


def _window_col(template: str, window: int) -> str:
    return template.format(W=window)


def compute_social_features(
    stocktwits: pd.DataFrame,
    *,
    tickers: list[str] | None = None,
    aggregation_window: int = DEFAULT_WINDOW,
    log1p_volume: bool = True,
) -> pd.DataFrame:
    """Roll per-message StockTwits data up to daily features.

    Parameters
    ----------
    stocktwits
        Per-message ``DataFrame`` from
        :func:`mmfp.data.loaders.load_stocktwits`. Required columns:
        ``Date``, ``Ticker``, ``sentiment`` (``+1.0`` bullish,
        ``-1.0`` bearish, ``NaN`` unlabelled).
    tickers
        Restrict to this ticker set. ``None`` (default) processes
        the full study universe.
    aggregation_window
        Trailing-window length for the ``_w{W}`` features. Study
        convention is 3 (DASF-Net optimum) matching the news path.
    log1p_volume
        Emit ``social_log_volume = np.log1p(social_volume)`` and also
        transform ``social_volume``/``social_volume_w{W}`` after the
        rolling sum when ``True``. Controls the parquet contents, not
        the :class:`FittedScaler` behaviour (that remains column-list
        driven).

    Returns
    -------
    pandas.DataFrame
        Per-(Ticker, Date) rows with the social feature columns above
        plus ``has_social``. Sorted ``(Ticker, Date)`` ascending.

    Raises
    ------
    ValueError
        If required columns are missing or ``aggregation_window < 1``.
    """
    if aggregation_window < 1:
        raise ValueError(
            f"aggregation_window must be >= 1; got {aggregation_window}"
        )
    for required in ("Date", "Ticker", "sentiment"):
        if required not in stocktwits.columns:
            raise ValueError(
                f"StockTwits DataFrame missing required column {required!r}"
            )

    df = stocktwits.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])

    wanted = list(tickers) if tickers is not None else list(ALL_TICKERS)

    all_chunks: list[pd.DataFrame] = []

    for ticker in wanted:
        ticker_df = df[df["Ticker"] == ticker]
        if ticker_df.empty:
            log.debug("compute_social_features: no data for %s", ticker)
            continue

        daily = ticker_df.groupby("Date", sort=True).agg(
            total=("Ticker", "count"),
            bullish=("sentiment", lambda x: (x == 1.0).sum()),
            bearish=("sentiment", lambda x: (x == -1.0).sum()),
            labeled=("sentiment", lambda x: x.notna().sum()),
        ).reset_index()

        daily["social_volume"] = daily["total"].astype(np.float64)
        daily["social_labeled_volume"] = daily["labeled"].astype(np.float64)
        daily["social_log_volume"] = np.log1p(daily["total"].astype(np.float64))

        eps = 1e-8  # avoid div-by-zero when no labelled messages
        daily["social_bullish_ratio"] = daily["bullish"] / (
            daily["labeled"] + eps
        )
        daily["social_net_sentiment"] = (
            daily["bullish"] - daily["bearish"]
        ) / (daily["labeled"] + eps)
        daily["social_sentiment_intensity"] = np.abs(
            daily["bullish"] - daily["bearish"]
        ) / (daily["labeled"] + eps)

        # v1-compatible sentiment columns.
        daily["social_sent_mean"] = daily["social_net_sentiment"]
        daily["social_sent_std"] = 0.0  # StockTwits has no per-msg scores
        daily["social_sent_min"] = np.where(
            daily["bearish"] > 0,
            -1.0,
            np.where(daily["bullish"] > 0, 1.0, 0.0),
        )
        daily["social_sent_max"] = np.where(
            daily["bullish"] > 0,
            1.0,
            np.where(daily["bearish"] > 0, -1.0, 0.0),
        )

        # Positive / negative share over ALL messages (not just labelled).
        total_plus_eps = daily["total"].astype(np.float64) + eps
        daily["social_pos_mean"] = daily["bullish"] / total_plus_eps
        daily["social_neg_mean"] = daily["bearish"] / total_plus_eps

        daily = daily.set_index("Date").sort_index()

        # Expand to full business-day grid; zero-fill matches v1.
        daily = daily.asfreq("B")

        # has_social set from the raw total (pre zero-fill).
        has_social = (daily["total"].fillna(0) > 0).astype(np.int8)
        daily = daily.fillna(0)
        daily["has_social"] = has_social

        # Rolling aggregates on the zero-filled frame.
        w = aggregation_window
        daily[_window_col("social_bullish_ratio_w{W}", w)] = (
            daily["social_bullish_ratio"].rolling(w, min_periods=1).mean()
        )
        daily[_window_col("social_net_sentiment_w{W}", w)] = (
            daily["social_net_sentiment"].rolling(w, min_periods=1).mean()
        )
        daily[_window_col("social_volume_w{W}", w)] = (
            daily["social_volume"].rolling(w, min_periods=1).sum()
        )
        daily[_window_col("social_sent_mean_w{W}", w)] = (
            daily["social_sent_mean"].rolling(w, min_periods=1).mean()
        )
        daily[_window_col("social_sent_std_w{W}", w)] = (
            daily["social_sent_std"].rolling(w, min_periods=1).mean()
        )

        daily["Ticker"] = ticker
        daily = daily.reset_index().rename(columns={"index": "Date"})
        all_chunks.append(daily)

    if not all_chunks:
        return _empty_social_frame(aggregation_window)

    out = pd.concat(all_chunks, ignore_index=True)

    # Apply log1p to volume columns when requested. This mirrors the
    # downstream scaler: we still emit the raw counts under
    # ``social_log_volume`` for inspection.
    if log1p_volume:
        out["social_volume"] = np.log1p(out["social_volume"].astype(np.float64))
        roll_vol_col = _window_col("social_volume_w{W}", aggregation_window)
        out[roll_vol_col] = np.log1p(out[roll_vol_col].astype(np.float64))
        out["social_labeled_volume"] = np.log1p(
            out["social_labeled_volume"].astype(np.float64)
        )

    feature_cols = _canonical_social_columns(aggregation_window)
    existing = [c for c in feature_cols if c in out.columns]
    out = out[["Date", "Ticker"] + existing]

    # Dtypes.
    for c in existing:
        if c == "has_social":
            out[c] = out[c].astype(np.int8)
        else:
            out[c] = out[c].astype(np.float32)

    out = out.sort_values(["Ticker", "Date"], kind="mergesort").reset_index(
        drop=True
    )

    log.debug(
        "compute_social_features: rows=%d tickers=%d window=%d log1p=%s",
        len(out),
        out["Ticker"].nunique(),
        aggregation_window,
        log1p_volume,
    )
    return out


def load_or_compute_social_features(
    stocktwits: pd.DataFrame | None = None,
    *,
    cached_path: Path = FEATURES_DIR / "social_features.parquet",
    aggregation_window: int = DEFAULT_WINDOW,
    log1p_volume: bool = True,
) -> pd.DataFrame:
    """Load cached social features or compute from raw StockTwits data."""
    if cached_path.exists():
        log.debug("Loading social features from %s", cached_path)
        return pd.read_parquet(cached_path)

    if stocktwits is None:
        raise FileNotFoundError(
            f"No cached social features at {cached_path} and no "
            "stocktwits DataFrame provided to recompute."
        )

    log.info(
        "Cache miss at %s; computing social features",
        cached_path,
    )
    return compute_social_features(
        stocktwits,
        aggregation_window=aggregation_window,
        log1p_volume=log1p_volume,
    )


def _canonical_social_columns(window: int) -> list[str]:
    """Canonical column ordering for the output parquet.

    Excludes ``Date`` / ``Ticker`` (added by the caller) but otherwise
    matches the v1 ``social_features.parquet`` schema with ``has_social``
    appended at the end.
    """
    return [
        "social_sent_mean",
        "social_sent_std",
        "social_sent_min",
        "social_sent_max",
        "social_pos_mean",
        "social_neg_mean",
        "social_volume",
        _window_col("social_sent_mean_w{W}", window),
        _window_col("social_sent_std_w{W}", window),
        _window_col("social_volume_w{W}", window),
        "social_bullish_ratio",
        "social_net_sentiment",
        "social_sentiment_intensity",
        "social_labeled_volume",
        "social_log_volume",
        _window_col("social_bullish_ratio_w{W}", window),
        _window_col("social_net_sentiment_w{W}", window),
        "has_social",
    ]


def _empty_social_frame(window: int) -> pd.DataFrame:
    cols = ["Date", "Ticker"] + _canonical_social_columns(window)
    empty = pd.DataFrame({c: pd.Series(dtype="float32") for c in cols})
    empty["Date"] = pd.to_datetime(empty["Date"])
    empty["Ticker"] = empty["Ticker"].astype("string")
    empty["has_social"] = empty["has_social"].astype("int8")
    return empty


__all__ = [
    "DEFAULT_WINDOW",
    "LOG1P_COLS",
    "PASSTHROUGH_COLS",
    "compute_social_features",
    "load_or_compute_social_features",
]
