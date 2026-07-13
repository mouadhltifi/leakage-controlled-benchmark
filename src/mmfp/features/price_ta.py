"""Technical-analysis indicators + rolling-z-score normalisation.

Thin port of v1 ``src/features/price_features.py`` with a
``load_or_compute`` helper so callers can reuse the cached
``price_features.parquet`` when the ticker universe and date range
match, and regenerate from raw OHLCV otherwise.

Features (match v1 exactly):

* Returns: ``return_1d``, ``return_5d``, ``return_20d``
* Volatility: ``volatility_20d`` (20-day rolling std of daily returns)
* Momentum: ``rsi_14``
* Trend: ``ema_ratio`` (EMA12/EMA26), ``macd_signal``
* Bands: ``bollinger_pctb``
* Range: ``atr_14``
* Volume: ``volume_ratio`` (volume / 20-day SMA volume)

All listed features are also z-scored with a 252-day rolling window
(min 60 day warm-up) into the ``_norm``-suffixed columns. These are
the columns consumed by the price encoder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mmfp.data.paths import FEATURES_DIR

log = logging.getLogger(__name__)

#: Raw feature columns (pre-normalisation). Matches v1
#: ``price_features.FEATURE_COLS`` for drop-in compatibility.
FEATURE_COLS: list[str] = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "rsi_14",
    "ema_ratio",
    "macd_signal",
    "bollinger_pctb",
    "atr_14",
    "volume_ratio",
]

#: Z-scored feature columns used downstream by the price encoder.
NORMALISED_FEATURE_COLS: list[str] = [f"{c}_norm" for c in FEATURE_COLS]


def compute_price_features_single_ticker(
    ohlcv: pd.DataFrame,
    *,
    rolling_window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Compute TA indicators + rolling z-scores for a single ticker.

    Parameters
    ----------
    ohlcv
        ``DataFrame`` with columns ``[Date, Open, High, Low, Close,
        Volume]`` for one ticker, sorted by ``Date``.
    rolling_window
        Z-score look-back (default 252 ~= one trading year). 252 keeps
        rolling stats stable while acknowledging regime drift.
    min_periods
        Minimum observations before emitting a rolling stat. Default 60
        matches v1.

    Returns
    -------
    pandas.DataFrame
        ``ohlcv`` columns + :data:`FEATURE_COLS` +
        :data:`NORMALISED_FEATURE_COLS`. The warm-up rows carry ``NaN``
        for any feature whose window is not yet full; downstream fold
        assembly slices past those.
    """
    try:
        import ta  # third-party TA library
    except ImportError as exc:  # pragma: no cover - clear error for dev setup
        raise ImportError(
            "The `ta` package is required for price feature computation. "
            "Install via `pip install ta>=0.11`."
        ) from exc

    df = ohlcv.copy().sort_values("Date").reset_index(drop=True)

    # Raw returns.
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # Rolling volatility on daily returns.
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # Technical indicators.
    df["rsi_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    ema12 = ta.trend.EMAIndicator(df["Close"], window=12).ema_indicator()
    ema26 = ta.trend.EMAIndicator(df["Close"], window=26).ema_indicator()
    df["ema_ratio"] = ema12 / ema26

    macd = ta.trend.MACD(
        df["Close"], window_slow=26, window_fast=12, window_sign=9
    )
    df["macd_signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(
        df["Close"], window=20, window_dev=2
    )
    df["bollinger_pctb"] = bb.bollinger_pband()

    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=14
    ).average_true_range()

    vol_sma = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / vol_sma

    # Rolling z-score normalisation.
    for col in FEATURE_COLS:
        rolling_mean = df[col].rolling(
            rolling_window, min_periods=min_periods
        ).mean()
        rolling_std = df[col].rolling(
            rolling_window, min_periods=min_periods
        ).std()
        df[f"{col}_norm"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    return df


def compute_price_features(
    ohlcv: pd.DataFrame,
    *,
    rolling_window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Compute price features for every ticker in ``ohlcv``.

    Parameters
    ----------
    ohlcv
        Tidy ``DataFrame`` from :func:`mmfp.data.loaders.load_ohlcv` with
        columns ``[Date, Ticker, Open, High, Low, Close, Volume]``.
    rolling_window, min_periods
        Forwarded to :func:`compute_price_features_single_ticker`.

    Returns
    -------
    pandas.DataFrame
        Per-(Ticker, Date) rows with OHLCV + raw TA features + z-scored
        features, sorted by ``(Ticker, Date)`` ascending.
    """
    if "Ticker" not in ohlcv.columns:
        raise ValueError(
            "ohlcv must have a 'Ticker' column (use load_ohlcv)."
        )

    chunks: list[pd.DataFrame] = []
    for ticker, grp in ohlcv.groupby("Ticker", sort=True):
        per_ticker = compute_price_features_single_ticker(
            grp,
            rolling_window=rolling_window,
            min_periods=min_periods,
        )
        per_ticker["Ticker"] = ticker
        chunks.append(per_ticker)

    out = pd.concat(chunks, ignore_index=True)
    # Guarantee a canonical column order: OHLCV, Ticker, features, norms.
    ordered_cols = (
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Ticker",
        ]
        + FEATURE_COLS
        + NORMALISED_FEATURE_COLS
    )
    existing = [c for c in ordered_cols if c in out.columns]
    extras = [c for c in out.columns if c not in existing]
    out = out[existing + extras]
    return out.sort_values(["Ticker", "Date"], kind="mergesort").reset_index(
        drop=True
    )


def load_or_compute_price_features(
    ohlcv: pd.DataFrame | None = None,
    *,
    cached_path: Path = FEATURES_DIR / "price_features.parquet",
    rolling_window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Load cached price features or compute them from OHLCV.

    Parameters
    ----------
    ohlcv
        If provided AND the cache is missing, compute from this. If
        both the cache AND ``ohlcv`` are provided, the cache wins; no
        recomputation happens (this keeps runs deterministic). Pass
        ``None`` to require the cache.
    cached_path
        Default: ``data/processed/features/price_features.parquet``.
    rolling_window, min_periods
        Forwarded to :func:`compute_price_features` when recomputing.

    Raises
    ------
    FileNotFoundError
        If ``cached_path`` is missing AND ``ohlcv`` is ``None``.
    """
    if cached_path.exists():
        log.debug("Loading price features from %s", cached_path)
        df = pd.read_parquet(cached_path)
        # Stable column ordering for downstream consumers.
        return df.sort_values(["Ticker", "Date"], kind="mergesort").reset_index(
            drop=True
        )

    if ohlcv is None:
        raise FileNotFoundError(
            f"No cached price features at {cached_path} and no OHLCV "
            "provided to recompute."
        )

    log.info(
        "Cache miss at %s; computing price features from OHLCV",
        cached_path,
    )
    return compute_price_features(
        ohlcv, rolling_window=rolling_window, min_periods=min_periods
    )


__all__ = [
    "FEATURE_COLS",
    "NORMALISED_FEATURE_COLS",
    "compute_price_features",
    "compute_price_features_single_ticker",
    "load_or_compute_price_features",
]
