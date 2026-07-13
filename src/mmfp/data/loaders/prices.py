"""OHLCV price loader.

Reads the combined parquet written by the v1 ``src/data/fetch_prices.py``
pipeline. The file contains one row per ``(Date, Ticker)`` with open,
high, low, close and volume columns.

This module does *no* feature engineering (returns, z-scores, TA
indicators). It is a pure I/O shim that hands back tidy rows for the
downstream feature pipeline to consume.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mmfp.data.paths import PRICES_PARQUET
from mmfp.data.universe import ALL_TICKERS

log = logging.getLogger(__name__)

#: Columns returned by :func:`load_ohlcv`, in this order.
OHLCV_COLUMNS: list[str] = [
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def load_ohlcv(
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    end: str = "2023-12-31",
    *,
    path: Path = PRICES_PARQUET,
) -> pd.DataFrame:
    """Return OHLCV rows as a tidy DataFrame.

    Parameters
    ----------
    tickers
        Symbols to keep. ``None`` (default) returns the full 55-stock
        universe. Symbols not present in the parquet produce a warning
        but do not raise.
    start, end
        Inclusive ISO-format date bounds (``YYYY-MM-DD``). Rows outside
        this range are dropped.
    path
        Override the default parquet location. Useful in tests.

    Returns
    -------
    pandas.DataFrame
        Columns ``[Date, Ticker, Open, High, Low, Close, Volume]`` with
        ``Date`` as ``datetime64[ns]`` and ``Ticker`` as ``str``. Sorted
        by ``(Ticker, Date)`` ascending. Index is a fresh ``RangeIndex``.

    Raises
    ------
    FileNotFoundError
        If the parquet at ``path`` does not exist.

    Notes
    -----
    Upstream schema (from v1 ``fetch_prices.py``):
    ``Date:datetime, Open:float64, High:float64, Low:float64,
    Close:float64, Volume:int64, Ticker:str``. Prices are yfinance
    ``auto_adjust=True`` close values (split- and dividend-adjusted).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"OHLCV parquet not found at {path}. "
            "Run v1 ``python -m src.data.fetch_prices`` first."
        )

    wanted = list(tickers) if tickers is not None else list(ALL_TICKERS)

    log.debug("Reading %s", path)
    df = pd.read_parquet(path)

    # Robustness: columns might arrive with a different order; coerce types.
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"OHLCV parquet missing expected columns {missing}; "
            f"got {list(df.columns)}"
        )
    df = df[OHLCV_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter tickers and date range.
    available = set(df["Ticker"].unique())
    unknown = [t for t in wanted if t not in available]
    if unknown:
        log.warning(
            "load_ohlcv: %d requested tickers not in parquet (%s)",
            len(unknown),
            unknown[:10] + (["..."] if len(unknown) > 10 else []),
        )
    df = df[df["Ticker"].isin(wanted)]

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f"start ({start}) must be <= end ({end})")
    df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)]

    df = df.sort_values(["Ticker", "Date"], kind="mergesort").reset_index(drop=True)
    return df


__all__ = ["OHLCV_COLUMNS", "load_ohlcv"]
