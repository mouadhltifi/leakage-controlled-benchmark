"""StockTwits raw message loader.

StockTwits supplies per-message Bullish/Bearish labels for a subset of
messages; the study uses these as a cheaper substitute for running
FinBERT over 14.6 M tweets. This loader reads the pre-filtered parquet
written by the study downloader (``work/data/external/stocktwits/``)
and hands back a tidy DataFrame.

No feature engineering happens here. Daily aggregation (bullish ratio,
net sentiment, volume, rolling windows) is the job of
:mod:`mmfp.features.social_features` (Milestone 3).

The downstream cached parquet ``social_features.parquet`` already
contains the aggregated daily features and should be preferred for
modelling. This loader is here for auditability and re-aggregation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mmfp.data.paths import STOCKTWITS_PARQUET
from mmfp.data.universe import ALL_TICKERS

log = logging.getLogger(__name__)

#: Columns returned by :func:`load_stocktwits`, in order.
STOCKTWITS_OUTPUT_COLUMNS: list[str] = [
    "Date",
    "Ticker",
    "message_id",
    "user_id",
    "sentiment",
]


def load_stocktwits(
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    end: str = "2023-12-31",
    *,
    path: Path = STOCKTWITS_PARQUET,
) -> pd.DataFrame:
    """Return StockTwits per-message rows for the selected tickers/dates.

    Parameters
    ----------
    tickers
        Canonical study tickers to keep. ``None`` = all 55.
    start, end
        Inclusive ISO-format date bounds.
    path
        Override the default parquet location.

    Returns
    -------
    pandas.DataFrame
        Columns ``[Date, Ticker, message_id, user_id, sentiment]``.

        * ``Date`` — ``datetime64[ns]`` (midnight of the trading day).
        * ``Ticker`` — canonical 55-universe symbol.
        * ``message_id`` — StockTwits primary key (``int64``).
        * ``user_id`` — posting user id (``int64``).
        * ``sentiment`` — ``+1.0`` (bullish), ``-1.0`` (bearish), or
          ``NaN`` (unlabelled; ~80 % of messages).

        Rows are sorted by ``(Ticker, Date, message_id)`` and the index
        is a fresh ``RangeIndex``.

    Raises
    ------
    FileNotFoundError
        If the parquet does not exist.
    ValueError
        If ``start > end`` or the parquet lacks expected columns.

    Notes
    -----
    Upstream schema (``stocktwits_our_tickers.parquet``):
    ``message_id:int64, user_id:int64, created_at:str, sentiment:float64,
    symbol_list:str, sym_number:int64, symbol:str, ticker:str``.
    This loader renames ``created_at`` -> ``Date`` and ``ticker`` ->
    ``Ticker`` to match platform conventions. ``created_at`` is a date
    only (no time), so we parse it at midnight in naive local time
    (consistent with the v1 pipeline).

    Coverage: 14.6 M messages, 55/55 tickers, 2008-2022. Messages after
    2022 exist but are sparse.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"StockTwits parquet not found at {path}. "
            "Expected upstream download at work/data/external/stocktwits/."
        )

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    wanted = list(tickers) if tickers is not None else list(ALL_TICKERS)

    log.debug("Reading %s", path)
    df = pd.read_parquet(
        path,
        columns=["message_id", "user_id", "created_at", "sentiment", "ticker"],
    )

    # Schema sanity check.
    expected = {"message_id", "user_id", "created_at", "sentiment", "ticker"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"StockTwits parquet missing columns {sorted(missing)}; "
            f"got {sorted(df.columns)}"
        )

    # Canonicalize types & names.
    df = df.rename(columns={"created_at": "Date", "ticker": "Ticker"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Filter universe and dates.
    df = df[df["Ticker"].isin(wanted)]
    df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)]

    df = df[STOCKTWITS_OUTPUT_COLUMNS]
    df = df.sort_values(
        ["Ticker", "Date", "message_id"], kind="mergesort"
    ).reset_index(drop=True)
    return df


__all__ = ["STOCKTWITS_OUTPUT_COLUMNS", "load_stocktwits"]
