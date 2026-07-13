"""FRED macroeconomic indicators loader.

Loads the parquet written by the v1 ``src/data/fetch_macro.py`` pipeline.
The file holds the six FRED series used by the study (Fed Funds Rate,
10Y Treasury, CPI, Unemployment, GDP, VIX) resampled to business-day
frequency and forward-filled so downstream code can align to any
trading day without extra logic.

This module does *no* feature engineering (rolling z-scores, FOMC
flags). Those live in ``mmfp.features.macro_events`` / ``macro_features``
(Milestone 3).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mmfp.data.paths import MACRO_PARQUET
from mmfp.data.universe import FRED_SERIES

log = logging.getLogger(__name__)


def load_macro_raw(
    start: str = "2015-01-01",
    end: str = "2023-12-31",
    *,
    path: Path = MACRO_PARQUET,
) -> pd.DataFrame:
    """Return the FRED macro series as a date-indexed DataFrame.

    Parameters
    ----------
    start, end
        Inclusive ISO-format date bounds. Rows outside this range are
        dropped.
    path
        Override the default parquet location. Useful in tests.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``Date`` (``datetime64[ns]``). Columns are the
        human-readable FRED series names:
        ``fed_funds_rate, treasury_10y, cpi, unemployment, gdp, vix``.
        Values are ``float64``; ``NaN`` indicates the upstream FRED
        release had not yet happened on that business day (the v1
        pipeline already forward-fills, so this mostly appears in the
        warm-up window).

    Raises
    ------
    FileNotFoundError
        If the parquet does not exist.
    ValueError
        If the parquet lacks any of the expected FRED columns.

    Notes
    -----
    Upstream preparation (``src/data/fetch_macro.py``):

    * business-day resampling via ``asfreq("B")``
    * forward-fill for monthly/quarterly series (CPI, UNRATE, GDP)
    * Fed Funds and VIX arrive daily; 10Y Treasury is business-day

    FOMC-event metadata lives in a separate CSV (``fomc_dates.csv``) and
    is consumed by :mod:`mmfp.features.macro_events` in M3.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Macro parquet not found at {path}. "
            "Run v1 ``python -m src.data.fetch_macro`` first."
        )

    log.debug("Reading %s", path)
    df = pd.read_parquet(path)

    # Expect FRED series names as columns, Date as index.
    expected = set(FRED_SERIES.keys())
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Macro parquet missing expected FRED columns {sorted(missing)}; "
            f"got {sorted(df.columns)}"
        )

    # Ensure the index is Date-typed.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f"start ({start}) must be <= end ({end})")
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    # Stable column ordering for downstream consumers.
    df = df[list(FRED_SERIES.keys())].copy()
    df = df.sort_index(kind="mergesort")
    return df


__all__ = ["load_macro_raw"]
