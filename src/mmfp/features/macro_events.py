"""Macro event flags + rolling-z-score normalisation.

Thin port of v1 ``src/features/macro_features.py`` with a
``load_or_compute`` helper so callers can reuse the cached
``macro_features.parquet`` when present. The v1 pipeline produces:

* FOMC-window binary flag (``is_fomc_window``) covering ``fomc_window``
  trading days on each side of every FOMC announcement date.
* Rolling-z-score normalisation of every numeric FRED series
  (``_norm``-suffixed). Binary ``is_*`` flags pass through unchanged
  and are declared in :data:`PASSTHROUGH_COLS` so the
  :class:`FittedScaler` leaves them alone.

The study only uses ``is_fomc_window`` (CPI/NFP event flags were
scoped out). The function signature accepts an optional
``release_dates`` mapping so the platform can grow more event families
without another port.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mmfp.data.paths import FEATURES_DIR, FOMC_CSV

log = logging.getLogger(__name__)

#: Binary flags that should pass through the :class:`FittedScaler`
#: unchanged (don't z-score a 0/1 indicator).
PASSTHROUGH_COLS: list[str] = ["is_fomc_window"]


def load_fomc_dates(
    *, path: Path = FOMC_CSV,
) -> pd.Series:
    """Return the hand-maintained list of FOMC meeting dates.

    Parameters
    ----------
    path
        CSV with columns ``date`` and ``is_fomc`` (True/False). Defaults
        to ``data/raw/macro/fomc_dates.csv``.

    Returns
    -------
    pandas.Series
        ``datetime64[ns]`` values, sorted ascending, duplicates removed.
    """
    if not path.exists():
        raise FileNotFoundError(f"FOMC dates CSV not found at {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    if "is_fomc" in df.columns:
        df = df[df["is_fomc"].astype(bool)]
    dates = (
        pd.to_datetime(df["date"]).dropna().drop_duplicates().sort_values()
    )
    dates.name = "Date"
    return dates.reset_index(drop=True)


def create_event_features(
    macro_raw: pd.DataFrame,
    *,
    fomc_dates: pd.Series | None = None,
    fomc_window_days: int = 1,
) -> pd.DataFrame:
    """Add ``is_fomc_window`` (and future event flags) to macro rows.

    Parameters
    ----------
    macro_raw
        Date-indexed macro DataFrame from
        :func:`mmfp.data.loaders.load_macro_raw`.
    fomc_dates
        Ordered series of FOMC announcement dates. Pass ``None`` to
        load from the default CSV.
    fomc_window_days
        Number of trading days on each side of an FOMC date to flag.
        ``1`` (default) flags the announcement day, the day before, and
        the day after.

    Returns
    -------
    pandas.DataFrame
        ``macro_raw`` with an ``is_fomc_window`` int8 column appended.
        ``0`` outside the window, ``1`` inside.

    Notes
    -----
    The window uses calendar days, not trading days, for parity with
    v1. Since ``macro_raw`` is already on a business-day index, calendar
    windows and trading windows line up for ``fomc_window_days`` up to
    ~3. Tighter windows would require a true trading-day walker; we
    have no research reason to change defaults.
    """
    if fomc_window_days < 0:
        raise ValueError(
            f"fomc_window_days must be >= 0; got {fomc_window_days}"
        )

    if fomc_dates is None:
        fomc_dates = load_fomc_dates()

    df = macro_raw.copy()
    # Ensure we have a DatetimeIndex to reason about.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    fomc_lookup = set(pd.Timestamp(d).normalize() for d in fomc_dates)

    flag = np.zeros(len(df), dtype=np.int8)
    for i, ts in enumerate(df.index):
        ts0 = pd.Timestamp(ts).normalize()
        for offset in range(-fomc_window_days, fomc_window_days + 1):
            if (ts0 + pd.Timedelta(days=offset)) in fomc_lookup:
                flag[i] = 1
                break

    df["is_fomc_window"] = flag
    return df


def normalize_macro_features(
    macro_with_events: pd.DataFrame,
    *,
    lookback: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Rolling-z-score numeric macro series into ``_norm`` columns.

    Passthrough columns (``is_*``) are left alone.
    """
    df = macro_with_events.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in PASSTHROUGH_COLS:
            continue
        if col.startswith("is_"):
            continue
        rolling_mean = df[col].rolling(lookback, min_periods=min_periods).mean()
        rolling_std = df[col].rolling(lookback, min_periods=min_periods).std()
        df[f"{col}_norm"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    return df


def compute_macro_features(
    macro_raw: pd.DataFrame,
    *,
    fomc_dates: pd.Series | None = None,
    fomc_window_days: int = 1,
    lookback: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Full macro feature pipeline: events + rolling z-score.

    Equivalent to calling :func:`create_event_features` followed by
    :func:`normalize_macro_features`; emits the canonical
    ``macro_features.parquet`` schema.
    """
    with_events = create_event_features(
        macro_raw,
        fomc_dates=fomc_dates,
        fomc_window_days=fomc_window_days,
    )
    return normalize_macro_features(
        with_events, lookback=lookback, min_periods=min_periods
    )


def load_or_compute_macro_features(
    macro_raw: pd.DataFrame | None = None,
    *,
    cached_path: Path = FEATURES_DIR / "macro_features.parquet",
    fomc_dates: pd.Series | None = None,
    fomc_window_days: int = 1,
    lookback: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Load cached macro features or compute from raw FRED series."""
    if cached_path.exists():
        log.debug("Loading macro features from %s", cached_path)
        df = pd.read_parquet(cached_path)
        # Preserve the upstream index name if present.
        return df

    if macro_raw is None:
        raise FileNotFoundError(
            f"No cached macro features at {cached_path} and no macro_raw "
            "provided to recompute."
        )

    log.info(
        "Cache miss at %s; computing macro features",
        cached_path,
    )
    return compute_macro_features(
        macro_raw,
        fomc_dates=fomc_dates,
        fomc_window_days=fomc_window_days,
        lookback=lookback,
        min_periods=min_periods,
    )


__all__ = [
    "PASSTHROUGH_COLS",
    "compute_macro_features",
    "create_event_features",
    "load_fomc_dates",
    "load_or_compute_macro_features",
    "normalize_macro_features",
]
