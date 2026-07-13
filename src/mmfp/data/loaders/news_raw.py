"""FNSPID news article loader.

Reads the FNSPID Financial News Dataset CSV (~25M rows on disk) in
chunks, filters to the study's 55-stock universe while normalising
ticker aliases (``FB`` -> ``META`` etc.), and returns a tidy DataFrame
of per-article rows.

This is a pure I/O shim. Feature extraction (FinBERT sentiment,
embeddings, per-day aggregation) is the job of
:mod:`mmfp.features.news_encode` / :mod:`mmfp.features.news_aggregate`
(Milestone 3).

For most downstream use the cached parquets
``news_per_article_sentiments.parquet`` and
``news_per_article_768.parquet`` should be preferred — those are
already aligned, typed, and include the encoder outputs. This raw
loader is here for rebuilding those caches or running a new encoder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mmfp.data.paths import FNSPID_CSV
from mmfp.data.universe import ALL_TICKERS, TICKER_ALIASES

log = logging.getLogger(__name__)

#: Columns read from the FNSPID CSV (ignoring many unused fields to save memory).
_FNSPID_READ_COLUMNS: list[str] = [
    "Date",
    "Article_title",
    "Stock_symbol",
    "Article",
]

#: Columns returned by :func:`load_fnspid`.
FNSPID_OUTPUT_COLUMNS: list[str] = [
    "Date",
    "Ticker",
    "Title",
    "Content",
]


def load_fnspid(
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    end: str = "2023-12-31",
    chunk_size: int = 100_000,
    *,
    path: Path = FNSPID_CSV,
) -> pd.DataFrame:
    """Return FNSPID articles for the selected tickers/date range.

    Parameters
    ----------
    tickers
        Canonical study tickers to keep (e.g. ``"META"``). Aliases in
        the CSV (e.g. ``"FB"``) are normalised to canonical form before
        filtering. ``None`` = full 55-stock universe.
    start, end
        Inclusive ISO-format date bounds. Rows outside are dropped.
    chunk_size
        Rows per ``pandas.read_csv`` chunk. FNSPID is large (~25 M rows)
        so streaming keeps peak memory under ~2 GB.
    path
        Override the default CSV location. Useful in tests.

    Returns
    -------
    pandas.DataFrame
        Columns ``[Date, Ticker, Title, Content]`` (order preserved).

        * ``Date`` — ``datetime64[ns, UTC]`` (FNSPID timestamps are UTC).
        * ``Ticker`` — canonical study symbol (aliases resolved).
        * ``Title`` — article headline (``str``).
        * ``Content`` — article body (``str``; may be empty).

        Rows with unparseable dates are dropped. Index is a fresh
        ``RangeIndex``.

    Raises
    ------
    FileNotFoundError
        If the FNSPID CSV is not present at ``path``.
    ValueError
        If ``start > end``.

    Notes
    -----
    **Aliases.** ``TICKER_ALIASES`` maps non-canonical symbols
    (``GOOG``, ``BRK.B``, ``FB``, ``UTX``, ``RTN``) to the study's
    canonical ticker. Both the alias and the canonical form are
    accepted in the CSV, then every hit is renamed to canonical.

    **Caveats.** The upstream CSV has a filename typo
    (``nasdaq_exteral_data.csv``, missing an ``n``); we preserve it.
    Article titles are the primary signal per v1 analysis — the
    ``Content`` body may be truncated or null in older rows.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"FNSPID CSV not found at {path}. "
            "See work/data/external/FNSPID_Financial_News_Dataset/README.md."
        )

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if start_ts > end_ts:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    wanted = list(tickers) if tickers is not None else list(ALL_TICKERS)
    wanted_set = set(wanted)

    # Union of canonical tickers + any aliases whose canonical form is wanted.
    accept_symbols = set(wanted_set)
    for alias, canonical in TICKER_ALIASES.items():
        if canonical in wanted_set:
            accept_symbols.add(alias)

    log.info(
        "Streaming FNSPID from %s (chunk_size=%d, %d symbols)",
        path,
        chunk_size,
        len(accept_symbols),
    )

    chunks: list[pd.DataFrame] = []
    n_read = 0
    n_matched = 0

    reader = pd.read_csv(
        path,
        usecols=_FNSPID_READ_COLUMNS,
        dtype={
            "Stock_symbol": "str",
            "Article_title": "str",
            "Article": "str",
        },
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in reader:
        n_read += len(chunk)
        mask = chunk["Stock_symbol"].isin(accept_symbols)
        if not mask.any():
            continue
        hit = chunk.loc[mask].copy()

        # Parse dates (UTC; drop unparseable rows).
        hit["Date"] = pd.to_datetime(hit["Date"], errors="coerce", utc=True)
        hit = hit.dropna(subset=["Date"])
        if hit.empty:
            continue

        # Apply canonical ticker mapping.
        hit["Stock_symbol"] = hit["Stock_symbol"].replace(TICKER_ALIASES)

        hit = hit[(hit["Date"] >= start_ts) & (hit["Date"] <= end_ts)]
        if hit.empty:
            continue

        chunks.append(hit)
        n_matched += len(hit)

    log.info("FNSPID: read %d rows, matched %d for our tickers", n_read, n_matched)

    if not chunks:
        # Return an empty DataFrame with the canonical schema so callers
        # can concat/merge without special-casing.
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(pd.Series(dtype="object"), utc=True),
                "Ticker": pd.Series(dtype="str"),
                "Title": pd.Series(dtype="str"),
                "Content": pd.Series(dtype="str"),
            }
        )

    out = pd.concat(chunks, ignore_index=True)
    out = out.rename(
        columns={
            "Stock_symbol": "Ticker",
            "Article_title": "Title",
            "Article": "Content",
        }
    )
    out = out[FNSPID_OUTPUT_COLUMNS]
    out = out.sort_values(["Ticker", "Date"], kind="mergesort").reset_index(drop=True)
    return out


__all__ = ["FNSPID_OUTPUT_COLUMNS", "load_fnspid"]
