"""Canonical stock universe, sector map, and FRED series IDs.

These constants are the single source of truth for the 55-stock universe
used throughout the study (2015-2023, 11 GICS sectors, 5 stocks each).

Duplicating them here (rather than importing from the legacy config module)
keeps the new ``mmfp`` package decoupled from the legacy ``src`` tree so
that ``src`` can be archived after the equivalence gate without breaking
loaders. The values are copy-equivalent to v1; tests in
:mod:`mmfp.tests.unit.test_loaders` compare them against the v1 list when
``src`` is still on ``PYTHONPATH``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stock universe: 11 GICS sectors x 5 stocks = 55 symbols.
# ---------------------------------------------------------------------------

STOCK_UNIVERSE: dict[str, list[str]] = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL"],
    "Health Care": ["UNH", "JNJ", "LLY", "ABBV", "MRK"],
    "Financials": ["BRK-B", "JPM", "V", "MA", "BAC"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA"],
    "Industrials": ["GE", "CAT", "UNP", "RTX", "HON"],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Utilities": ["NEE", "SO", "DUK", "SRE", "AEP"],
    "Materials": ["LIN", "APD", "SHW", "FCX", "NEM"],
    "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "SPG"],
}

#: Canonical ticker list, sorted alphabetically (stable across runs).
ALL_TICKERS: list[str] = sorted(
    t for tickers in STOCK_UNIVERSE.values() for t in tickers
)

#: Ticker -> GICS sector lookup.
TICKER_TO_SECTOR: dict[str, str] = {
    t: sector for sector, tickers in STOCK_UNIVERSE.items() for t in tickers
}

#: Number of stocks (used for adjacency matrix shapes).
N_STOCKS: int = len(ALL_TICKERS)


# ---------------------------------------------------------------------------
# FRED macro series (ported from v1 ``src/utils/config.py``).
# ---------------------------------------------------------------------------

FRED_SERIES: dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "treasury_10y": "DGS10",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "gdp": "GDP",
    "vix": "VIXCLS",
}


# ---------------------------------------------------------------------------
# FNSPID ticker aliases (ported from v1 ``src/features/encode_news.py``).
# ---------------------------------------------------------------------------
#: Non-canonical symbols that appear in FNSPID and map to one of our 55 tickers.
#:
#: * ``GOOG`` is Alphabet Class C shares (we use Class A, ``GOOGL``).
#: * ``BRK.B`` is Berkshire Hathaway Class B with a dot separator (we use ``BRK-B``).
#: * ``FB`` was Meta's pre-2021 ticker.
#: * ``UTX``/``RTN`` pre-dated the 2020 Raytheon-UTC merger which produced ``RTX``.
TICKER_ALIASES: dict[str, str] = {
    "GOOG": "GOOGL",
    "BRK.B": "BRK-B",
    "FB": "META",
    "UTX": "RTX",
    "RTN": "RTX",
}

__all__ = [
    "ALL_TICKERS",
    "FRED_SERIES",
    "N_STOCKS",
    "STOCK_UNIVERSE",
    "TICKER_ALIASES",
    "TICKER_TO_SECTOR",
]
