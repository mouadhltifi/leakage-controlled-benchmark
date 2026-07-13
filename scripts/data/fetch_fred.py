#!/usr/bin/env python3
"""Re-fetch the six FRED macro series (regenerate path).

This is a thin, documented **stub** for the from-scratch reproduction path. The
artifact ALREADY ships the processed macro parquet
(``data/raw/macro/macro_indicators.parquet`` — FRED is U.S. public domain), so
you only need this if you want to refetch the raw series yourself. See
``data/README.md``.

What it does
------------
Downloads the six series used in the study from FRED, resamples them to business
days, forward-fills, and writes a date-indexed parquet to the path
``mmfp.data.paths.MACRO_PARQUET`` resolves to
(``data/raw/macro/macro_indicators.parquet``):

    FEDFUNDS  -> fed_funds_rate     (monthly average)
    DGS10     -> treasury_10y       (daily)
    CPIAUCSL  -> cpi                (monthly)
    UNRATE    -> unemployment       (monthly)
    GDP       -> gdp                (quarterly)
    VIXCLS    -> vix                (daily)

Note: this fetches the **current revised** series indexed by reference date. The
leakage-free study additionally applies realistic publication lags — see
``scripts/audits/macro_lag/apply_macro_publication_lag.py``.

Requirements
------------
A free FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html) exported
as ``FRED_API_KEY`` (never commit it), plus ``pip install fredapi``. Needs
network access.

Usage (from the artifact root)::

    export FRED_API_KEY=...        # never commit this
    python scripts/data/fetch_fred.py
    python scripts/data/fetch_fred.py --start 2015-02-03 --end 2023-12-31

This is a stub: it is import-/``--help``-clean without a key, network, or
``fredapi``, but the actual download requires all three.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Packages live under <root>/src.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

START = "2015-02-03"
END = "2023-12-31"

# FRED series id -> internal column name.
SERIES = {
    "FEDFUNDS": "fed_funds_rate",
    "DGS10": "treasury_10y",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
    "GDP": "gdp",
    "VIXCLS": "vix",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=START, help=f"Start date (default {START}).")
    p.add_argument("--end", default=END, help=f"End date (default {END}).")
    p.add_argument(
        "--out", default=None,
        help="Output parquet (default: data/raw/macro/macro_indicators.parquet).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise SystemExit(
            "set FRED_API_KEY (free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html); never commit it."
        )

    # Imports deferred so --help works without these installed.
    try:
        import pandas as pd
        from fredapi import Fred
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "fetch_fred needs pandas + fredapi: `pip install fredapi`. "
            f"(import failed: {exc})"
        )

    from mmfp.data.paths import MACRO_PARQUET

    out = Path(args.out) if args.out else MACRO_PARQUET
    out.parent.mkdir(parents=True, exist_ok=True)

    fred = Fred(api_key=api_key)  # pragma: no cover - network path
    bdays = pd.date_range(args.start, args.end, freq="B")
    cols = {}
    for sid, name in SERIES.items():  # pragma: no cover - network path
        s = fred.get_series(sid, observation_start=args.start, observation_end=args.end)
        cols[name] = s.reindex(s.index.union(bdays)).sort_index().ffill().reindex(bdays)

    pd.DataFrame(cols, index=bdays).to_parquet(out)  # pragma: no cover
    print(f"wrote {out} ({len(SERIES)} series, {args.start}..{args.end})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
