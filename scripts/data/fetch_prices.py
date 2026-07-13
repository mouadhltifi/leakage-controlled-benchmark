#!/usr/bin/env python3
"""Fetch raw Yahoo Finance OHLCV for the 55-stock universe (regenerate path).

This is a thin, documented **stub** for the from-scratch reproduction path. The
artifact does NOT ship raw prices (Yahoo terms: link, don't redistribute), and
the headline numbers do not need them — see ``data/README.md``. Run this only if
you want to regenerate the price feature parquet from scratch.

What it does
------------
Downloads daily OHLCV for every ticker in ``mmfp.data.universe.ALL_TICKERS``
over the study window (2015-02-03 .. 2023-12-31, ``auto_adjust=True``) and writes
a single tidy parquet with columns ``[Date, Open, High, Low, Close, Volume,
Ticker]`` to ``data/raw/prices/ohlcv_all.parquet`` (the path
``mmfp.data.paths.PRICES_PARQUET`` resolves to). The price feature pipeline in
``src/mmfp/features/`` then consumes that file.

Requirements
------------
``pip install yfinance`` (kept out of the core ``requirements.txt`` because the
headline path does not need it). Needs network access.

Usage (from the artifact root)::

    python scripts/data/fetch_prices.py                 # full universe + window
    python scripts/data/fetch_prices.py --tickers AAPL MSFT --start 2020-01-01

This is a stub: it is import-/``--help``-clean without network or yfinance, but
the actual download requires both.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Packages live under <root>/src.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

START = "2015-02-03"
END = "2023-12-31"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers to fetch (default: the full 55-stock universe).",
    )
    p.add_argument("--start", default=START, help=f"Start date (default {START}).")
    p.add_argument("--end", default=END, help=f"End date (default {END}).")
    p.add_argument(
        "--out", default=None,
        help="Output parquet (default: data/raw/prices/ohlcv_all.parquet).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Imports deferred so --help works without these installed.
    try:
        import pandas as pd  # noqa: F401
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "fetch_prices needs pandas + yfinance: `pip install yfinance`. "
            f"(import failed: {exc})"
        )

    from mmfp.data.paths import PRICES_PARQUET
    from mmfp.data.universe import ALL_TICKERS

    tickers = args.tickers or list(ALL_TICKERS)
    out = Path(args.out) if args.out else PRICES_PARQUET
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for tk in tickers:  # pragma: no cover - network path
        df = yf.download(
            tk, start=args.start, end=args.end, auto_adjust=True, progress=False
        )
        if df.empty:
            print(f"[warn] no data for {tk}", file=sys.stderr)
            continue
        df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df["Ticker"] = tk
        frames.append(df)

    if not frames:  # pragma: no cover
        raise SystemExit("no tickers fetched")

    import pandas as pd

    pd.concat(frames, ignore_index=True).to_parquet(out)
    print(f"wrote {out} ({len(tickers)} tickers, {args.start}..{args.end})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
