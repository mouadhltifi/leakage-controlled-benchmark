"""Filesystem paths for raw and processed data artifacts.

Paths are resolved relative to the repository root (discovered by walking up
from this file until a directory containing ``data/`` is found). Consumers that
need a custom root should override by passing explicit paths to loaders; these
constants are the defaults for the bundled data.
"""

from __future__ import annotations

from pathlib import Path


def _find_root() -> Path:
    """Walk up from this module to the repository root.

    The root is identified as the first ancestor that holds a ``data/raw``
    directory (the dataset root), which is unambiguous -- it skips this
    package's own ``mmfp/data`` subpackage. Works for both the original layout
    (``<root>/mmfp/data/paths.py``, root has ``data/raw``) and the artifact
    layout (``<root>/src/mmfp/data/paths.py``, root has ``data/raw``).
    An explicit override via the ``MMFP_DATA_ROOT`` env var wins.
    """
    import os

    env = os.environ.get("MMFP_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "raw").is_dir():
            return parent
    # fall back to the legacy assumption (two levels up).
    return here.parents[2]


#: Repository root (the directory that contains ``data/raw``).
WORK_ROOT: Path = _find_root()

#: Raw data root (upstream of feature engineering).
DATA_RAW: Path = WORK_ROOT / "data" / "raw"

#: Processed data root (downstream of feature engineering).
DATA_PROCESSED: Path = WORK_ROOT / "data" / "processed"

#: External third-party data root (large, un-redistributable).
DATA_EXTERNAL: Path = WORK_ROOT / "data" / "external"

#: Pre-computed daily per-stock feature parquets (reused across configs).
FEATURES_DIR: Path = DATA_PROCESSED / "features"

#: Pre-computed graph artifacts: static adjacency + dynamic snapshots.
GRAPHS_DIR: Path = DATA_PROCESSED / "graphs"

# ---------------------------------------------------------------------------
# Individual files referenced by raw loaders.
# ---------------------------------------------------------------------------

#: All-ticker OHLCV parquet: columns ``[Date, Open, High, Low, Close, Volume, Ticker]``.
PRICES_PARQUET: Path = DATA_RAW / "prices" / "ohlcv_all.parquet"

#: FRED indicators parquet: date-indexed float64 DataFrame.
MACRO_PARQUET: Path = DATA_RAW / "macro" / "macro_indicators.parquet"

#: FOMC dates CSV (hand-maintained).
FOMC_CSV: Path = DATA_RAW / "macro" / "fomc_dates.csv"

#: Raw FNSPID dump (very large; per-article rows).
FNSPID_CSV: Path = (
    DATA_EXTERNAL
    / "FNSPID_Financial_News_Dataset"
    / "data"
    / "nasdaq_exteral_data.csv"  # upstream typo preserved
)

#: StockTwits raw messages filtered to our 55 tickers.
STOCKTWITS_PARQUET: Path = DATA_EXTERNAL / "stocktwits" / "stocktwits_our_tickers.parquet"

#: Static same-sector adjacency (55x55, float32) — the RELEASED graph:
#: Fama-French-12 taxonomy (see DATA-STATEMENTS "Sector structure"). The
#: GICS sensitivity arm is materialized on demand via
#: ``scripts/data/kdd_sector_map.py --emit-gics``.
STATIC_ADJ_NPY: Path = GRAPHS_DIR / "sector_adjacency.npy"

#: Dynamic correlation snapshots directory (one ``graph_YYYY-MM-DD.npy`` per snapshot).
DYNAMIC_GRAPHS_DIR: Path = GRAPHS_DIR / "dynamic"

#: JSON index: date-key -> filename, written by v1 ``build_graphs.py``.
DYNAMIC_GRAPHS_INDEX: Path = DYNAMIC_GRAPHS_DIR / "index.json"

__all__ = [
    "DATA_EXTERNAL",
    "DATA_PROCESSED",
    "DATA_RAW",
    "DYNAMIC_GRAPHS_DIR",
    "DYNAMIC_GRAPHS_INDEX",
    "FEATURES_DIR",
    "FNSPID_CSV",
    "FOMC_CSV",
    "GRAPHS_DIR",
    "MACRO_PARQUET",
    "PRICES_PARQUET",
    "STATIC_ADJ_NPY",
    "STOCKTWITS_PARQUET",
    "WORK_ROOT",
]
