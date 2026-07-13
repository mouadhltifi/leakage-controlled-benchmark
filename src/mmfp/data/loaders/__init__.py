"""One loader per raw data source.

Milestone 2 deliverable. See spec Section 3.2.

Public entry points
-------------------

* :func:`load_ohlcv` — yfinance-provenance OHLCV parquet.
* :func:`load_macro_raw` — FRED macro indicator parquet.
* :func:`load_fnspid` — FNSPID news CSV (chunked).
* :func:`load_stocktwits` — StockTwits per-message parquet.
* :func:`build_static_adjacency` / :func:`load_static_adjacency` — GICS
  sector co-membership adjacency.
* :func:`load_dynamic_snapshots` /
  :func:`build_dynamic_snapshots_from_returns` — rolling-correlation
  graph snapshots.
"""

from mmfp.data.loaders.graph_dynamic import (
    build_dynamic_snapshots_from_returns,
    load_dynamic_snapshots,
    pick_snapshot_for_date,
)
from mmfp.data.loaders.graph_static import (
    build_static_adjacency,
    load_static_adjacency,
)
from mmfp.data.loaders.macro import load_macro_raw
from mmfp.data.loaders.news_raw import load_fnspid
from mmfp.data.loaders.prices import load_ohlcv
from mmfp.data.loaders.social_raw import load_stocktwits

__all__ = [
    "build_dynamic_snapshots_from_returns",
    "build_static_adjacency",
    "load_dynamic_snapshots",
    "load_fnspid",
    "load_macro_raw",
    "load_ohlcv",
    "load_static_adjacency",
    "load_stocktwits",
    "pick_snapshot_for_date",
]
