"""v3 windowed multimodal dataset + graph precompute (M4).

Exports:

* :class:`~forecast.datasets.multimodal.ForecastDataset` — windowed
  variant that emits ``(lookback, F_m)`` sequences per active modality
  plus static categorical ids (ticker, sector) and scalar targets.
* :class:`~forecast.datasets.graph_precompute.GraphNodeCache` +
  :func:`~forecast.datasets.graph_precompute.build_graph_node_cache` —
  per-fold GAT node embedding precompute (the architecture spec).
"""

from __future__ import annotations

from forecast.datasets.graph_precompute import (
    GraphNodeCache,
    build_graph_node_cache,
)
from forecast.datasets.multimodal import ForecastDataset, Split

__all__ = [
    "ForecastDataset",
    "GraphNodeCache",
    "Split",
    "build_graph_node_cache",
]
