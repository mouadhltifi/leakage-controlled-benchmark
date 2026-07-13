"""Dynamic correlation graph snapshot loader.

The v1 pipeline (``src/data/build_graphs.py``) computes rolling-window
return-correlation graphs every 20 trading days and persists each
snapshot as a separate ``graph_YYYY-MM-DD.npy`` under
``work/data/processed/graphs/dynamic/`` with a JSON index mapping
ISO-date keys to file names. This loader reads them back into memory.

For datasets where the snapshots have not been written, callers can
fall back to :func:`build_dynamic_snapshots_from_returns`, which ports
the v1 computation (cosine-similarity thresholded) to the platform.

This module does *no* feature engineering. Graph features (node
attributes, edge indices consumed by PyG) are produced at
fold-assembly time in Milestone 4.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from mmfp.data.paths import DYNAMIC_GRAPHS_DIR, DYNAMIC_GRAPHS_INDEX

log = logging.getLogger(__name__)


def load_dynamic_snapshots(
    index_path: Path = DYNAMIC_GRAPHS_INDEX,
    *,
    graphs_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Load all pre-computed dynamic graph snapshots from disk.

    Parameters
    ----------
    index_path
        Override the default ``index.json`` location.
    graphs_dir
        Directory holding the per-snapshot ``.npy`` files. Defaults to
        the directory containing ``index_path``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping of ISO date string (``"YYYY-MM-DD"``) to a
        ``(N, N)`` float32 adjacency matrix. For dates not in the map,
        callers should use the most recent ``<= date`` snapshot.

    Raises
    ------
    FileNotFoundError
        If ``index_path`` does not exist or a referenced ``.npy`` is
        missing.
    ValueError
        If the index is empty or loaded matrices disagree in shape.
    """
    if not index_path.exists():
        raise FileNotFoundError(
            f"Dynamic graphs index not found at {index_path}. "
            "Run v1 ``python -m src.data.build_graphs`` to regenerate."
        )

    graphs_dir = graphs_dir or index_path.parent

    log.debug("Reading %s", index_path)
    with index_path.open("r") as fh:
        index: dict[str, str] = json.load(fh)

    if not index:
        raise ValueError(f"Dynamic graphs index {index_path} is empty")

    snapshots: dict[str, np.ndarray] = {}
    first_shape: tuple[int, int] | None = None
    for date_key, filename in index.items():
        npy_path = graphs_dir / filename
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Dynamic graph file missing: {npy_path} "
                f"(referenced by {index_path} for date {date_key})"
            )
        mat = np.load(npy_path).astype(np.float32, copy=False)
        if first_shape is None:
            first_shape = mat.shape
        elif mat.shape != first_shape:
            raise ValueError(
                f"Shape mismatch in dynamic graphs: {date_key} has "
                f"{mat.shape}, expected {first_shape}"
            )
        snapshots[date_key] = mat

    log.info(
        "Loaded %d dynamic graph snapshots from %s",
        len(snapshots),
        graphs_dir,
    )
    return snapshots


def pick_snapshot_for_date(
    target_date: str,
    snapshots: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Return the most recent snapshot on or before ``target_date``.

    Parameters
    ----------
    target_date
        ISO date string (``YYYY-MM-DD``).
    snapshots
        As returned by :func:`load_dynamic_snapshots`.

    Returns
    -------
    numpy.ndarray | None
        The adjacency matrix for the latest ``<= target_date`` snapshot,
        or ``None`` if no snapshot precedes the target date.
    """
    keys = sorted(snapshots.keys())
    best: str | None = None
    for k in keys:
        if k <= target_date:
            best = k
        else:
            break
    if best is None:
        return None
    return snapshots[best]


def build_dynamic_snapshots_from_returns(
    returns: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.3,
) -> dict[str, np.ndarray]:
    """Compute dynamic graph snapshots from a returns DataFrame.

    Port of v1 ``src/features/graph_features.py::build_dynamic_graph``
    for auditability. Consumers normally prefer
    :func:`load_dynamic_snapshots` (reads the cached files).

    Parameters
    ----------
    returns
        Date-indexed DataFrame with one column per ticker (stable
        ordering defines row/column order in the output matrices).
        Must have no ``NaN`` (drop or fill before calling).
    window
        Rolling window length in rows. Snapshots are produced every
        ``window`` rows.
    threshold
        Cosine-similarity threshold for edge creation. Edges are
        weighted by similarity when above the threshold; otherwise
        zero.

    Returns
    -------
    dict[str, numpy.ndarray]
        Same schema as :func:`load_dynamic_snapshots`.

    Raises
    ------
    ValueError
        If ``window < 2`` or ``returns`` is empty.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if returns.empty:
        raise ValueError("returns DataFrame is empty")

    n = returns.shape[1]
    dates = returns.index
    snapshots: dict[str, np.ndarray] = {}

    for start_idx in range(window, len(dates), window):
        end_idx = start_idx
        date_key = str(pd.Timestamp(dates[end_idx]).date())
        window_returns = returns.iloc[start_idx - window : start_idx].values

        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                sim = 1.0 - cosine(window_returns[:, i], window_returns[:, j])
                if sim > threshold:
                    adj[i, j] = np.float32(sim)
                    adj[j, i] = np.float32(sim)

        snapshots[date_key] = adj

    return snapshots


__all__ = [
    "build_dynamic_snapshots_from_returns",
    "load_dynamic_snapshots",
    "pick_snapshot_for_date",
]
