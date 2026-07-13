"""Static GICS sector adjacency builder.

The static graph connects two stocks if they belong to the same GICS
sector. All entries are ``0`` or ``1``; the matrix is symmetric with
zero diagonal.

This loader computes the adjacency directly from the universe sector
map, so it is deterministic and self-contained — no on-disk artifact is
strictly required. For convenience, :func:`load_static_adjacency` can
also read the cached ``sector_adjacency.npy`` written by the v1
pipeline; the two paths agree by construction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mmfp.data.paths import STATIC_ADJ_NPY
from mmfp.data.universe import ALL_TICKERS, TICKER_TO_SECTOR

log = logging.getLogger(__name__)


def build_static_adjacency(tickers: list[str] | None = None) -> np.ndarray:
    """Build the GICS sector co-membership adjacency matrix.

    Parameters
    ----------
    tickers
        Symbols defining row/column order. ``None`` (default) uses
        :data:`ALL_TICKERS` (55 symbols, alphabetical).

    Returns
    -------
    numpy.ndarray
        Shape ``(N, N)`` float32 matrix with:

        * symmetric (``adj == adj.T``),
        * zero diagonal,
        * ``adj[i, j] == 1`` iff ``tickers[i]`` and ``tickers[j]``
          belong to the same GICS sector.

    Raises
    ------
    KeyError
        If any ticker is not in the study universe (no sector lookup
        available). Callers that want a sparse sub-universe should
        pass only study tickers.

    Notes
    -----
    The v1 pipeline wrote this matrix to
    ``work/data/processed/graphs/sector_adjacency.npy``. Results agree
    bit-for-bit with that file for the default ticker list.
    """
    syms = list(tickers) if tickers is not None else list(ALL_TICKERS)

    missing = [t for t in syms if t not in TICKER_TO_SECTOR]
    if missing:
        raise KeyError(
            f"Unknown tickers (no GICS sector on file): {missing}. "
            "Add them to mmfp.data.universe.STOCK_UNIVERSE first."
        )

    n = len(syms)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        s_i = TICKER_TO_SECTOR[syms[i]]
        for j in range(i + 1, n):
            if TICKER_TO_SECTOR[syms[j]] == s_i:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    return adj


def load_static_adjacency(path: Path = STATIC_ADJ_NPY) -> np.ndarray:
    """Load the cached static adjacency from disk.

    Parameters
    ----------
    path
        Override the default NPY location.

    Returns
    -------
    numpy.ndarray
        The cached ``(55, 55)`` float32 adjacency matrix. By
        construction this is identical to
        ``build_static_adjacency(ALL_TICKERS)``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Static adjacency not found at {path}. "
            "Run v1 ``python -m src.data.build_graphs`` first, or call "
            "build_static_adjacency() to compute from the universe."
        )
    log.debug("Reading %s", path)
    return np.load(path).astype(np.float32, copy=False)


__all__ = ["build_static_adjacency", "load_static_adjacency"]
