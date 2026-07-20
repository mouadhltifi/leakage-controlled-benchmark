"""Static same-sector adjacency: released graph vs. universe partition.

Two distinct static graphs exist, deliberately:

* :func:`load_static_adjacency` reads the RELEASED static graph,
  ``data/processed/graphs/sector_adjacency.npy`` — the Fama–French-12
  taxonomy (133 edges), which the harness consumes
  (``assemble.py``) and the benchmark's graph reference rows use.
* :func:`build_static_adjacency` constructs the same-sector partition of
  the universe's 11×5 sector-balanced construction (110 edges, GICS-era
  membership) from the in-code map — the sensitivity arm
  (``results/sector/sector_gics_*.csv``) and the historical grid.

The two taxonomies differ materially (edge Jaccard 0.365); the paired
FF12-vs-GICS comparison ships in ``results/sector/`` (paper Appendix B).
All entries are ``0`` or ``1``; matrices are symmetric with zero diagonal.
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
    This is the universe's GICS-partition graph (the sensitivity arm),
    NOT the released FF12 graph committed at
    ``data/processed/graphs/sector_adjacency.npy`` — see the module
    docstring and DATA-STATEMENTS "Sector structure".
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
        The released ``(55, 55)`` float32 static graph (Fama–French-12
        same-sector adjacency, 133 edges) — deliberately distinct from
        ``build_static_adjacency(ALL_TICKERS)``, the universe's GICS
        partition.

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
