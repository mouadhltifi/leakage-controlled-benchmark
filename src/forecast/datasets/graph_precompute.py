"""Per-fold graph node embedding precompute (the architecture spec).

The v3 TFT architecture consumes a ``(lookback, graph_node_dim)``
sequence per sample as one of its past-observed covariates. Running the
full ``GraphGATEncoder`` once per sample is prohibitively expensive
(``B * L * N_stocks`` GAT forward passes per mini-batch), so v3 takes
the standard pre-embedding shortcut: run the GAT once per distinct
snapshot per fold and cache the ``(N_tickers, graph_node_dim)``
embedding keyed by date.

Implementation details
----------------------

* The GAT weights are **not** trained in v3 M4. We load
  :class:`~mmfp.models.encoders.graph_gat.GraphGATEncoder` with random
  init, freeze it (``requires_grad=False``), and apply it as a
  structural encoder. This gives us a deterministic, cheap function
  ``(adjacency, node_features) -> (N, H)`` without any joint-training
  coupling. A future research question (documented in the spec) is
  whether trained-GAT embeddings help; for M4 we prioritise cache
  simplicity and leakage safety.

* **Leakage discipline is load-bearing.** The cache for date ``d`` uses
  only adjacencies dated ``<= d`` (dynamic snapshot lookup rounds down
  per v2's :func:`~mmfp.data.loaders.graph_dynamic.pick_snapshot_for_date`).
  For ``graph.source == "static_gics"`` this is trivially satisfied —
  the adjacency is date-independent and the same for every date. For
  dynamic sources we loop snapshot-by-snapshot so no future snapshot
  ever enters the embedding for an earlier date.

* The disk cache key includes the fold index plus the graph-section
  config hash (``graph.source``, ``graph.dynamic_window``,
  ``graph.dynamic_threshold``, ``graph.dynamic_refresh_every``) so
  different graph configurations produce different caches.

* Missing dates (samples before the first available snapshot under
  pure dynamic mode) return a zero vector rather than raising — the
  assemble step already drops samples predating the first snapshot, so
  this only bites in edge cases (tests and custom date ranges).

Why this is safe from test-split leakage
----------------------------------------

The GAT reads **node features** (v2 uses the per-ticker price block at
date ``d``) and **adjacencies** (static GICS or a rolling-correlation
snapshot dated ``<= d``). Both inputs are already clean of future
information under v2's assemble discipline. The only new surface v3
introduces is the precompute step, and the leakage-test module
:mod:`forecast.tests.leakage.test_graph_precompute_leakage` asserts
that surface remains clean.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from forecast.config.schema import V3ExperimentConfig
from mmfp.data.assemble import FoldArtifacts
from mmfp.data.loaders.graph_dynamic import pick_snapshot_for_date
from mmfp.models.encoders.graph_gat import GraphGATEncoder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GraphNodeCache
# ---------------------------------------------------------------------------


@dataclass
class GraphNodeCache:
    """In-memory cache of GAT node embeddings keyed by date.

    Attributes
    ----------
    data
        Mapping ``date (YYYY-MM-DD) -> (N_tickers, graph_node_dim)`` float32.
        Every date that produced a sample in the fold should be a key;
        :meth:`get_sequence` additionally tolerates missing keys by
        returning a zero vector.
    dim
        Embedding width (``graph_node_dim``). Returned on a missing-key
        lookup so shape contracts hold for callers.
    n_tickers
        Number of tickers the per-date tensors cover (rows of
        ``data[date]``).
    """

    data: dict[str, np.ndarray] = field(default_factory=dict)
    dim: int = 64
    n_tickers: int = 55

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sequence(
        self,
        ticker_id: int,
        dates: list[str] | tuple[str, ...],
    ) -> np.ndarray:
        """Return a ``(len(dates), graph_node_dim)`` sequence for one ticker.

        Missing dates are zero-filled — the caller's assemble step has
        already dropped samples that cannot satisfy the lookback window,
        so in practice zero fills bite only for the earliest samples of
        a dynamic-only run whose first lookback days fall before the
        first snapshot.

        Parameters
        ----------
        ticker_id
            Integer in ``[0, n_tickers)``; row index into the cached
            per-date matrices.
        dates
            Ordered list of ISO date strings (``YYYY-MM-DD``).

        Returns
        -------
        np.ndarray
            Float32 matrix of shape ``(len(dates), dim)``.
        """
        if not 0 <= ticker_id < self.n_tickers:
            raise IndexError(
                f"ticker_id {ticker_id} out of range [0, {self.n_tickers})"
            )
        out = np.zeros((len(dates), self.dim), dtype=np.float32)
        for i, date in enumerate(dates):
            mat = self.data.get(date)
            if mat is None:
                continue
            out[i] = mat[ticker_id]
        return out


# ---------------------------------------------------------------------------
# Precompute entry point
# ---------------------------------------------------------------------------


def _graph_config_hash(cfg: V3ExperimentConfig) -> str:
    """Stable short hash of the graph-section config fields that matter.

    Included fields: ``source``, ``dynamic_window``, ``dynamic_threshold``,
    ``dynamic_refresh_every``, plus ``forecast.graph_node_dim`` (cache
    shape) and ``forecast.hidden_dim`` (GAT hidden width, since the GAT
    reuses :attr:`cfg.model.hidden_dim` which is independent — but the
    safest key pins both).
    """
    g = cfg.graph
    payload = "|".join(
        (
            str(g.source),
            str(g.dynamic_window),
            str(g.dynamic_threshold),
            str(g.dynamic_refresh_every),
            str(cfg.forecast.graph_node_dim),
            str(cfg.model.hidden_dim),
        )
    )
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _cache_path(cache_dir: Path, fold_idx: int, config_hash: str) -> Path:
    """Return the on-disk cache file path (``.npz``)."""
    return cache_dir / f"fold_{fold_idx}_graph_{config_hash}.npz"


def _init_seed(cfg_seed: int, fold_idx: int, config_hash: str) -> int:
    """Deterministic seed for the frozen GAT init.

    Distinct from ``cfg.seed`` so the GAT's init does not drift with
    changes to the training seed. Composed from the config hash so
    different graph configs get different (but still repeatable)
    weights.
    """
    # Fold mix-in and hash prefix keep the seed range inside int32.
    h = int(hashlib.sha1(config_hash.encode("utf-8"), usedforsecurity=False).hexdigest()[:8], 16)
    return (cfg_seed * 0x100_0001 + fold_idx * 997 + h) & 0x7FFF_FFFF


def _load_from_disk(path: Path) -> dict[str, np.ndarray] | None:
    """Load a cached ``.npz`` if present; otherwise return ``None``."""
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            return {key: data[key].astype(np.float32, copy=False) for key in data.files}
    except (OSError, ValueError) as exc:  # pragma: no cover - corrupt cache
        log.warning(
            "graph_precompute: failed to read cache %s (%s); will rebuild.",
            path,
            exc,
        )
        return None


def _save_to_disk(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Atomically write the cache ``.npz`` to disk.

    :func:`numpy.savez_compressed` appends ``.npz`` to its ``file``
    argument when it doesn't already end with ``.npz``. We open the
    target explicitly via :func:`open` to avoid that suffix massaging
    and to get true atomic-rename semantics.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    tmp.replace(path)


def _resolve_dates_and_node_features(
    artifacts: FoldArtifacts,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Collect the sorted distinct dates and per-date node features.

    Node features for date ``d`` are the scaled per-ticker price vectors
    sourced from :attr:`FoldArtifacts.price_features_by_stock_day`. For
    tickers without a row on ``d`` we fall back to zeros so the GAT
    always sees ``(N_tickers, F_price)``.
    """
    tickers = list(artifacts.tickers)
    n_tickers = len(tickers)

    # Infer F_price from the first available entry.
    if not artifacts.price_features_by_stock_day:
        raise ValueError(
            "graph_precompute: FoldArtifacts.price_features_by_stock_day is "
            "empty — cannot build graph node cache."
        )
    any_feat = next(iter(artifacts.price_features_by_stock_day.values()))
    f_price = int(any_feat.shape[0])

    # Gather unique dates from the price index.
    dates: set[str] = set()
    for _, date in artifacts.price_features_by_stock_day.keys():
        dates.add(date)
    sorted_dates = sorted(dates)

    per_date_features: dict[str, np.ndarray] = {}
    for date in sorted_dates:
        mat = np.zeros((n_tickers, f_price), dtype=np.float32)
        for i, ticker in enumerate(tickers):
            feats = artifacts.price_features_by_stock_day.get((ticker, date))
            if feats is not None:
                mat[i] = feats
        per_date_features[date] = mat
    return sorted_dates, per_date_features


def _adj_to_edge_index(adj: np.ndarray) -> np.ndarray:
    """Convert ``(N, N)`` adjacency to ``(2, E)`` int64 COO layout."""
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"Adjacency must be square 2-D; got shape {adj.shape}"
        )
    rows, cols = np.nonzero(adj)
    return np.stack([rows, cols], axis=0).astype(np.int64, copy=False)


def _pick_dynamic_snapshot(
    date: str,
    dynamic_snapshots: dict[str, np.ndarray] | None,
) -> np.ndarray | None:
    """Leakage-safe lookup of the most recent snapshot dated ``<= date``.

    Delegates to v2's :func:`pick_snapshot_for_date`, which enforces the
    ``<=`` invariant. Returns ``None`` if no snapshot qualifies (e.g. the
    sample predates the first snapshot) — the caller then returns a zero
    embedding, matching :meth:`GraphNodeCache.get_sequence`'s behaviour
    for missing dates.
    """
    if dynamic_snapshots is None:
        return None
    return pick_snapshot_for_date(date, dynamic_snapshots)


def _build_stub_gat_cfg(cfg: V3ExperimentConfig) -> V3ExperimentConfig:
    """Return a config whose ``model.hidden_dim`` matches ``graph_node_dim``.

    v2's :class:`GraphGATEncoder` reads ``cfg.model.hidden_dim`` for its
    output width. v3 wants that width to equal
    ``cfg.forecast.graph_node_dim``; we create a lightweight copy rather
    than mutating the caller's config so the rest of the run sees the
    original ``hidden_dim``.
    """
    # Fast path: if they already agree, use as-is.
    if cfg.model.hidden_dim == cfg.forecast.graph_node_dim:
        return cfg
    patched = cfg.model_copy(deep=True)
    patched.model.hidden_dim = cfg.forecast.graph_node_dim
    return patched


def build_graph_node_cache(
    artifacts: FoldArtifacts,
    cfg: V3ExperimentConfig,
    cache_dir: Path,
) -> GraphNodeCache:
    """Compute (or load) the per-fold graph node-embedding cache.

    Parameters
    ----------
    artifacts
        Assembled :class:`FoldArtifacts` for the fold. Must carry a
        populated ``price_features_by_stock_day`` and, if the config
        enables dynamic sources, ``dynamic_snapshots`` / ``static_adj``.
    cfg
        Validated :class:`V3ExperimentConfig`. The ``graph.*`` section
        and ``forecast.graph_node_dim`` drive the cache key and the GAT
        output width.
    cache_dir
        Directory to store the ``.npz`` cache. Created if missing.

    Returns
    -------
    GraphNodeCache
        In-memory cache ready for
        :meth:`GraphNodeCache.get_sequence` lookups.

    Raises
    ------
    ValueError
        If ``cfg.graph.enabled`` is ``False`` (caller bug — callers
        should short-circuit graph-disabled configs before reaching
        here).
    """
    if not cfg.graph.enabled:
        raise ValueError(
            "build_graph_node_cache: cfg.graph.enabled is False. "
            "Do not call this function for graph-disabled configs."
        )

    cache_dir = Path(cache_dir)
    config_hash = _graph_config_hash(cfg)
    path = _cache_path(cache_dir, artifacts.fold_idx, config_hash)

    n_tickers = len(artifacts.tickers)
    dim = int(cfg.forecast.graph_node_dim)

    # Try to load a prior cache.
    disk = _load_from_disk(path)
    if disk is not None:
        return GraphNodeCache(data=disk, dim=dim, n_tickers=n_tickers)

    sorted_dates, per_date_features = _resolve_dates_and_node_features(artifacts)
    f_price = next(iter(per_date_features.values())).shape[1]

    # Build GAT encoder (random init, frozen). Deterministic construction:
    # seed-gate the init so repeated builds with the same (cfg.seed,
    # config_hash, fold_idx) triple produce identical encoder weights.
    # The leakage tests rely on this — they vary adjacency / node
    # features across two calls and require that *other* inputs produce
    # bit-identical embeddings.
    init_seed = _init_seed(cfg.seed, artifacts.fold_idx, config_hash)
    gat_cfg = _build_stub_gat_cfg(cfg)
    gen = torch.Generator()
    gen.manual_seed(init_seed)
    # Use a context that temporarily seeds the default PyTorch RNG so
    # ``GraphGATEncoder`` module init picks up deterministic weights.
    prior_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(init_seed)
        encoder = GraphGATEncoder(cfg=gat_cfg, input_dim=f_price)
    finally:
        torch.random.set_rng_state(prior_state)
    encoder.train(False)  # inference mode
    for p in encoder.parameters():
        p.requires_grad = False

    # Pre-compute static edge index (reused across dates when applicable).
    static_adj = artifacts.static_adj
    static_edge_index_t: torch.Tensor | None = None
    if cfg.graph.source in ("static_gics", "static_plus_dynamic"):
        if static_adj is None:
            raise ValueError(
                "build_graph_node_cache: static source requested but "
                "artifacts.static_adj is None."
            )
        static_edge_index_t = torch.as_tensor(
            _adj_to_edge_index(static_adj), dtype=torch.long
        )

    dynamic_snapshots = artifacts.dynamic_snapshots
    if (
        cfg.graph.source in ("dynamic_corr", "static_plus_dynamic")
        and dynamic_snapshots is None
    ):
        raise ValueError(
            "build_graph_node_cache: dynamic source requested but "
            "artifacts.dynamic_snapshots is None."
        )

    # Cache dynamic edge indices by snapshot identity (k dates share one
    # snapshot) to avoid recomputing the COO layout N times.
    dyn_edge_cache: dict[str, torch.Tensor] = {}

    data: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for date in sorted_dates:
            node_features = torch.as_tensor(
                per_date_features[date], dtype=torch.float32
            )

            if cfg.graph.source == "static_gics":
                assert static_edge_index_t is not None
                out = encoder(node_features, edge_index=static_edge_index_t)
            elif cfg.graph.source == "dynamic_corr":
                dyn_adj = _pick_dynamic_snapshot(date, dynamic_snapshots)
                if dyn_adj is None:
                    # No snapshot yet — zero embedding for this date.
                    data[date] = np.zeros((n_tickers, dim), dtype=np.float32)
                    continue
                key = _snapshot_identity(date, dyn_adj, dynamic_snapshots)
                if key not in dyn_edge_cache:
                    dyn_edge_cache[key] = torch.as_tensor(
                        _adj_to_edge_index(dyn_adj), dtype=torch.long
                    )
                out = encoder(node_features, edge_index=dyn_edge_cache[key])
            elif cfg.graph.source == "static_plus_dynamic":
                assert static_edge_index_t is not None
                dyn_adj = _pick_dynamic_snapshot(date, dynamic_snapshots)
                if dyn_adj is None:
                    # Fall back to static-only for this early date by
                    # passing the static edge index to both stacks —
                    # matches v2's degenerate path.
                    dyn_edge_index_t = static_edge_index_t
                else:
                    key = _snapshot_identity(date, dyn_adj, dynamic_snapshots)
                    if key not in dyn_edge_cache:
                        dyn_edge_cache[key] = torch.as_tensor(
                            _adj_to_edge_index(dyn_adj), dtype=torch.long
                        )
                    dyn_edge_index_t = dyn_edge_cache[key]
                out = encoder(
                    node_features,
                    edge_index_static=static_edge_index_t,
                    edge_index_dynamic=dyn_edge_index_t,
                )
            else:  # pragma: no cover - pydantic forbids unknown sources
                raise ValueError(
                    f"build_graph_node_cache: unknown graph.source "
                    f"{cfg.graph.source!r}"
                )

            arr = out.detach().cpu().numpy().astype(np.float32, copy=False)
            if arr.shape != (n_tickers, dim):
                raise RuntimeError(
                    f"GraphGATEncoder output has shape {arr.shape}; "
                    f"expected ({n_tickers}, {dim}). "
                    "Check graph_node_dim / model.hidden_dim wiring."
                )
            data[date] = arr

    _save_to_disk(path, data)
    log.info(
        "graph_precompute: fold=%d dates=%d hash=%s wrote %s",
        artifacts.fold_idx,
        len(data),
        config_hash,
        path,
    )
    return GraphNodeCache(data=data, dim=dim, n_tickers=n_tickers)


def _snapshot_identity(
    date: str,
    adj: np.ndarray,
    snapshots: dict[str, np.ndarray] | None,
) -> str:
    """Stable cache key for a dynamic adjacency snapshot.

    ``pick_snapshot_for_date`` returns the array *object* held by
    ``snapshots``, so scanning for identity (``is``) uniquely names the
    snapshot regardless of which date we asked for.
    """
    if snapshots is not None:
        for k, v in snapshots.items():
            if v is adj:
                return k
    # Fallback: content-hash (only reached if snapshots drift from the
    # dict — defensive).
    return hashlib.sha1(adj.tobytes(), usedforsecurity=False).hexdigest()[:16]


__all__ = ["GraphNodeCache", "build_graph_node_cache"]
