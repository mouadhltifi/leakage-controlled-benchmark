"""PyTorch :class:`Dataset` over pre-assembled :class:`FoldArtifacts`.

This module deliberately performs **no** feature engineering.
:mod:`mmfp.data.assemble` did that once and handed us arrays. Our job is
to index.

What ``__getitem__`` returns
----------------------------

Keys depend on which modalities the config enabled. For every active
modality listed below the dataset emits one tensor per sample:

* ``price`` — ``(F_price,)`` when ``cfg.data.lookback == 1``, or
  ``(lookback, F_price)`` when ``lookback > 1``. The LSTM encoder reads
  the sequence shape; the feedforward encoder reads the flat shape.
* ``macro``, ``news``, ``social`` — ``(F_mod,)`` one vector per sample.
* ``graph_features`` — ``(N_stocks, F_price)`` float32; the node
  features for the graph encoder derived from the per-stock-per-day
  price block. Emitted only when ``cfg.graph.enabled``.
* ``edge_index`` or ``edge_index_static`` / ``edge_index_dynamic`` —
  ``(2, E)`` long tensors in COO layout. Static+dynamic mode emits both.
* ``stock_idx`` — scalar long, ``cfg.data.universe`` ticker index.
* Target keys — ``cls_target`` (long), ``reg_target`` (float),
  ``vol_target`` (float). Only present when the corresponding target is
  active in ``cfg.head.targets``.

Invariants
----------

* CPU tensors only. The trainer handles device transfer.
* Deadzone rows are already filtered by assembly when
  ``"direction"`` is a target, so ``cls_target`` is always in
  ``{0, 1}`` (never ``-1``).
* NaN label rows are already filtered by assembly for the active
  targets.

See the design spec for the authoritative
interface.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from mmfp.config.schema import ExperimentConfig
from mmfp.data.assemble import FoldArtifacts
from mmfp.data.loaders.graph_dynamic import pick_snapshot_for_date
from mmfp.data.universe import ALL_TICKERS

log = logging.getLogger(__name__)


Split = Literal["train", "val", "test"]


class MultiModalDataset(Dataset):
    """Fold-resident :class:`torch.utils.data.Dataset`.

    Parameters
    ----------
    artifacts
        Output of :func:`mmfp.data.assemble.assemble_fold`. This dataset
        holds a reference — no copying — so the caller should keep the
        artifacts object alive for the dataset's lifetime.
    split
        ``"train"``, ``"val"`` or ``"test"``. Indexing uses the
        corresponding ``{split}_*`` arrays on the artifacts object.
    cfg
        The :class:`ExperimentConfig` whose ``assemble_fold`` produced
        ``artifacts``. Only the ``data`` and ``graph`` sections are
        consulted; passing any other config is a caller bug.

    Attributes
    ----------
    split : Split
        The split name passed at construction.
    artifacts : FoldArtifacts
        The underlying artifacts reference.

    Notes
    -----
    Sequence reconstruction for ``lookback > 1`` relies on
    ``artifacts.price_features_by_stock_day``. A sample for ticker ``t``
    on date ``d`` requires the prior ``lookback-1`` trading days'
    entries. :func:`mmfp.data.assemble.assemble_fold` guarantees that
    filter — the constructor additionally sanity-checks it.
    """

    def __init__(
        self,
        artifacts: FoldArtifacts,
        split: Split,
        cfg: ExperimentConfig,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be one of 'train', 'val', 'test'; got {split!r}"
            )
        self.split: Split = split
        self.artifacts: FoldArtifacts = artifacts
        self._cfg = cfg

        self._features: np.ndarray = getattr(artifacts, f"{split}_features")
        self._labels_cls: np.ndarray = getattr(artifacts, f"{split}_labels_cls")
        self._labels_reg: np.ndarray = getattr(artifacts, f"{split}_labels_reg")
        self._labels_vol: np.ndarray = getattr(artifacts, f"{split}_labels_vol")
        self._dates: np.ndarray = getattr(artifacts, f"{split}_dates")
        self._stock_idx: np.ndarray = getattr(artifacts, f"{split}_stock_idx")

        # Graph support.
        self._has_graph = cfg.graph.enabled
        self._graph_source = cfg.graph.source
        self._static_adj = artifacts.static_adj
        self._dynamic_snapshots = artifacts.dynamic_snapshots

        if self._has_graph:
            # Pre-compute static edge_index once (COO format, symmetric).
            if self._static_adj is None and self._graph_source in (
                "static_gics",
                "static_plus_dynamic",
            ):
                raise ValueError(
                    f"graph.source={self._graph_source!r} requires a static "
                    "adjacency but FoldArtifacts.static_adj is None."
                )
            if (
                self._dynamic_snapshots is None
                and self._graph_source in ("dynamic_corr", "static_plus_dynamic")
            ):
                raise ValueError(
                    f"graph.source={self._graph_source!r} requires dynamic "
                    "snapshots but FoldArtifacts.dynamic_snapshots is None."
                )
            if self._static_adj is not None:
                self._static_edge_index = _adj_to_edge_index(self._static_adj)
            else:
                self._static_edge_index = None
            # Per-date dynamic edge_index caching — only fill lazily in
            # __getitem__. Storing the full cache here would waste memory
            # for no wall-clock win.
            self._dynamic_edge_cache: dict[str, np.ndarray] = {}

        # Price column slice, used for lookback reconstruction + graph
        # node features.
        self._price_slice = artifacts.feature_schema.range_for("price")

        # Cache a per-(ticker, date) map of the scaled price block so
        # lookback sequences can be pulled without re-joining. This
        # includes samples dropped by deadzone/terminal filters so a
        # sample's "prior 19 trading days" are always available.
        self._price_by_day = artifacts.price_features_by_stock_day

        # Pre-compute per-(ticker)-sorted (date, row_idx) for lookback
        # lookup. Only needed when lookback > 1.
        self._lookback = cfg.data.lookback
        if self._lookback > 1:
            self._stock_to_sorted_dates = _build_ticker_date_index(self._price_by_day)

        # Targets controlling which label keys are emitted.
        self._emit_cls = "direction" in cfg.head.targets
        self._emit_reg = "return" in cfg.head.targets
        self._emit_vol = "volatility" in cfg.head.targets

        # Modality slices (absent slices → don't emit the key).
        self._macro_slice = _safe_range(artifacts.feature_schema, "macro")
        self._news_slice = _safe_range(artifacts.feature_schema, "news")
        self._social_slice = _safe_range(artifacts.feature_schema, "social")

        # Sanity-check length alignment.
        n = self._features.shape[0]
        for name, arr in (
            ("labels_cls", self._labels_cls),
            ("labels_reg", self._labels_reg),
            ("labels_vol", self._labels_vol),
            ("dates", self._dates),
            ("stock_idx", self._stock_idx),
        ):
            if arr.shape[0] != n:
                raise ValueError(
                    f"FoldArtifacts.{split}_{name} length {arr.shape[0]} "
                    f"does not match {split}_features length {n}."
                )

        self._tickers = artifacts.tickers

    # ------------------------------------------------------------------
    # Core Dataset interface.
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._features.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for {self.split} split")

        row = self._features[idx]

        out: dict[str, torch.Tensor] = {}

        # Price — flat or sequence.
        price_flat = row[self._price_slice]
        if self._lookback > 1:
            out["price"] = self._build_price_sequence(idx, self._lookback)
        else:
            out["price"] = torch.as_tensor(price_flat, dtype=torch.float32)

        # Other flat modalities.
        if self._macro_slice is not None:
            out["macro"] = torch.as_tensor(row[self._macro_slice], dtype=torch.float32)
        if self._news_slice is not None:
            out["news"] = torch.as_tensor(row[self._news_slice], dtype=torch.float32)
        if self._social_slice is not None:
            out["social"] = torch.as_tensor(row[self._social_slice], dtype=torch.float32)

        # Stock index — always emitted; graph and non-graph models both
        # use it (graph encoders to pick the right node, non-graph
        # encoders to log per-stock metrics).
        out["stock_idx"] = torch.as_tensor(int(self._stock_idx[idx]), dtype=torch.long)

        # Graph tensors.
        if self._has_graph:
            graph_date = str(self._dates[idx])
            out["graph_features"] = self._build_graph_features(graph_date)
            if self._graph_source in ("static_gics",):
                assert self._static_edge_index is not None
                out["edge_index"] = torch.as_tensor(
                    self._static_edge_index, dtype=torch.long
                )
            elif self._graph_source in ("dynamic_corr",):
                out["edge_index"] = torch.as_tensor(
                    self._get_dynamic_edge_index(graph_date), dtype=torch.long
                )
            elif self._graph_source == "static_plus_dynamic":
                assert self._static_edge_index is not None
                out["edge_index_static"] = torch.as_tensor(
                    self._static_edge_index, dtype=torch.long
                )
                out["edge_index_dynamic"] = torch.as_tensor(
                    self._get_dynamic_edge_index(graph_date), dtype=torch.long
                )

        # Targets.
        if self._emit_cls:
            cls_val = int(self._labels_cls[idx])
            if cls_val == -1:
                raise RuntimeError(
                    "Dataset emitted a deadzone row (cls==-1) with direction "
                    "in targets. Assembly should have filtered this sample."
                )
            out["cls_target"] = torch.as_tensor(cls_val, dtype=torch.long)
        if self._emit_reg:
            out["reg_target"] = torch.as_tensor(
                float(self._labels_reg[idx]), dtype=torch.float32
            )
        if self._emit_vol:
            out["vol_target"] = torch.as_tensor(
                float(self._labels_vol[idx]), dtype=torch.float32
            )

        return out

    # ------------------------------------------------------------------
    # Lookback sequence reconstruction.
    # ------------------------------------------------------------------

    def _build_price_sequence(self, idx: int, lookback: int) -> torch.Tensor:
        """Return ``(lookback, F_price)`` ending at sample ``idx``.

        Sequences are drawn from
        :attr:`FoldArtifacts.price_features_by_stock_day` which spans
        the whole processed horizon (including rows dropped by
        deadzone/terminal filters). Assembly guarantees every retained
        sample has at least ``lookback-1`` prior trading days for its
        ticker; we additionally assert it here.
        """
        ticker = self._tickers[int(self._stock_idx[idx])]
        day = str(self._dates[idx])

        sorted_dates = self._stock_to_sorted_dates[ticker]
        # Fast path: binary search for ``day`` in the sorted date list.
        pos = _bisect_index(sorted_dates, day)
        if sorted_dates[pos] != day:
            raise RuntimeError(
                f"Lookback lookup: sample date {day!r} not in the "
                f"per-ticker date index for {ticker!r}."
            )
        if pos < lookback - 1:
            raise RuntimeError(
                f"Lookback lookup: sample for {ticker!r} on {day} has only "
                f"{pos + 1} prior trading days; needs {lookback}. "
                "Assembly lookback filter is broken."
            )

        rows = [
            self._price_by_day[(ticker, sorted_dates[pos - lookback + 1 + k])]
            for k in range(lookback)
        ]
        arr = np.stack(rows, axis=0)
        return torch.as_tensor(arr, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Graph helpers.
    # ------------------------------------------------------------------

    def _build_graph_features(self, iso_date: str) -> torch.Tensor:
        """Return ``(N_stocks, F_price)`` node features for ``iso_date``.

        Tickers without a price row on that date receive zeros (they
        will never be queried by the sample's ``stock_idx`` — they exist
        because GAT needs a fixed node count per forward pass).
        """
        n_stocks = len(self._tickers)
        f_price = self._price_slice.stop - self._price_slice.start
        arr = np.zeros((n_stocks, f_price), dtype=np.float32)
        for i, ticker in enumerate(self._tickers):
            feats = self._price_by_day.get((ticker, iso_date))
            if feats is not None:
                arr[i] = feats
        return torch.as_tensor(arr, dtype=torch.float32)

    def _get_dynamic_edge_index(self, iso_date: str) -> np.ndarray:
        """Resolve and cache the dynamic edge_index for ``iso_date``."""
        assert self._dynamic_snapshots is not None
        # Round down to the most recent snapshot on-or-before iso_date.
        # ``pick_snapshot_for_date`` returns an adjacency or None.
        adj = pick_snapshot_for_date(iso_date, self._dynamic_snapshots)
        if adj is None:
            raise RuntimeError(
                f"No dynamic snapshot on or before {iso_date}. "
                "Assembly's dynamic-coverage filter should have dropped "
                "this sample before it reached the dataset."
            )
        # ``pick_snapshot_for_date`` returns an ndarray; we need a stable
        # cache key that matches the snapshot identity (not the date).
        # The snapshots dict keys are ISO strings; scan for equality.
        # For the tiny n of snapshots (~113) linear is fine.
        snapshot_key: str | None = None
        for k, v in self._dynamic_snapshots.items():
            if v is adj:
                snapshot_key = k
                break
        cache_key = snapshot_key if snapshot_key is not None else iso_date
        if cache_key not in self._dynamic_edge_cache:
            self._dynamic_edge_cache[cache_key] = _adj_to_edge_index(adj)
        return self._dynamic_edge_cache[cache_key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_range(schema, modality: str) -> slice | None:
    """Return ``schema.range_for(modality)`` or ``None`` if absent."""
    try:
        return schema.range_for(modality)
    except KeyError:
        return None


def _adj_to_edge_index(adj: np.ndarray) -> np.ndarray:
    """Convert an adjacency matrix to a ``(2, E)`` int64 COO edge index."""
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"Adjacency matrix must be square 2-D; got shape {adj.shape}"
        )
    rows, cols = np.nonzero(adj)
    return np.stack([rows, cols], axis=0).astype(np.int64, copy=False)


def _build_ticker_date_index(
    price_by_day: dict[tuple[str, str], np.ndarray],
) -> dict[str, list[str]]:
    """Group the ``(ticker, date)`` keys into per-ticker sorted date lists."""
    per_ticker: dict[str, list[str]] = {}
    for ticker, date in price_by_day.keys():
        per_ticker.setdefault(ticker, []).append(date)
    for ticker in per_ticker:
        per_ticker[ticker].sort()
    return per_ticker


def _bisect_index(sorted_strings: list[str], key: str) -> int:
    """Return the index of ``key`` in ``sorted_strings`` (linear search).

    Binary search would be faster asymptotically; for the study's ~2000
    trading days per ticker the linear path is still sub-millisecond and
    avoids pulling in a tiny dependency.
    """
    # Use bisect_left for true binary search efficiency.
    import bisect

    i = bisect.bisect_left(sorted_strings, key)
    if i == len(sorted_strings):
        return i - 1  # clamp; caller verifies equality
    return i


__all__ = ["MultiModalDataset", "Split"]
