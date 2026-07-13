"""Windowed multi-modal dataset for v3's TFT training (the architecture spec).

Unlike v2's :class:`~mmfp.datasets.multimodal.MultiModalDataset`, which
emits flat per-sample vectors (optionally with a price sequence), this
dataset emits a full ``(lookback, F_m)`` sequence for **every active
modality** plus the static categorical ids (ticker, sector) that the
TFT body consumes as its static-covariate inputs.

Emitted keys per sample
-----------------------

* ``price_seq``    — ``(lookback, F_price)`` float32, always present.
* ``news_seq``     — ``(lookback, F_news)`` float32, when ``cfg.news.enabled``.
* ``macro_seq``    — ``(lookback, F_macro)`` float32, when ``cfg.macro.enabled``.
* ``social_seq``   — ``(lookback, F_social)`` float32, when ``cfg.social.enabled``.
* ``graph_seq``    — ``(lookback, graph_node_dim)`` float32, when ``cfg.graph.enabled``.
* ``static_categorical`` — ``(2,)`` int64 ``[ticker_id, sector_id]``.
* ``y_return``     — scalar float32, ``log(p_{T+1} / p_T)``.
* ``y_direction``  — scalar int64, ``sign(y_return)`` in ``{0, 1}`` post-
  deadzone filter, or ``-1`` on deadzone rows (kept for optional
  auxiliaries when the split has not filtered them).
* ``y_volatility`` — scalar float32, ``|y_return|``.

Sequence reconstruction
-----------------------

Price uses :attr:`FoldArtifacts.price_features_by_stock_day` (populated
by v2 assembly for every processed trading day, including warm-up and
rows dropped by target filters).

News / macro / social reconstruct per-(ticker, date) dicts by scanning
train + val + test split feature arrays. Dates dropped by target
filters (deadzone / NaN-vol) become zero-fill entries on lookup — a
minor approximation, matching v2's empty-day policy for news. The TFT's
VSN can downweight such timesteps.

Graph uses the per-fold :class:`GraphNodeCache` produced by
:func:`build_graph_node_cache` at experiment-runner time.
"""

from __future__ import annotations

import bisect
import logging
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from forecast.config.schema import V3ExperimentConfig
from forecast.datasets.graph_precompute import GraphNodeCache
from mmfp.data.assemble import FoldArtifacts
from mmfp.data.universe import TICKER_TO_SECTOR

log = logging.getLogger(__name__)


Split = Literal["train", "val", "test"]


# ---------------------------------------------------------------------------
# Sector encoding — sorted alphabetic unique sectors from TICKER_TO_SECTOR.
# ---------------------------------------------------------------------------

_SORTED_SECTORS: tuple[str, ...] = tuple(sorted(set(TICKER_TO_SECTOR.values())))
_SECTOR_TO_ID: dict[str, int] = {s: i for i, s in enumerate(_SORTED_SECTORS)}


def _sector_id_for_ticker(ticker: str) -> int:
    """Deterministic 0-indexed sector id for a ticker.

    Raises :class:`KeyError` if ``ticker`` is not in the 55-stock
    universe.
    """
    return _SECTOR_TO_ID[TICKER_TO_SECTOR[ticker]]


# ---------------------------------------------------------------------------
# ForecastDataset
# ---------------------------------------------------------------------------


class ForecastDataset(Dataset):
    """Windowed multi-modal dataset for the TFT training loop.

    Parameters
    ----------
    artifacts
        Assembled :class:`FoldArtifacts` for this fold. The dataset
        holds a reference (no copy) and indexes into the per-split
        arrays.
    split
        ``"train"``, ``"val"``, or ``"test"``.
    cfg
        Validated :class:`V3ExperimentConfig`. Modality enablement
        flags plus :attr:`ForecastConfig.lookback` drive the emission
        contract.
    graph_node_cache
        Per-fold :class:`GraphNodeCache`. Required when
        ``cfg.graph.enabled``; may be ``None`` otherwise.

    Attributes
    ----------
    split : Split
    artifacts : FoldArtifacts
    lookback : int
    active_modalities : tuple[str, ...]
    """

    def __init__(
        self,
        artifacts: FoldArtifacts,
        split: Split,
        cfg: V3ExperimentConfig,
        graph_node_cache: GraphNodeCache | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be one of 'train','val','test'; got {split!r}"
            )
        self.split: Split = split
        self.artifacts = artifacts
        self.cfg = cfg

        self.lookback: int = int(cfg.forecast.lookback)
        if self.lookback < 1:
            raise ValueError(
                f"forecast.lookback must be >= 1; got {self.lookback}"
            )

        # Per-split label / index arrays.
        self._features: np.ndarray = getattr(artifacts, f"{split}_features")
        self._labels_cls: np.ndarray = getattr(artifacts, f"{split}_labels_cls")
        self._labels_reg: np.ndarray = getattr(artifacts, f"{split}_labels_reg")
        self._labels_vol: np.ndarray = getattr(artifacts, f"{split}_labels_vol")
        self._dates: np.ndarray = getattr(artifacts, f"{split}_dates")
        self._stock_idx: np.ndarray = getattr(artifacts, f"{split}_stock_idx")

        # Feature schema
        schema = artifacts.feature_schema
        self._price_slice = schema.range_for("price")
        self._macro_slice = _safe_range(schema, "macro")
        self._news_slice = _safe_range(schema, "news")
        self._social_slice = _safe_range(schema, "social")

        # Tickers and static categorical ids.
        self._tickers: list[str] = list(artifacts.tickers)
        self._ticker_sector_ids: np.ndarray = np.array(
            [_sector_id_for_ticker(t) for t in self._tickers],
            dtype=np.int64,
        )

        # Active modality list (sorted for deterministic consumer order).
        active: set[str] = {"price"}
        if cfg.news.enabled and self._news_slice is not None:
            active.add("news")
        if cfg.macro.enabled and self._macro_slice is not None:
            active.add("macro")
        if cfg.social.enabled and self._social_slice is not None:
            active.add("social")
        if cfg.graph.enabled:
            active.add("graph")
        self.active_modalities: tuple[str, ...] = tuple(sorted(active))

        # Graph cache validation.
        if "graph" in self.active_modalities:
            if graph_node_cache is None:
                raise ValueError(
                    "ForecastDataset: cfg.graph.enabled is True but "
                    "graph_node_cache is None. Pass a "
                    "GraphNodeCache built via build_graph_node_cache."
                )
            if graph_node_cache.dim != cfg.forecast.graph_node_dim:
                raise ValueError(
                    f"graph_node_cache.dim={graph_node_cache.dim} does not "
                    f"match cfg.forecast.graph_node_dim="
                    f"{cfg.forecast.graph_node_dim}."
                )
        self._graph_cache = graph_node_cache

        # Per-ticker sorted date index (price has the full history).
        self._price_by_day = artifacts.price_features_by_stock_day
        self._stock_to_sorted_dates: dict[str, list[str]] = (
            _build_ticker_date_index(self._price_by_day)
        )

        # Per-(ticker, date) dicts for non-price modalities, reconstructed
        # from the concatenation of train + val + test splits. Dates
        # dropped by target filters default to zero.
        self._macro_by_day: dict[tuple[str, str], np.ndarray] | None = None
        self._news_by_day: dict[tuple[str, str], np.ndarray] | None = None
        self._social_by_day: dict[tuple[str, str], np.ndarray] | None = None

        if "macro" in self.active_modalities:
            self._macro_by_day = _build_modality_by_day(
                artifacts, slice_=self._macro_slice, tickers=self._tickers
            )
        if "news" in self.active_modalities:
            self._news_by_day = _build_modality_by_day(
                artifacts, slice_=self._news_slice, tickers=self._tickers
            )
        if "social" in self.active_modalities:
            self._social_by_day = _build_modality_by_day(
                artifacts, slice_=self._social_slice, tickers=self._tickers
            )

        # Sanity: split alignments.
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

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._features.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"index {idx} out of range for {self.split} split"
            )

        ticker_id = int(self._stock_idx[idx])
        ticker = self._tickers[ticker_id]
        day = str(self._dates[idx])

        # Resolve the lookback window of dates ending at ``day``.
        window_dates = self._resolve_window_dates(ticker, day)

        out: dict[str, torch.Tensor] = {}

        # --- price --------------------------------------------------------
        price_seq = self._build_modality_sequence(
            self._price_by_day, ticker, window_dates, default_width=self._flat_width(self._price_slice)
        )
        out["price_seq"] = torch.as_tensor(price_seq, dtype=torch.float32)

        # --- other flat modalities ----------------------------------------
        if "macro" in self.active_modalities:
            assert self._macro_by_day is not None
            arr = self._build_modality_sequence(
                self._macro_by_day, ticker, window_dates,
                default_width=self._flat_width(self._macro_slice),
            )
            out["macro_seq"] = torch.as_tensor(arr, dtype=torch.float32)
        if "news" in self.active_modalities:
            assert self._news_by_day is not None
            arr = self._build_modality_sequence(
                self._news_by_day, ticker, window_dates,
                default_width=self._flat_width(self._news_slice),
            )
            out["news_seq"] = torch.as_tensor(arr, dtype=torch.float32)
        if "social" in self.active_modalities:
            assert self._social_by_day is not None
            arr = self._build_modality_sequence(
                self._social_by_day, ticker, window_dates,
                default_width=self._flat_width(self._social_slice),
            )
            out["social_seq"] = torch.as_tensor(arr, dtype=torch.float32)

        # --- graph --------------------------------------------------------
        if "graph" in self.active_modalities:
            assert self._graph_cache is not None
            graph_arr = self._graph_cache.get_sequence(ticker_id, window_dates)
            out["graph_seq"] = torch.as_tensor(graph_arr, dtype=torch.float32)

        # --- static categoricals ------------------------------------------
        sector_id = int(self._ticker_sector_ids[ticker_id])
        out["static_categorical"] = torch.as_tensor(
            [ticker_id, sector_id], dtype=torch.long
        )

        # --- targets ------------------------------------------------------
        out["y_return"] = torch.as_tensor(
            float(self._labels_reg[idx]), dtype=torch.float32
        )
        out["y_direction"] = torch.as_tensor(
            int(self._labels_cls[idx]), dtype=torch.long
        )
        out["y_volatility"] = torch.as_tensor(
            float(self._labels_vol[idx]), dtype=torch.float32
        )

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flat_width(self, sl: slice | None) -> int:
        """Width of a schema slice, or raise if absent."""
        if sl is None:
            raise ValueError(
                "ForecastDataset: requested slice width for an absent modality."
            )
        return sl.stop - sl.start

    def _resolve_window_dates(self, ticker: str, day: str) -> list[str]:
        """Return the sorted list of ``lookback`` trading days ending at ``day``.

        Raises :class:`RuntimeError` if there are fewer than
        ``lookback`` prior days in the per-ticker date index (assembly's
        lookback filter should have dropped this sample).
        """
        sorted_dates = self._stock_to_sorted_dates.get(ticker)
        if sorted_dates is None:
            raise RuntimeError(
                f"Lookback lookup: ticker {ticker!r} absent from the price "
                "index (price_features_by_stock_day)."
            )
        pos = bisect.bisect_left(sorted_dates, day)
        if pos >= len(sorted_dates) or sorted_dates[pos] != day:
            raise RuntimeError(
                f"Lookback lookup: sample date {day!r} not in the per-ticker "
                f"date index for {ticker!r}."
            )
        if pos < self.lookback - 1:
            raise RuntimeError(
                f"Lookback lookup: sample for {ticker!r} on {day} has only "
                f"{pos + 1} prior trading days; needs {self.lookback}. "
                "Assembly lookback filter is broken."
            )
        start = pos - self.lookback + 1
        return sorted_dates[start : pos + 1]

    @staticmethod
    def _build_modality_sequence(
        by_day: dict[tuple[str, str], np.ndarray],
        ticker: str,
        window_dates: list[str],
        *,
        default_width: int,
    ) -> np.ndarray:
        """Stack per-day vectors into ``(L, F)``; zero-fill missing dates."""
        L = len(window_dates)
        out = np.zeros((L, default_width), dtype=np.float32)
        for i, date in enumerate(window_dates):
            vec = by_day.get((ticker, date))
            if vec is not None:
                out[i] = vec
        return out


# ---------------------------------------------------------------------------
# Helpers (module-level)
# ---------------------------------------------------------------------------


def _safe_range(schema, modality: str) -> slice | None:
    """Return ``schema.range_for(modality)`` or ``None`` if absent."""
    try:
        return schema.range_for(modality)
    except KeyError:
        return None


def _build_ticker_date_index(
    price_by_day: dict[tuple[str, str], np.ndarray],
) -> dict[str, list[str]]:
    """Group ``(ticker, date)`` keys into per-ticker sorted date lists."""
    per_ticker: dict[str, list[str]] = {}
    for ticker, date in price_by_day.keys():
        per_ticker.setdefault(ticker, []).append(date)
    for ticker in per_ticker:
        per_ticker[ticker].sort()
    return per_ticker


def _build_modality_by_day(
    artifacts: FoldArtifacts,
    *,
    slice_: slice | None,
    tickers: list[str],
) -> dict[tuple[str, str], np.ndarray]:
    """Reconstruct ``{(ticker, iso_date): feature_vec}`` for a flat modality.

    Concatenates the train + val + test split arrays for the requested
    slice. Rows dropped by target filters (deadzone / NaN vol) are
    absent from the result; callers must treat missing keys as zero.

    Parameters
    ----------
    artifacts
        The fold's :class:`FoldArtifacts`.
    slice_
        Column slice in ``*_features`` for the target modality. Must be
        non-None (caller should have checked).
    tickers
        Canonical ticker list in ``stock_idx`` order.

    Returns
    -------
    dict
        Mapping ``(ticker, date) -> (F_mod,) float32``. Duplicate keys
        across splits (impossible by construction — a date/ticker is in
        exactly one split) would silently pick the last write; we do
        not special-case because assembly guarantees disjointness.
    """
    if slice_ is None:
        raise ValueError(
            "_build_modality_by_day called with slice_=None."
        )
    out: dict[tuple[str, str], np.ndarray] = {}
    for split in ("train", "val", "test"):
        features: np.ndarray = getattr(artifacts, f"{split}_features")
        dates: np.ndarray = getattr(artifacts, f"{split}_dates")
        stock_idx: np.ndarray = getattr(artifacts, f"{split}_stock_idx")
        if features.size == 0:
            continue
        block = features[:, slice_]
        for i in range(block.shape[0]):
            ticker = tickers[int(stock_idx[i])]
            date = str(dates[i])
            out[(ticker, date)] = block[i].astype(np.float32, copy=False)
    return out


__all__ = ["ForecastDataset", "Split"]
