"""Per-fold dataset assembly — the central integration point of the platform.

This module answers one question: given an :class:`ExperimentConfig`, build
the fold's train/val/test feature tensors, labels, scalers and graph
artifacts, with **all train-only fit discipline** enforced. Every audit
finding that concerns fold-aware feature computation (news PCA, modality
scaling, terminal-row handling, lookback filtering) is implemented here.

Pipeline at a glance
--------------------

1. Resolve fold boundaries from :attr:`DataConfig.fold_idx`.
2. Load per-modality raw/feature caches via ports.
3. Compute the news block (11-dim stats OR per-fold PCA on 768-dim).
4. Compute labels: ``cls`` (direction, deadzone-aware), ``reg`` (log
   return), ``vol`` (|log return|).
5. Align everything on the (Date, Ticker) grid.
6. Mask into train / val / test by fold boundaries.
7. Fit :class:`FittedScaler` per modality on the train block; transform
   val/test with the frozen parameters.
8. Apply lookback filtering: drop samples that don't have ``lookback-1``
   prior trading days for their ticker.
9. Drop terminal rows (``NaN`` log return) when ``"volatility"`` in targets.
10. Drop deadzone samples when ``"direction"`` in targets.

The output is a :class:`FoldArtifacts` dataclass holding flat numpy arrays
plus a :class:`FeatureSchema` that records which column range belongs to
which modality. Downstream :class:`~mmfp.datasets.multimodal.MultiModalDataset`
indexes into these arrays; it does no feature computation of its own.

Fold boundaries
---------------

Ported from v1 ``src/utils/config.py::TEMPORAL_FOLDS`` (5 expanding
windows, 2019-2023 test periods). See :data:`FOLD_BOUNDARIES`.

The train period starts at ``data.start_date`` (default ``2015-02-03``
following the 252-day TA warm-up) and ends at ``fold.train_end``. The
last ``training.val_fraction`` fraction of those calendar days forms the
validation set — this matches v1's ``_compute_val_start`` in
``src/training/experiment_runner_v2.py``.

Per-fold PCA (audit B.4 fix)
-----------------------------

For ``news.encoder == "finbert_cls_768"`` (and the other 768-dim
encoders) the platform fits :class:`sklearn.decomposition.PCA` on the
train-fold *articles* only — not on v1's fixed ``2018-12-31`` cutoff.
The PCA basis is then applied to val + test articles. This is
implemented inline (no caching) so every fold carries its own basis and
the legacy ``news_features_768_*_pca32.parquet`` is left untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from mmfp.config.schema import ExperimentConfig
from mmfp.data.loaders.graph_dynamic import (
    load_dynamic_snapshots,
    pick_snapshot_for_date,
)
from mmfp.data.loaders.graph_static import load_static_adjacency
from mmfp.data.universe import ALL_TICKERS
from mmfp.features.macro_events import load_or_compute_macro_features
from mmfp.features.news_aggregate import aggregate_daily
from mmfp.features.news_stats import (
    LOG1P_COLS as NEWS_STATS_LOG1P_COLS,
    PASSTHROUGH_COLS as NEWS_STATS_PASSTHROUGH_COLS,
    build_news_stats_11dim,
)
from mmfp.features.price_ta import (
    NORMALISED_FEATURE_COLS as PRICE_NORM_COLS,
    load_or_compute_price_features,
)
from mmfp.features.scalers import FittedScaler
from mmfp.features.social_features import (
    LOG1P_COLS as SOCIAL_LOG1P_COLS,
    PASSTHROUGH_COLS as SOCIAL_PASSTHROUGH_COLS,
    load_or_compute_social_features,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fold boundaries — ported verbatim from v1 ``src/utils/config.py``.
# ---------------------------------------------------------------------------


#: Train start date used by the v1 runner (``2016-01-04``), one calendar
#: year after ``data.start_date=2015-02-03`` so the 252-day TA warm-up
#: plus rolling-z-score min_periods=60 is complete before training begins.
#:
#: Overridable via :attr:`DataConfig.start_date` does NOT change this —
#: the warm-up window lives between ``start_date`` and ``TRAIN_START``.
#: See v1 ``src/training/experiment_runner_v2.py::run_single_experiment``.
TRAIN_START: str = "2016-01-04"


#: Fold boundaries matching v1 ``TEMPORAL_FOLDS``. Each entry:
#:
#: * ``train_end`` — last day of training period (inclusive).
#: * ``test_start`` — first day of test period (inclusive).
#: * ``test_end`` — last day of test period (inclusive).
#:
#: Val is carved out of train via ``training.val_fraction`` (v1 convention
#: mirrored in :func:`_compute_val_start`).
FOLD_BOUNDARIES: list[dict[str, str]] = [
    {"train_end": "2019-06-30", "test_start": "2019-07-01", "test_end": "2020-06-30"},
    {"train_end": "2020-06-30", "test_start": "2020-07-01", "test_end": "2021-06-30"},
    {"train_end": "2021-06-30", "test_start": "2021-07-01", "test_end": "2022-06-30"},
    {"train_end": "2022-06-30", "test_start": "2022-07-01", "test_end": "2023-06-30"},
    {"train_end": "2023-06-30", "test_start": "2023-07-01", "test_end": "2023-12-31"},
]


def _compute_val_start(
    train_start: str,
    train_end: str,
    val_fraction: float,
) -> str:
    """Carve the last ``val_fraction`` of the train window for validation.

    Ported from v1 ``src/training/experiment_runner_v2.py``. Uses calendar
    days rather than trading days for parity; the resulting split date is
    rounded to the nearest day. ``val_start`` is inclusive.
    """
    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)
    total_days = (end - start).days
    val_days = int(total_days * val_fraction)
    val_start = end - pd.Timedelta(days=val_days)
    return str(val_start.date())


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------


#: Canonical order in which modalities are concatenated into the feature
#: array. Graph nodes are NOT part of this flat layout (the graph modality
#: carries its own per-date node-feature tensor in :class:`FoldArtifacts`).
_CANONICAL_MODALITY_ORDER: tuple[str, ...] = ("price", "macro", "news", "social")


@dataclass(frozen=True)
class FeatureSchema:
    """Slice map for per-modality column ranges in the concatenated feature array.

    The concatenation order is ``(price, macro, news, social)``. Absent
    modalities have a ``None`` slice so callers can introspect which
    modalities were materialised for a fold.

    Attributes
    ----------
    price
        Slice covering price columns (always present).
    macro, news, social
        Slice covering that modality, or ``None`` if the modality was
        disabled in the config.
    """

    price: slice
    macro: slice | None = None
    news: slice | None = None
    social: slice | None = None

    @property
    def modalities(self) -> list[str]:
        """Names of modalities with a non-empty slice, in canonical order."""
        active: list[str] = []
        for name in _CANONICAL_MODALITY_ORDER:
            sl = getattr(self, name)
            if sl is not None:
                active.append(name)
        return active

    def range_for(self, modality: str) -> slice:
        """Return the column slice for ``modality`` or raise ``KeyError``.

        ``modality`` must be one of the canonical names ("price",
        "macro", "news", "social"). Graph is not a flat-feature modality.
        """
        if modality not in _CANONICAL_MODALITY_ORDER:
            raise KeyError(
                f"Unknown modality {modality!r}. "
                f"Expected one of {_CANONICAL_MODALITY_ORDER}."
            )
        sl = getattr(self, modality)
        if sl is None:
            raise KeyError(
                f"Modality {modality!r} was not materialised for this fold."
            )
        return sl

    def total_dim(self) -> int:
        """Total concatenated feature dimension across active modalities."""
        active = [getattr(self, m) for m in _CANONICAL_MODALITY_ORDER if getattr(self, m) is not None]
        if not active:
            return 0
        return max(sl.stop for sl in active)


# ---------------------------------------------------------------------------
# FoldArtifacts
# ---------------------------------------------------------------------------


@dataclass
class FoldArtifacts:
    """Everything downstream datasets and models need for one fold.

    Flat arrays hold one row per ``(Date, Ticker)`` sample post-filter.
    The ``price_features_by_stock_day`` mapping stores *every* processed
    trading day per ticker (including samples dropped by deadzone or
    terminal-row filters) so lookback sequence windows can be reconstructed
    without re-reading from the train/val/test arrays.

    Attributes
    ----------
    {train,val,test}_features
        ``(N, F_total)`` float32 arrays. Column layout matches
        :attr:`feature_schema`.
    {train,val,test}_labels_cls
        ``(N,)`` int64; values in ``{-1, 0, 1}`` (``-1`` = deadzone and
        only present when ``"direction"`` is NOT a target). When
        ``"direction"`` is a target, assembly drops deadzone rows so the
        returned labels are ``{0, 1}``.
    {train,val,test}_labels_reg
        ``(N,)`` float32 log return. NaNs already removed if
        ``"return"`` in targets.
    {train,val,test}_labels_vol
        ``(N,)`` float32 ``|log return|``. NaNs removed if
        ``"volatility"`` in targets.
    {train,val,test}_dates
        ``(N,)`` ``"<U10"`` ISO date strings (``YYYY-MM-DD``).
    {train,val,test}_stock_idx
        ``(N,)`` int64 indices into :attr:`tickers`.
    feature_schema
        :class:`FeatureSchema` describing per-modality column ranges.
    scalers
        ``{modality_name: FittedScaler}`` fit on the train split only.
    static_adj
        ``(N_stocks, N_stocks)`` float32 adjacency matrix, or ``None``
        if graph is disabled.
    dynamic_snapshots
        ``{ISO_date: (N_stocks, N_stocks) adjacency}``, or ``None``.
        Ordering matches the snapshot dates on disk; callers resolve a
        sample's snapshot via
        :func:`mmfp.data.loaders.graph_dynamic.pick_snapshot_for_date`.
    tickers
        Ticker order that defines ``stock_idx`` (matches
        :data:`~mmfp.data.universe.ALL_TICKERS`).
    price_features_by_stock_day
        ``{(ticker, iso_date): (F_price,) float32}`` over every processed
        trading day for every ticker (including warm-up and terminal
        rows). Used by :class:`~mmfp.datasets.multimodal.MultiModalDataset`
        to reconstruct lookback sequences.
    fold_idx, train_end, val_end, test_end
        Fold metadata for downstream logging / caching.
    """

    train_features: np.ndarray
    train_labels_cls: np.ndarray
    train_labels_reg: np.ndarray
    train_labels_vol: np.ndarray
    train_dates: np.ndarray
    train_stock_idx: np.ndarray

    val_features: np.ndarray
    val_labels_cls: np.ndarray
    val_labels_reg: np.ndarray
    val_labels_vol: np.ndarray
    val_dates: np.ndarray
    val_stock_idx: np.ndarray

    test_features: np.ndarray
    test_labels_cls: np.ndarray
    test_labels_reg: np.ndarray
    test_labels_vol: np.ndarray
    test_dates: np.ndarray
    test_stock_idx: np.ndarray

    feature_schema: FeatureSchema
    scalers: dict[str, FittedScaler]
    static_adj: np.ndarray | None
    dynamic_snapshots: dict[str, np.ndarray] | None
    tickers: list[str]

    price_features_by_stock_day: dict[tuple[str, str], np.ndarray] = field(
        default_factory=dict
    )

    fold_idx: int = 0
    train_start: str = ""
    train_end: str = ""
    val_end: str = ""
    test_end: str = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def assemble_fold(cfg: ExperimentConfig) -> FoldArtifacts:
    """Assemble a fold from raw/feature caches into :class:`FoldArtifacts`.

    Parameters
    ----------
    cfg
        Validated :class:`ExperimentConfig`. The fold index, active
        modalities, news encoder/aggregation, graph source, lookback,
        deadzone and target set all feed into assembly.

    Returns
    -------
    FoldArtifacts
        Train/val/test arrays, labels, scalers, graph artifacts and a
        :class:`FeatureSchema` describing the concatenated feature layout.

    Raises
    ------
    ValueError
        If the fold index is out of range or the resulting train/val/test
        split is empty (suggests a data-coverage bug).
    FileNotFoundError
        If a required cached feature/raw parquet is missing. Messages
        include the expected paths so callers can rebuild with the v1
        pipeline or the ``load_or_compute_*`` fallbacks.
    NotImplementedError
        If :attr:`IntradayConfig.enabled` is ``True`` — the v2 intraday
        loader is a stub (see spec Section 9 non-goal #3).
    """
    _reject_stubs(cfg)

    fold_idx = cfg.data.fold_idx
    if not 0 <= fold_idx < len(FOLD_BOUNDARIES):
        raise ValueError(
            f"fold_idx {fold_idx} out of range [0, {len(FOLD_BOUNDARIES) - 1}]"
        )

    fold = FOLD_BOUNDARIES[fold_idx]
    train_end = fold["train_end"]
    test_start = fold["test_start"]
    test_end = fold["test_end"]
    val_start = _compute_val_start(
        TRAIN_START, train_end, cfg.training.val_fraction
    )

    log.info(
        "assemble_fold: fold=%d train=[%s..%s) val=[%s..%s] test=[%s..%s]",
        fold_idx,
        TRAIN_START,
        val_start,
        val_start,
        train_end,
        test_start,
        test_end,
    )

    # ------------------------------------------------------------------
    # 1. Price features (always present).
    # ------------------------------------------------------------------
    price_df = _load_price_frame(cfg)

    # price_df columns: Date, Ticker, price TA features + labels
    # Price features consumed downstream are the ``_norm`` columns.
    price_feat_cols = PRICE_NORM_COLS

    # ------------------------------------------------------------------
    # 2. Macro features (optional, shared across tickers).
    # ------------------------------------------------------------------
    macro_df, macro_feat_cols, macro_passthrough_cols = _load_macro_frame(cfg)

    # ------------------------------------------------------------------
    # 3. News features (optional). Two paths: 11-dim stats OR 768-dim PCA.
    # ------------------------------------------------------------------
    news_df, news_feat_cols, news_log1p_cols, news_passthrough_cols = (
        _load_news_frame(cfg, train_end=train_end)
    )

    # ------------------------------------------------------------------
    # 4. Social features (optional).
    # ------------------------------------------------------------------
    social_df, social_feat_cols, social_log1p_cols, social_passthrough_cols = (
        _load_social_frame(cfg)
    )

    # ------------------------------------------------------------------
    # 5. Build the big (Date, Ticker) table: one row per (stock, trading day).
    # ------------------------------------------------------------------
    aligned = _align_modalities(
        price_df=price_df,
        price_feat_cols=price_feat_cols,
        macro_df=macro_df,
        macro_feat_cols=macro_feat_cols,
        news_df=news_df,
        news_feat_cols=news_feat_cols,
        social_df=social_df,
        social_feat_cols=social_feat_cols,
        cfg=cfg,
    )

    if aligned.empty:
        raise ValueError(
            "assemble_fold: aligned frame empty after load. "
            "Check data.start_date / end_date."
        )

    # ------------------------------------------------------------------
    # 6. Labels: derive cls/reg/vol once, fold-agnostic.
    # ------------------------------------------------------------------
    aligned = _add_labels(aligned, deadzone=cfg.data.deadzone)

    # ------------------------------------------------------------------
    # 7. Dynamic graph — if dynamic is required, drop samples predating
    #    the first snapshot.
    # ------------------------------------------------------------------
    static_adj: np.ndarray | None = None
    dynamic_snapshots: dict[str, np.ndarray] | None = None
    if cfg.graph.enabled:
        if cfg.graph.source in ("static_gics", "static_plus_dynamic"):
            static_adj = load_static_adjacency()
        if cfg.graph.source in ("dynamic_corr", "static_plus_dynamic"):
            dynamic_snapshots = load_dynamic_snapshots()
            aligned = _filter_by_dynamic_coverage(aligned, dynamic_snapshots, cfg)

    # ------------------------------------------------------------------
    # 8. Lookback filter (uses the FULL per-ticker history, including
    #    warm-up rows before TRAIN_START). Drop samples with insufficient
    #    history for their ticker. This MUST run before the train/val/
    #    test masks — a sample on the first day of training still needs
    #    its prior (lookback-1) trading days.
    # ------------------------------------------------------------------
    aligned = _apply_lookback_filter(aligned, lookback=cfg.data.lookback)

    # ------------------------------------------------------------------
    # 9. Fold masks. Warm-up rows (Date < TRAIN_START) are excluded from
    #    every split, but their feature vectors remain accessible via
    #    ``price_features_by_stock_day`` for lookback reconstruction.
    # ------------------------------------------------------------------
    train_mask = (aligned["Date"] >= pd.Timestamp(TRAIN_START)) & (
        aligned["Date"] < pd.Timestamp(val_start)
    )
    val_mask = (aligned["Date"] >= pd.Timestamp(val_start)) & (
        aligned["Date"] <= pd.Timestamp(train_end)
    )
    test_mask = (aligned["Date"] >= pd.Timestamp(test_start)) & (
        aligned["Date"] <= pd.Timestamp(test_end)
    )

    # Apply target-aware row filters. Direction drops deadzone;
    # volatility drops NaN ``label_vol``.
    target_mask = _target_mask(aligned, targets=cfg.head.targets)

    train_df = aligned[train_mask & target_mask].copy()
    val_df = aligned[val_mask & target_mask].copy()
    test_df = aligned[test_mask & target_mask].copy()

    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if split_df.empty:
            raise ValueError(
                f"assemble_fold: {split_name} split is empty for fold {fold_idx}. "
                "Check fold boundaries and target filters."
            )

    # ------------------------------------------------------------------
    # 9. Fit per-modality scalers on TRAIN rows only; transform val/test.
    # ------------------------------------------------------------------
    feature_blocks: dict[str, dict[str, np.ndarray]] = {}
    scalers: dict[str, FittedScaler] = {}

    schema_kwargs: dict[str, slice | None] = {
        "price": slice(0, 0),
        "macro": None,
        "news": None,
        "social": None,
    }

    offset = 0
    for modality in _CANONICAL_MODALITY_ORDER:
        cols, log1p_cols, passthrough_cols = _modality_columns(
            modality,
            price_feat_cols=price_feat_cols,
            macro_feat_cols=macro_feat_cols,
            macro_passthrough_cols=macro_passthrough_cols,
            news_feat_cols=news_feat_cols,
            news_log1p_cols=news_log1p_cols,
            news_passthrough_cols=news_passthrough_cols,
            social_feat_cols=social_feat_cols,
            social_log1p_cols=social_log1p_cols,
            social_passthrough_cols=social_passthrough_cols,
        )
        if not cols:
            schema_kwargs[modality] = None
            continue

        scaler = FittedScaler(
            log1p_cols=log1p_cols, passthrough_cols=passthrough_cols,
        )
        train_block = scaler.fit_transform(train_df[cols])
        val_block = scaler.transform(val_df[cols])
        test_block = scaler.transform(test_df[cols])

        feature_blocks[modality] = {
            "train": train_block,
            "val": val_block,
            "test": test_block,
        }
        scalers[modality] = scaler

        width = len(cols)
        schema_kwargs[modality] = slice(offset, offset + width)
        offset += width

    feature_schema = FeatureSchema(**schema_kwargs)  # type: ignore[arg-type]

    # Concatenate blocks in canonical order.
    def _concat(split: str) -> np.ndarray:
        pieces = [
            feature_blocks[m][split]
            for m in _CANONICAL_MODALITY_ORDER
            if m in feature_blocks
        ]
        return np.concatenate(pieces, axis=1).astype(np.float32, copy=False)

    train_features = _concat("train")
    val_features = _concat("val")
    test_features = _concat("test")

    # ------------------------------------------------------------------
    # 10. Build per-(ticker, day) price-feature map for lookback windows.
    # ------------------------------------------------------------------
    price_by_stock_day = _build_price_by_stock_day(
        aligned, price_feat_cols=price_feat_cols, scaler=scalers["price"],
    )

    # ------------------------------------------------------------------
    # 11. Pack labels / metadata for each split.
    # ------------------------------------------------------------------
    def _pack(df: pd.DataFrame) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        cls = df["label_cls"].to_numpy(dtype=np.int64)
        reg = df["label_reg"].to_numpy(dtype=np.float32)
        vol = df["label_vol"].to_numpy(dtype=np.float32)
        dates = df["Date"].dt.strftime("%Y-%m-%d").to_numpy(dtype="<U10")
        stock = df["stock_idx"].to_numpy(dtype=np.int64)
        return cls, reg, vol, dates, stock

    tr_cls, tr_reg, tr_vol, tr_dates, tr_stock = _pack(train_df)
    va_cls, va_reg, va_vol, va_dates, va_stock = _pack(val_df)
    te_cls, te_reg, te_vol, te_dates, te_stock = _pack(test_df)

    artifacts = FoldArtifacts(
        train_features=train_features,
        train_labels_cls=tr_cls,
        train_labels_reg=tr_reg,
        train_labels_vol=tr_vol,
        train_dates=tr_dates,
        train_stock_idx=tr_stock,
        val_features=val_features,
        val_labels_cls=va_cls,
        val_labels_reg=va_reg,
        val_labels_vol=va_vol,
        val_dates=va_dates,
        val_stock_idx=va_stock,
        test_features=test_features,
        test_labels_cls=te_cls,
        test_labels_reg=te_reg,
        test_labels_vol=te_vol,
        test_dates=te_dates,
        test_stock_idx=te_stock,
        feature_schema=feature_schema,
        scalers=scalers,
        static_adj=static_adj,
        dynamic_snapshots=dynamic_snapshots,
        tickers=list(ALL_TICKERS),
        price_features_by_stock_day=price_by_stock_day,
        fold_idx=fold_idx,
        train_start=TRAIN_START,
        train_end=val_start,  # train *rows* end at val_start (exclusive).
        val_end=train_end,
        test_end=test_end,
    )
    log.info(
        "assemble_fold: fold=%d n_train=%d n_val=%d n_test=%d F_total=%d",
        fold_idx,
        train_features.shape[0],
        val_features.shape[0],
        test_features.shape[0],
        feature_schema.total_dim(),
    )
    return artifacts


# ---------------------------------------------------------------------------
# Per-modality loaders.
# ---------------------------------------------------------------------------


def _reject_stubs(cfg: ExperimentConfig) -> None:
    """Refuse to assemble when the config sets unsupported stub flags."""
    if cfg.intraday.enabled:
        raise NotImplementedError(
            "IntradayConfig.enabled=True is a v2 stub — see spec Section 9 "
            "non-goal #3. No intraday loader is implemented; unset the flag."
        )


def _load_price_frame(cfg: ExperimentConfig) -> pd.DataFrame:
    """Load (and narrow) the price feature parquet."""
    df = load_or_compute_price_features()
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # Universe narrow.
    df = df[df["Ticker"].isin(ALL_TICKERS)]
    # Date narrow to ``data.start_date .. end_date``.
    start = pd.Timestamp(cfg.data.start_date)
    end = pd.Timestamp(cfg.data.end_date)
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]

    # Sanity: normalised feature columns exist.
    missing = [c for c in PRICE_NORM_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"price_features.parquet missing normalised columns {missing}. "
            "Re-run mmfp.features.price_ta.compute_price_features."
        )

    return df


def _load_macro_frame(
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Load the macro feature frame and declare log1p / passthrough cols."""
    if not cfg.macro.enabled:
        return None, [], []

    df = load_or_compute_macro_features()
    df = df.copy()
    # The cached parquet keeps ``Date`` as the index; promote to column.
    if "Date" not in df.columns:
        df.index.name = "Date"
        df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    # Only the ``_norm`` columns + ``is_fomc_window`` are consumed.
    candidate_cols = [c for c in df.columns if c.endswith("_norm")]
    passthrough_cols: list[str] = []
    if "is_fomc_window" in df.columns:
        candidate_cols.append("is_fomc_window")
        passthrough_cols.append("is_fomc_window")

    # Keep only what we need for join + consumption.
    keep = ["Date"] + candidate_cols
    df = df[keep].drop_duplicates(subset=["Date"]).sort_values("Date")

    return df, candidate_cols, passthrough_cols


def _load_news_frame(
    cfg: ExperimentConfig, *, train_end: str,
) -> tuple[pd.DataFrame | None, list[str], list[str], list[str]]:
    """Load the news feature frame; dispatches between 11-dim and 768-dim.

    The 768-dim path fits PCA on train-fold articles only (audit B.4 fix).
    """
    if not cfg.news.enabled:
        return None, [], [], []

    pca_train_end = cfg.news.pca_train_end or train_end

    if cfg.news.encoder == "finbert_11dim":
        df, feat_cols, log1p_cols, passthrough_cols = _load_news_11dim(cfg)
    else:
        df, feat_cols, log1p_cols, passthrough_cols = _load_news_embedding_path(
            cfg, pca_train_end=pca_train_end,
        )

    # All encoders: shift by ``lag_days`` business days so day T gets news
    # from day T-lag_days. v1 convention; prevents same-day commentary
    # leakage (which had r=0.14 same-day vs r=0.005 next-day in Phase 1-R).
    if cfg.news.lag_days > 0:
        df = df.copy()
        df["Date"] = df["Date"] + pd.offsets.BDay(cfg.news.lag_days)

    return df, feat_cols, log1p_cols, passthrough_cols


def _load_news_11dim(
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Load or compute the 11-dim statistical news path."""
    from mmfp.data.paths import FEATURES_DIR

    cache = FEATURES_DIR / "news_features.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        df["Date"] = pd.to_datetime(df["Date"])
        feat_cols = [c for c in df.columns if c not in ("Date", "Ticker")]
        # Resolve parametrised column names in LOG1P/PASSTHROUGH lists.
        window = cfg.social.aggregation_window  # news & social share W=3 default.
        # NewsConfig has no explicit window knob; the cached parquet uses
        # _w3 suffixes which is the study default.
        log1p_cols = [c.replace("{W}", "3") for c in NEWS_STATS_LOG1P_COLS]
        passthrough_cols = list(NEWS_STATS_PASSTHROUGH_COLS)
        # Only keep columns that actually exist in the cached parquet.
        log1p_cols = [c for c in log1p_cols if c in feat_cols]
        passthrough_cols = [c for c in passthrough_cols if c in feat_cols]
        # If the cached parquet predates the log1p fix, honour cfg flag
        # by applying log1p downstream via FittedScaler (already handled).
        return df[["Date", "Ticker"] + feat_cols], feat_cols, log1p_cols, passthrough_cols

    # Cache miss: recompute. Requires per-article sentiment parquet.
    sentiments_cache = FEATURES_DIR / "news_per_article_sentiments.parquet"
    if not sentiments_cache.exists():
        raise FileNotFoundError(
            f"Neither {cache} nor {sentiments_cache} exists. "
            "Run v1 ``python -m src.features.encode_news`` to rebuild."
        )
    per_article = pd.read_parquet(sentiments_cache)
    df = build_news_stats_11dim(
        per_article,
        log1p_volume=cfg.news.log1p_volume,
        empty_day_policy=cfg.news.empty_day_policy,
        rolling_window=3,
    )
    feat_cols = [c for c in df.columns if c not in ("Date", "Ticker")]
    log1p_cols = [c.replace("{W}", "3") for c in NEWS_STATS_LOG1P_COLS]
    log1p_cols = [c for c in log1p_cols if c in feat_cols]
    passthrough_cols = [c for c in NEWS_STATS_PASSTHROUGH_COLS if c in feat_cols]
    return df, feat_cols, log1p_cols, passthrough_cols


def _load_news_embedding_path(
    cfg: ExperimentConfig, *, pca_train_end: str,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Aggregate per-article 768-dim embeddings and PCA to ``cfg.news.pca_dims``.

    The PCA basis is fit on *articles* (not per-day vectors) with
    ``Date <= pca_train_end``. The basis is then applied to every
    article; per-day aggregation happens on the reduced representation.
    """
    from mmfp.data.paths import FEATURES_DIR
    from sklearn.decomposition import PCA

    # Locate the per-article embedding parquet for the chosen encoder.
    encoder_to_parquet = {
        "finbert_cls_768": FEATURES_DIR / "news_per_article_768.parquet",
        # Older v1 parquets named by encoder; we keep them if present:
        "bge_base": FEATURES_DIR / "news_embeddings_bge.parquet",
        "qwen3_embedding": FEATURES_DIR / "news_embeddings_qwen3.parquet",
        "deberta_v3_financial": FEATURES_DIR / "news_embeddings_deberta.parquet",
    }
    parquet = encoder_to_parquet.get(cfg.news.encoder)
    if parquet is None:
        raise ValueError(
            f"No cached 768-dim embedding parquet known for encoder "
            f"{cfg.news.encoder!r}. Run mmfp.features.news_encode to build one."
        )
    if not parquet.exists():
        raise FileNotFoundError(
            f"Embedding parquet {parquet} missing for encoder "
            f"{cfg.news.encoder!r}. Run mmfp.features.news_encode first."
        )

    per_article = pd.read_parquet(parquet)
    # Normalise column names / dtypes.
    if "sent_score" not in per_article.columns:
        # Fallback: derive from p_pos/p_neg if present (parity with
        # news_stats path).
        if {"p_pos", "p_neg"}.issubset(per_article.columns):
            per_article = per_article.assign(
                sent_score=per_article["p_pos"].astype(np.float32)
                - per_article["p_neg"].astype(np.float32),
            )
        else:
            # Not every embedding parquet carries sentiment; attention /
            # max_sentiment strategies need it. When absent we zero-fill
            # and warn; arithmetic / recency / spherical still work.
            log.warning(
                "Embedding parquet %s has no sent_score; zero-filling. "
                "Attention/max_sentiment aggregation will reduce to arithmetic.",
                parquet,
            )
            per_article = per_article.assign(sent_score=0.0)

    # Normalise Date dtype (some parquets arrive tz-aware UTC).
    per_article["Date"] = pd.to_datetime(per_article["Date"], errors="coerce")
    if getattr(per_article["Date"].dtype, "tz", None) is not None:
        per_article["Date"] = (
            per_article["Date"].dt.tz_convert("UTC").dt.tz_localize(None)
        )
    per_article["Date"] = per_article["Date"].dt.normalize()

    emb_cols = sorted(
        (c for c in per_article.columns if c.startswith("emb_")),
        key=lambda c: int(c.split("_", 1)[1]),
    )
    if not emb_cols:
        raise ValueError(
            f"{parquet} has no ``emb_*`` columns; cannot run 768-dim path."
        )

    # Per-fold PCA fit on train-fold articles only.
    pca_dims = cfg.news.pca_dims
    if pca_dims is None:
        raise ValueError(
            "NewsConfig.pca_dims must be set for 768-dim encoders "
            "(schema validator enforces pca_dims=None only for 11-dim)."
        )
    train_cutoff = pd.Timestamp(pca_train_end)
    train_mask = per_article["Date"] <= train_cutoff
    train_matrix = per_article.loc[train_mask, emb_cols].to_numpy(
        dtype=np.float32, copy=False
    )
    if train_matrix.shape[0] < pca_dims:
        raise ValueError(
            f"Per-fold PCA needs >= {pca_dims} train articles; got "
            f"{train_matrix.shape[0]} on or before {pca_train_end}."
        )

    pca = PCA(n_components=pca_dims, random_state=cfg.seed)
    pca.fit(train_matrix)
    log.info(
        "news PCA: 768->%d fit on %d articles (<=%s); ev_sum=%.4f",
        pca_dims,
        train_matrix.shape[0],
        pca_train_end,
        float(pca.explained_variance_ratio_.sum()),
    )

    all_matrix = per_article[emb_cols].to_numpy(dtype=np.float32, copy=False)
    reduced = pca.transform(all_matrix).astype(np.float32)

    # Rebuild a per-article DataFrame with the reduced embedding columns.
    red_cols = [f"emb_{i}" for i in range(pca_dims)]
    reduced_df = pd.DataFrame(reduced, columns=red_cols)
    reduced_df.insert(0, "Ticker", per_article["Ticker"].to_numpy())
    reduced_df.insert(1, "Date", per_article["Date"].to_numpy())
    reduced_df["sent_score"] = per_article["sent_score"].to_numpy(dtype=np.float32)
    if "article_order" in per_article.columns:
        reduced_df["article_order"] = per_article["article_order"].to_numpy()

    # Aggregate to per-(Ticker, Date) using the configured strategy.
    agg = aggregate_daily(
        reduced_df,
        strategy=cfg.news.aggregation,
        dispersion_feature=cfg.news.dispersion_feature
        and cfg.news.aggregation == "spherical_mean",
    )
    # ``has_news=1`` on every aggregated row; assembly later joins
    # against the full trading grid and zero-fills absent (Date, Ticker)
    # pairs — so ``has_news`` is reconstructed from news_volume at join
    # time.
    feat_cols = [c for c in agg.columns if c not in ("Date", "Ticker")]
    # log1p on the *aggregated* news_volume uses the downstream
    # FittedScaler; we only declare the column here.
    log1p_cols = ["news_volume"] if cfg.news.log1p_volume else []
    passthrough_cols = ["has_news"] if "has_news" in feat_cols else []
    return agg, feat_cols, log1p_cols, passthrough_cols


def _load_social_frame(
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame | None, list[str], list[str], list[str]]:
    """Load the social (StockTwits) feature frame."""
    if not cfg.social.enabled:
        return None, [], [], []

    df = load_or_compute_social_features(
        aggregation_window=cfg.social.aggregation_window,
        log1p_volume=cfg.social.log1p_volume,
    )
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    feat_cols = [c for c in df.columns if c not in ("Date", "Ticker")]
    window = cfg.social.aggregation_window
    log1p_cols = [c.replace("{W}", str(window)) for c in SOCIAL_LOG1P_COLS]
    log1p_cols = [c for c in log1p_cols if c in feat_cols]
    passthrough_cols = [c for c in SOCIAL_PASSTHROUGH_COLS if c in feat_cols]

    # Apply lag (same rationale as news).
    if cfg.social.lag_days > 0:
        df["Date"] = df["Date"] + pd.offsets.BDay(cfg.social.lag_days)

    return df, feat_cols, log1p_cols, passthrough_cols


# ---------------------------------------------------------------------------
# Alignment and labels
# ---------------------------------------------------------------------------


def _align_modalities(
    *,
    price_df: pd.DataFrame,
    price_feat_cols: list[str],
    macro_df: pd.DataFrame | None,
    macro_feat_cols: list[str],
    news_df: pd.DataFrame | None,
    news_feat_cols: list[str],
    social_df: pd.DataFrame | None,
    social_feat_cols: list[str],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Left-join every modality onto the price (Date, Ticker) grid.

    ``next_day_return`` and ``log_return`` come from the price frame and
    are carried through for label derivation.
    """
    keep_cols = ["Date", "Ticker"] + price_feat_cols + [
        c for c in ("next_day_return", "log_return", "movement") if c in price_df.columns
    ]
    existing = [c for c in keep_cols if c in price_df.columns]
    result = price_df[existing].copy()

    # Price ``_norm`` columns can have NaN during warm-up; zero-fill.
    result[price_feat_cols] = result[price_feat_cols].fillna(0.0)

    ticker_to_idx = {t: i for i, t in enumerate(ALL_TICKERS)}
    result["stock_idx"] = result["Ticker"].map(ticker_to_idx)

    if macro_df is not None:
        # Date-only join (macro is ticker-agnostic).
        result = result.merge(macro_df, on="Date", how="left")
        result[macro_feat_cols] = result[macro_feat_cols].fillna(0.0)

    if news_df is not None:
        # (Date, Ticker) join with lag already applied upstream.
        news_cols_present = [c for c in news_feat_cols if c in news_df.columns]
        result = result.merge(
            news_df[["Date", "Ticker"] + news_cols_present],
            on=["Date", "Ticker"],
            how="left",
        )
        if "has_news" not in news_cols_present:
            # Derive has_news from news_volume presence to preserve v1 behaviour.
            if "news_volume" in news_cols_present:
                result["has_news"] = (
                    result["news_volume"].fillna(0) > 0
                ).astype(np.float32)
            else:
                result["has_news"] = (~result[news_cols_present[0]].isna()).astype(
                    np.float32
                )
            if "has_news" not in news_feat_cols:
                news_feat_cols.append("has_news")
        result[news_feat_cols] = result[news_feat_cols].fillna(0.0)

    if social_df is not None:
        social_cols_present = [c for c in social_feat_cols if c in social_df.columns]
        result = result.merge(
            social_df[["Date", "Ticker"] + social_cols_present],
            on=["Date", "Ticker"],
            how="left",
        )
        if "has_social" not in social_cols_present:
            if "social_volume" in social_cols_present:
                result["has_social"] = (
                    result["social_volume"].fillna(0) > 0
                ).astype(np.float32)
            else:
                result["has_social"] = (
                    ~result[social_cols_present[0]].isna()
                ).astype(np.float32)
            if "has_social" not in social_feat_cols:
                social_feat_cols.append("has_social")
        result[social_feat_cols] = result[social_feat_cols].fillna(0.0)

    # Stable ordering for downstream indexing.
    result = result.sort_values(["Ticker", "Date"], kind="mergesort").reset_index(
        drop=True
    )
    return result


def _add_labels(frame: pd.DataFrame, *, deadzone: float) -> pd.DataFrame:
    """Derive ``label_cls`` (deadzone-aware), ``label_reg``, ``label_vol``.

    * ``label_reg`` is ``next_day_return`` from the price frame.
    * ``label_cls`` is ``1`` above deadzone, ``0`` below negative deadzone,
      ``-1`` within the deadzone and for rows with NaN returns. Direction
      target downstream drops the ``-1`` rows; volatility retains them.
    * ``label_vol`` is ``|next_day_return|`` with NaNs preserved so the
      volatility target filter can drop them later.
    """
    if "next_day_return" not in frame.columns:
        raise ValueError(
            "Aligned frame missing ``next_day_return`` column. Price parquet "
            "needs to carry next_day_return for label derivation."
        )
    out = frame.copy()
    ret = out["next_day_return"].astype(np.float64)

    cls = np.full(len(out), -1, dtype=np.int64)
    up_mask = ret.to_numpy() > deadzone
    down_mask = ret.to_numpy() < -deadzone
    cls[up_mask] = 1
    cls[down_mask] = 0
    nan_mask = np.isnan(ret.to_numpy())
    cls[nan_mask] = -1
    out["label_cls"] = cls

    out["label_reg"] = ret.astype(np.float32)
    out["label_vol"] = ret.abs().astype(np.float32)
    return out


def _target_mask(frame: pd.DataFrame, *, targets: list[str]) -> pd.Series:
    """Row-keep mask honouring the active targets.

    * ``"direction"`` in targets → drop deadzone (``label_cls == -1``).
    * ``"volatility"`` in targets → drop NaN ``label_vol``.
    * ``"return"`` in targets → drop NaN ``label_reg`` (also covers
      deadzone rows' NaN when ``"direction"`` absent; no-op otherwise).
    """
    keep = pd.Series(True, index=frame.index)
    if "direction" in targets:
        keep &= frame["label_cls"] != -1
    if "return" in targets:
        keep &= ~frame["label_reg"].isna()
    if "volatility" in targets:
        keep &= ~frame["label_vol"].isna()
    return keep


def _apply_lookback_filter(frame: pd.DataFrame, *, lookback: int) -> pd.DataFrame:
    """Drop rows that lack ``lookback-1`` prior trading days for their ticker.

    For ``lookback=1`` this is a no-op — every row has itself.
    """
    if lookback <= 1:
        return frame
    # Build the per-ticker position ordinal (0-indexed, sorted by Date).
    frame = frame.sort_values(["Ticker", "Date"], kind="mergesort").reset_index(
        drop=True
    )
    position = frame.groupby("Ticker", sort=False).cumcount()
    keep = position >= (lookback - 1)
    filtered = frame[keep.to_numpy()].reset_index(drop=True)
    return filtered


def _filter_by_dynamic_coverage(
    frame: pd.DataFrame,
    snapshots: dict[str, np.ndarray],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Drop rows whose date predates every dynamic snapshot.

    When ``cfg.graph.source == "static_plus_dynamic"`` a sample can still
    be kept even when dynamic coverage fails (the static fallback is
    always present); in that case this function is a no-op. For pure
    dynamic mode we log the drop count and filter.
    """
    if cfg.graph.source == "static_plus_dynamic":
        return frame
    if not snapshots:
        raise ValueError("Dynamic snapshots dict is empty.")
    first_snapshot = min(snapshots.keys())
    mask = frame["Date"].dt.strftime("%Y-%m-%d") >= first_snapshot
    dropped = int((~mask).sum())
    if dropped > 0:
        log.warning(
            "assemble_fold: dropped %d rows predating first dynamic snapshot %s",
            dropped,
            first_snapshot,
        )
    return frame[mask].reset_index(drop=True)


def _modality_columns(
    modality: str,
    *,
    price_feat_cols: list[str],
    macro_feat_cols: list[str],
    macro_passthrough_cols: list[str],
    news_feat_cols: list[str],
    news_log1p_cols: list[str],
    news_passthrough_cols: list[str],
    social_feat_cols: list[str],
    social_log1p_cols: list[str],
    social_passthrough_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(cols, log1p_cols, passthrough_cols)`` for ``modality``."""
    if modality == "price":
        # Price features are already rolling-z-scored upstream; the
        # per-fold scaler does a (small) re-standardisation on the
        # already-normalised columns — treat the z-scored columns as
        # plain numeric with no log1p, no passthrough.
        return list(price_feat_cols), [], []
    if modality == "macro":
        return list(macro_feat_cols), [], list(macro_passthrough_cols)
    if modality == "news":
        return list(news_feat_cols), list(news_log1p_cols), list(news_passthrough_cols)
    if modality == "social":
        return list(social_feat_cols), list(social_log1p_cols), list(social_passthrough_cols)
    raise ValueError(f"Unknown modality {modality!r}")


def _build_price_by_stock_day(
    aligned: pd.DataFrame,
    *,
    price_feat_cols: list[str],
    scaler: FittedScaler,
) -> dict[tuple[str, str], np.ndarray]:
    """Produce ``{(ticker, iso_date): scaled (F_price,) float32}`` for every row.

    Applying the **already-fit** price scaler to the full aligned frame
    keeps lookback-window reconstruction consistent with the per-split
    feature arrays. Val/test rows pass through the train-fit scaler —
    the same invariant enforced in :meth:`FittedScaler.transform`.
    """
    out: dict[tuple[str, str], np.ndarray] = {}
    cols = list(price_feat_cols)
    # Transform in one pass; slice per-row afterwards.
    transformed = scaler.transform(aligned[cols])
    tickers = aligned["Ticker"].to_numpy()
    dates = aligned["Date"].dt.strftime("%Y-%m-%d").to_numpy(dtype="<U10")
    for i in range(transformed.shape[0]):
        out[(str(tickers[i]), str(dates[i]))] = transformed[i].astype(
            np.float32, copy=False
        )
    return out


__all__ = [
    "FOLD_BOUNDARIES",
    "FeatureSchema",
    "FoldArtifacts",
    "TRAIN_START",
    "assemble_fold",
]
