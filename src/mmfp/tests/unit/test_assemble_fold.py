"""Unit tests for :func:`mmfp.data.assemble.assemble_fold`.

The tests hit real cached parquets under ``work/data/processed/features/``.
If those are missing the tests skip gracefully; CI that owns the data
will exercise them.

Key contracts exercised here:

* Smoke: an A7 price-only config produces a valid
  :class:`FoldArtifacts` with all three splits non-empty.
* Modality slicing: enabling news (11-dim) grows the feature dim and
  returns the correct :class:`FeatureSchema` slice for each modality.
* Scaler discipline: every active modality yields a fitted
  :class:`FittedScaler` keyed by modality name.
* Stock ordering: ``stock_idx`` in the artifact indexes into
  :data:`~mmfp.data.universe.ALL_TICKERS` (canonical alphabetical order).
* Lookback filter: ``lookback=20`` drops samples whose ticker has
  fewer than 19 prior trading days on the sample date.
* Deadzone filter: when ``"direction"`` is a target, class labels are
  strictly in ``{0, 1}``.
* Terminal-row filter: when ``"volatility"`` is a target, labels are
  never NaN.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.data.assemble import (
    FOLD_BOUNDARIES,
    FeatureSchema,
    FoldArtifacts,
    TRAIN_START,
    assemble_fold,
)
from mmfp.data.paths import FEATURES_DIR
from mmfp.data.universe import ALL_TICKERS


# ---------------------------------------------------------------------------
# Fixtures and helpers.
# ---------------------------------------------------------------------------


#: Require cached feature parquets to be present; skip cleanly if not.
_REQUIRED_FILES: tuple[Path, ...] = (
    FEATURES_DIR / "price_features.parquet",
)


pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED_FILES),
    reason=(
        "assemble_fold tests require cached feature parquets at "
        f"{[str(p) for p in _REQUIRED_FILES]}. Run v1 feature builders first."
    ),
)


def _base_cfg_dict(name: str) -> dict:
    """Return a deep-copied defaults dict with ``name`` set."""
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = name
    return d


@pytest.fixture
def price_only_cfg() -> ExperimentConfig:
    """A7-equivalent config: price only, fold 0, FF (lookback=1)."""
    d = _base_cfg_dict("price_only_fold0")
    d["data"]["lookback"] = 1
    d["data"]["fold_idx"] = 0
    d["model"]["price_encoder"] = "feedforward"
    return ExperimentConfig.model_validate(d)


# ---------------------------------------------------------------------------
# Schema tests (cheap; don't touch disk).
# ---------------------------------------------------------------------------


def test_feature_schema_range_for_active_modality() -> None:
    schema = FeatureSchema(
        price=slice(0, 10),
        macro=slice(10, 17),
        news=None,
        social=None,
    )
    assert schema.range_for("price") == slice(0, 10)
    assert schema.range_for("macro") == slice(10, 17)
    assert schema.total_dim() == 17
    assert schema.modalities == ["price", "macro"]


def test_feature_schema_missing_modality_raises() -> None:
    schema = FeatureSchema(price=slice(0, 10))
    with pytest.raises(KeyError, match="news"):
        schema.range_for("news")


def test_feature_schema_unknown_modality_raises() -> None:
    schema = FeatureSchema(price=slice(0, 10))
    with pytest.raises(KeyError, match="Unknown modality"):
        schema.range_for("graph")


def test_fold_boundaries_count_matches_spec() -> None:
    """Five expanding-window folds ported from v1 ``TEMPORAL_FOLDS``."""
    assert len(FOLD_BOUNDARIES) == 5
    for f in FOLD_BOUNDARIES:
        assert {"train_end", "test_start", "test_end"} <= set(f.keys())
        # Test window always starts the day after train ends.
        assert (
            pd.Timestamp(f["test_start"]) - pd.Timestamp(f["train_end"])
        ).days == 1


def test_train_start_after_data_start() -> None:
    """TRAIN_START must leave the TA warm-up period before it."""
    assert pd.Timestamp(TRAIN_START) > pd.Timestamp(DEFAULT_CONFIG["data"]["start_date"])


# ---------------------------------------------------------------------------
# Smoke tests over real data.
# ---------------------------------------------------------------------------


def test_assemble_price_only_fold0_smoke(price_only_cfg: ExperimentConfig) -> None:
    """End-to-end smoke on fold 0 with price only — the A7 baseline."""
    art = assemble_fold(price_only_cfg)
    assert isinstance(art, FoldArtifacts)
    assert art.fold_idx == 0
    assert art.train_features.shape[0] > 0
    assert art.val_features.shape[0] > 0
    assert art.test_features.shape[0] > 0
    # Price is the only active modality → total dim = price dim.
    assert art.feature_schema.modalities == ["price"]
    assert art.feature_schema.total_dim() == art.train_features.shape[1]
    # No NaNs / Infs in features.
    for arr in (art.train_features, art.val_features, art.test_features):
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()


def test_assemble_scaler_fitted_on_train_only(
    price_only_cfg: ExperimentConfig,
) -> None:
    """Every active modality must produce a fitted :class:`FittedScaler`."""
    art = assemble_fold(price_only_cfg)
    assert "price" in art.scalers
    scaler = art.scalers["price"]
    assert scaler.is_fitted is True
    assert scaler.feature_names_in_ is not None
    # Train block post-scaling has approximately zero-mean per column.
    # (Price cols are already rolling-z-scored so re-scaling yields a
    # tight near-zero mean.)
    train_means = art.train_features.mean(axis=0)
    assert np.abs(train_means).max() < 1e-4


def test_assemble_stock_idx_uses_canonical_universe(
    price_only_cfg: ExperimentConfig,
) -> None:
    """``stock_idx`` values must index into :data:`ALL_TICKERS`."""
    art = assemble_fold(price_only_cfg)
    idx_max = max(
        int(art.train_stock_idx.max()),
        int(art.val_stock_idx.max()),
        int(art.test_stock_idx.max()),
    )
    assert idx_max < len(ALL_TICKERS)
    assert art.tickers == ALL_TICKERS


def test_assemble_deadzone_filter_when_direction_active(
    price_only_cfg: ExperimentConfig,
) -> None:
    """``direction`` in targets ⇒ cls labels strictly in ``{0, 1}``."""
    art = assemble_fold(price_only_cfg)
    # Default targets = ["direction", "return"]; deadzone must be dropped.
    assert art.train_labels_cls.min() >= 0
    assert art.train_labels_cls.max() <= 1
    assert art.val_labels_cls.min() >= 0
    assert art.test_labels_cls.min() >= 0


def test_assemble_volatility_only_no_nan_labels() -> None:
    """Volatility-only target ⇒ ``label_vol`` has zero NaNs."""
    d = _base_cfg_dict("vol_only_fold0")
    d["data"]["fold_idx"] = 0
    d["head"]["targets"] = ["volatility"]
    d["head"]["architecture"] = "single_task"
    cfg = ExperimentConfig.model_validate(d)
    art = assemble_fold(cfg)
    for arr in (art.train_labels_vol, art.val_labels_vol, art.test_labels_vol):
        assert not np.isnan(arr).any()


def test_assemble_news_11dim_grows_feature_dim() -> None:
    """Enabling news (11-dim) adds news columns to the feature schema."""
    d = _base_cfg_dict("price_news11_fold0")
    d["data"]["fold_idx"] = 0
    d["news"]["enabled"] = True
    d["news"]["encoder"] = "finbert_11dim"
    d["news"]["pca_dims"] = None
    cfg = ExperimentConfig.model_validate(d)
    try:
        art = assemble_fold(cfg)
    except FileNotFoundError as exc:  # pragma: no cover
        pytest.skip(f"news cache missing: {exc}")

    assert art.feature_schema.news is not None
    news_slice = art.feature_schema.range_for("news")
    news_width = news_slice.stop - news_slice.start
    # 11-dim path: 6 stats + 3 rolling + volume + has_news = 11 columns.
    assert news_width == 11
    assert "news" in art.scalers
    assert art.scalers["news"].is_fitted


def test_assemble_macro_adds_passthrough_cols() -> None:
    """``is_fomc_window`` is declared passthrough in the macro scaler."""
    d = _base_cfg_dict("price_macro_fold0")
    d["data"]["fold_idx"] = 0
    d["macro"]["enabled"] = True
    cfg = ExperimentConfig.model_validate(d)
    try:
        art = assemble_fold(cfg)
    except FileNotFoundError as exc:  # pragma: no cover
        pytest.skip(f"macro cache missing: {exc}")
    assert "is_fomc_window" in art.scalers["macro"].passthrough_cols


def test_assemble_invalid_fold_idx_raises() -> None:
    """Fold idx outside ``[0, 4]`` raises at assembly time."""
    d = _base_cfg_dict("bad_fold")
    # Can't set fold_idx=7 via schema (ge=0, le=4); pass through model_validate.
    d["data"]["fold_idx"] = 4  # valid
    cfg = ExperimentConfig.model_validate(d)
    # Mutate the underlying validated object to force the runtime check.
    object.__setattr__(cfg.data, "fold_idx", 99)
    with pytest.raises(ValueError, match="out of range"):
        assemble_fold(cfg)


def test_assemble_intraday_stub_raises() -> None:
    """Intraday enabled raises :class:`NotImplementedError` (v2 stub)."""
    d = _base_cfg_dict("intraday_stub")
    d["intraday"]["enabled"] = True
    cfg = ExperimentConfig.model_validate(d)
    with pytest.raises(NotImplementedError, match="Intraday"):
        assemble_fold(cfg)


def test_assemble_graph_static_populates_adj() -> None:
    """Graph-static mode populates ``static_adj`` and leaves dynamic None."""
    d = _base_cfg_dict("price_graph_static")
    d["data"]["fold_idx"] = 0
    d["data"]["lookback"] = 1
    d["model"]["price_encoder"] = "feedforward"
    d["graph"]["enabled"] = True
    d["graph"]["source"] = "static_gics"
    cfg = ExperimentConfig.model_validate(d)
    try:
        art = assemble_fold(cfg)
    except FileNotFoundError as exc:  # pragma: no cover
        pytest.skip(f"graph cache missing: {exc}")
    assert art.static_adj is not None
    assert art.static_adj.shape == (55, 55)
    assert art.dynamic_snapshots is None


def test_assemble_lookback_20_fold0_history_available() -> None:
    """For ``lookback=20``, every retained sample has 19 prior days."""
    d = _base_cfg_dict("lookback20_fold0")
    d["data"]["fold_idx"] = 0
    d["data"]["lookback"] = 20
    cfg = ExperimentConfig.model_validate(d)
    art = assemble_fold(cfg)
    # All train samples have a non-NaN label.
    assert not np.isnan(art.train_labels_reg).any()
    # Every (ticker, date) in train split is preceded by >= 19 earlier
    # dates for that ticker in price_features_by_stock_day.
    per_ticker_dates: dict[str, list[str]] = {}
    for (ticker, date) in art.price_features_by_stock_day:
        per_ticker_dates.setdefault(ticker, []).append(date)
    for lst in per_ticker_dates.values():
        lst.sort()
    # Sample 100 random train rows (all-set check is O(N log N)).
    rng = np.random.default_rng(0)
    n = min(100, art.train_features.shape[0])
    idxs = rng.choice(art.train_features.shape[0], size=n, replace=False)
    for idx in idxs:
        tkr = ALL_TICKERS[int(art.train_stock_idx[idx])]
        day = str(art.train_dates[idx])
        pos = per_ticker_dates[tkr].index(day)
        assert pos >= 19, (
            f"Sample for {tkr} on {day}: only {pos} prior days available "
            "— assembly lookback filter did not run."
        )


def test_assemble_split_dates_respect_fold_boundaries(
    price_only_cfg: ExperimentConfig,
) -> None:
    """Train dates < val dates < test dates, all within fold bounds."""
    art = assemble_fold(price_only_cfg)
    fold = FOLD_BOUNDARIES[0]
    train_end_ts = pd.Timestamp(fold["train_end"])
    test_start_ts = pd.Timestamp(fold["test_start"])
    test_end_ts = pd.Timestamp(fold["test_end"])

    # `numpy.max` doesn't support `<U10` on some numpy versions; sort
    # via a native list comparison instead.
    train_dates_sorted = sorted(art.train_dates.tolist())
    val_dates_sorted = sorted(art.val_dates.tolist())
    test_dates_sorted = sorted(art.test_dates.tolist())
    train_max = pd.Timestamp(train_dates_sorted[-1])
    val_min = pd.Timestamp(val_dates_sorted[0])
    val_max = pd.Timestamp(val_dates_sorted[-1])
    test_min = pd.Timestamp(test_dates_sorted[0])
    test_max = pd.Timestamp(test_dates_sorted[-1])

    assert train_max < val_min, "train and val overlap"
    assert val_max <= train_end_ts, "val extends past fold.train_end"
    assert test_start_ts <= test_min <= test_max <= test_end_ts


def test_assemble_return_only_no_nan_labels() -> None:
    """Return-only target filters NaN ``label_reg``."""
    d = _base_cfg_dict("return_only")
    d["data"]["fold_idx"] = 0
    d["head"]["targets"] = ["return"]
    d["head"]["architecture"] = "single_task"
    cfg = ExperimentConfig.model_validate(d)
    art = assemble_fold(cfg)
    for arr in (art.train_labels_reg, art.val_labels_reg, art.test_labels_reg):
        assert not np.isnan(arr).any()
