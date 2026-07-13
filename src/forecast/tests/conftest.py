"""Shared pytest fixtures for v3 ``forecast`` tests.

Parallels :mod:`mmfp.tests.conftest` and is intentionally small and
dependency-free. Fixtures exported:

* :data:`minimal_cfg_dict`     — deep copy of :data:`DEFAULT_CONFIG`.
* :data:`minimal_cfg`           — validated :class:`V3ExperimentConfig`.
* :data:`defaults_toml_path`    — filesystem path to ``configs/defaults.toml``.
* :data:`a7_toml_path`          — filesystem path to the A7 reference config.
* :data:`synthetic_artifacts`  — factory for tiny in-memory
  :class:`FoldArtifacts` used by M4 dataset / trainer tests.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

# Ensure the v3 package is importable when pytest is launched from ``work/``
# or from the repo root. Same pattern as v2's conftest.
_WORK_DIR = Path(__file__).resolve().parents[2]
if str(_WORK_DIR) not in sys.path:
    sys.path.insert(0, str(_WORK_DIR))

from forecast.config.defaults import DEFAULT_CONFIG
from forecast.config.schema import V3ExperimentConfig
from mmfp.data.assemble import FeatureSchema, FoldArtifacts
from mmfp.data.universe import ALL_TICKERS


@pytest.fixture
def minimal_cfg_dict() -> dict[str, Any]:
    """Return a deep-copied canonical v3 defaults dict.

    Tests can freely mutate the copy without affecting sibling tests.
    """
    return copy.deepcopy(DEFAULT_CONFIG)


@pytest.fixture
def minimal_cfg(minimal_cfg_dict: dict[str, Any]) -> V3ExperimentConfig:
    """Return a validated :class:`V3ExperimentConfig` from canonical defaults."""
    return V3ExperimentConfig.model_validate(minimal_cfg_dict)


@pytest.fixture
def defaults_toml_path() -> Path:
    """Filesystem path to the canonical v3 defaults TOML."""
    return _WORK_DIR / "forecast" / "configs" / "defaults.toml"


@pytest.fixture
def a7_toml_path() -> Path:
    """Filesystem path to the A7 price-only TFT reference config."""
    return (
        _WORK_DIR
        / "forecast"
        / "configs"
        / "experiments"
        / "A7_price_only_tft.toml"
    )


# ---------------------------------------------------------------------------
# Synthetic FoldArtifacts factory for M4 dataset / trainer tests.
# ---------------------------------------------------------------------------


def _build_synthetic_artifacts(
    *,
    n_tickers: int = 3,
    n_days: int = 200,
    f_price: int = 13,
    f_macro: int = 9,
    f_news: int = 129,
    f_social: int = 7,
    include_macro: bool = False,
    include_news: bool = False,
    include_social: bool = False,
    include_graph_adj: bool = False,
    lookback: int = 60,
    deadzone: float = 0.0,
    seed: int = 0,
) -> FoldArtifacts:
    """Construct a small in-memory :class:`FoldArtifacts` for tests.

    Produces ``n_tickers * (n_days - (lookback - 1))`` samples in total,
    split 60/20/20 by date (train/val/test). Each split's feature
    columns follow the canonical ``[price, macro, news, social]`` order.
    Graph adjacency / dynamic snapshots are optional — most dataset
    tests don't need them.

    Returns a ``FoldArtifacts`` compatible with both v2 dataset code
    (for cross-checks) and v3's :class:`ForecastDataset`.
    """
    rng = np.random.default_rng(seed)

    tickers = list(ALL_TICKERS[:n_tickers])
    # Date grid — business days starting at 2016-01-04.
    start = pd.Timestamp("2016-01-04")
    date_grid = pd.bdate_range(start, periods=n_days)
    date_strs = [d.strftime("%Y-%m-%d") for d in date_grid]

    # Per-(ticker, date) price features.
    price_by_day: dict[tuple[str, str], np.ndarray] = {}
    for ticker in tickers:
        for date in date_strs:
            price_by_day[(ticker, date)] = rng.standard_normal(
                f_price
            ).astype(np.float32)

    # Feature schema construction.
    offset = 0
    price_slice = slice(offset, offset + f_price)
    offset += f_price
    macro_slice = None
    if include_macro:
        macro_slice = slice(offset, offset + f_macro)
        offset += f_macro
    news_slice = None
    if include_news:
        news_slice = slice(offset, offset + f_news)
        offset += f_news
    social_slice = None
    if include_social:
        social_slice = slice(offset, offset + f_social)
        offset += f_social

    schema = FeatureSchema(
        price=price_slice,
        macro=macro_slice,
        news=news_slice,
        social=social_slice,
    )
    total_dim = schema.total_dim()

    # Split boundaries: last (lookback-1) days are unavailable for
    # samples (lookback filter), so the usable date range is
    # date_strs[lookback - 1 :]. Within that, 60% train / 20% val / 20% test.
    usable_dates = date_strs[lookback - 1 :]
    n_usable = len(usable_dates)
    n_train = int(n_usable * 0.6)
    n_val = int(n_usable * 0.2)
    train_dates_ = usable_dates[:n_train]
    val_dates_ = usable_dates[n_train : n_train + n_val]
    test_dates_ = usable_dates[n_train + n_val :]

    def _pack_split(
        dates_for_split: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        dates_arr: list[str] = []
        stock_idxs: list[int] = []
        reg_labels: list[float] = []
        cls_labels: list[int] = []
        vol_labels: list[float] = []
        for date in dates_for_split:
            for si, ticker in enumerate(tickers):
                vec = np.zeros(total_dim, dtype=np.float32)
                vec[price_slice] = price_by_day[(ticker, date)]
                if macro_slice is not None:
                    vec[macro_slice] = rng.standard_normal(f_macro).astype(
                        np.float32
                    )
                if news_slice is not None:
                    vec[news_slice] = rng.standard_normal(f_news).astype(
                        np.float32
                    )
                if social_slice is not None:
                    vec[social_slice] = rng.standard_normal(f_social).astype(
                        np.float32
                    )
                rows.append(vec)
                dates_arr.append(date)
                stock_idxs.append(si)
                ret = float(rng.standard_normal() * 0.02)
                reg_labels.append(ret)
                if deadzone > 0 and abs(ret) <= deadzone:
                    cls_labels.append(-1)
                else:
                    cls_labels.append(1 if ret > 0 else 0)
                vol_labels.append(abs(ret))
        features = np.stack(rows, axis=0) if rows else np.zeros(
            (0, total_dim), dtype=np.float32
        )
        return (
            features.astype(np.float32),
            np.asarray(cls_labels, dtype=np.int64),
            np.asarray(reg_labels, dtype=np.float32),
            np.asarray(vol_labels, dtype=np.float32),
            np.asarray(dates_arr, dtype="<U10"),
            np.asarray(stock_idxs, dtype=np.int64),
        )

    tr_feat, tr_cls, tr_reg, tr_vol, tr_dates_arr, tr_stock = _pack_split(
        train_dates_
    )
    va_feat, va_cls, va_reg, va_vol, va_dates_arr, va_stock = _pack_split(
        val_dates_
    )
    te_feat, te_cls, te_reg, te_vol, te_dates_arr, te_stock = _pack_split(
        test_dates_
    )

    static_adj = None
    if include_graph_adj:
        adj = np.eye(n_tickers, dtype=np.float32)
        static_adj = adj

    artifacts = FoldArtifacts(
        train_features=tr_feat,
        train_labels_cls=tr_cls,
        train_labels_reg=tr_reg,
        train_labels_vol=tr_vol,
        train_dates=tr_dates_arr,
        train_stock_idx=tr_stock,
        val_features=va_feat,
        val_labels_cls=va_cls,
        val_labels_reg=va_reg,
        val_labels_vol=va_vol,
        val_dates=va_dates_arr,
        val_stock_idx=va_stock,
        test_features=te_feat,
        test_labels_cls=te_cls,
        test_labels_reg=te_reg,
        test_labels_vol=te_vol,
        test_dates=te_dates_arr,
        test_stock_idx=te_stock,
        feature_schema=schema,
        scalers={},
        static_adj=static_adj,
        dynamic_snapshots=None,
        tickers=tickers,
        price_features_by_stock_day=price_by_day,
        fold_idx=0,
        train_start="2016-01-04",
        train_end=train_dates_[-1] if train_dates_ else "",
        val_end=val_dates_[-1] if val_dates_ else "",
        test_end=test_dates_[-1] if test_dates_ else "",
    )
    return artifacts


@pytest.fixture
def synthetic_artifacts_factory() -> Callable[..., FoldArtifacts]:
    """Factory fixture returning the synthetic-artifact builder.

    Pass kwargs (``n_tickers``, ``n_days``, ``include_macro``, ...) to
    tailor to a given test case. Example::

        def test_something(synthetic_artifacts_factory):
            arts = synthetic_artifacts_factory(n_days=150, include_news=True)
    """
    return _build_synthetic_artifacts


@pytest.fixture
def synthetic_artifacts() -> FoldArtifacts:
    """Default synthetic artifacts: 3 tickers × 200 days, price only."""
    return _build_synthetic_artifacts()
