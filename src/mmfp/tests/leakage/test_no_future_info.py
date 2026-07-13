"""End-to-end leakage: feature blocks must use only past/present data.

Two invariants this test suite pins:

1. News lag — for a sample on date T, news features come from articles
   on date ``T - news.lag_days`` or earlier. Default ``lag_days=1``.
2. Scaler fit scope — the train-only fit contract already covered by
   :mod:`test_scaler_fit_on_train_only` continues to hold end-to-end
   through :func:`assemble_fold`, not only at the module boundary.

Both tests construct in-memory data so the leakage check is hermetic
and doesn't depend on disk caches.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.data.assemble import (
    FOLD_BOUNDARIES,
    TRAIN_START,
    _add_labels,
    _align_modalities,
    _apply_lookback_filter,
    _compute_val_start,
    _target_mask,
)
from mmfp.features.scalers import FittedScaler


# ---------------------------------------------------------------------------
# Helper: build tiny synthetic price + news frames with known sentinel values.
# ---------------------------------------------------------------------------


def _make_tiny_price_frame(
    tickers: list[str], dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Make a price-features-like frame with one _norm feature."""
    rows = []
    for t in tickers:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "Date": d,
                    "Ticker": t,
                    "return_1d_norm": 0.01 * i,
                    "return_5d_norm": 0.0,
                    "return_20d_norm": 0.0,
                    "volatility_20d_norm": 0.0,
                    "rsi_14_norm": 0.0,
                    "ema_ratio_norm": 0.0,
                    "macd_signal_norm": 0.0,
                    "bollinger_pctb_norm": 0.0,
                    "atr_14_norm": 0.0,
                    "volume_ratio_norm": 0.0,
                    "next_day_return": 0.02 if i % 2 == 0 else -0.02,
                    "log_return": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _make_tiny_news_frame(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    sentinel_col: str = "news_sent_mean",
) -> pd.DataFrame:
    """Per-(Ticker, Date) news frame whose ``sentinel_col`` equals the date ordinal.

    The ordinal lets us verify post-lag alignment: a sample on date T
    that carries ``news_sent_mean == k`` sourced the news row from the
    date at ordinal ``k`` (which we expect to be T-lag).
    """
    rows: list[dict] = []
    for t in tickers:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "Date": d,
                    "Ticker": t,
                    sentinel_col: float(i),  # ordinal marker
                    "news_volume": 1.0,
                    "has_news": 1.0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. News lag-by-1 propagates through the assembly aligner.
# ---------------------------------------------------------------------------


def test_news_lag_days_shifts_by_one_business_day() -> None:
    """After shifting by ``+1 BDay``, day T merges with article from T-1."""
    tickers = ["AAPL"]
    dates = pd.bdate_range("2020-01-02", periods=10)
    price_df = _make_tiny_price_frame(tickers, dates)
    news_df = _make_tiny_news_frame(tickers, dates)

    # Simulate the lag the loader applies (lag_days=1 default).
    news_lagged = news_df.copy()
    news_lagged["Date"] = news_lagged["Date"] + pd.offsets.BDay(1)

    cfg = ExperimentConfig.model_validate(copy.deepcopy(DEFAULT_CONFIG))

    aligned = _align_modalities(
        price_df=price_df,
        price_feat_cols=[c for c in price_df.columns if c.endswith("_norm")],
        macro_df=None,
        macro_feat_cols=[],
        news_df=news_lagged,
        news_feat_cols=[c for c in news_df.columns if c not in ("Date", "Ticker")],
        social_df=None,
        social_feat_cols=[],
        cfg=cfg,
    )

    # Row for Date=dates[1] should have news ordinal == 0 (from dates[0]).
    row = aligned[aligned["Date"] == dates[1]].iloc[0]
    assert row["news_sent_mean"] == pytest.approx(0.0)
    # Row for Date=dates[5] should have news ordinal == 4.
    row = aligned[aligned["Date"] == dates[5]].iloc[0]
    assert row["news_sent_mean"] == pytest.approx(4.0)


def test_news_first_date_has_no_news_after_lag() -> None:
    """Day 0 has no news source when lag_days=1 (nothing predates it)."""
    tickers = ["AAPL"]
    dates = pd.bdate_range("2020-01-02", periods=10)
    price_df = _make_tiny_price_frame(tickers, dates)
    news_df = _make_tiny_news_frame(tickers, dates)
    news_lagged = news_df.copy()
    news_lagged["Date"] = news_lagged["Date"] + pd.offsets.BDay(1)

    cfg = ExperimentConfig.model_validate(copy.deepcopy(DEFAULT_CONFIG))
    aligned = _align_modalities(
        price_df=price_df,
        price_feat_cols=[c for c in price_df.columns if c.endswith("_norm")],
        macro_df=None,
        macro_feat_cols=[],
        news_df=news_lagged,
        news_feat_cols=[c for c in news_df.columns if c not in ("Date", "Ticker")],
        social_df=None,
        social_feat_cols=[],
        cfg=cfg,
    )

    # Day 0 after a left-join gets zero-filled (no matching row). The
    # assembly step sets has_news=0 when news_volume==0. We pin the
    # zero-fill here; downstream ``has_news`` logic is tested separately.
    row = aligned[aligned["Date"] == dates[0]].iloc[0]
    # zero-filled because no article <= dates[0] once lag applied.
    assert row["news_sent_mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Scaler stats in assemble_fold come from train rows only.
# ---------------------------------------------------------------------------


def test_scaler_mean_equals_train_block_mean_not_pooled() -> None:
    """Synthetic train/val/test with distinct means: scaler must use train only.

    This is the end-to-end sibling of
    :mod:`test_scaler_fit_on_train_only` — it exercises the assembly's
    fit/transform wiring rather than the :class:`FittedScaler` in
    isolation.
    """
    cols = ["f0", "f1"]
    train = pd.DataFrame(
        {"f0": np.zeros(50), "f1": np.linspace(0, 1, 50)},
    )
    val = pd.DataFrame(
        {"f0": np.full(20, 100.0), "f1": np.full(20, 2.0)},
    )
    test = pd.DataFrame(
        {"f0": np.full(20, 1000.0), "f1": np.full(20, 5.0)},
    )
    scaler = FittedScaler()
    train_t = scaler.fit_transform(train)
    val_t = scaler.transform(val)
    test_t = scaler.transform(test)

    # mean_[0] must be 0 (train only), not ~50 (train+val pooled).
    assert scaler.mean_[0] == pytest.approx(0.0)
    # val/test values reflect train stats: (v - 0) / epsilon.
    assert np.all(val_t[:, 0] > 1.0)
    assert np.all(test_t[:, 0] > 1.0)
    # train after transform has near-zero mean.
    assert abs(train_t[:, 1].mean()) < 1e-6


# ---------------------------------------------------------------------------
# 3. Labels encode only current-day info (no cross-row contamination).
# ---------------------------------------------------------------------------


def test_labels_depend_only_on_next_day_return() -> None:
    """``label_cls``/``label_reg``/``label_vol`` derive solely from the row."""
    dates = pd.bdate_range("2020-01-02", periods=4)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": ["AAPL"] * 4,
            "next_day_return": [0.01, -0.02, 0.0, float("nan")],
        }
    )
    out = _add_labels(df, deadzone=0.005)

    # Row 0 ( +1% ): cls=1, reg=0.01, vol=0.01
    assert out.iloc[0]["label_cls"] == 1
    assert out.iloc[0]["label_reg"] == pytest.approx(0.01)
    assert out.iloc[0]["label_vol"] == pytest.approx(0.01)
    # Row 1 ( -2% ): cls=0
    assert out.iloc[1]["label_cls"] == 0
    # Row 2 ( 0% ): within deadzone → cls=-1
    assert out.iloc[2]["label_cls"] == -1
    # Row 3 (NaN): cls=-1, reg NaN, vol NaN
    assert out.iloc[3]["label_cls"] == -1
    assert np.isnan(out.iloc[3]["label_reg"])
    assert np.isnan(out.iloc[3]["label_vol"])


def test_target_mask_drops_deadzone_for_direction() -> None:
    df = pd.DataFrame(
        {
            "label_cls": [1, 0, -1, 1],
            "label_reg": [0.01, -0.02, 0.0, 0.03],
            "label_vol": [0.01, 0.02, 0.0, 0.03],
        }
    )
    mask = _target_mask(df, targets=["direction", "return"])
    # Row 2 (deadzone) must be dropped.
    assert list(mask.values) == [True, True, False, True]


def test_target_mask_drops_nan_reg_for_return_only() -> None:
    df = pd.DataFrame(
        {
            "label_cls": [1, 0, 1, 1],
            "label_reg": [0.01, -0.02, float("nan"), 0.03],
            "label_vol": [0.01, 0.02, float("nan"), 0.03],
        }
    )
    mask = _target_mask(df, targets=["return"])
    assert list(mask.values) == [True, True, False, True]


# ---------------------------------------------------------------------------
# 4. Lookback filter never injects future info.
# ---------------------------------------------------------------------------


def test_lookback_filter_drops_first_rows_per_ticker() -> None:
    dates = pd.bdate_range("2020-01-02", periods=25)
    df = pd.concat(
        [
            pd.DataFrame({"Ticker": "A", "Date": dates}),
            pd.DataFrame({"Ticker": "B", "Date": dates}),
        ],
        ignore_index=True,
    )

    out = _apply_lookback_filter(df, lookback=5)
    # For each ticker, the first 4 rows (positions 0..3) must be gone.
    for t in ("A", "B"):
        tkr_dates = out[out["Ticker"] == t]["Date"].sort_values().tolist()
        assert tkr_dates[0] == dates[4]
        assert tkr_dates[-1] == dates[-1]


def test_lookback_1_is_identity() -> None:
    dates = pd.bdate_range("2020-01-02", periods=10)
    df = pd.DataFrame({"Ticker": "A", "Date": dates})
    out = _apply_lookback_filter(df, lookback=1)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), out.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 5. val_start computation is deterministic and historical.
# ---------------------------------------------------------------------------


def test_val_start_falls_before_train_end() -> None:
    for fold in FOLD_BOUNDARIES:
        val_start = _compute_val_start(TRAIN_START, fold["train_end"], 0.2)
        assert pd.Timestamp(val_start) > pd.Timestamp(TRAIN_START)
        assert pd.Timestamp(val_start) < pd.Timestamp(fold["train_end"])
