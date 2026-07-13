"""Unit tests for :mod:`mmfp.features.social_features`.

Smoke-tests the StockTwits daily aggregator with a synthetic
message frame and the cache loader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmfp.data.paths import FEATURES_DIR
from mmfp.features.social_features import (
    DEFAULT_WINDOW,
    LOG1P_COLS,
    PASSTHROUGH_COLS,
    compute_social_features,
    load_or_compute_social_features,
)


def _synthetic_stocktwits(seed: int = 4) -> pd.DataFrame:
    """Build a small per-message StockTwits-like frame for AAPL and MSFT."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2020-01-02", periods=5)
    for ticker in ("AAPL", "MSFT"):
        for d in dates:
            n = int(rng.integers(5, 20))
            for msg_i in range(n):
                # 40% bullish, 20% bearish, 40% unlabelled.
                r = rng.random()
                if r < 0.4:
                    sent = 1.0
                elif r < 0.6:
                    sent = -1.0
                else:
                    sent = np.nan
                rows.append(
                    {
                        "Date": d,
                        "Ticker": ticker,
                        "message_id": len(rows),
                        "user_id": int(rng.integers(1, 100)),
                        "sentiment": sent,
                    }
                )
    return pd.DataFrame(rows)


def test_compute_returns_expected_shape() -> None:
    st = _synthetic_stocktwits()
    out = compute_social_features(
        st, tickers=["AAPL", "MSFT"], aggregation_window=3,
    )
    assert len(out) > 0
    assert set(out["Ticker"]) == {"AAPL", "MSFT"}


def test_compute_emits_canonical_columns() -> None:
    st = _synthetic_stocktwits()
    out = compute_social_features(
        st, tickers=["AAPL"], aggregation_window=3,
    )
    for col in (
        "social_sent_mean",
        "social_volume",
        "social_bullish_ratio",
        "social_net_sentiment",
        "social_log_volume",
        "has_social",
        "social_volume_w3",
        "social_net_sentiment_w3",
    ):
        assert col in out.columns


def test_has_social_flag_agrees_with_nonzero_volume() -> None:
    st = _synthetic_stocktwits()
    out = compute_social_features(
        st, tickers=["AAPL"], aggregation_window=3, log1p_volume=False,
    )
    # log1p_volume=False -> social_volume is the raw count.
    # A day with volume > 0 has has_social=1.
    positive = out["social_volume"] > 0
    assert (out.loc[positive, "has_social"] == 1).all()
    # And absent days (filler) have has_social=0.
    absent = out["social_volume"] == 0
    if absent.any():
        assert (out.loc[absent, "has_social"] == 0).all()


def test_log1p_volume_flag_changes_volume_scale() -> None:
    st = _synthetic_stocktwits()
    out_log = compute_social_features(
        st, tickers=["AAPL"], aggregation_window=3, log1p_volume=True,
    )
    out_raw = compute_social_features(
        st, tickers=["AAPL"], aggregation_window=3, log1p_volume=False,
    )
    # On days with volume > 0, log-transformed volume < raw volume.
    positive_raw = out_raw[out_raw["social_volume"] > 0]
    positive_log = out_log[out_raw["social_volume"] > 0]
    assert (positive_log["social_volume"] < positive_raw["social_volume"]).all()


def test_rolling_aggregates_are_finite() -> None:
    st = _synthetic_stocktwits()
    out = compute_social_features(
        st, tickers=["AAPL"], aggregation_window=3,
    )
    for col in (
        "social_sent_mean_w3",
        "social_volume_w3",
        "social_net_sentiment_w3",
    ):
        assert np.isfinite(out[col]).all(), f"{col} contains non-finite values"


def test_rolling_sum_of_volume_matches_manual_computation() -> None:
    # Fully controlled input: 3 labelled messages on day 1, 2 on day 2.
    rows = []
    d1 = pd.Timestamp("2020-01-02")
    d2 = pd.Timestamp("2020-01-03")
    for _ in range(3):
        rows.append({"Date": d1, "Ticker": "AAPL", "message_id": len(rows), "user_id": 1, "sentiment": 1.0})
    for _ in range(2):
        rows.append({"Date": d2, "Ticker": "AAPL", "message_id": len(rows), "user_id": 2, "sentiment": -1.0})
    df = pd.DataFrame(rows)
    out = compute_social_features(
        df, tickers=["AAPL"], aggregation_window=3, log1p_volume=False,
    )
    # On d2, rolling sum = 3 + 2 = 5.
    row = out[out["Date"] == d2].iloc[0]
    assert row["social_volume_w3"] == pytest.approx(5.0, abs=1e-5)


def test_missing_columns_raises() -> None:
    df = pd.DataFrame({"Date": [pd.Timestamp("2020-01-02")]})
    with pytest.raises(ValueError, match="Ticker"):
        compute_social_features(df)


def test_zero_window_raises() -> None:
    st = _synthetic_stocktwits()
    with pytest.raises(ValueError, match=">= 1"):
        compute_social_features(st, aggregation_window=0)


def test_unknown_ticker_returns_empty() -> None:
    st = _synthetic_stocktwits()
    out = compute_social_features(st, tickers=["UNKNOWN_TICKER_XYZ"])
    assert len(out) == 0


def test_load_or_compute_reads_cache() -> None:
    cache = FEATURES_DIR / "social_features.parquet"
    if not cache.exists():
        pytest.skip("social_features.parquet missing")
    out = load_or_compute_social_features()
    assert "social_sent_mean" in out.columns


def test_load_or_compute_recomputes(tmp_path: Path) -> None:
    st = _synthetic_stocktwits()
    out = load_or_compute_social_features(
        stocktwits=st, cached_path=tmp_path / "nope.parquet",
    )
    assert "social_volume" in out.columns


def test_load_or_compute_errors_without_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_or_compute_social_features(cached_path=tmp_path / "nope.parquet")


def test_default_window_constant() -> None:
    assert DEFAULT_WINDOW == 3


def test_log1p_passthrough_constants() -> None:
    """The wiring constants mirror the scaler contract."""
    assert any("social_volume" in c for c in LOG1P_COLS)
    assert "has_social" in PASSTHROUGH_COLS
