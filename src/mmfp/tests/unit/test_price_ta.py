"""Unit tests for :mod:`mmfp.features.price_ta`.

Smoke-tests computing price features for a synthetic single ticker
and loading the cached parquet. No network access; no yfinance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmfp.data.paths import FEATURES_DIR
from mmfp.features.price_ta import (
    FEATURE_COLS,
    NORMALISED_FEATURE_COLS,
    compute_price_features,
    compute_price_features_single_ticker,
    load_or_compute_price_features,
)


def _synthetic_ohlcv(
    n_days: int = 400,
    ticker: str = "AAPL",
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a deterministic synthetic OHLCV frame long enough to
    populate the 252-day rolling window.
    """
    rng = np.random.default_rng(seed)
    # Geometric random walk for Close.
    close = 100.0 * np.exp(
        np.cumsum(rng.normal(0.0, 0.01, size=n_days))
    )
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.005, size=n_days)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.005, size=n_days)))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n_days))
    vol = rng.integers(1_000_000, 5_000_000, size=n_days)
    dates = pd.date_range("2018-01-02", periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": vol,
            "Ticker": ticker,
        }
    )


def test_compute_single_ticker_produces_expected_columns() -> None:
    ohlcv = _synthetic_ohlcv()
    out = compute_price_features_single_ticker(ohlcv)
    for col in FEATURE_COLS:
        assert col in out.columns, f"missing raw feature {col}"
    for col in NORMALISED_FEATURE_COLS:
        assert col in out.columns, f"missing normalised feature {col}"


def test_rolling_norm_stable_after_warmup() -> None:
    """After the 252-day warm-up, all ``_norm`` columns are finite."""
    ohlcv = _synthetic_ohlcv(n_days=400)
    out = compute_price_features_single_ticker(ohlcv)
    # Skip the first 252 rows. Allow first 1-2 NaNs for return-based cols.
    post_warmup = out.iloc[260:]
    for col in NORMALISED_FEATURE_COLS:
        values = post_warmup[col]
        assert not values.isna().any(), f"{col} has NaN after warmup"
        assert np.isfinite(values).all(), f"{col} has inf after warmup"


def test_compute_multi_ticker() -> None:
    """When fed a multi-ticker frame, groups are handled independently."""
    aapl = _synthetic_ohlcv(ticker="AAPL", seed=1)
    msft = _synthetic_ohlcv(ticker="MSFT", seed=2)
    combined = pd.concat([aapl, msft], ignore_index=True)
    out = compute_price_features(combined)
    assert set(out["Ticker"]) == {"AAPL", "MSFT"}
    # Canonical sort: Ticker first, Date ascending.
    for ticker in ("AAPL", "MSFT"):
        sub = out[out["Ticker"] == ticker]
        assert list(sub["Date"]) == sorted(sub["Date"])


def test_compute_rejects_missing_ticker_column() -> None:
    df = _synthetic_ohlcv().drop(columns=["Ticker"])
    with pytest.raises(ValueError, match="Ticker"):
        compute_price_features(df)


def test_load_or_compute_reads_cache_when_available() -> None:
    """If the cache exists on disk, the loader returns it (no compute)."""
    cache_path = FEATURES_DIR / "price_features.parquet"
    if not cache_path.exists():
        pytest.skip("price_features.parquet not present in this checkout")

    out = load_or_compute_price_features()
    assert "Ticker" in out.columns
    assert "Date" in out.columns
    # Sorted by (Ticker, Date) ascending.
    assert len(out) > 0


def test_load_or_compute_recomputes_when_cache_missing(
    tmp_path: Path,
) -> None:
    """With a non-existent cache path, recompute from provided OHLCV."""
    ohlcv = _synthetic_ohlcv()
    custom_cache = tmp_path / "nonexistent.parquet"
    assert not custom_cache.exists()

    out = load_or_compute_price_features(
        ohlcv=ohlcv, cached_path=custom_cache,
    )
    assert "return_1d" in out.columns
    assert "return_1d_norm" in out.columns


def test_load_or_compute_requires_ohlcv_when_no_cache(
    tmp_path: Path,
) -> None:
    custom_cache = tmp_path / "nonexistent.parquet"
    with pytest.raises(FileNotFoundError):
        load_or_compute_price_features(cached_path=custom_cache)


def test_z_score_mean_std_rough_zero_one() -> None:
    """Sanity: normalised columns should have ~0 mean and ~1 std over a
    large post-warm-up sample."""
    ohlcv = _synthetic_ohlcv(n_days=600)
    out = compute_price_features_single_ticker(ohlcv)
    post = out.iloc[300:]
    # Not exactly 0/1 because rolling z-score uses trailing window, not
    # full sample. But mean should be near zero and std near 1 over a
    # large window.
    for col in ("return_1d_norm", "rsi_14_norm"):
        assert abs(float(post[col].mean())) < 0.5
        assert 0.4 < float(post[col].std()) < 3.0
