"""Unit tests for :mod:`mmfp.features.macro_events`.

Smoke-tests FOMC window flags, rolling-z-score normalisation, and
the ``load_or_compute`` helper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmfp.data.paths import FEATURES_DIR, FOMC_CSV
from mmfp.features.macro_events import (
    PASSTHROUGH_COLS,
    compute_macro_features,
    create_event_features,
    load_fomc_dates,
    load_or_compute_macro_features,
    normalize_macro_features,
)


def _synthetic_macro(n_days: int = 500, seed: int = 3) -> pd.DataFrame:
    """Date-indexed macro DataFrame with realistic-ish series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-02", periods=n_days, freq="B")
    fed_funds = np.cumsum(rng.normal(0.0, 0.05, size=n_days)) + 2.0
    ten_y = fed_funds + rng.normal(0.5, 0.1, size=n_days)
    cpi = 260.0 + np.cumsum(rng.normal(0.0, 0.1, size=n_days))
    unemployment = 4.0 + np.cumsum(rng.normal(0.0, 0.01, size=n_days))
    gdp = 20000.0 + np.cumsum(rng.normal(0.0, 1.0, size=n_days))
    vix = np.abs(rng.normal(18.0, 3.0, size=n_days))
    return pd.DataFrame(
        {
            "fed_funds_rate": fed_funds,
            "treasury_10y": ten_y,
            "cpi": cpi,
            "unemployment": unemployment,
            "gdp": gdp,
            "vix": vix,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def test_create_event_features_adds_is_fomc_window_column() -> None:
    macro = _synthetic_macro()
    # Put an FOMC date squarely in the middle of our series.
    anchor = macro.index[len(macro) // 2]
    fomc = pd.Series([anchor])
    out = create_event_features(macro, fomc_dates=fomc, fomc_window_days=1)
    assert "is_fomc_window" in out.columns
    assert out["is_fomc_window"].dtype == np.int8


def test_fomc_window_flag_centred_on_date() -> None:
    # 3-day window: anchor + and - 1 are flagged.
    macro = _synthetic_macro(n_days=21)
    anchor = macro.index[10]
    fomc = pd.Series([anchor])
    out = create_event_features(macro, fomc_dates=fomc, fomc_window_days=1)
    # Anchor and its two neighbours flagged.
    assert int(out.loc[anchor, "is_fomc_window"]) == 1
    # Index is BDay so the neighbours are index 9 and 11.
    assert int(out.iloc[9]["is_fomc_window"]) == 1
    assert int(out.iloc[11]["is_fomc_window"]) == 1
    # Outside the window: no flag.
    assert int(out.iloc[0]["is_fomc_window"]) == 0
    assert int(out.iloc[20]["is_fomc_window"]) == 0


def test_fomc_window_zero_days_flags_anchor_only() -> None:
    macro = _synthetic_macro(n_days=21)
    anchor = macro.index[10]
    fomc = pd.Series([anchor])
    out = create_event_features(macro, fomc_dates=fomc, fomc_window_days=0)
    assert int(out.loc[anchor, "is_fomc_window"]) == 1
    assert int(out.iloc[9]["is_fomc_window"]) == 0
    assert int(out.iloc[11]["is_fomc_window"]) == 0


def test_fomc_window_negative_raises() -> None:
    macro = _synthetic_macro(n_days=5)
    with pytest.raises(ValueError, match=">= 0"):
        create_event_features(macro, fomc_dates=pd.Series(dtype="datetime64[ns]"), fomc_window_days=-1)


def test_normalize_does_not_touch_passthrough() -> None:
    macro = _synthetic_macro()
    anchor = macro.index[len(macro) // 2]
    with_events = create_event_features(
        macro, fomc_dates=pd.Series([anchor]), fomc_window_days=1,
    )
    normed = normalize_macro_features(with_events)
    # Passthrough columns unchanged.
    for col in PASSTHROUGH_COLS:
        np.testing.assert_array_equal(
            normed[col].values, with_events[col].values
        )


def test_normalize_emits_norm_columns() -> None:
    macro = _synthetic_macro()
    normed = normalize_macro_features(macro)
    for col in (
        "fed_funds_rate_norm",
        "treasury_10y_norm",
        "cpi_norm",
        "unemployment_norm",
        "gdp_norm",
        "vix_norm",
    ):
        assert col in normed.columns


def test_compute_macro_features_full_pipeline() -> None:
    macro = _synthetic_macro()
    anchor = macro.index[len(macro) // 2]
    out = compute_macro_features(
        macro, fomc_dates=pd.Series([anchor]), fomc_window_days=1,
    )
    assert "is_fomc_window" in out.columns
    assert "vix_norm" in out.columns


def test_load_fomc_dates_reads_csv() -> None:
    if not FOMC_CSV.exists():
        pytest.skip("FOMC dates CSV missing in this checkout")
    dates = load_fomc_dates()
    assert len(dates) > 0
    assert dates.is_monotonic_increasing
    assert np.issubdtype(dates.dtype, np.datetime64)


def test_load_or_compute_reads_cache_when_available() -> None:
    cache = FEATURES_DIR / "macro_features.parquet"
    if not cache.exists():
        pytest.skip("macro_features.parquet not present")
    out = load_or_compute_macro_features()
    assert "is_fomc_window" in out.columns


def test_load_or_compute_recomputes_when_cache_missing(tmp_path: Path) -> None:
    macro = _synthetic_macro()
    anchor = macro.index[len(macro) // 2]
    out = load_or_compute_macro_features(
        macro_raw=macro,
        cached_path=tmp_path / "nope.parquet",
        fomc_dates=pd.Series([anchor]),
        fomc_window_days=1,
    )
    assert "is_fomc_window" in out.columns


def test_load_or_compute_requires_input_when_no_cache(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_or_compute_macro_features(cached_path=tmp_path / "nope.parquet")
