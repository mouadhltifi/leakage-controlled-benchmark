"""Smoke tests for :mod:`mmfp.data.loaders`.

Each loader has:
* a happy-path "does it return a sensible DataFrame / array" test
* a schema test pinning the documented column names and dtypes
* a purity test (same args twice -> equal output)
* an edge-case test for unknown tickers and bad date ranges

Heavy loaders (FNSPID, StockTwits full scan) are marked ``slow``. The
repo's ``pytest.ini`` doesn't auto-skip ``slow`` today; they pass in
under a couple of seconds thanks to pre-filtering, so leaving them in
the default run is fine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmfp.data.loaders import (
    build_dynamic_snapshots_from_returns,
    build_static_adjacency,
    load_dynamic_snapshots,
    load_fnspid,
    load_macro_raw,
    load_ohlcv,
    load_static_adjacency,
    load_stocktwits,
    pick_snapshot_for_date,
)
from mmfp.data.loaders.news_raw import FNSPID_OUTPUT_COLUMNS
from mmfp.data.loaders.prices import OHLCV_COLUMNS
from mmfp.data.loaders.social_raw import STOCKTWITS_OUTPUT_COLUMNS
from mmfp.data.paths import (
    DYNAMIC_GRAPHS_INDEX,
    FNSPID_CSV,
    MACRO_PARQUET,
    PRICES_PARQUET,
    STATIC_ADJ_NPY,
    STOCKTWITS_PARQUET,
)
from mmfp.data.universe import ALL_TICKERS, FRED_SERIES, N_STOCKS


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


def test_universe_has_55_tickers() -> None:
    assert len(ALL_TICKERS) == 55
    assert N_STOCKS == 55


def test_universe_sorted() -> None:
    assert ALL_TICKERS == sorted(ALL_TICKERS)


def test_universe_unique() -> None:
    assert len(set(ALL_TICKERS)) == len(ALL_TICKERS)


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PRICES_PARQUET.exists(),
    reason="OHLCV parquet not present; run fetch_prices.py first",
)
class TestPrices:
    def test_schema(self) -> None:
        df = load_ohlcv(["AAPL"], "2020-01-01", "2020-01-31")
        assert list(df.columns) == OHLCV_COLUMNS
        # Accept any datetime resolution (pandas 3 defaults to ms for parquet).
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])
        assert df.index.equals(pd.RangeIndex(len(df)))

    def test_non_empty(self) -> None:
        df = load_ohlcv(["AAPL"], "2020-01-01", "2020-01-31")
        assert not df.empty

    def test_default_tickers_is_universe(self) -> None:
        df = load_ohlcv(start="2020-01-02", end="2020-01-03")
        # One row per ticker per business day.
        assert set(df["Ticker"].unique()) == set(ALL_TICKERS)

    def test_date_filter_is_inclusive(self) -> None:
        df = load_ohlcv(["AAPL"], "2020-01-02", "2020-01-10")
        assert df["Date"].min() >= pd.Timestamp("2020-01-02")
        assert df["Date"].max() <= pd.Timestamp("2020-01-10")

    def test_deterministic(self) -> None:
        a = load_ohlcv(["AAPL", "MSFT"], "2020-06-01", "2020-06-15")
        b = load_ohlcv(["AAPL", "MSFT"], "2020-06-01", "2020-06-15")
        pd.testing.assert_frame_equal(a, b)

    def test_unknown_ticker_warns_returns_empty(self, caplog) -> None:
        """Tickers not in parquet produce a warning, not a raise."""
        import logging

        caplog.set_level(logging.WARNING)
        df = load_ohlcv(["ZZZZ_NOT_A_TICKER"], "2020-01-01", "2020-01-31")
        assert df.empty
        assert any("not in parquet" in rec.message for rec in caplog.records)

    def test_sorted_by_ticker_then_date(self) -> None:
        df = load_ohlcv(["AAPL", "MSFT"], "2020-01-01", "2020-01-31")
        assert df.equals(df.sort_values(["Ticker", "Date"], kind="mergesort").reset_index(drop=True))

    def test_bad_date_range_raises(self) -> None:
        with pytest.raises(ValueError, match="start .* must be"):
            load_ohlcv(["AAPL"], "2023-01-01", "2022-01-01")

    def test_missing_parquet_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_ohlcv(["AAPL"], path=tmp_path / "nowhere.parquet")


# ---------------------------------------------------------------------------
# macro
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not MACRO_PARQUET.exists(),
    reason="Macro parquet not present; run fetch_macro.py first",
)
class TestMacro:
    def test_schema(self) -> None:
        df = load_macro_raw("2020-01-01", "2020-03-31")
        assert list(df.columns) == list(FRED_SERIES.keys())
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "Date"

    def test_non_empty(self) -> None:
        df = load_macro_raw("2020-01-01", "2020-03-31")
        assert not df.empty

    def test_sorted(self) -> None:
        df = load_macro_raw("2020-01-01", "2020-03-31")
        assert df.index.is_monotonic_increasing

    def test_deterministic(self) -> None:
        a = load_macro_raw("2021-01-01", "2021-01-31")
        b = load_macro_raw("2021-01-01", "2021-01-31")
        pd.testing.assert_frame_equal(a, b)

    def test_bad_date_range_raises(self) -> None:
        with pytest.raises(ValueError):
            load_macro_raw("2023-01-01", "2022-01-01")

    def test_missing_parquet_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_macro_raw(path=tmp_path / "nowhere.parquet")


# ---------------------------------------------------------------------------
# static graph
# ---------------------------------------------------------------------------


class TestStaticAdjacency:
    def test_build_shape_and_symmetry(self) -> None:
        adj = build_static_adjacency()
        assert adj.shape == (55, 55)
        assert adj.dtype == np.float32
        # Symmetric.
        np.testing.assert_array_equal(adj, adj.T)
        # Zero diagonal.
        np.testing.assert_array_equal(np.diag(adj), np.zeros(55, dtype=np.float32))
        # Entries are 0 or 1.
        assert set(np.unique(adj).tolist()).issubset({0.0, 1.0})

    def test_known_pair_same_sector(self) -> None:
        """AAPL and MSFT share the IT sector; adjacency entry must be 1."""
        adj = build_static_adjacency()
        i = ALL_TICKERS.index("AAPL")
        j = ALL_TICKERS.index("MSFT")
        assert adj[i, j] == 1.0

    def test_known_pair_different_sector(self) -> None:
        """AAPL (IT) vs XOM (Energy) -> 0."""
        adj = build_static_adjacency()
        i = ALL_TICKERS.index("AAPL")
        j = ALL_TICKERS.index("XOM")
        assert adj[i, j] == 0.0

    def test_deterministic(self) -> None:
        a = build_static_adjacency()
        b = build_static_adjacency()
        np.testing.assert_array_equal(a, b)

    def test_unknown_ticker_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown tickers"):
            build_static_adjacency(["AAPL", "NOT_A_TICKER"])

    @pytest.mark.skipif(
        not STATIC_ADJ_NPY.exists(),
        reason="Cached static adjacency NPY not present",
    )
    def test_disk_is_released_ff12_graph(self) -> None:
        """The committed NPY is the RELEASED static graph (Fama-French-12
        taxonomy, 133 edges), deliberately distinct from the in-code
        GICS-partition build (110 edges, the sensitivity arm); the paired
        comparison ships in results/sector/ (paper Appendix B)."""
        disk = load_static_adjacency()
        assert disk.shape == (55, 55)
        np.testing.assert_array_equal(disk, disk.T)
        np.testing.assert_array_equal(
            np.diag(disk), np.zeros(55, dtype=np.float32))
        assert int(disk.sum()) // 2 == 133  # FF12 same-sector edge count
        built = build_static_adjacency()   # universe GICS partition
        assert int(built.sum()) // 2 == 110
        assert not np.array_equal(built, disk)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_static_adjacency(path=tmp_path / "nope.npy")


# ---------------------------------------------------------------------------
# dynamic graph
# ---------------------------------------------------------------------------


class TestDynamicSnapshots:
    @pytest.mark.skipif(
        not DYNAMIC_GRAPHS_INDEX.exists(),
        reason="Dynamic graphs index.json not present",
    )
    def test_load_returns_dict_of_square_matrices(self) -> None:
        snaps = load_dynamic_snapshots()
        assert len(snaps) > 0
        first = next(iter(snaps.values()))
        assert first.ndim == 2
        assert first.shape[0] == first.shape[1]
        # All snapshots share the same shape.
        for mat in snaps.values():
            assert mat.shape == first.shape

    @pytest.mark.skipif(
        not DYNAMIC_GRAPHS_INDEX.exists(),
        reason="Dynamic graphs index.json not present",
    )
    def test_keys_are_iso_dates(self) -> None:
        snaps = load_dynamic_snapshots()
        for key in snaps:
            # "YYYY-MM-DD" length.
            assert len(key) == 10
            # Parseable.
            pd.Timestamp(key)

    @pytest.mark.skipif(
        not DYNAMIC_GRAPHS_INDEX.exists(),
        reason="Dynamic graphs index.json not present",
    )
    def test_pick_snapshot_returns_most_recent(self) -> None:
        snaps = load_dynamic_snapshots()
        keys = sorted(snaps.keys())
        # A date after the latest snapshot.
        picked = pick_snapshot_for_date("2030-01-01", snaps)
        np.testing.assert_array_equal(picked, snaps[keys[-1]])

    @pytest.mark.skipif(
        not DYNAMIC_GRAPHS_INDEX.exists(),
        reason="Dynamic graphs index.json not present",
    )
    def test_pick_snapshot_before_first_returns_none(self) -> None:
        snaps = load_dynamic_snapshots()
        picked = pick_snapshot_for_date("1990-01-01", snaps)
        assert picked is None

    def test_missing_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dynamic_snapshots(index_path=tmp_path / "nowhere.json")

    def test_build_from_returns_toy(self) -> None:
        """Synthetic toy returns for 3 stocks produce a well-formed snapshot."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=60)
        # Build two correlated series and one anti-correlated.
        base = rng.normal(0.0, 0.01, size=60)
        returns = pd.DataFrame(
            {
                "A": base,
                "B": base + rng.normal(0.0, 0.001, size=60),  # highly correlated
                "C": -base,  # anti-correlated
            },
            index=dates,
        )
        snaps = build_dynamic_snapshots_from_returns(returns, window=20, threshold=0.3)

        assert len(snaps) > 0
        for mat in snaps.values():
            assert mat.shape == (3, 3)
            assert mat.dtype == np.float32
            # Symmetric.
            np.testing.assert_array_equal(mat, mat.T)
            # Zero diagonal.
            np.testing.assert_array_equal(np.diag(mat), np.zeros(3, dtype=np.float32))
            # A-B correlated -> edge exists.
            assert mat[0, 1] > 0.3

    def test_build_rejects_bad_window(self) -> None:
        with pytest.raises(ValueError, match="window"):
            build_dynamic_snapshots_from_returns(
                pd.DataFrame({"A": [0.0, 1.0]}), window=1
            )

    def test_build_rejects_empty_returns(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_dynamic_snapshots_from_returns(
                pd.DataFrame(columns=["A"]), window=2
            )


# ---------------------------------------------------------------------------
# FNSPID (news_raw)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not FNSPID_CSV.exists(),
    reason="FNSPID CSV not present; see data/external/FNSPID_.../README.md",
)
class TestFnspid:
    """FNSPID is multi-GB; we keep tests to one small slice."""

    def test_schema_small_slice(self) -> None:
        df = load_fnspid(["AAPL"], "2023-01-01", "2023-01-07", chunk_size=50_000)
        assert list(df.columns) == FNSPID_OUTPUT_COLUMNS
        # Empty is acceptable for a 1-week slice, but if rows exist they must
        # be typed correctly.
        if not df.empty:
            assert pd.api.types.is_datetime64_any_dtype(df["Date"])
            assert df["Ticker"].dtype == object or pd.api.types.is_string_dtype(
                df["Ticker"]
            )

    def test_bad_date_range_raises(self) -> None:
        with pytest.raises(ValueError):
            load_fnspid(["AAPL"], "2023-01-01", "2022-01-01")

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_fnspid(["AAPL"], path=tmp_path / "nowhere.csv")


# ---------------------------------------------------------------------------
# StockTwits (social_raw)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not STOCKTWITS_PARQUET.exists(),
    reason="StockTwits parquet not present",
)
class TestStockTwits:
    def test_schema(self) -> None:
        df = load_stocktwits(["AAPL"], "2020-01-01", "2020-01-31")
        assert list(df.columns) == STOCKTWITS_OUTPUT_COLUMNS
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_non_empty_slice(self) -> None:
        df = load_stocktwits(["AAPL"], "2020-01-01", "2020-01-31")
        # AAPL has StockTwits volume throughout 2020, so this must be non-empty.
        assert not df.empty

    def test_sentiment_is_plus_minus_one_or_nan(self) -> None:
        df = load_stocktwits(["AAPL"], "2020-01-01", "2020-01-03")
        labelled = df["sentiment"].dropna().unique()
        assert set(labelled.tolist()).issubset({-1.0, 1.0})

    def test_deterministic(self) -> None:
        a = load_stocktwits(["MSFT"], "2020-06-01", "2020-06-15")
        b = load_stocktwits(["MSFT"], "2020-06-01", "2020-06-15")
        pd.testing.assert_frame_equal(a, b)

    def test_unknown_ticker_returns_empty(self) -> None:
        df = load_stocktwits(["NOT_A_TICKER"], "2020-01-01", "2020-01-10")
        assert df.empty
        # Schema preserved even when empty.
        assert list(df.columns) == STOCKTWITS_OUTPUT_COLUMNS

    def test_bad_date_range_raises(self) -> None:
        with pytest.raises(ValueError):
            load_stocktwits(["AAPL"], "2023-01-01", "2022-01-01")

    def test_missing_parquet_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_stocktwits(["AAPL"], path=tmp_path / "nowhere.parquet")
