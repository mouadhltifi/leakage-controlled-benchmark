"""Unit tests for :mod:`mmfp.features.news_stats`.

Focus on the audit-critical fixes: ``log1p`` on volume columns,
explicit ``empty_day_policy`` handling, and the ``has_news`` flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmfp.features.news_stats import (
    LOG1P_COLS,
    NEWS_STATS_FEATURES,
    PASSTHROUGH_COLS,
    build_news_stats_11dim,
)


def _make_sentiments_df(
    articles_per_day: dict[tuple[str, str], list[tuple[float, float, float]]],
) -> pd.DataFrame:
    """Construct a per-article sentiments DataFrame.

    Parameters
    ----------
    articles_per_day
        Keyed by ``(Ticker, Date_str)``; values are lists of
        ``(p_pos, p_neg, p_neu)`` tuples.
    """
    rows: list[dict[str, object]] = []
    for (ticker, date_str), probs_list in articles_per_day.items():
        for p_pos, p_neg, p_neu in probs_list:
            rows.append(
                {
                    "Ticker": ticker,
                    "Date": pd.Timestamp(date_str),
                    "p_pos": p_pos,
                    "p_neg": p_neg,
                    "p_neu": p_neu,
                }
            )
    return pd.DataFrame(rows)


def test_output_columns_exact() -> None:
    df = _make_sentiments_df(
        {("AAPL", "2020-01-02"): [(0.6, 0.2, 0.2), (0.3, 0.5, 0.2)]}
    )
    out = build_news_stats_11dim(df, rolling_window=3)
    expected = [
        "Ticker",
        "Date",
        "news_sent_mean",
        "news_sent_std",
        "news_sent_min",
        "news_sent_max",
        "news_pos_mean",
        "news_neg_mean",
        "news_volume",
        "news_sent_mean_w3",
        "news_sent_std_w3",
        "news_volume_w3",
        "has_news",
    ]
    assert list(out.columns) == expected


def test_single_day_sentiment_stats() -> None:
    df = _make_sentiments_df(
        {("AAPL", "2020-01-02"): [(0.6, 0.2, 0.2), (0.3, 0.5, 0.2)]}
    )
    out = build_news_stats_11dim(df, log1p_volume=False, rolling_window=3)
    # Only check the explicitly-written day; the business-day reindex
    # may add neighbouring zero-filled days.
    day = out[out["Date"] == pd.Timestamp("2020-01-02")].iloc[0]
    # sent_score = p_pos - p_neg -> [0.4, -0.2]; mean = 0.1, min = -0.2, max = 0.4.
    assert day["news_sent_mean"] == pytest.approx(0.1, abs=1e-5)
    assert day["news_sent_min"] == pytest.approx(-0.2, abs=1e-5)
    assert day["news_sent_max"] == pytest.approx(0.4, abs=1e-5)
    assert day["news_volume"] == 2  # log1p_volume=False -> raw count
    assert day["has_news"] == 1


def test_log1p_volume_applied_when_flag_true() -> None:
    df = _make_sentiments_df(
        {("AAPL", "2020-01-02"): [(0.5, 0.3, 0.2)] * 4}  # volume = 4
    )
    out_log = build_news_stats_11dim(df, log1p_volume=True, rolling_window=3)
    out_raw = build_news_stats_11dim(df, log1p_volume=False, rolling_window=3)

    day_log = out_log[out_log["Date"] == pd.Timestamp("2020-01-02")].iloc[0]
    day_raw = out_raw[out_raw["Date"] == pd.Timestamp("2020-01-02")].iloc[0]
    # log1p(4) = log(5)
    assert day_log["news_volume"] == pytest.approx(np.log1p(4), abs=1e-5)
    assert day_raw["news_volume"] == pytest.approx(4.0, abs=1e-5)


def test_log1p_monotonic_on_volume() -> None:
    """The more articles, the higher ``log1p(news_volume)``."""
    df = _make_sentiments_df(
        {
            ("AAPL", "2020-01-02"): [(0.5, 0.3, 0.2)] * 2,
            ("AAPL", "2020-01-03"): [(0.5, 0.3, 0.2)] * 5,
            ("AAPL", "2020-01-06"): [(0.5, 0.3, 0.2)] * 10,
        }
    )
    out = build_news_stats_11dim(df, log1p_volume=True, rolling_window=3)
    # Filter to the days we explicitly populated; skip asfreq B-day fillers.
    populated = out[out["Date"].isin(pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-06"]
    ))].sort_values("Date")
    vols = populated["news_volume"].tolist()
    assert vols == sorted(vols)  # monotonic increasing


def test_empty_day_policy_zero_fill_vs_sentinel() -> None:
    df = _make_sentiments_df(
        {("AAPL", "2020-01-02"): [(0.6, 0.2, 0.2)]}
    )
    # Ensure there are adjacent business days with no articles.
    out_zero = build_news_stats_11dim(
        df, empty_day_policy="zero_fill_has_news_flag", rolling_window=3,
    )
    out_sent = build_news_stats_11dim(
        df, empty_day_policy="sentinel_vector", rolling_window=3,
    )

    # Both must contain the populated day and at least one unpopulated
    # business day (the reindex forces contiguous B-days per ticker).
    zero_empty = out_zero[out_zero["Date"] != pd.Timestamp("2020-01-02")]
    sent_empty = out_sent[out_sent["Date"] != pd.Timestamp("2020-01-02")]
    # If there's only one business day in the range, add another day:
    if len(zero_empty) == 0:
        # Extend with a later-dated article so asfreq has neighbours.
        df2 = pd.concat(
            [
                df,
                _make_sentiments_df(
                    {("AAPL", "2020-01-10"): [(0.6, 0.2, 0.2)]}
                ),
            ]
        )
        out_zero = build_news_stats_11dim(
            df2, empty_day_policy="zero_fill_has_news_flag", rolling_window=3,
        )
        out_sent = build_news_stats_11dim(
            df2, empty_day_policy="sentinel_vector", rolling_window=3,
        )
        zero_empty = out_zero[
            ~out_zero["Date"].isin(pd.to_datetime(
                ["2020-01-02", "2020-01-10"]
            ))
        ]
        sent_empty = out_sent[
            ~out_sent["Date"].isin(pd.to_datetime(
                ["2020-01-02", "2020-01-10"]
            ))
        ]

    assert len(zero_empty) > 0

    # Zero-fill: every sentiment column is 0 on empty days.
    assert (zero_empty["news_sent_mean"] == 0.0).all()
    assert (zero_empty["news_sent_min"] == 0.0).all()
    assert (zero_empty["news_volume"] == 0.0).all()
    # has_news is 0 on empty days under both policies.
    assert (zero_empty["has_news"] == 0).all()

    # Sentinel: sentiment columns are NaN on empty days.
    assert sent_empty["news_sent_mean"].isna().all()
    assert sent_empty["news_sent_min"].isna().all()
    # Volume is still 0 (not NaN): it's a count, 0 is interpretable.
    assert (sent_empty["news_volume"] == 0.0).all()
    # has_news agrees.
    assert (sent_empty["has_news"] == 0).all()


def test_has_news_flag_correct() -> None:
    df = _make_sentiments_df(
        {
            ("AAPL", "2020-01-02"): [(0.6, 0.2, 0.2)],
            ("AAPL", "2020-01-10"): [(0.3, 0.5, 0.2), (0.7, 0.1, 0.2)],
        }
    )
    out = build_news_stats_11dim(df, rolling_window=3)
    # Populated days have has_news=1; filler B-days have has_news=0.
    populated_days = pd.to_datetime(["2020-01-02", "2020-01-10"])
    pop_mask = out["Date"].isin(populated_days)
    assert (out.loc[pop_mask, "has_news"] == 1).all()
    assert (out.loc[~pop_mask, "has_news"] == 0).all()


def test_rolling_window_sum_of_volume() -> None:
    df = _make_sentiments_df(
        {
            ("AAPL", "2020-01-02"): [(0.5, 0.3, 0.2)] * 3,  # Thu
            ("AAPL", "2020-01-03"): [(0.5, 0.3, 0.2)] * 2,  # Fri
            ("AAPL", "2020-01-06"): [(0.5, 0.3, 0.2)] * 1,  # Mon
        }
    )
    # log1p_volume=False so we can predict exact sums.
    out = build_news_stats_11dim(
        df, log1p_volume=False, rolling_window=3,
    )
    # Window of 3: on Mon(06) rolling sum = 3+2+1 = 6.
    mon = out[out["Date"] == pd.Timestamp("2020-01-06")].iloc[0]
    assert mon["news_volume_w3"] == pytest.approx(6.0, abs=1e-5)


def test_rolling_window_w_log1p() -> None:
    df = _make_sentiments_df(
        {
            ("AAPL", "2020-01-02"): [(0.5, 0.3, 0.2)] * 3,
            ("AAPL", "2020-01-03"): [(0.5, 0.3, 0.2)] * 2,
        }
    )
    out = build_news_stats_11dim(
        df, log1p_volume=True, rolling_window=3,
    )
    fri = out[out["Date"] == pd.Timestamp("2020-01-03")].iloc[0]
    # Raw rolling sum on Fri = 3 + 2 = 5 -> log1p(5).
    assert fri["news_volume_w3"] == pytest.approx(np.log1p(5), abs=1e-5)


def test_rolling_window_min_one() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        build_news_stats_11dim(
            _make_sentiments_df({("AAPL", "2020-01-02"): [(0.5, 0.3, 0.2)]}),
            rolling_window=0,
        )


def test_unknown_empty_day_policy_raises() -> None:
    df = _make_sentiments_df({("AAPL", "2020-01-02"): [(0.5, 0.3, 0.2)]})
    with pytest.raises(ValueError, match="Unknown empty_day_policy"):
        build_news_stats_11dim(df, empty_day_policy="bogus")  # type: ignore[arg-type]


def test_missing_probability_columns_raises() -> None:
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL"],
            "Date": [pd.Timestamp("2020-01-02")],
        }
    )
    with pytest.raises(ValueError, match="p_pos"):
        build_news_stats_11dim(df)


def test_empty_input_returns_empty_frame() -> None:
    df = pd.DataFrame(
        {
            "Ticker": pd.Series(dtype="string"),
            "Date": pd.to_datetime(pd.Series(dtype="object")),
            "p_pos": pd.Series(dtype="float32"),
            "p_neg": pd.Series(dtype="float32"),
            "p_neu": pd.Series(dtype="float32"),
        }
    )
    out = build_news_stats_11dim(df)
    assert len(out) == 0
    # Schema contract: 11 feature columns + Ticker + Date.
    assert "has_news" in out.columns
    assert "news_volume" in out.columns


def test_log1p_passthrough_defaults_match_spec() -> None:
    """The LOG1P_COLS / PASSTHROUGH_COLS constants document the scaler wiring."""
    assert "news_volume" in LOG1P_COLS
    assert "news_volume_w{W}" in LOG1P_COLS
    assert "has_news" in PASSTHROUGH_COLS


def test_eleven_feature_names_template() -> None:
    """The canonical 11 feature names are documented in the constant."""
    # Strip placeholders when counting; 11 concrete names.
    assert len(NEWS_STATS_FEATURES) == 11


def test_sorted_by_ticker_date() -> None:
    df = _make_sentiments_df(
        {
            ("MSFT", "2020-01-02"): [(0.5, 0.3, 0.2)],
            ("AAPL", "2020-01-03"): [(0.5, 0.3, 0.2)],
            ("AAPL", "2020-01-02"): [(0.5, 0.3, 0.2)],
        }
    )
    out = build_news_stats_11dim(df, rolling_window=3)
    tickers = list(out["Ticker"])
    # AAPL rows precede MSFT rows.
    aapl_indices = [i for i, t in enumerate(tickers) if t == "AAPL"]
    msft_indices = [i for i, t in enumerate(tickers) if t == "MSFT"]
    assert max(aapl_indices) < min(msft_indices)
    # Within a ticker, Date is ascending.
    for ticker in ("AAPL", "MSFT"):
        sub = out[out["Ticker"] == ticker]
        assert list(sub["Date"]) == sorted(sub["Date"])
