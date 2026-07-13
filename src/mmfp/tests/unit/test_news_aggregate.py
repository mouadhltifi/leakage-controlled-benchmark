"""Unit tests for :mod:`mmfp.features.news_aggregate`.

Covers every aggregation strategy + the DataFrame-level driver
:func:`aggregate_daily`. Numerical edge cases (e.g. zero-vector
spherical mean) have dedicated tests in
``mmfp/tests/numerical/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmfp.features.news_aggregate import (
    AGGREGATION_REGISTRY,
    ArithmeticMean,
    AttentionWeighted,
    MaxSentiment,
    RecencyWeighted,
    SphericalMean,
    aggregate_daily,
    build_strategy,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_all_five_strategies() -> None:
    assert set(AGGREGATION_REGISTRY.keys()) == {
        "arithmetic_mean",
        "spherical_mean",
        "attention_weighted",
        "recency_weighted",
        "max_sentiment",
    }


def test_registry_maps_to_classes() -> None:
    assert AGGREGATION_REGISTRY["arithmetic_mean"] is ArithmeticMean
    assert AGGREGATION_REGISTRY["spherical_mean"] is SphericalMean
    assert AGGREGATION_REGISTRY["attention_weighted"] is AttentionWeighted
    assert AGGREGATION_REGISTRY["recency_weighted"] is RecencyWeighted
    assert AGGREGATION_REGISTRY["max_sentiment"] is MaxSentiment


def test_build_strategy_factory() -> None:
    for name in AGGREGATION_REGISTRY:
        strat = build_strategy(name)
        assert hasattr(strat, "aggregate")


def test_build_strategy_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown aggregation"):
        build_strategy("not_a_strategy")


def test_build_strategy_respects_kwargs() -> None:
    recency = build_strategy("recency_weighted", decay=0.5)
    assert isinstance(recency, RecencyWeighted)
    assert recency.decay == pytest.approx(0.5)
    att = build_strategy("attention_weighted", attention_tau=2.0)
    assert isinstance(att, AttentionWeighted)
    assert att.tau == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Synthetic 3-article batch used across strategies
# ---------------------------------------------------------------------------


@pytest.fixture
def three_articles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three 4-dim embeddings with distinguishable sentiment scores."""
    emb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    sent = np.array([0.1, -0.9, 0.4], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.int64)
    return emb, sent, idx


# ---------------------------------------------------------------------------
# ArithmeticMean
# ---------------------------------------------------------------------------


def test_arithmetic_mean_matches_numpy_mean(three_articles) -> None:
    emb, sent, idx = three_articles
    out, extras = ArithmeticMean().aggregate(emb, sent, idx)
    np.testing.assert_allclose(out, emb.mean(axis=0), rtol=1e-6)
    assert extras == {}


def test_arithmetic_mean_output_shape(three_articles) -> None:
    emb, sent, idx = three_articles
    out, _ = ArithmeticMean().aggregate(emb, sent, idx)
    assert out.shape == (4,)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# SphericalMean
# ---------------------------------------------------------------------------


def test_spherical_mean_produces_unit_norm_when_mean_nonzero(three_articles) -> None:
    emb, sent, idx = three_articles
    out, extras = SphericalMean().aggregate(emb, sent, idx)
    # Non-zero input: output must be unit-norm.
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, rtol=1e-6)
    # Dispersion is the pre-normalisation L2 norm of the mean.
    expected_norm = np.linalg.norm(emb.mean(axis=0))
    assert extras["dispersion"] == pytest.approx(float(expected_norm), rel=1e-5)


def test_spherical_mean_reduces_to_unit_vector_for_single_direction() -> None:
    """If every article is the same direction, spherical mean returns that
    direction unit-normalised."""
    d = 6
    v = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb = np.tile(v, (5, 1))
    sent = np.zeros(5, dtype=np.float32)
    idx = np.arange(5, dtype=np.int64)
    out, extras = SphericalMean().aggregate(emb, sent, idx)
    np.testing.assert_allclose(
        out,
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        rtol=1e-6,
    )
    # Dispersion equals the original magnitude (0.2).
    assert extras["dispersion"] == pytest.approx(0.2, rel=1e-5)


@pytest.mark.parametrize(
    "noise_scale,expected_disp_lt",
    [(0.0, None), (0.5, None), (2.0, None)],
)
def test_spherical_mean_dispersion_monotonic_in_noise(
    noise_scale: float, expected_disp_lt: None
) -> None:
    """Dispersion decreases as semantic spread grows.

    Construct a cluster of near-identical unit vectors then perturb by
    increasing Gaussian noise. The pre-normalisation mean norm
    (``dispersion``) must fall monotonically.
    """
    rng = np.random.default_rng(42)
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Normalise to unit length (typical for BGE / Qwen3 embeddings).
    base /= np.linalg.norm(base)

    disp_values: list[float] = []
    for scale in [0.0, 0.3, 1.0, 3.0]:
        emb = np.tile(base, (10, 1)) + rng.normal(
            scale=scale, size=(10, base.size)
        ).astype(np.float32)
        # Renormalise so each embedding is still unit-norm (mirrors
        # l2_normalize=True in encode_articles_embeddings).
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)

        _, extras = SphericalMean().aggregate(
            emb,
            np.zeros(10, dtype=np.float32),
            np.arange(10, dtype=np.int64),
        )
        disp_values.append(extras["dispersion"])

    # Strictly non-increasing over the noise ladder.
    assert disp_values == sorted(disp_values, reverse=True)


def test_spherical_mean_zero_input_yields_zero_output() -> None:
    """Critical robustness case: zero vector in -> zero vector out, no NaN."""
    emb = np.zeros((4, 16), dtype=np.float32)
    sent = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    idx = np.arange(4, dtype=np.int64)
    out, extras = SphericalMean().aggregate(emb, sent, idx)
    np.testing.assert_array_equal(out, np.zeros(16, dtype=np.float32))
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert extras["dispersion"] == 0.0


# ---------------------------------------------------------------------------
# AttentionWeighted
# ---------------------------------------------------------------------------


def test_attention_weighted_reduces_to_uniform_when_sent_all_zero(
    three_articles,
) -> None:
    """Zero sentiment everywhere -> softmax(0/tau) = uniform -> arithmetic mean."""
    emb, _, idx = three_articles
    zeros = np.zeros_like(idx, dtype=np.float32)
    out_att, _ = AttentionWeighted().aggregate(emb, zeros, idx)
    out_arith, _ = ArithmeticMean().aggregate(emb, zeros, idx)
    np.testing.assert_allclose(out_att, out_arith, rtol=1e-6)


def test_attention_weighted_emphasises_extreme_sentiment() -> None:
    """The strongest |sent| article dominates the weighted sum."""
    emb = np.eye(3, 4, dtype=np.float32)
    sent = np.array([0.01, 5.0, 0.0], dtype=np.float32)
    idx = np.arange(3, dtype=np.int64)
    # Low temperature amplifies selection.
    out, _ = AttentionWeighted(tau=1.0).aggregate(emb, sent, idx)
    # Mass is concentrated on the second article's embedding.
    assert out[1] > out[0]
    assert out[1] > out[2]


def test_attention_tau_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tau must be > 0"):
        AttentionWeighted(tau=0.0)


# ---------------------------------------------------------------------------
# RecencyWeighted
# ---------------------------------------------------------------------------


def test_recency_weighted_decay_one_equals_arithmetic_mean(three_articles) -> None:
    emb, sent, idx = three_articles
    out_rec, _ = RecencyWeighted(decay=1.0).aggregate(emb, sent, idx)
    out_arith, _ = ArithmeticMean().aggregate(emb, sent, idx)
    np.testing.assert_allclose(out_rec, out_arith, rtol=1e-6)


def test_recency_weighted_emphasises_latest_article() -> None:
    """With small decay, the last article dominates."""
    emb = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    sent = np.zeros(3, dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.int64)  # 2 is latest
    out, _ = RecencyWeighted(decay=0.1).aggregate(emb, sent, idx)
    # Second coordinate (from the latest article) must dominate.
    assert out[1] > out[0]
    assert out[1] > 0.8  # heavy concentration on the final article


def test_recency_weighted_respects_article_idx_not_row_order() -> None:
    """Row order != article order: recency uses article_idx, not position."""
    emb = np.array(
        [
            [1.0, 0.0],  # row 0 but article_idx 2 = latest
            [0.0, 1.0],  # row 1, article_idx 0 = earliest
            [0.0, 0.0],  # row 2, article_idx 1
        ],
        dtype=np.float32,
    )
    sent = np.zeros(3, dtype=np.float32)
    idx = np.array([2, 0, 1], dtype=np.int64)
    out, _ = RecencyWeighted(decay=0.01).aggregate(emb, sent, idx)
    # The "latest" article (row 0) should dominate -> first coordinate wins.
    assert out[0] > out[1]


def test_recency_decay_range_enforced() -> None:
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        RecencyWeighted(decay=0.0)
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        RecencyWeighted(decay=1.5)


# ---------------------------------------------------------------------------
# MaxSentiment
# ---------------------------------------------------------------------------


def test_max_sentiment_picks_extreme_article() -> None:
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    sent = np.array([0.1, -0.9, 0.2], dtype=np.float32)  # |0.9| is max
    idx = np.arange(3, dtype=np.int64)
    out, extras = MaxSentiment().aggregate(emb, sent, idx)
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert extras == {}


def test_max_sentiment_first_occurrence_tiebreak() -> None:
    """Two articles with equal |sent| magnitude -> the first wins."""
    emb = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    sent = np.array([0.5, 0.5], dtype=np.float32)
    idx = np.arange(2, dtype=np.int64)
    out, _ = MaxSentiment().aggregate(emb, sent, idx)
    np.testing.assert_allclose(out, np.array([1.0, 0.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# Single-article day: every strategy returns that one vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy_name",
    [
        "arithmetic_mean",
        "spherical_mean",
        "attention_weighted",
        "recency_weighted",
        "max_sentiment",
    ],
)
def test_single_article_day_returns_that_vector(strategy_name: str) -> None:
    v = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    emb = v.reshape(1, -1)
    sent = np.array([0.3], dtype=np.float32)
    idx = np.array([0], dtype=np.int64)
    strat = build_strategy(strategy_name)
    out, _ = strat.aggregate(emb, sent, idx)

    if strategy_name == "spherical_mean":
        # Unit-normalised version of v.
        expected = v / np.linalg.norm(v)
    else:
        expected = v
    np.testing.assert_allclose(out, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


def test_aggregate_rejects_empty_input() -> None:
    emb = np.zeros((0, 5), dtype=np.float32)
    sent = np.zeros(0, dtype=np.float32)
    idx = np.zeros(0, dtype=np.int64)
    with pytest.raises(ValueError, match="zero articles"):
        ArithmeticMean().aggregate(emb, sent, idx)


def test_aggregate_rejects_non_2d_embeddings() -> None:
    emb = np.zeros(5, dtype=np.float32)  # 1D
    sent = np.zeros(5, dtype=np.float32)
    idx = np.zeros(5, dtype=np.int64)
    with pytest.raises(ValueError, match="2D"):
        ArithmeticMean().aggregate(emb, sent, idx)  # type: ignore[arg-type]


def test_aggregate_rejects_mismatched_sent_length() -> None:
    emb = np.zeros((3, 5), dtype=np.float32)
    sent = np.zeros(2, dtype=np.float32)  # wrong length
    idx = np.zeros(3, dtype=np.int64)
    with pytest.raises(ValueError, match="sent_scores shape"):
        ArithmeticMean().aggregate(emb, sent, idx)


def test_aggregate_rejects_mismatched_idx_length() -> None:
    emb = np.zeros((3, 5), dtype=np.float32)
    sent = np.zeros(3, dtype=np.float32)
    idx = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="article_idx shape"):
        ArithmeticMean().aggregate(emb, sent, idx)


# ---------------------------------------------------------------------------
# aggregate_daily driver
# ---------------------------------------------------------------------------


def _make_per_article_df(
    ticker_dates_articles: list[tuple[str, str, int]],
    d: int = 3,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a synthetic per-article DataFrame.

    Each tuple is ``(Ticker, Date_str, n_articles_for_that_day)``.
    """
    rng = rng or np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    article_idx_counter: dict[tuple[str, str], int] = {}
    for ticker, date_str, n in ticker_dates_articles:
        for _ in range(n):
            row: dict[str, object] = {
                "Ticker": ticker,
                "Date": pd.Timestamp(date_str),
            }
            emb = rng.normal(size=d).astype(np.float32)
            for i, v in enumerate(emb):
                row[f"emb_{i}"] = float(v)
            row["sent_score"] = float(rng.uniform(-1.0, 1.0))
            key = (ticker, date_str)
            row["article_idx"] = article_idx_counter.get(key, 0)
            article_idx_counter[key] = row["article_idx"] + 1
            rows.append(row)
    return pd.DataFrame(rows)


def test_aggregate_daily_returns_one_row_per_ticker_date() -> None:
    df = _make_per_article_df(
        [
            ("AAPL", "2020-01-02", 3),
            ("AAPL", "2020-01-03", 2),
            ("MSFT", "2020-01-02", 5),
        ]
    )
    out = aggregate_daily(df, strategy="arithmetic_mean")
    assert len(out) == 3
    assert set(out["Ticker"]) == {"AAPL", "MSFT"}


def test_aggregate_daily_news_volume_and_has_news_columns() -> None:
    df = _make_per_article_df([("AAPL", "2020-01-02", 4)])
    out = aggregate_daily(df, strategy="arithmetic_mean")
    assert int(out["news_volume"].iloc[0]) == 4
    assert int(out["has_news"].iloc[0]) == 1


def test_aggregate_daily_preserves_date_ordering_within_ticker() -> None:
    # Shuffled input; output must still be sorted ascending.
    df = _make_per_article_df(
        [
            ("AAPL", "2020-01-05", 1),
            ("AAPL", "2020-01-02", 2),
            ("AAPL", "2020-01-03", 1),
        ]
    )
    out = aggregate_daily(df, strategy="arithmetic_mean")
    aapl = out[out["Ticker"] == "AAPL"]
    dates = list(aapl["Date"])
    assert dates == sorted(dates)


def test_aggregate_daily_dispersion_feature_only_with_spherical() -> None:
    df = _make_per_article_df([("AAPL", "2020-01-02", 3)])
    out_arith = aggregate_daily(
        df, strategy="arithmetic_mean", dispersion_feature=True,
    )
    # Flag has no effect for non-spherical strategies.
    assert "dispersion" not in out_arith.columns

    out_sph = aggregate_daily(
        df, strategy="spherical_mean", dispersion_feature=True,
    )
    assert "dispersion" in out_sph.columns
    assert out_sph["dispersion"].iloc[0] >= 0.0


def test_aggregate_daily_spherical_without_flag_no_dispersion() -> None:
    df = _make_per_article_df([("AAPL", "2020-01-02", 3)])
    out = aggregate_daily(df, strategy="spherical_mean")
    assert "dispersion" not in out.columns


def test_aggregate_daily_rejects_missing_columns() -> None:
    # sent_score missing -> clear error referring to required column.
    df_missing_sent = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02"]),
            "Ticker": ["AAPL"],
            "emb_0": [0.0],
        }
    )
    with pytest.raises(ValueError, match="sent_score"):
        aggregate_daily(df_missing_sent, strategy="arithmetic_mean")

    # emb_* missing -> separate clear error.
    df_missing_emb = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02"]),
            "Ticker": ["AAPL"],
            "sent_score": [0.1],
        }
    )
    with pytest.raises(ValueError, match="no columns named 'emb_\\*'"):
        aggregate_daily(df_missing_emb, strategy="arithmetic_mean")


def test_aggregate_daily_rejects_unknown_strategy() -> None:
    df = _make_per_article_df([("AAPL", "2020-01-02", 1)])
    with pytest.raises(ValueError, match="Unknown aggregation"):
        aggregate_daily(df, strategy="nonexistent")


def test_aggregate_daily_accepts_tz_aware_dates() -> None:
    """Upstream parquets store Date as UTC tz-aware; output is tz-naive."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL", "AAPL"],
            "Date": pd.to_datetime(
                ["2020-01-02 15:30:00", "2020-01-02 20:00:00"], utc=True,
            ),
            "emb_0": rng.normal(size=2).astype(np.float32),
            "emb_1": rng.normal(size=2).astype(np.float32),
            "sent_score": np.array([0.2, -0.1], dtype=np.float32),
            "article_idx": [0, 1],
        }
    )
    out = aggregate_daily(df, strategy="arithmetic_mean")
    assert len(out) == 1
    # Date is tz-naive in the output (unit may be ns or us depending on pandas).
    assert np.issubdtype(out["Date"].dtype, np.datetime64)
    assert getattr(out["Date"].dtype, "tz", None) is None


def test_aggregate_daily_fallback_to_row_order_when_no_idx() -> None:
    """Missing article_idx and article_order: row order is used."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL"] * 3,
            "Date": pd.to_datetime(["2020-01-02"] * 3),
            "emb_0": [1.0, 0.0, 0.0],
            "emb_1": [0.0, 1.0, 0.0],
            "emb_2": [0.0, 0.0, 1.0],
            "sent_score": [0.0, 0.0, 0.0],
        }
    )
    out = aggregate_daily(
        df, strategy="recency_weighted", decay=0.01,
    )
    # Last row (emb_2) dominates under heavy decay.
    assert out["emb_2"].iloc[0] > out["emb_0"].iloc[0]


def test_aggregate_daily_emb_ordering_is_numeric() -> None:
    """emb_10 must come after emb_2 in the output, not before."""
    rng = np.random.default_rng(0)
    d = 12
    rows: list[dict[str, object]] = []
    for _ in range(2):
        row: dict[str, object] = {
            "Ticker": "AAPL",
            "Date": pd.Timestamp("2020-01-02"),
            "sent_score": 0.0,
            "article_idx": len(rows),
        }
        emb = rng.normal(size=d).astype(np.float32)
        for i, v in enumerate(emb):
            row[f"emb_{i}"] = float(v)
        rows.append(row)
    df = pd.DataFrame(rows)
    # Shuffle columns to make sure inference doesn't depend on input order.
    shuffled_cols = list(df.columns)
    np.random.default_rng(0).shuffle(shuffled_cols)
    df = df[shuffled_cols]

    out = aggregate_daily(df, strategy="arithmetic_mean")
    emb_cols_out = [c for c in out.columns if c.startswith("emb_")]
    # Numerically sorted: emb_0, emb_1, ..., emb_10, emb_11
    assert emb_cols_out == [f"emb_{i}" for i in range(d)]


def test_aggregate_daily_empty_input_returns_empty_frame() -> None:
    df = pd.DataFrame(
        {
            "Ticker": pd.Series(dtype="string"),
            "Date": pd.to_datetime(pd.Series(dtype="object")),
            "emb_0": pd.Series(dtype="float32"),
            "sent_score": pd.Series(dtype="float32"),
        }
    )
    out = aggregate_daily(df, strategy="arithmetic_mean")
    assert len(out) == 0
    assert "Ticker" in out.columns
    assert "Date" in out.columns
    assert "emb_0" in out.columns
    assert "news_volume" in out.columns
    assert "has_news" in out.columns
