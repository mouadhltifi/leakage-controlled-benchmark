"""Critical robustness case: spherical mean on zero embeddings.

Why this matters
----------------

The spherical mean is defined as ``v_mean / ||v_mean||``. When every
per-article embedding is the zero vector (or they cancel out exactly),
``||v_mean|| == 0`` and the naive division yields ``NaN`` across the
entire aggregated row. Downstream standardisation (z-score) propagates
those NaNs into the batch and the loss becomes ``NaN`` on the first
gradient step — silently killing a run with no useful error.

v2 must therefore return the zero vector (and zero dispersion)
whenever the input's mean has norm below the epsilon floor. This test
file exists specifically to enforce that invariant in isolation so any
regression fails loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmfp.features.news_aggregate import SphericalMean, aggregate_daily


def test_spherical_mean_all_zero_input_yields_zero_output() -> None:
    """All-zero embeddings -> all-zero output, no NaN/Inf, dispersion = 0."""
    emb = np.zeros((5, 32), dtype=np.float32)
    sent = np.zeros(5, dtype=np.float32)
    idx = np.arange(5, dtype=np.int64)

    out, extras = SphericalMean().aggregate(emb, sent, idx)

    assert out.shape == (32,)
    assert out.dtype == np.float32
    assert not np.isnan(out).any(), "SphericalMean produced NaN on zero input"
    assert not np.isinf(out).any(), "SphericalMean produced Inf on zero input"
    np.testing.assert_array_equal(out, np.zeros(32, dtype=np.float32))
    assert extras == {"dispersion": 0.0}


def test_spherical_mean_antipodal_cancellation() -> None:
    """Perfectly cancelling articles also yield zero mean -> zero output."""
    # Two articles, exactly opposite; mean is zero.
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb = np.stack([v, -v], axis=0)
    sent = np.zeros(2, dtype=np.float32)
    idx = np.array([0, 1], dtype=np.int64)

    out, extras = SphericalMean().aggregate(emb, sent, idx)

    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    np.testing.assert_allclose(out, np.zeros(4, dtype=np.float32), atol=1e-6)
    assert extras["dispersion"] < 1e-6


def test_aggregate_daily_spherical_zero_vectors_in_batch() -> None:
    """End-to-end: aggregate_daily with spherical + zero-vector day."""
    # Day 1: two zero-vector articles (edge case).
    # Day 2: two real articles.
    rng = np.random.default_rng(0)
    d = 8
    rows: list[dict[str, object]] = []
    for i in range(2):
        row = {
            "Ticker": "AAPL",
            "Date": pd.Timestamp("2020-01-02"),
            "sent_score": 0.0,
            "article_idx": i,
        }
        for k in range(d):
            row[f"emb_{k}"] = 0.0
        rows.append(row)
    for i in range(2):
        row = {
            "Ticker": "AAPL",
            "Date": pd.Timestamp("2020-01-03"),
            "sent_score": 0.5,
            "article_idx": i,
        }
        emb_vec = rng.normal(size=d).astype(np.float32)
        for k, v in enumerate(emb_vec):
            row[f"emb_{k}"] = float(v)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = aggregate_daily(
        df, strategy="spherical_mean", dispersion_feature=True,
    )
    # Day 1 row: all-zero embedding, zero dispersion.
    day1 = out[out["Date"] == pd.Timestamp("2020-01-02")].iloc[0]
    for k in range(d):
        assert day1[f"emb_{k}"] == 0.0
    assert day1["dispersion"] == 0.0
    # Day 2 row: unit-norm embedding.
    day2 = out[out["Date"] == pd.Timestamp("2020-01-03")].iloc[0]
    vec = np.array([day2[f"emb_{k}"] for k in range(d)], dtype=np.float64)
    assert np.isfinite(vec).all()
    # Close to unit norm (allow float32 rounding).
    np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-4)
