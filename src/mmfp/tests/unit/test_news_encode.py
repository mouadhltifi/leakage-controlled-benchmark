"""Unit tests for :mod:`mmfp.features.news_encode`.

These tests download HuggingFace weights, which is too slow for default
CI. Gated behind the ``RUN_ENCODING_TESTS`` environment variable so
they run only when explicitly requested.

The ``_clean_text_batch`` helper has cheap pure-Python tests that
always run.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mmfp.features.news_encode import _clean_text_batch, _select_device


# ---------------------------------------------------------------------------
# Cheap tests (always run)
# ---------------------------------------------------------------------------


def test_clean_text_batch_coerces_nan_and_none() -> None:
    out = _clean_text_batch([None, float("nan"), "hello", 42])
    assert out == ["", "", "hello", "42"]


def test_clean_text_batch_truncates_long_strings() -> None:
    long = "a" * 2000
    out = _clean_text_batch([long], max_char_len=100)
    assert out == ["a" * 100]


def test_select_device_auto_returns_known_device() -> None:
    d = _select_device("auto")
    assert d in {"mps", "cuda", "cpu"}


def test_select_device_cpu_always_ok() -> None:
    assert _select_device("cpu") == "cpu"


# ---------------------------------------------------------------------------
# Slow tests — require RUN_ENCODING_TESTS=1 and network access.
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.slow


@pytest.mark.skipif(
    not os.environ.get("RUN_ENCODING_TESTS"),
    reason="Requires RUN_ENCODING_TESTS=1 and HuggingFace model download.",
)
def test_encode_articles_sentiments_shape() -> None:
    """End-to-end smoke: encode 3 dummy articles with FinBERT."""
    from mmfp.features.news_encode import encode_articles_sentiments

    articles = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-01-03"]
            ),
            "Ticker": ["AAPL", "AAPL", "MSFT"],
            "Title": [
                "Apple posts record earnings",
                "Apple faces lawsuit",
                "Microsoft cloud growth exceeds estimates",
            ],
        }
    )
    out = encode_articles_sentiments(articles, batch_size=2, device="cpu")
    assert list(out.columns) == ["Date", "Ticker", "p_pos", "p_neg", "p_neu"]
    assert len(out) == 3
    # Sum of probabilities ~ 1 per row.
    probs = out[["p_pos", "p_neg", "p_neu"]].sum(axis=1).to_numpy()
    np.testing.assert_allclose(probs, 1.0, atol=1e-4)


@pytest.mark.skipif(
    not os.environ.get("RUN_ENCODING_TESTS"),
    reason="Requires RUN_ENCODING_TESTS=1 and HuggingFace model download.",
)
def test_encode_articles_embeddings_shape() -> None:
    """End-to-end smoke: encode 3 dummy articles into embeddings."""
    from mmfp.features.news_encode import encode_articles_embeddings

    articles = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-01-03"]
            ),
            "Ticker": ["AAPL", "AAPL", "MSFT"],
            "Title": [
                "Apple posts record earnings",
                "Apple faces lawsuit",
                "Microsoft cloud growth exceeds estimates",
            ],
        }
    )
    out = encode_articles_embeddings(
        articles,
        model_name="ProsusAI/finbert",
        batch_size=2,
        device="cpu",
        pooling="cls",
        l2_normalize=True,
    )
    # Expect FinBERT's 768-dim.
    d = sum(1 for c in out.columns if c.startswith("emb_"))
    assert d == 768
    # l2_normalize=True -> row norms ~ 1.
    emb_cols = [c for c in out.columns if c.startswith("emb_")]
    norms = np.linalg.norm(out[emb_cols].to_numpy(), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_encode_sentiments_rejects_empty_df() -> None:
    from mmfp.features.news_encode import encode_articles_sentiments

    with pytest.raises(ValueError, match="empty"):
        encode_articles_sentiments(
            pd.DataFrame(
                {
                    "Date": pd.to_datetime(pd.Series(dtype="object")),
                    "Ticker": pd.Series(dtype="string"),
                    "Title": pd.Series(dtype="string"),
                }
            )
        )


def test_encode_sentiments_rejects_missing_text_column() -> None:
    from mmfp.features.news_encode import encode_articles_sentiments

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02"]),
            "Ticker": ["AAPL"],
        }
    )
    with pytest.raises(ValueError, match="Title"):
        encode_articles_sentiments(df)


def test_encode_embeddings_rejects_bad_pooling() -> None:
    from mmfp.features.news_encode import encode_articles_embeddings

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02"]),
            "Ticker": ["AAPL"],
            "Title": ["hi"],
        }
    )
    with pytest.raises(ValueError, match="pooling must be"):
        encode_articles_embeddings(
            df, model_name="ProsusAI/finbert", pooling="bogus",  # type: ignore[arg-type]
        )
