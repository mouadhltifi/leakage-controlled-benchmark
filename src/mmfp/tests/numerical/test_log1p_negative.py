"""Numerical-stability test: log1p on negative inputs raises a clear error.

Motivation
----------

``numpy.log1p(x)`` is finite only for ``x >= -1``; for the study's
count features (news volume, social volume) the domain is ``x >= 0``
and any negative value signals a bug upstream (e.g. bad subtraction,
mis-reindex, corrupted cache). v2 must raise a clear ``ValueError`` at
fit/transform time rather than silently propagating ``NaN``.

This complements the happy-path log1p tests in
``mmfp/tests/unit/test_scalers.py`` and
``mmfp/tests/unit/test_news_stats.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmfp.features.scalers import FittedScaler


def test_fit_log1p_negative_value_raises() -> None:
    df = pd.DataFrame({"vol": [0.0, 1.0, 2.0, -0.1]})
    scaler = FittedScaler(log1p_cols=["vol"])
    with pytest.raises(ValueError, match="negative value"):
        scaler.fit(df)


def test_transform_log1p_negative_value_raises() -> None:
    train = pd.DataFrame({"vol": [0.0, 1.0, 2.0]})
    val = pd.DataFrame({"vol": [-0.01, 1.0]})
    scaler = FittedScaler(log1p_cols=["vol"]).fit(train)
    with pytest.raises(ValueError, match="negative value"):
        scaler.transform(val)


def test_fit_log1p_handles_large_positive_values() -> None:
    """Very large positive volumes work (log1p stays finite)."""
    df = pd.DataFrame({"vol": np.array([0.0, 1e6, 1e9, 1e12], dtype=np.float64)})
    scaler = FittedScaler(log1p_cols=["vol"])
    scaler.fit(df)
    out = scaler.transform(df)
    assert np.isfinite(out).all()


def test_fit_log1p_boundary_negative_one_raises() -> None:
    """``log1p(-1) = -inf``: we conservatively raise on any negative value."""
    df = pd.DataFrame({"vol": [0.0, -1.0]})
    scaler = FittedScaler(log1p_cols=["vol"])
    with pytest.raises(ValueError, match="negative value"):
        scaler.fit(df)
