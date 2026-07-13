"""Audit-critical leakage test for :class:`FittedScaler`.

If this test fails, standardization is leaking val/test statistics into
the train-fit parameters. This is the exact bug that Section 3.3 of the
rewrite spec identifies as the audit-critical finding. The test must
stay green across refactors, which is why it lives in a dedicated
``leakage/`` directory separate from unit tests.

This is one of the leakage tests: the standardization contract requires that
scaler statistics are fit on the train block only, and that val/test rows are
transformed with the train parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmfp.features.scalers import FittedScaler


@pytest.fixture
def train_all_zeros_val_all_hundred() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic train/val block with dramatically different means.

    Train block: ``feature_a`` = 0.0 for every row, ``feature_b``
    linspace-0..1 for sanity.
    Val block: ``feature_a`` = 100.0 for every row, ``feature_b`` = 2.0.

    If the scaler is ever accidentally fit on train+val, the learnt
    mean for ``feature_a`` will be ~50, not 0. The test below pins the
    learnt mean to 0 exactly.
    """
    n_train = 50
    n_val = 50
    train = pd.DataFrame(
        {
            "feature_a": np.zeros(n_train, dtype=np.float64),
            "feature_b": np.linspace(0.0, 1.0, n_train),
        }
    )
    val = pd.DataFrame(
        {
            "feature_a": np.full(n_val, 100.0, dtype=np.float64),
            "feature_b": np.full(n_val, 2.0, dtype=np.float64),
        }
    )
    return train, val


def test_mean_fit_on_train_only_not_pooled(
    train_all_zeros_val_all_hundred: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """``scaler.mean_`` for ``feature_a`` must equal 0.0, not ~50.0.

    This is the canonical leakage check. A scaler that silently pooled
    train+val would learn mean = 50 for ``feature_a`` (since train is
    all zeros and val is all hundreds). The scaler must refuse.
    """
    train, _val = train_all_zeros_val_all_hundred

    scaler = FittedScaler()
    scaler.fit(train)

    assert scaler.mean_ is not None
    # Column 0 is feature_a (all zeros in train).
    assert scaler.mean_[0] == pytest.approx(0.0, abs=1e-12), (
        "FittedScaler.mean_ for feature_a must be exactly 0.0 (train block "
        "was all zeros). If this is ~50, the scaler is pooling train+val — "
        "that is the leakage bug this class exists to prevent."
    )


def test_transform_val_uses_train_stats_not_val_stats(
    train_all_zeros_val_all_hundred: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Val-block ``feature_a`` after transform must reflect train stats.

    With train mean 0 and train std 0 (clamped to ``_STD_EPSILON``),
    the val column is ``(100 - 0) / epsilon = 1e10``. That enormous
    value is *correct* for this test: it proves the scaler used
    train-only parameters, not val statistics. If pooling had occurred
    the val values would be ``(100 - 50) / 50 = 1.0``, which the
    assertion below explicitly rules out.
    """
    train, val = train_all_zeros_val_all_hundred

    scaler = FittedScaler()
    scaler.fit(train)
    val_out = scaler.transform(val)

    # feature_a column (index 0) of the val output.
    val_a = val_out[:, 0]

    # Guard against the "leaked" outcome: if the scaler pooled train+val,
    # feature_a val output would be around +1 (one std above mean).
    # We reject that band explicitly.
    assert not np.all(np.abs(val_a - 1.0) < 0.1), (
        "Val-block feature_a transformed to ~1.0 — this indicates the "
        "scaler learnt pooled train+val statistics (mean ~50, std ~50)."
    )

    # The correct (leak-free) outcome: (100 - 0) / max(0, epsilon) is
    # huge. We only assert it's much larger than the "leaked" band.
    assert np.all(val_a > 100.0), (
        "Val-block feature_a should be >> 1 after train-only fit "
        "(train std was 0 and was clamped; so (100-0)/epsilon >> 1)."
    )


def test_transform_val_feature_b_uses_train_stats(
    train_all_zeros_val_all_hundred: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """``feature_b`` has non-trivial train variance; verify arithmetic.

    Train ``feature_b`` is ``linspace(0, 1, 50)``: mean ~0.5, population
    std ~0.29. Val ``feature_b`` is all 2.0. After transform:
    ``(2.0 - 0.5) / 0.29 ~= 5.1``. The exact value isn't the point —
    the test pins it to a narrow band to catch a pooling bug (which
    would give ``(2.0 - 1.25) / 0.75 ~= 1.0``).
    """
    train, val = train_all_zeros_val_all_hundred

    scaler = FittedScaler()
    scaler.fit(train)

    train_mean_b = float(train["feature_b"].mean())
    train_std_b = float(train["feature_b"].std(ddof=0))  # population

    val_out = scaler.transform(val)
    val_b = val_out[:, 1]

    expected = (2.0 - train_mean_b) / train_std_b
    np.testing.assert_allclose(val_b, expected, rtol=1e-4)


def test_train_transform_has_zero_mean(
    train_all_zeros_val_all_hundred: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Sanity: applying the transform to the train block yields mean 0 / std 1.

    Not a leakage check per se, but confirms the scaler is actually
    doing z-scoring rather than being an accidental no-op.
    """
    train, _val = train_all_zeros_val_all_hundred

    scaler = FittedScaler()
    out = scaler.fit_transform(train)

    # feature_b mean should be 0 post-z-score.
    assert out[:, 1].mean() == pytest.approx(0.0, abs=1e-6)
    # feature_a train values were all zero; after (0 - 0) / epsilon = 0.
    assert out[:, 0].mean() == pytest.approx(0.0, abs=1e-6)


def test_refit_raises_protects_against_accidental_pooling() -> None:
    """Re-fitting after an earlier fit must raise.

    If callers could silently refit, the common-in-practice pattern
    ``fit(train).fit(val)`` (say, from a copy-paste error) would
    overwrite train stats with val stats — an even more severe leakage
    failure than pooling. We make that pattern explicit by raising.
    """
    scaler = FittedScaler()
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    scaler.fit(df)

    with pytest.raises(RuntimeError, match="fit called twice"):
        scaler.fit(df)


def test_transform_before_fit_raises() -> None:
    """Transforming before fit must raise, not return untransformed data.

    A silent pass-through would cause downstream code to train on raw
    (unstandardised) features without warning. We reject that.
    """
    scaler = FittedScaler()
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    with pytest.raises(RuntimeError, match="before fit"):
        scaler.transform(df)
