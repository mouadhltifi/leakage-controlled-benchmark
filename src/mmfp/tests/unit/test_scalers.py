"""Unit tests for :class:`mmfp.features.scalers.FittedScaler`.

Leakage-specific assertions live separately in
``mmfp/tests/leakage/test_scaler_fit_on_train_only.py``. This module
covers the happy path, edge cases, ``log1p`` / ``passthrough``
column handling, serialization round-trip, and column-order preservation.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from mmfp.features.scalers import FittedScaler


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_fit_transform_yields_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(5.0, 2.0, size=500),
            "b": rng.normal(-1.0, 0.5, size=500),
        }
    )
    scaler = FittedScaler()
    out = scaler.fit_transform(df)

    assert out.shape == df.shape
    # After z-score, each column has mean ~0 and std ~1.
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=0, ddof=0), 1.0, atol=1e-5)


def test_output_dtype_is_float32() -> None:
    """Downstream consumers expect float32 arrays."""
    df = pd.DataFrame({"x": np.linspace(0.0, 1.0, 20)})
    out = FittedScaler().fit_transform(df)
    assert out.dtype == np.float32


def test_fit_returns_self_for_chaining() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    scaler = FittedScaler()
    returned = scaler.fit(df)
    assert returned is scaler


def test_is_fitted_flag_flips_on_fit() -> None:
    scaler = FittedScaler()
    assert scaler.is_fitted is False
    scaler.fit(pd.DataFrame({"x": [1.0, 2.0]}))
    assert scaler.is_fitted is True


# ---------------------------------------------------------------------------
# Fit-time / transform-time discipline
# ---------------------------------------------------------------------------


def test_fit_twice_raises() -> None:
    scaler = FittedScaler()
    df = pd.DataFrame({"x": [1.0, 2.0]})
    scaler.fit(df)
    with pytest.raises(RuntimeError, match="fit called twice"):
        scaler.fit(df)


def test_transform_before_fit_raises() -> None:
    scaler = FittedScaler()
    with pytest.raises(RuntimeError, match="before fit"):
        scaler.transform(pd.DataFrame({"x": [1.0]}))


def test_fit_on_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty DataFrame"):
        FittedScaler().fit(pd.DataFrame({"x": pd.Series(dtype=float)}))


def test_fit_rejects_non_dataframe() -> None:
    with pytest.raises(ValueError, match="requires a DataFrame"):
        FittedScaler().fit(np.array([[1.0, 2.0]]))  # type: ignore[arg-type]


def test_transform_rejects_non_dataframe() -> None:
    scaler = FittedScaler()
    scaler.fit(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="requires a DataFrame"):
        scaler.transform(np.array([[1.0, 2.0]]))  # type: ignore[arg-type]


def test_feature_names_out_before_fit_raises() -> None:
    scaler = FittedScaler()
    with pytest.raises(RuntimeError, match="fitted scaler"):
        _ = scaler.feature_names_out_


# ---------------------------------------------------------------------------
# Column mismatch handling at transform time
# ---------------------------------------------------------------------------


def test_transform_missing_column_raises() -> None:
    scaler = FittedScaler().fit(
        pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    )
    with pytest.raises(ValueError, match="column mismatch"):
        scaler.transform(pd.DataFrame({"a": [1.0, 2.0]}))


def test_transform_extra_column_raises() -> None:
    scaler = FittedScaler().fit(pd.DataFrame({"a": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="column mismatch"):
        scaler.transform(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))


def test_transform_reorders_columns_to_fit_order() -> None:
    """If the val DataFrame has the same columns in a different order,
    the scaler reorders to the fit order before applying statistics."""
    train = pd.DataFrame({"a": [0.0, 10.0], "b": [100.0, 110.0]})
    val = pd.DataFrame({"b": [105.0], "a": [5.0]})  # reversed order

    scaler = FittedScaler().fit(train)
    out = scaler.transform(val)

    # Column order in output = fit order = [a, b].
    # Train: a mean=5 std=5, b mean=105 std=5.
    # Val after transform: a=(5-5)/5=0.0, b=(105-105)/5=0.0.
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# log1p_cols
# ---------------------------------------------------------------------------


def test_log1p_applied_before_standardization() -> None:
    """For ``log1p_cols``, the recorded mean/std must be on log1p(x), not x."""
    # Raw values 0..9; log1p gives [log1p(0), log1p(1), ..., log1p(9)].
    df = pd.DataFrame({"vol": np.arange(10, dtype=np.float64)})
    scaler = FittedScaler(log1p_cols=["vol"])
    scaler.fit(df)

    # Compare learnt mean/std to the mean/std of log1p(x).
    expected_mean = float(np.log1p(df["vol"]).mean())
    expected_std = float(np.log1p(df["vol"]).std(ddof=0))
    assert scaler.mean_ is not None and scaler.scale_ is not None
    assert scaler.mean_[0] == pytest.approx(expected_mean)
    assert scaler.scale_[0] == pytest.approx(expected_std)


def test_log1p_negative_in_fit_raises() -> None:
    df = pd.DataFrame({"vol": [0.0, 1.0, -0.5]})
    scaler = FittedScaler(log1p_cols=["vol"])
    with pytest.raises(ValueError, match="negative value"):
        scaler.fit(df)


def test_log1p_negative_in_transform_raises() -> None:
    train = pd.DataFrame({"vol": [0.0, 1.0, 2.0]})
    val = pd.DataFrame({"vol": [-0.1, 1.0]})
    scaler = FittedScaler(log1p_cols=["vol"]).fit(train)
    with pytest.raises(ValueError, match="negative value"):
        scaler.transform(val)


def test_log1p_missing_column_in_fit_raises() -> None:
    df = pd.DataFrame({"other": [1.0, 2.0]})
    with pytest.raises(ValueError, match="log1p_cols not present"):
        FittedScaler(log1p_cols=["missing"]).fit(df)


# ---------------------------------------------------------------------------
# passthrough_cols
# ---------------------------------------------------------------------------


def test_passthrough_column_values_unchanged() -> None:
    """Binary flags pass through unchanged."""
    train = pd.DataFrame(
        {
            "x": np.arange(10, dtype=np.float64),
            "flag": np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64),
        }
    )
    scaler = FittedScaler(passthrough_cols=["flag"]).fit(train)
    out = scaler.transform(train)

    # Second column should equal the raw flag values.
    np.testing.assert_allclose(out[:, 1], train["flag"].values.astype(np.float32))
    # First column should be z-scored (mean 0, std 1).
    assert out[:, 0].mean() == pytest.approx(0.0, abs=1e-6)
    assert out[:, 0].std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_passthrough_mean_zero_scale_one() -> None:
    df = pd.DataFrame({"flag": [0.0, 1.0, 1.0, 0.0]})
    scaler = FittedScaler(passthrough_cols=["flag"]).fit(df)
    assert scaler.mean_ is not None and scaler.scale_ is not None
    assert scaler.mean_[0] == 0.0
    assert scaler.scale_[0] == 1.0


def test_passthrough_missing_column_in_fit_raises() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="passthrough_cols not present"):
        FittedScaler(passthrough_cols=["nope"]).fit(df)


def test_log1p_and_passthrough_overlap_rejected_in_init() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        FittedScaler(log1p_cols=["x"], passthrough_cols=["x"])


# ---------------------------------------------------------------------------
# Column order preservation
# ---------------------------------------------------------------------------


def test_feature_names_out_matches_fit_order() -> None:
    """The output column order equals the input column order."""
    cols = ["c", "a", "b", "d", "z"]
    df = pd.DataFrame({c: np.linspace(i, i + 1.0, 10) for i, c in enumerate(cols)})
    scaler = FittedScaler().fit(df)
    assert scaler.feature_names_out_ == cols


def test_transform_output_column_order_matches_fit() -> None:
    cols = ["c", "a", "b"]
    df = pd.DataFrame({c: np.arange(10, dtype=np.float64) for c in cols})
    scaler = FittedScaler().fit(df)
    # Column 0 of output corresponds to "c" (first fit column). Inject a
    # distinctive large value into "c" and check column 0.
    probe = pd.DataFrame(
        {"a": [0.0], "b": [0.0], "c": [1000.0]}
    )  # reversed order
    out = scaler.transform(probe)
    # Train mean/std for c equal those of arange(10): mean 4.5, std ~2.87.
    expected_c = (1000.0 - 4.5) / np.std(np.arange(10), ddof=0)
    assert out[0, 0] == pytest.approx(np.float32(expected_c), rel=1e-4)


# ---------------------------------------------------------------------------
# Constant column (zero-std) handling
# ---------------------------------------------------------------------------


def test_constant_train_column_does_not_blow_up() -> None:
    """A train column with zero variance is clamped; transform returns 0."""
    df = pd.DataFrame({"constant": np.full(20, 3.14, dtype=np.float64)})
    scaler = FittedScaler().fit(df)
    out = scaler.transform(df)
    # (3.14 - 3.14) / epsilon = 0 exactly.
    np.testing.assert_allclose(out, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Serialization round-trip (via __getstate__/__setstate__)
# ---------------------------------------------------------------------------


def test_getstate_setstate_round_trip_preserves_stats() -> None:
    """Round-trip through ``__getstate__`` / ``__setstate__``.

    This is the contract any serializer (pickle, joblib, cloudpickle)
    relies on. The round trip below is performed explicitly via
    ``copy.deepcopy`` which exercises the state hooks without going
    through a byte-level serialization layer.
    """
    rng = np.random.default_rng(7)
    train = pd.DataFrame(
        {
            "vol": np.abs(rng.normal(3.0, 1.0, size=100)),
            "x": rng.normal(0.0, 1.0, size=100),
            "flag": (rng.random(100) > 0.5).astype(np.float64),
        }
    )
    scaler = FittedScaler(log1p_cols=["vol"], passthrough_cols=["flag"]).fit(train)

    revived = copy.deepcopy(scaler)

    assert isinstance(revived, FittedScaler)
    assert revived is not scaler
    assert revived.is_fitted is True
    assert revived.log1p_cols == ("vol",)
    assert revived.passthrough_cols == ("flag",)
    assert revived.feature_names_out_ == ["vol", "x", "flag"]
    np.testing.assert_array_equal(revived.mean_, scaler.mean_)
    np.testing.assert_array_equal(revived.scale_, scaler.scale_)

    # Transforms agree.
    val = pd.DataFrame(
        {
            "vol": [1.0, 5.0, 10.0],
            "x": [0.0, 1.0, -1.0],
            "flag": [0.0, 1.0, 0.0],
        }
    )
    np.testing.assert_array_equal(revived.transform(val), scaler.transform(val))


def test_manual_state_dict_round_trip() -> None:
    """Reconstructing from ``__getstate__`` mirrors the original scaler.

    This documents the serialization contract for any external tool
    (joblib, cloudpickle, hdf5, etc.) — dumping ``__getstate__`` and
    later restoring via ``__setstate__`` is sufficient.
    """
    train = pd.DataFrame({"vol": [0.0, 1.0, 2.0, 3.0], "flag": [0.0, 1.0, 0.0, 1.0]})
    scaler = FittedScaler(log1p_cols=["vol"], passthrough_cols=["flag"]).fit(train)

    state = scaler.__getstate__()
    # State must be a plain dict of JSON-compatible or ndarray entries
    # (no references to the original object).
    assert isinstance(state, dict)
    assert set(state.keys()) == {
        "log1p_cols",
        "passthrough_cols",
        "is_fitted",
        "feature_names_in_",
        "mean_",
        "scale_",
    }

    revived = FittedScaler.__new__(FittedScaler)
    revived.__setstate__(state)
    assert revived.is_fitted is True
    np.testing.assert_array_equal(revived.mean_, scaler.mean_)
    np.testing.assert_array_equal(revived.scale_, scaler.scale_)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_fits_on_same_data_match() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0.0, 1.0, 50)})
    a = FittedScaler().fit_transform(df)
    b = FittedScaler().fit_transform(df)
    np.testing.assert_array_equal(a, b)


def test_log1p_plus_passthrough_plus_z_score_combined() -> None:
    """A three-kind column DataFrame: log1p, passthrough, and plain z-score."""
    train = pd.DataFrame(
        {
            "vol": np.arange(20, dtype=np.float64),  # log1p target
            "x": np.linspace(-1.0, 1.0, 20),  # plain z-score
            "flag": np.array([0, 1] * 10, dtype=np.float64),  # passthrough
        }
    )
    scaler = FittedScaler(log1p_cols=["vol"], passthrough_cols=["flag"]).fit(train)
    out = scaler.transform(train)

    # vol column: log1p then z-score -> mean ~0, std ~1.
    assert out[:, 0].mean() == pytest.approx(0.0, abs=1e-5)
    assert out[:, 0].std(ddof=0) == pytest.approx(1.0, abs=1e-5)
    # x: z-score -> mean ~0, std ~1.
    assert out[:, 1].mean() == pytest.approx(0.0, abs=1e-5)
    assert out[:, 1].std(ddof=0) == pytest.approx(1.0, abs=1e-5)
    # flag: unchanged.
    np.testing.assert_allclose(
        out[:, 2], train["flag"].values.astype(np.float32)
    )
