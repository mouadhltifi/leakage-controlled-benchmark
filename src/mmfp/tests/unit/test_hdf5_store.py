"""Tests for :mod:`mmfp.data.hdf5_store`.

Covers the save/load round-trip contract: nothing is lost, nothing is
changed, and a killed writer never leaves a half-written file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmfp.data.hdf5_store import (
    SCHEMA_VERSION,
    load_fold_artifacts,
    save_fold_artifacts,
)


def _toy_artifacts() -> dict:
    """A mixed bag that exercises every supported entry type."""
    return {
        "features": np.random.default_rng(0).normal(size=(8, 3)).astype(np.float32),
        "labels_cls": np.array([0, 1, -1, 0, 1, 1, 0, -1], dtype=np.int64),
        "labels_reg": np.linspace(-0.01, 0.01, 8, dtype=np.float32),
        "static_adj": np.eye(4, dtype=np.float32),
        # scalar attributes
        "n_train": 100,
        "n_val": 25,
        "fold_id": "fold_2",
        "rmse": 0.1234,
        "used_pca": True,
        # dict-of-arrays (dynamic graphs style)
        "dynamic_snapshots": {
            "2020-01-15": np.ones((4, 4), dtype=np.float32),
            "2020-02-15": np.eye(4, dtype=np.float32),
        },
        # list / tuple
        "ticker_order": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    }


def test_save_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "fold.h5"
    artifacts = _toy_artifacts()
    save_fold_artifacts(path, artifacts)

    loaded = load_fold_artifacts(path)

    # Schema version surfaced under the reserved key.
    assert loaded["schema_version"] == SCHEMA_VERSION

    # Arrays match exactly.
    np.testing.assert_array_equal(loaded["features"], artifacts["features"])
    np.testing.assert_array_equal(loaded["labels_cls"], artifacts["labels_cls"])
    np.testing.assert_array_equal(loaded["labels_reg"], artifacts["labels_reg"])
    np.testing.assert_array_equal(loaded["static_adj"], artifacts["static_adj"])

    # Scalars decoded.
    assert loaded["n_train"] == 100
    assert loaded["n_val"] == 25
    assert loaded["fold_id"] == "fold_2"
    assert loaded["rmse"] == pytest.approx(0.1234)
    assert bool(loaded["used_pca"]) is True

    # Dict-of-arrays preserved.
    assert isinstance(loaded["dynamic_snapshots"], dict)
    assert set(loaded["dynamic_snapshots"].keys()) == {
        "2020-01-15",
        "2020-02-15",
    }
    np.testing.assert_array_equal(
        loaded["dynamic_snapshots"]["2020-01-15"],
        artifacts["dynamic_snapshots"]["2020-01-15"],
    )
    np.testing.assert_array_equal(
        loaded["dynamic_snapshots"]["2020-02-15"],
        artifacts["dynamic_snapshots"]["2020-02-15"],
    )

    # List becomes array of strings.
    tickers = loaded["ticker_order"]
    assert list(tickers.astype(str)) == artifacts["ticker_order"]


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "sub" / "fold.h5"
    save_fold_artifacts(path, {"x": np.zeros(3)})
    assert path.exists()


def test_save_is_atomic_no_tmp_leftover(tmp_path: Path) -> None:
    path = tmp_path / "fold.h5"
    save_fold_artifacts(path, {"x": np.zeros(3)})
    assert not (tmp_path / "fold.h5.tmp").exists()


def test_save_overwrites_existing(tmp_path: Path) -> None:
    path = tmp_path / "fold.h5"
    save_fold_artifacts(path, {"x": np.zeros(3)})
    save_fold_artifacts(path, {"x": np.ones(5)})
    loaded = load_fold_artifacts(path)
    np.testing.assert_array_equal(loaded["x"], np.ones(5))


def test_schema_version_written(tmp_path: Path) -> None:
    path = tmp_path / "fold.h5"
    save_fold_artifacts(path, {"x": np.zeros(3)})
    loaded = load_fold_artifacts(path)
    assert loaded["schema_version"] == SCHEMA_VERSION


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_fold_artifacts(tmp_path / "nowhere.h5")


def test_load_without_schema_version_raises(tmp_path: Path) -> None:
    """Files not written by ``save_fold_artifacts`` should be rejected."""
    import h5py

    path = tmp_path / "foreign.h5"
    with h5py.File(path, "w") as h:
        h.create_dataset("x", data=np.zeros(3))

    with pytest.raises(ValueError, match="schema_version"):
        load_fold_artifacts(path)


def test_save_rejects_unsupported_type(tmp_path: Path) -> None:
    class Weird:
        pass

    with pytest.raises(TypeError, match="Unsupported artifact type"):
        save_fold_artifacts(tmp_path / "f.h5", {"obj": Weird()})


def test_save_rejects_non_ndarray_dict_value(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="non-ndarray"):
        save_fold_artifacts(
            tmp_path / "f.h5",
            {"graphs": {"2020-01-01": [1, 2, 3]}},  # list, not ndarray
        )


def test_empty_dict_saves_and_loads(tmp_path: Path) -> None:
    """A minimal artifact set still writes and reads schema_version."""
    path = tmp_path / "f.h5"
    save_fold_artifacts(path, {})
    loaded = load_fold_artifacts(path)
    assert loaded == {"schema_version": SCHEMA_VERSION}


def test_schema_version_constant_is_documented() -> None:
    """The schema version string must match the documented convention
    ``mmfp-v<major>.<minor>`` so downstream consumers can detect breakage.
    """
    assert SCHEMA_VERSION.startswith("mmfp-v")


def test_save_cleans_tmp_on_failure(tmp_path: Path, monkeypatch) -> None:
    """If ``save_fold_artifacts`` raises mid-write, no ``.tmp`` remains."""
    path = tmp_path / "fold.h5"

    # Mixin an unsupported value after a supported one so the write
    # starts and then fails partway through.
    bad_artifacts = {
        "ok": np.zeros(3),
        "bad": object(),
    }
    with pytest.raises(TypeError):
        save_fold_artifacts(path, bad_artifacts)

    # Target file was never created, and no tmp leftover.
    assert not path.exists()
    assert not (tmp_path / "fold.h5.tmp").exists()
