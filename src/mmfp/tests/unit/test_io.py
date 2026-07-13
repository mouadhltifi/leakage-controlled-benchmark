"""Tests for :mod:`mmfp.utils.io` atomic helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mmfp.utils.io import atomic_csv_append, get_git_sha, save_json


def test_atomic_csv_append_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "results.csv"
    atomic_csv_append(out, {"a": 1, "b": "hello"})
    assert out.exists()

    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows == [{"a": "1", "b": "hello"}]


def test_atomic_csv_append_appends_subsequent_rows(tmp_path: Path) -> None:
    out = tmp_path / "r.csv"
    atomic_csv_append(out, {"a": 1, "b": 2})
    atomic_csv_append(out, {"a": 3, "b": 4})
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows == [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    ]


def test_atomic_csv_append_schema_mismatch_raises(tmp_path: Path) -> None:
    out = tmp_path / "r.csv"
    atomic_csv_append(out, {"a": 1, "b": 2})
    with pytest.raises(ValueError, match="Schemas must agree"):
        atomic_csv_append(out, {"a": 3, "c": 9})


def test_atomic_csv_append_no_tmp_leftover(tmp_path: Path) -> None:
    out = tmp_path / "r.csv"
    atomic_csv_append(out, {"a": 1})
    assert not (tmp_path / "r.csv.tmp").exists()


def test_save_json(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "obj.json"
    save_json(out, {"k": [1, 2, 3], "x": "y"})
    with out.open() as fh:
        data = json.load(fh)
    assert data == {"k": [1, 2, 3], "x": "y"}


def test_get_git_sha_returns_string() -> None:
    """Running inside the Study repo should return a short sha.

    Test does not check specific value (will change over time) but only
    that we get a non-empty string.
    """
    sha = get_git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0


def test_get_git_sha_default_when_no_git(tmp_path: Path, monkeypatch) -> None:
    """Without ``git`` on PATH, we fall back to the default value."""
    monkeypatch.setenv("PATH", "/nonexistent")
    sha = get_git_sha(default="fallback")
    assert sha == "fallback"
