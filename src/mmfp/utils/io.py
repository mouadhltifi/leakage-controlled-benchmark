"""Atomic IO helpers: CSV append, JSON save, git-sha probe.

CSV append is atomic to keep sweep results correct under ``Ctrl-C`` and
parallel writers: the whole file is rewritten to a ``.tmp`` sibling then
:func:`os.replace`\\-d into place.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)


def atomic_csv_append(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append a row to a CSV file atomically.

    If the file does not exist, a header is written using the row's keys.
    Otherwise, the existing file is read, the new row is appended, and the
    result is written to ``path.tmp`` then atomically renamed.

    This is conservative but correct: it avoids corruption if the process
    is killed mid-append. For small CSVs (thousands of rows) the overhead
    is negligible; our sweep CSVs are well under that.

    Parameters
    ----------
    path
        Target CSV path. Parent directory is created if missing.
    row
        Mapping of column name to value. Keys define the header on first
        write; subsequent rows must have a compatible key set.

    Raises
    ------
    ValueError
        If the existing CSV has a different header than the new row's
        keys.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    existing_rows: list[dict[str, Any]] = []
    existing_fieldnames: list[str] | None = None
    if path.exists():
        with path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    new_fieldnames = list(row.keys())
    fieldnames = existing_fieldnames or new_fieldnames

    if existing_fieldnames is not None and set(new_fieldnames) != set(
        existing_fieldnames
    ):
        raise ValueError(
            f"atomic_csv_append: new row has keys {sorted(new_fieldnames)}; "
            f"existing CSV {path} has header {sorted(existing_fieldnames)}. "
            "Schemas must agree."
        )

    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow(existing)
        writer.writerow({k: row.get(k, "") for k in fieldnames})

    os.replace(tmp, path)


def save_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    """Dump an object to JSON atomically.

    Parameters
    ----------
    path
        Target JSON path. Parent directory is created if missing.
    obj
        Any JSON-serialisable object.
    indent
        ``json.dump`` indent. Defaults to 2 for human readability.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=indent, sort_keys=True, default=str)
    os.replace(tmp, path)


def get_git_sha(default: str = "unknown") -> str:
    """Return the current git HEAD short sha, or ``default`` if unavailable.

    Parameters
    ----------
    default
        Value returned if not inside a git repo or ``git`` is missing.

    Returns
    -------
    str
        12-char short sha, or ``default``.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or default
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        log.debug("get_git_sha failed: %s", exc)
        return default


__all__ = ["atomic_csv_append", "get_git_sha", "save_json"]
