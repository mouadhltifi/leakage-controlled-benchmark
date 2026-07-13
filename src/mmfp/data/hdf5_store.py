"""Thin HDF5 read/write wrapper for per-fold artifacts.

Milestone 2 provides a minimal surface: atomic save, simple load, and a
schema-version root attribute. The :class:`~mmfp.data.assemble.FoldArtifacts`
dataclass and ``assemble_fold`` entry point land in Milestone 4.

Conventions
-----------

* One HDF5 file per ``(config_hash, fold_idx)``.
* Every root attribute is a simple scalar or short string — richer
  metadata goes in a dedicated ``"meta"`` group.
* Every dataset is numpy-typed. Dict-of-arrays mappings (e.g. per-date
  dynamic graphs) are stored as a sub-group, one dataset per key.

Atomic writes: we write to ``path.tmp`` then ``os.replace`` into place
so a killed process leaves either the old file or the new file, never a
half-written file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

log = logging.getLogger(__name__)

#: Schema version written as the ``schema_version`` root attribute. Bump on
#: breaking changes to the artifact layout. Readers compare against this.
SCHEMA_VERSION: str = "mmfp-v2.0"

#: Key under which nested dict-of-array mappings are stored.
_DICT_GROUP_MARKER: str = "__dict__"


def save_fold_artifacts(
    path: str | Path,
    artifacts: Mapping[str, Any],
    *,
    compression: str | None = "gzip",
) -> None:
    """Atomically write ``artifacts`` to an HDF5 file.

    Parameters
    ----------
    path
        Target ``.h5`` path. Parent directory is created if missing.
    artifacts
        Mapping of key -> value. Supported value types:

        * ``numpy.ndarray`` — written as an HDF5 dataset.
        * ``bytes`` / ``str`` / ``int`` / ``float`` / ``bool`` — written
          as a scalar attribute under the root.
        * ``dict[str, numpy.ndarray]`` — written as a sub-group, one
          dataset per inner key. Useful for dynamic graph snapshots.
        * ``list`` / ``tuple`` of numeric scalars or strings — written
          as a 1-D array (string lists become ``S``-typed arrays).

        Any other type raises ``TypeError``.

    compression
        Compression filter for array datasets. ``None`` disables it.
        ``"gzip"`` (default) matches the v1 convention.

    Raises
    ------
    TypeError
        If an artifact value is of an unsupported type.
    OSError
        If the write fails (no space, permissions, etc.).

    Notes
    -----
    The root attribute ``schema_version`` is always written and equals
    :data:`SCHEMA_VERSION`. Use :func:`load_fold_artifacts` to read it
    back and verify.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    # If a prior run crashed mid-write, clean up the leftover.
    if tmp.exists():
        tmp.unlink()

    try:
        with h5py.File(tmp, "w") as h:
            h.attrs["schema_version"] = SCHEMA_VERSION
            for key, value in artifacts.items():
                _write_entry(h, key, value, compression=compression)
    except BaseException:
        # Clean up the temp file on any failure (including KeyboardInterrupt)
        # so we don't leave a half-written file behind.
        if tmp.exists():
            tmp.unlink()
        raise

    os.replace(tmp, path)
    log.debug("save_fold_artifacts: wrote %s (%d top-level entries)", path, len(artifacts))


def load_fold_artifacts(path: str | Path) -> dict[str, Any]:
    """Read an HDF5 artifact file back into a dict.

    Parameters
    ----------
    path
        Path to an ``.h5`` file previously written by
        :func:`save_fold_artifacts`.

    Returns
    -------
    dict[str, Any]
        Keys and values reconstructed. Datasets become ``numpy.ndarray``;
        groups whose members are datasets become ``dict[str, ndarray]``;
        root attributes (other than ``schema_version``) become scalars.
        The ``schema_version`` attribute is surfaced under the reserved
        key ``"schema_version"``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file lacks the ``schema_version`` root attribute (not a
        valid artifact file).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 artifact not found at {path}")

    out: dict[str, Any] = {}
    with h5py.File(path, "r") as h:
        if "schema_version" not in h.attrs:
            raise ValueError(
                f"{path} is missing the ``schema_version`` root attribute; "
                "it is not a valid mmfp artifact file."
            )
        out["schema_version"] = _decode_scalar(h.attrs["schema_version"])

        for attr_key, attr_val in h.attrs.items():
            if attr_key == "schema_version":
                continue
            out[attr_key] = _decode_scalar(attr_val)

        for key, item in h.items():
            if isinstance(item, h5py.Dataset):
                out[key] = np.asarray(item)
            elif isinstance(item, h5py.Group):
                out[key] = {
                    sub_key: np.asarray(sub_item)
                    for sub_key, sub_item in item.items()
                    if isinstance(sub_item, h5py.Dataset)
                }
            else:
                log.warning("load_fold_artifacts: skipping unknown HDF5 entry %r", key)

    return out


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


def _write_entry(
    h: h5py.File, key: str, value: Any, *, compression: str | None
) -> None:
    """Dispatch an artifact value to the appropriate HDF5 writer."""
    if isinstance(value, np.ndarray):
        _create_array_dataset(h, key, value, compression=compression)
    elif isinstance(value, (bytes, str, int, float, bool, np.integer, np.floating, np.bool_)):
        h.attrs[key] = value
    elif isinstance(value, dict):
        grp = h.create_group(key)
        for sub_key, sub_val in value.items():
            if not isinstance(sub_val, np.ndarray):
                raise TypeError(
                    f"dict artifact {key!r} contains a non-ndarray value "
                    f"for inner key {sub_key!r} (got {type(sub_val).__name__})"
                )
            _create_array_dataset(grp, sub_key, sub_val, compression=compression)
    elif isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value)
        except Exception as exc:
            raise TypeError(
                f"Could not convert list/tuple artifact {key!r} to array: {exc}"
            ) from exc
        _create_array_dataset(h, key, arr, compression=compression)
    else:
        raise TypeError(
            f"Unsupported artifact type for key {key!r}: {type(value).__name__}"
        )


def _create_array_dataset(
    parent: h5py.Group, key: str, arr: np.ndarray, *, compression: str | None
) -> None:
    """Create an HDF5 dataset handling numpy string dtypes safely.

    h5py does not directly accept ``<U*`` (unicode) numpy arrays; we
    encode them as fixed-length byte strings. On read, consumers apply
    ``.astype(str)`` to recover.
    """
    if arr.dtype.kind == "U":
        arr = np.char.encode(arr, encoding="utf-8")
    parent.create_dataset(key, data=arr, compression=compression)


def _decode_scalar(val: Any) -> Any:
    """Normalise an HDF5 attribute scalar to a native Python type."""
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, np.generic):
        return val.item()
    return val


__all__ = [
    "SCHEMA_VERSION",
    "load_fold_artifacts",
    "save_fold_artifacts",
]
