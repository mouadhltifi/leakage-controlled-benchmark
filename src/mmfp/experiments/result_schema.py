"""Typed :class:`ResultRecord` + atomic CSV helpers for experiment outputs.

One row per experiment. Every column is typed; metrics that do not
apply for a given target set (e.g. :attr:`sharpe_ratio` when ``direction``
is not active) are ``None``.

Spec reference: the design spec

The :func:`config_hash` helper gives us a stable fingerprint for a
validated :class:`~mmfp.config.schema.ExperimentConfig`. It is computed
from the canonical JSON dump (pydantic ``mode='json'`` with
``sort_keys=True``) so logically equivalent configs produce the same
hash regardless of dict insertion order.

The :func:`record_fingerprint` helper combines the four identifying
fields ``(experiment_name, config_hash, seed, fold_idx)`` into a single
string that :func:`~mmfp.experiments.sweep.run_sweep` uses to skip
already-recorded rows on resume.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from mmfp.config.schema import ExperimentConfig
from mmfp.utils.io import atomic_csv_append

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ResultRecord dataclass
# ---------------------------------------------------------------------------


@dataclass
class ResultRecord:
    """Per-experiment result row.

    All metric fields that do not apply for a given ``cfg.head.targets``
    must be set to ``None`` so downstream analysis can distinguish
    "unmeasured" from "measured as zero".

    See spec Section 3.12 for field-level documentation.
    """

    # ---- identification --------------------------------------------
    experiment_name: str
    config_hash: str
    seed: int
    fold_idx: int

    # ---- axis fingerprint ------------------------------------------
    news_encoder: str
    news_aggregation: str
    fusion_strategy: str
    head_architecture: str
    targets: str  # comma-separated, e.g. "direction,return"
    graph_source: str
    lookback: int
    price_encoder: str

    # ---- direction metrics -----------------------------------------
    mcc: float | None
    accuracy: float | None
    f1: float | None

    # ---- regression metrics ----------------------------------------
    r2: float | None
    rmse: float | None
    sharpe_ratio: float | None

    # ---- volatility metrics ----------------------------------------
    vol_rmse: float | None
    vol_r2: float | None

    # ---- meta ------------------------------------------------------
    n_train: int
    n_val: int
    n_test: int
    n_params: int
    epochs_trained: int
    best_val_metric: float
    elapsed_seconds: float
    platform_version: str
    git_sha: str

    # ----------------------------------------------------------------
    # Serialisation helpers
    # ----------------------------------------------------------------

    def to_row(self) -> dict[str, Any]:
        """Return a dict suitable for :func:`append_result` / ``csv.DictWriter``.

        ``None`` values are emitted as the empty string so the CSV round-trips
        cleanly through :func:`load_results`.
        """
        row: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            row[f.name] = "" if value is None else value
        return row

    @classmethod
    def column_names(cls) -> list[str]:
        """Ordered list of dataclass field names (= CSV header)."""
        return [f.name for f in fields(cls)]


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------


def config_hash(cfg: ExperimentConfig) -> str:
    """Return a sha256 digest of the config's canonical JSON dump.

    Hash is computed over :meth:`ExperimentConfig.model_dump_json` with
    stable key ordering so logically equivalent configs produce the same
    hash regardless of field insertion order.

    Parameters
    ----------
    cfg
        Validated experiment config.

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    payload = cfg.model_dump(mode="json")
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def record_fingerprint(rec: ResultRecord) -> str:
    """Deterministic dedup key for an experiment run.

    Combines ``(experiment_name, config_hash, seed, fold_idx)`` into a
    single string. The choice of delimiter (``|``) avoids collisions
    with any field value: none of the four components admit a literal
    pipe.

    Parameters
    ----------
    rec
        A :class:`ResultRecord`.

    Returns
    -------
    str
        Fingerprint string used by
        :func:`~mmfp.experiments.sweep.run_sweep` to skip already-run
        experiments.
    """
    return "|".join(
        (
            str(rec.experiment_name),
            str(rec.config_hash),
            str(int(rec.seed)),
            str(int(rec.fold_idx)),
        )
    )


def fingerprint_from_row(row: dict[str, Any]) -> str:
    """Fingerprint helper for dicts loaded from CSV.

    Mirrors :func:`record_fingerprint` but accepts a plain ``dict``
    (e.g. the output of :func:`load_results`). Keeps the sweep's resume
    logic symmetric between already-loaded rows and freshly-computed
    :class:`ResultRecord` instances.
    """
    try:
        return "|".join(
            (
                str(row["experiment_name"]),
                str(row["config_hash"]),
                str(int(row["seed"])),
                str(int(row["fold_idx"])),
            )
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(
            "fingerprint_from_row: row is missing one of "
            "('experiment_name', 'config_hash', 'seed', 'fold_idx') "
            f"or has non-integer seed/fold_idx. Got {sorted(row.keys())}."
        ) from exc


# ---------------------------------------------------------------------------
# CSV IO
# ---------------------------------------------------------------------------


def append_result(path: str | Path, record: ResultRecord) -> None:
    """Atomically append a :class:`ResultRecord` to ``path``.

    Parameters
    ----------
    path
        Target CSV. Parent directory is created if missing.
    record
        The row to write.

    Notes
    -----
    Delegates to :func:`mmfp.utils.io.atomic_csv_append` for the atomic
    write; the header is auto-generated from the dataclass field order
    on first write (and enforced against existing files afterwards).
    """
    atomic_csv_append(path, record.to_row())


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load an existing results CSV into a list of dicts.

    Returns an empty list when ``path`` does not exist — useful in the
    resume path of :func:`~mmfp.experiments.sweep.run_sweep` where the
    first run of a sweep has no prior data.

    Parameters
    ----------
    path
        CSV path previously written by :func:`append_result`.

    Returns
    -------
    list[dict[str, Any]]
        One dict per row, keys preserved as strings. Callers can
        post-process numeric fields themselves — we intentionally do not
        coerce types here because the record schema has mixed
        numeric/string columns and downstream uses differ.
    """
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


__all__ = [
    "ResultRecord",
    "append_result",
    "config_hash",
    "fingerprint_from_row",
    "load_results",
    "record_fingerprint",
]
