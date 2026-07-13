"""v3 sweep wrapper: ``run_forecast_sweep`` over ``run_one_forecast_experiment``.

This module is forked from :mod:`mmfp.experiments.sweep`. The fork is
deliberate and minimal:

* **Why fork.** v2's :func:`mmfp.experiments.sweep.run_sweep` hard-codes
  ``from mmfp.experiments.runner import run_one_experiment`` in the
  worker entry point. Injecting a runner callable would require either
  a kwarg on the public v2 API (spreading v3 concerns into v2) or a
  module-level rebind (fragile under ``spawn`` workers, which re-import
  ``mmfp`` cleanly and would miss the monkeypatch). Forking isolates
  the change to ``forecast/`` and keeps ``mmfp/`` untouched per the
  architecture-spec reuse map (§7).
* **What is copied.** Fingerprint-based resume, dotted-key override
  application, deep-merge of the base TOML onto defaults, serial /
  ``multiprocessing.get_context("spawn")`` pool execution, device
  strategy dispatch, atomic CSV append, dataframe return.
* **What is changed.** Base config is v3's
  :class:`~forecast.config.schema.V3ExperimentConfig` validated through
  :func:`~forecast.config.validate.validate_forecast_config`. Defaults
  come from :data:`forecast.config.defaults.DEFAULT_CONFIG`. Worker
  calls :func:`~forecast.experiments.runner.run_one_forecast_experiment`.
  Record type is :class:`ForecastResultRecord`.

Mechanically the two modules are line-for-line equivalent once the
package paths are swapped; readers who already understand v2's sweep
will recognise every block here.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from forecast.config.defaults import DEFAULT_CONFIG
from forecast.config.schema import V3ExperimentConfig
from forecast.config.validate import validate_forecast_config
from forecast.experiments.result_schema import (
    ForecastResultRecord,
    append_result,
    config_hash,
    fingerprint_from_row,
    load_results,
    record_fingerprint,
)
# [M5 forked from mmfp.experiments.sweep.run_sweep; diff: runner fn + config
# class + defaults + validator only.]
from forecast.experiments.runner import run_one_forecast_experiment
from mmfp.config.load import load_toml
from mmfp.experiments.sweep import apply_override

log = logging.getLogger(__name__)


#: Device strategy literal used by :func:`run_forecast_sweep`. Mirrors
#: v2's :data:`mmfp.experiments.sweep.DeviceStrategy`.
DeviceStrategy = Literal["cpu", "mps", "auto"]


# ---------------------------------------------------------------------------
# Worker-side machinery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkerTask:
    """Serialisable unit of work dispatched to a spawn worker.

    Holds only picklable primitives. The validated
    :class:`V3ExperimentConfig` is materialised inside the worker so
    pydantic validation runs on both sides of the process boundary,
    surfacing schema drift in worker logs rather than at dispatch time.
    """

    base_dict: dict[str, Any]
    override: dict[str, Any]
    device_strategy: str
    sweep_index: int
    sweep_total: int


@dataclass(frozen=True)
class _WorkerSuccess:
    """Successful worker return. Holds only serialisable primitives."""

    sweep_index: int
    record: ForecastResultRecord


@dataclass(frozen=True)
class _WorkerFailure:
    """Worker run failed. The parent logs and continues."""

    sweep_index: int
    experiment_name: str
    error_repr: str


def _run_worker_task(task: _WorkerTask) -> _WorkerSuccess | _WorkerFailure:
    """Module-level worker entry point for :class:`multiprocessing.pool.Pool`.

    Must be importable by fully-qualified name (``forecast.experiments.
    sweep._run_worker_task``) because ``spawn`` pools pass functions by
    name, not by bound closure. Takes a serialisable :class:`_WorkerTask`
    and returns a serialisable result.
    """
    try:
        cfg = _materialise_cfg(task.base_dict, task.override)
        cfg = _apply_device_strategy(cfg, task.device_strategy)
        log.info(
            "forecast sweep worker [%d/%d]: running %s (seed=%d fold=%d device=%s)",
            task.sweep_index + 1,
            task.sweep_total,
            cfg.name,
            cfg.seed,
            cfg.data.fold_idx,
            cfg.device,
        )
        record = run_one_forecast_experiment(cfg)
        return _WorkerSuccess(sweep_index=task.sweep_index, record=record)
    except Exception as exc:  # noqa: BLE001 -- uniform failure reporting
        experiment_name = "<unknown>"
        try:
            experiment_name = str(
                task.override.get("name", task.base_dict.get("name", "<unknown>"))
            )
        except Exception:  # pragma: no cover - defensive
            pass
        return _WorkerFailure(
            sweep_index=task.sweep_index,
            experiment_name=experiment_name,
            error_repr=f"{type(exc).__name__}: {exc}",
        )


def _apply_device_strategy(
    cfg: V3ExperimentConfig, device_strategy: str,
) -> V3ExperimentConfig:
    """Return a copy of ``cfg`` whose ``device`` matches ``device_strategy``.

    ``"auto"`` is passthrough: the base config's existing ``device`` value
    (possibly ``"auto"`` itself) is honoured. Any other value is written
    to the config and re-validated so schema changes are caught here.
    """
    if device_strategy == "auto":
        return cfg
    if device_strategy == cfg.device:
        return cfg
    merged = cfg.model_dump(mode="python")
    merged["device"] = device_strategy
    return validate_forecast_config(V3ExperimentConfig.model_validate(merged))


# ---------------------------------------------------------------------------
# Public sweep entry point
# ---------------------------------------------------------------------------


def run_forecast_sweep(
    base_cfg_path: str | Path,
    overrides: list[dict[str, Any]],
    output_csv: str | Path,
    *,
    resume: bool = True,
    parallelism: int = 1,
    device_strategy: DeviceStrategy = "cpu",
) -> pd.DataFrame:
    """Run every override (optionally in parallel) and append to one CSV.

    Parameters
    ----------
    base_cfg_path
        Path to a TOML config loaded by
        :func:`forecast.config.load.load_config`. An empty / non-existent
        path falls back to :data:`DEFAULT_CONFIG`.
    overrides
        List of dotted-key override dicts applied one by one to the base
        config. An empty dict means "use the base config as-is"; an empty
        list short-circuits and returns whatever is already on disk.
    output_csv
        Destination CSV. Created atomically if it doesn't exist. Each
        row is appended via :func:`append_result`.
    resume
        When ``True`` (default), any row whose fingerprint already appears
        in ``output_csv`` is skipped. When ``False``, every override runs
        even if a matching row exists on disk.
    parallelism
        Number of worker processes. ``1`` (default) runs serially with no
        pool overhead. Values greater than ``1`` dispatch to a
        ``spawn``-context :class:`multiprocessing.pool.Pool`.
    device_strategy
        How workers pick their compute device (``cpu`` / ``mps`` /
        ``auto``). Only ``cpu`` is guaranteed bit-identical to a serial
        run; MPS tolerance follows v2's reproducibility docs.

    Returns
    -------
    pandas.DataFrame
        The full set of rows in ``output_csv`` after the sweep. Empty
        when no rows were ever written.

    Raises
    ------
    ValueError
        If ``parallelism < 1`` or ``device_strategy`` is not one of
        ``"cpu" / "mps" / "auto"``.
    """
    if parallelism < 1:
        raise ValueError(
            f"run_forecast_sweep: parallelism must be >= 1 (got {parallelism})."
        )
    if device_strategy not in ("cpu", "mps", "auto"):
        raise ValueError(
            "run_forecast_sweep: device_strategy must be one of "
            f"('cpu', 'mps', 'auto'); got {device_strategy!r}."
        )

    base_cfg_path = Path(base_cfg_path)
    output_csv = Path(output_csv)

    # Load the base config as a dict. v3 defaults live in
    # ``forecast.config.defaults.DEFAULT_CONFIG``.
    base_dict = copy.deepcopy(DEFAULT_CONFIG)
    if str(base_cfg_path):
        toml_data = load_toml(base_cfg_path)
        base_dict = _deep_merge(base_dict, toml_data)

    # Resume: collect fingerprints already on disk.
    existing_rows = load_results(output_csv) if resume else []
    existing_fingerprints: set[str] = set()
    for row in existing_rows:
        try:
            existing_fingerprints.add(fingerprint_from_row(row))
        except ValueError as exc:
            log.warning(
                "run_forecast_sweep: skipping unrecognised row in %s: %s",
                output_csv, exc,
            )

    if existing_fingerprints:
        log.info(
            "run_forecast_sweep: resume enabled; %d existing rows on disk.",
            len(existing_fingerprints),
        )

    # Pre-flight: compute prospective fingerprint for each override so
    # workers never start runs that would be skipped anyway.
    pending: list[_WorkerTask] = []
    skipped = 0
    total = len(overrides)

    for i, override in enumerate(overrides):
        prospective_cfg = _materialise_cfg(base_dict, override)
        # Device strategy participates in config_hash via cfg.device, so
        # the prospective fingerprint must be computed after it is
        # applied. Matches v2's contract.
        prospective_cfg = _apply_device_strategy(prospective_cfg, device_strategy)
        fp = _prospective_fingerprint(prospective_cfg)
        if fp in existing_fingerprints:
            skipped += 1
            log.info(
                "run_forecast_sweep: [%d/%d] SKIP already-recorded %s",
                i + 1, total, fp,
            )
            continue
        pending.append(
            _WorkerTask(
                base_dict=base_dict,
                override=override,
                device_strategy=device_strategy,
                sweep_index=i,
                sweep_total=total,
            )
        )

    if parallelism == 1:
        completed, failed = _run_serial(
            pending=pending,
            output_csv=output_csv,
            existing_fingerprints=existing_fingerprints,
        )
    else:
        completed, failed = _run_parallel(
            pending=pending,
            output_csv=output_csv,
            existing_fingerprints=existing_fingerprints,
            parallelism=parallelism,
        )

    log.info(
        "run_forecast_sweep: done; %d completed, %d skipped, %d failed out of %d overrides.",
        completed, skipped, failed, total,
    )

    if output_csv.exists():
        return pd.read_csv(output_csv)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Execution strategies
# ---------------------------------------------------------------------------


def _run_serial(
    *,
    pending: list[_WorkerTask],
    output_csv: Path,
    existing_fingerprints: set[str],
) -> tuple[int, int]:
    """Run pending tasks one after another in the current process."""
    completed = 0
    failed = 0
    for task in pending:
        result = _run_worker_task(task)
        if isinstance(result, _WorkerFailure):
            log.error(
                "run_forecast_sweep: [%d/%d] FAILED (%s): %s",
                task.sweep_index + 1,
                task.sweep_total,
                result.experiment_name,
                result.error_repr,
            )
            failed += 1
            continue
        append_result(output_csv, result.record)
        existing_fingerprints.add(record_fingerprint(result.record))
        completed += 1
    return completed, failed


def _run_parallel(
    *,
    pending: list[_WorkerTask],
    output_csv: Path,
    existing_fingerprints: set[str],
    parallelism: int,
) -> tuple[int, int]:
    """Dispatch pending tasks to a ``spawn``-context pool."""
    if not pending:
        return 0, 0

    ctx = mp.get_context("spawn")
    completed = 0
    failed = 0
    with ctx.Pool(processes=parallelism) as pool:
        for result in pool.imap_unordered(_run_worker_task, pending, chunksize=1):
            if isinstance(result, _WorkerFailure):
                log.error(
                    "run_forecast_sweep (parallel): sweep_index=%d FAILED (%s): %s",
                    result.sweep_index,
                    result.experiment_name,
                    result.error_repr,
                )
                failed += 1
                continue
            append_result(output_csv, result.record)
            existing_fingerprints.add(record_fingerprint(result.record))
            completed += 1
    return completed, failed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (parity with v2)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _materialise_cfg(
    base_dict: dict[str, Any], override: dict[str, Any],
) -> V3ExperimentConfig:
    """Apply ``override`` to ``base_dict`` and return a validated v3 config."""
    merged = apply_override(base_dict, override)
    cfg = V3ExperimentConfig.model_validate(merged)
    return validate_forecast_config(cfg)


def _prospective_fingerprint(cfg: V3ExperimentConfig) -> str:
    """Mirror :func:`record_fingerprint` for a cfg before it runs."""
    return "|".join(
        (
            str(cfg.name),
            str(config_hash(cfg)),
            str(int(cfg.seed)),
            str(int(cfg.data.fold_idx)),
        )
    )


__all__ = [
    "DeviceStrategy",
    "run_forecast_sweep",
]
