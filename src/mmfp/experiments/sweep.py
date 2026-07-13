"""Thin orchestration layer over :func:`run_one_experiment`.

A "sweep" is a list of config overrides applied to a shared base
config, each producing one :class:`~mmfp.experiments.result_schema.ResultRecord`
appended atomically to an output CSV. Resume support is fingerprint-based
so re-running the same sweep skips rows already on disk.

Parallelism (Milestone 9)
-------------------------

The original spec Q14 chose serial-with-resume over multiprocessing on
determinism grounds. Milestone 9 revises that recommendation: on CPU,
:func:`run_one_experiment` is bit-identically reproducible (see the
reproducibility test suite), so workers that each run one experiment
cannot diverge from a serial reference. MPS is left as a documented
tolerance path — its GPU queue serialises cross-process anyway, so
there is little to gain from running multiple MPS workers.

Process model
~~~~~~~~~~~~~

* ``parallelism == 1``: serial execution, identical to the pre-M9
  behaviour.
* ``parallelism > 1``: ``multiprocessing.get_context("spawn").Pool`` —
  each worker is a fresh Python interpreter that imports ``mmfp``
  independently. No shared mutable state. Results are collected via
  ``imap_unordered`` and appended to the CSV as they arrive.

``fork`` is deliberately not used: PyTorch's caching allocator retains
state across ``fork`` that can cause non-deterministic hangs on macOS.

Device strategy
~~~~~~~~~~~~~~~

Workers may override the base config's ``device`` field to control
where the compute actually runs. The default is ``"cpu"`` because that
is the only path validated as bit-identical against a serial run; MPS
and ``"auto"`` are opt-in and carry the usual MPS tolerance caveats
from the reproducibility docs.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.load import load_toml
from mmfp.config.schema import ExperimentConfig
from mmfp.config.validate import validate_experiment_config
from mmfp.experiments.result_schema import (
    ResultRecord,
    append_result,
    config_hash,
    fingerprint_from_row,
    load_results,
    record_fingerprint,
)
from mmfp.experiments.runner import run_one_experiment

log = logging.getLogger(__name__)


#: Device-strategy literal used by :func:`run_sweep` and the CLI. Each
#: worker applies the strategy as the value of ``cfg.device`` before
#: calling :func:`~mmfp.utils.device.resolve_device`. ``"auto"`` honours
#: whatever is already in the base config (``"auto"`` itself bubbles up
#: to the MPS > CUDA > CPU priority in ``resolve_device``).
DeviceStrategy = Literal["cpu", "mps", "auto"]


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------


def apply_override(cfg_dict: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Apply a dotted-key override dict to a deep-copied base config.

    Parameters
    ----------
    cfg_dict
        Base config as a nested dict (e.g. a TOML parse). Not mutated.
    override
        Mapping from dotted keys (``"training.batch_size"``) to values.
        Keys with dots are interpreted as nested paths; plain keys are
        applied at the top level.

    Returns
    -------
    dict[str, Any]
        Deep-copied ``cfg_dict`` with ``override`` applied.

    Raises
    ------
    ValueError
        If a dotted-path segment would overwrite a non-dict node with a
        dict (indicates a typo or a schema change).
    """
    out = copy.deepcopy(cfg_dict)
    for key, value in override.items():
        parts = str(key).split(".")
        if any(not p for p in parts):
            raise ValueError(f"apply_override: malformed key {key!r}")
        cursor = out
        for segment in parts[:-1]:
            if segment in cursor and not isinstance(cursor[segment], dict):
                raise ValueError(
                    f"apply_override: cannot descend into key {segment!r} "
                    f"(existing value is {type(cursor[segment]).__name__})"
                )
            if segment not in cursor:
                cursor[segment] = {}
            cursor = cursor[segment]
        cursor[parts[-1]] = copy.deepcopy(value)
    return out


# ---------------------------------------------------------------------------
# Worker-side machinery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkerTask:
    """Serialisable unit of work dispatched to a spawn worker.

    Must hold only picklable primitives — the base dict is a plain nested
    ``dict``, the override dict is a user-supplied mapping of dotted keys
    to JSON-native values, and ``device_strategy`` / ``sweep_index`` are
    primitives. Pydantic ``ExperimentConfig`` instances are not passed
    across the process boundary: the worker re-materialises + re-validates
    from ``base_dict`` + ``override`` on arrival so that any schema drift
    surfaces in the worker log rather than at dispatch time.
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
    record: ResultRecord


@dataclass(frozen=True)
class _WorkerFailure:
    """Worker run failed. The parent logs and continues."""

    sweep_index: int
    experiment_name: str
    error_repr: str


def _run_worker_task(task: _WorkerTask) -> _WorkerSuccess | _WorkerFailure:
    """Module-level worker entry point for :class:`multiprocessing.pool.Pool`.

    Must be importable by name from the parent module (``spawn`` pools
    pass functions by fully-qualified name, not by bound closure). Takes
    a single serialisable :class:`_WorkerTask` and returns a
    serialisable result.

    The worker:

    1. Materialises a validated :class:`ExperimentConfig` from
       ``base_dict`` + ``override``.
    2. Applies ``device_strategy`` by overwriting ``cfg.device`` when
       the strategy is not ``"auto"`` (which is passthrough).
    3. Runs :func:`run_one_experiment`.
    4. Wraps the :class:`ResultRecord` in :class:`_WorkerSuccess`, or
       captures the exception as :class:`_WorkerFailure` so the parent
       can log + skip rather than losing the whole pool to one bad run.
    """
    try:
        cfg = _materialise_cfg(task.base_dict, task.override)
        cfg = _apply_device_strategy(cfg, task.device_strategy)
        log.info(
            "sweep worker [%d/%d]: running %s (seed=%d fold=%d device=%s)",
            task.sweep_index + 1,
            task.sweep_total,
            cfg.name,
            cfg.seed,
            cfg.data.fold_idx,
            cfg.device,
        )
        record = run_one_experiment(cfg)
        return _WorkerSuccess(sweep_index=task.sweep_index, record=record)
    except Exception as exc:  # noqa: BLE001 — uniform failure reporting
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
    cfg: ExperimentConfig, device_strategy: str,
) -> ExperimentConfig:
    """Return a copy of ``cfg`` whose ``device`` field matches ``device_strategy``.

    ``"auto"`` is passthrough: the base config's existing ``device`` value
    (which itself may be ``"auto"``) is honoured. Any other value is
    written to the config and re-validated.
    """
    if device_strategy == "auto":
        return cfg
    if device_strategy == cfg.device:
        return cfg
    merged = cfg.model_dump(mode="python")
    merged["device"] = device_strategy
    return validate_experiment_config(ExperimentConfig.model_validate(merged))


# ---------------------------------------------------------------------------
# Public sweep entry point
# ---------------------------------------------------------------------------


def run_sweep(
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
        :func:`mmfp.config.load.load_config`. An empty/non-existent path
        falls back to :data:`DEFAULT_CONFIG`.
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
        ``spawn``-context :class:`multiprocessing.pool.Pool`. Values less
        than ``1`` raise :class:`ValueError`.
    device_strategy
        How workers pick their compute device:

        * ``"cpu"`` (default): every worker forces ``cfg.device = "cpu"``.
          This is the only strategy the reproducibility suite guarantees
          to be bit-identical to a serial run.
        * ``"mps"``: every worker forces ``cfg.device = "mps"``. Apple's
          MPS has a single GPU queue that serialises across processes, so
          high parallelism numbers rarely help on MPS. Metric tolerance
          is 1e-4 per the MPS reproducibility test.
        * ``"auto"``: workers keep whatever device the base config
          requests (which itself may be ``"auto"``).

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

    Notes
    -----
    The parent process is the sole writer: workers return in-memory
    :class:`ResultRecord` instances and the parent appends them to the
    CSV via :func:`append_result`. This keeps the atomic-append contract
    single-writer without cross-process locking.
    """
    if parallelism < 1:
        raise ValueError(
            f"run_sweep: parallelism must be >= 1 (got {parallelism})."
        )
    if device_strategy not in ("cpu", "mps", "auto"):
        raise ValueError(
            "run_sweep: device_strategy must be one of "
            f"('cpu', 'mps', 'auto'); got {device_strategy!r}."
        )

    base_cfg_path = Path(base_cfg_path)
    output_csv = Path(output_csv)

    # Load the base config as a dict once — we apply overrides at the
    # dict level so every worker re-validates the pydantic config.
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
                "run_sweep: skipping unrecognised row in %s: %s",
                output_csv, exc,
            )

    if existing_fingerprints:
        log.info(
            "run_sweep: resume enabled; %d existing rows on disk.",
            len(existing_fingerprints),
        )

    # Pre-flight: compute the prospective fingerprint for each override
    # in the parent so workers never start runs that would be skipped.
    pending: list[_WorkerTask] = []
    skipped = 0
    total = len(overrides)

    for i, override in enumerate(overrides):
        prospective_cfg = _materialise_cfg(base_dict, override)
        # Device strategy participates in config_hash via cfg.device, so
        # the prospective fingerprint must be computed after it is
        # applied. Otherwise resume would miss rows that were written by
        # a prior run with a different strategy.
        prospective_cfg = _apply_device_strategy(prospective_cfg, device_strategy)
        fp = _prospective_fingerprint(prospective_cfg)
        if fp in existing_fingerprints:
            skipped += 1
            log.info(
                "run_sweep: [%d/%d] SKIP already-recorded %s",
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
        "run_sweep: done; %d completed, %d skipped, %d failed out of %d overrides.",
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
    """Run pending tasks one after another in the current process.

    Returns ``(completed, failed)`` counts so the caller can log a
    consolidated summary.
    """
    completed = 0
    failed = 0
    for task in pending:
        result = _run_worker_task(task)
        if isinstance(result, _WorkerFailure):
            log.error(
                "run_sweep: [%d/%d] FAILED (%s): %s",
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
    """Dispatch pending tasks to a ``spawn``-context pool.

    Results stream back via :meth:`~multiprocessing.pool.Pool.imap_unordered`
    so a single slow run does not block its faster peers. Each completion
    is appended to the CSV in the parent, keeping the atomic-append
    contract single-writer.
    """
    if not pending:
        return 0, 0

    ctx = mp.get_context("spawn")
    completed = 0
    failed = 0
    # chunksize=1: dispatch fine-grained to keep load balanced when
    # individual runs differ in wall-clock cost.
    with ctx.Pool(processes=parallelism) as pool:
        for result in pool.imap_unordered(_run_worker_task, pending, chunksize=1):
            if isinstance(result, _WorkerFailure):
                log.error(
                    "run_sweep (parallel): sweep_index=%d FAILED (%s): %s",
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
    """Mirror :func:`mmfp.config.load._deep_merge` so we do not import a private."""
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
) -> ExperimentConfig:
    """Apply ``override`` to ``base_dict`` and return a validated config."""
    merged = apply_override(base_dict, override)
    cfg = ExperimentConfig.model_validate(merged)
    return validate_experiment_config(cfg)


def _prospective_fingerprint(cfg: ExperimentConfig) -> str:
    """Compute the same fingerprint that :func:`record_fingerprint` will emit.

    We need this before running the experiment so the resume path can
    skip it. Mirrors the exact field set of
    :func:`mmfp.experiments.result_schema.record_fingerprint`.
    """
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
    "apply_override",
    "run_sweep",
]
