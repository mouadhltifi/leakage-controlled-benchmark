#!/usr/bin/env python3
"""Thin CLI wrapping :mod:`forecast.experiments.runner` and
:mod:`forecast.experiments.sweep`.

Usage
-----

Single experiment (appends one row to ``output`` if supplied):

    python scripts/run_forecast.py run \\
        --config forecast/configs/experiments/A7_price_only_tft.toml \\
        [--set training.max_epochs=3 training.min_epochs=0 ...] \\
        [--output results.csv]

Sweep (appends one row per grid point):

    python scripts/run_forecast.py sweep \\
        --config forecast/configs/experiments/A7_price_only_tft.toml \\
        --grid path/to/grid.json \\
        --out results.csv \\
        [--no-resume]

The ``grid`` JSON file must be a list of dicts; each dict is a dotted-
key override applied to the base config. Example::

    [
      {"seed": 42, "data.fold_idx": 0},
      {"seed": 43, "data.fold_idx": 0}
    ]

Mirrors ``scripts/run_platform.py`` (the v2 CLI) so operators can keep
their muscle memory; only the import roots differ.
"""

from __future__ import annotations

import os
# Pinned-state thread configuration (S3.2 determinism scope): thread count
# changes floating-point reduction order and can move a single run by up
# to ~0.04 MCC. The shipped grids ran under this pin; keep it for
# bit-identical reproduction (override only knowingly).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import logging
import sys
from pathlib import Path

# Make both ``mmfp`` and ``forecast`` importable: the packages live under ``<root>/src``.
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from forecast.config.load import load_config  # noqa: E402
from forecast.experiments.result_schema import append_result  # noqa: E402
from forecast.experiments.runner import run_one_forecast_experiment  # noqa: E402
from forecast.experiments.sweep import run_forecast_sweep  # noqa: E402

log = logging.getLogger("forecast.scripts.run_forecast")


def _configure_logging(level: str = "INFO") -> None:
    """Configure a simple root logger. Safe to call multiple times."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a single v3 experiment."""
    cfg = load_config(args.config, overrides=args.set or [])
    log.info("run_forecast: config loaded (name=%r)", cfg.name)
    record = run_one_forecast_experiment(cfg)

    if args.output is not None:
        append_result(args.output, record)
        log.info("run_forecast: appended row to %s", args.output)
    else:
        log.info("run_forecast: %s", record)
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Execute a JSON-driven v3 sweep."""
    grid_path = Path(args.grid)
    with grid_path.open("r") as fh:
        grid = json.load(fh)

    if not isinstance(grid, list) or not all(isinstance(x, dict) for x in grid):
        raise ValueError(
            f"{grid_path}: grid JSON must be a list of dicts; got {type(grid).__name__}"
        )

    df = run_forecast_sweep(
        base_cfg_path=args.config,
        overrides=grid,
        output_csv=args.out,
        resume=args.resume,
        parallelism=args.parallelism,
        device_strategy=args.device_strategy,
    )
    log.info("run_forecast: sweep done; %d rows in %s", len(df), args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_forecast",
        description="v3 forecast experiment runner and sweep CLI",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run --
    p_run = sub.add_parser("run", help="Run a single v3 experiment.")
    p_run.add_argument(
        "--config", required=True,
        help="Path to an experiment TOML.",
    )
    p_run.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Dotted-key override (repeatable).",
    )
    p_run.add_argument(
        "--output", default=None,
        help="Optional CSV to which the result row is appended.",
    )
    p_run.set_defaults(func=_cmd_run)

    # -- sweep --
    p_sweep = sub.add_parser("sweep", help="Run a JSON-driven v3 sweep.")
    p_sweep.add_argument(
        "--config", required=True,
        help="Path to the base experiment TOML.",
    )
    p_sweep.add_argument(
        "--grid", required=True,
        help="Path to a JSON file listing dotted-key override dicts.",
    )
    p_sweep.add_argument(
        "--out", required=True,
        help="Path to the output CSV (created if missing).",
    )
    p_sweep.add_argument(
        "--no-resume", action="store_false", dest="resume",
        help="Re-run every grid point even if the CSV already has it.",
    )
    p_sweep.add_argument(
        "--parallelism", type=int, default=1,
        help=(
            "Number of worker processes. 1 (default) runs serially with no "
            "pool overhead; values >1 dispatch to a spawn-context pool."
        ),
    )
    p_sweep.add_argument(
        "--device-strategy",
        choices=("cpu", "mps", "auto"),
        default="cpu",
        dest="device_strategy",
        help=(
            "Device each worker uses. 'cpu' (default) is bit-identical to "
            "serial; 'mps' is fastest per-run but the GPU queue serialises "
            "across processes; 'auto' honours the base config's device field."
        ),
    )
    p_sweep.set_defaults(func=_cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
