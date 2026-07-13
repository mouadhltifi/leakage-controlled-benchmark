#!/usr/bin/env python3
"""Thin CLI wrapping :mod:`mmfp.experiments.runner` and
:mod:`mmfp.experiments.sweep`.

Usage
-----

Single experiment (appends one row to ``output``):

    python scripts/run_platform.py run \
        --config path/to/experiment.toml \
        [--set training.batch_size=128 ...] \
        [--output results.csv]

Sweep (appends one row per grid point):

    python scripts/run_platform.py sweep \
        --config path/to/base.toml \
        --grid path/to/grid.json \
        --out results.csv \
        [--no-resume]

The ``grid`` JSON file must be a list of dicts; each dict is a dotted-
key override applied to the base config. Example::

    [
      {"seed": 42, "data.fold_idx": 0},
      {"seed": 43, "data.fold_idx": 0},
      {"seed": 42, "data.fold_idx": 1}
    ]

This is intentionally minimal. The rich sweep-generation helpers (axis
cross-products, seed ladders, ablation matrices) live in analysis
notebooks / scripts that emit grid JSONs; this CLI is just the engine.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make ``mmfp`` importable: the packages live under ``<root>/src``.
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mmfp.config.load import load_config  # noqa: E402
from mmfp.experiments.result_schema import append_result  # noqa: E402
from mmfp.experiments.runner import run_one_experiment  # noqa: E402
from mmfp.experiments.sweep import run_sweep  # noqa: E402

log = logging.getLogger("mmfp.scripts.run_platform")


def _configure_logging(level: str = "INFO") -> None:
    """Configure a simple root logger. Safe to call multiple times."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a single experiment."""
    cfg = load_config(args.config, overrides=args.set or [])
    log.info("run_platform: config loaded (name=%r)", cfg.name)
    record = run_one_experiment(cfg)

    if args.output is not None:
        append_result(args.output, record)
        log.info("run_platform: appended row to %s", args.output)
    else:
        log.info("run_platform: %s", record)
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Execute a JSON-driven sweep."""
    grid_path = Path(args.grid)
    with grid_path.open("r") as fh:
        grid = json.load(fh)

    if not isinstance(grid, list) or not all(isinstance(x, dict) for x in grid):
        raise ValueError(
            f"{grid_path}: grid JSON must be a list of dicts; got {type(grid).__name__}"
        )

    df = run_sweep(
        base_cfg_path=args.config,
        overrides=grid,
        output_csv=args.out,
        resume=args.resume,
        parallelism=args.parallelism,
        device_strategy=args.device_strategy,
    )
    log.info("run_platform: sweep done; %d rows in %s", len(df), args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_platform",
        description="mmfp experiment runner and sweep CLI",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run --
    p_run = sub.add_parser("run", help="Run a single experiment.")
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
    p_sweep = sub.add_parser("sweep", help="Run a JSON-driven sweep.")
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
            "pool overhead; values >1 dispatch to a spawn-context pool. "
            "Bit-identical to serial on CPU path only."
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
