#!/usr/bin/env python3
"""Milestone 8 equivalence-gate driver.

Executes the 15-point A7 price-only sweep (5 folds x 3 seeds) on the
mmfp v2 platform and compares the aggregated results to the archived
v1 reference in ``results/analysis/phase5_full_750core.md``.

Outputs
-------
* CSV   : the equivalence-gate CSV
         (one row per run via :func:`mmfp.experiments.sweep.run_sweep`).
* Report: the equivalence-gate report
         (markdown rendered by :func:`mmfp.experiments.equivalence.format_report`).

The script is idempotent: re-running it resumes from whatever rows are
already in the CSV (per the sweep's ``resume=True`` default).

Exit code
---------
* ``0`` — gate passed, CSV + report written.
* ``1`` — gate failed, CSV + report written, no further action.
* ``2`` — execution error (exception surfaced before the gate could run).
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Pin the thread configuration BEFORE torch/numpy import: the reference
# results (and the artifact's determinism claim) are scoped to the pinned
# environment including thread count — thread count changes floating-point
# summation order and moves single-config fold means by up to ~0.015 MCC
# while grand means hold. The gate must run under the same pin it certifies.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

# Make ``mmfp`` importable: the packages live under ``<root>/src``
# (this script is at ``<root>/scripts/audits/``).
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent.parent
_SRC_DIR = _ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mmfp.experiments.equivalence import (  # noqa: E402
    compare_to_reference,
    format_report,
)
from mmfp.experiments.sweep import run_sweep  # noqa: E402


log = logging.getLogger("mmfp.scripts.run_equivalence_gate")


# ---------------------------------------------------------------------------
# Paths and grid definition
# ---------------------------------------------------------------------------

CONFIG_PATH = _ROOT / "configs" / "mmfp" / "experiments" / "A7_price_only.toml"
# The gate WRITES a fresh CSV + report; it is a re-run, not part of the committed
# evidence, so it defaults to a temp dir (override with EQUIV_GATE_OUT). This keeps
# the documented command from clobbering anything tracked under results/.
OUTPUT_DIR = Path(
    os.environ.get("EQUIV_GATE_OUT", tempfile.gettempdir() + "/mmfp_equivalence_gate")
)
CSV_PATH = OUTPUT_DIR / "a7_price_only.csv"
REPORT_PATH = OUTPUT_DIR / "report.md"

#: Task-specified seeds for the equivalence gate.
#:
#: Note: v1's archived 750-core sweep used seeds [42, 123, 456]; the
#: task brief for this milestone overrides that to [42, 123, 2024]. The
#: gate tolerances (+-0.005 on mean, +-0.015 per fold) were chosen with
#: seed variance in mind, so this should not affect pass/fail.
SEEDS: tuple[int, ...] = (42, 123, 2024)
FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def build_grid() -> list[dict[str, int]]:
    """Return the 15-entry overrides list for the sweep.

    Each entry sets ``seed`` and ``data.fold_idx``; all other knobs come
    from ``A7_price_only.toml``. The ordering is
    fold-major (all seeds for fold 0, then fold 1, ...) so a partial
    run shows progress fold by fold rather than seed by seed.
    """
    grid: list[dict[str, int]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            grid.append({"data.fold_idx": int(fold), "seed": int(seed)})
    return grid


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging(level: str = "INFO") -> None:
    """Attach a simple stderr handler so sweep progress is visible."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    _configure_logging("INFO")
    log.info("equivalence gate: starting")
    log.info("  config : %s", CONFIG_PATH)
    log.info("  csv    : %s", CSV_PATH)
    log.info("  report : %s", REPORT_PATH)

    if not CONFIG_PATH.exists():
        log.error("equivalence gate: config missing at %s", CONFIG_PATH)
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = build_grid()
    log.info("equivalence gate: %d overrides queued", len(grid))

    t_start = time.time()
    try:
        run_sweep(
            base_cfg_path=CONFIG_PATH,
            overrides=grid,
            output_csv=CSV_PATH,
            resume=True,
            parallelism=1,
        )
    except Exception as exc:  # pragma: no cover — surfaced to caller
        log.exception("equivalence gate: sweep execution failed: %s", exc)
        return 2
    elapsed = time.time() - t_start
    log.info("equivalence gate: sweep done in %.1fs", elapsed)

    # Compare.
    try:
        result = compare_to_reference(CSV_PATH)
    except FileNotFoundError as exc:
        log.error("equivalence gate: %s", exc)
        return 2

    report = format_report(result)
    report_with_timing = (
        report
        + "\n"
        + f"## Wall-time\n\n- Sweep elapsed: {elapsed:.1f} seconds\n"
    )
    REPORT_PATH.write_text(report_with_timing)
    log.info("equivalence gate: report written to %s", REPORT_PATH)

    # Echo the headline to stderr so a CI / terminal user sees the
    # outcome without having to open the file.
    headline = "PASSED" if result.passed else "FAILED"
    log.info(
        "equivalence gate: %s (v2 mean=%+.4f, v1 mean=%+.4f, diff=%+.4f)",
        headline,
        result.mean_mcc_v2,
        result.mean_mcc_v1,
        result.mean_diff,
    )
    for note in result.notes:
        log.info("equivalence gate: %s", note)

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
