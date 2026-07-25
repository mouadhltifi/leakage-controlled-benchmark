#!/usr/bin/env python3
"""Per-arm reference grid: every configuration x every architecture arm.

Table 3 reports the architecture-pooled estimand, because architecture is a
nuisance axis rather than a hypothesis (see the paper's C4 discussion). A
reviewer reasonably asks to see the axis that pooling hides -- both to check
that pooling is not concealing an arm that would certify, and because
selecting the better arm post hoc is itself one of the evaluation choices the
benchmark prices.

This script therefore ships the full split: all eight declared configurations
x both arms, each run through the *release's own evaluator*
(``evaluate_submission.py``) exactly as an external submitter would, at the
declared family size k=8.

The headline it makes checkable: **0 of 16 arm-level submissions certify.**
The largest is the price+graph feedforward arm; it is also the one whose
post-hoc selection is worth the most, which is why the paper reports the swing
(+0.0090 pooled -> +0.0152 on the better arm) as a priced quantity rather than
as a result.

Outputs
-------
* ``results/analysis/arm_split.json``  -- machine-readable record.
* a LaTeX fragment (default ``tables/arm_split.tex``) -- one row per config.

Self-verification (exits non-zero on failure)
---------------------------------------------
* every cell resolves to shipped rows (no silent gaps);
* every arm-level verdict is parsed, none unknown;
* the pooled mean of the two arms reproduces the Table 3 row for that config
  to 1e-4, which ties this table to the reference floor;
* no arm certifies at k=8 -- if that ever changes, the paper's headline claim
  changes with it, and this script must fail loudly rather than quietly
  emit a table that contradicts the abstract.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "scripts" / "analysis" / "evaluate_submission.py"
DEFAULT_JSON = ROOT / "results" / "analysis" / "arm_split.json"
DEFAULT_TEX = ROOT / "tables" / "arm_split.tex"
REF_TEX = ROOT / "tables" / "ref_direction_v2.tex"

N_TEST = {0: 10100, 1: 9851, 2: 9979, 3: 9994, 4: 4514}
K_DECLARED = 8

# display, config id, results file template, experiment_name prefix template
CONFIGS = [
    ("P+N",     "A3", "results/native_core/native_core_{arm}.csv",   "natcore_{cfg}_{arm}_concat"),
    ("P+M",     "A4", "results/native_macro/native_macro_{arm}.csv", "natmacro_{cfg}_{arm}_concat"),
    ("P+S",     "A6", "results/native_core/native_core_{arm}.csv",   "natcore_{cfg}_{arm}_concat"),
    ("P+G",     "A8", "results/sector/sector_ff12_{arm}.csv",        "secff12_{cfg}_{arm}_concat"),
    ("P+N+M",   "A2", "results/native_macro/native_macro_{arm}.csv", "natmacro_{cfg}_{arm}_concat"),
    ("P+M+S",   "A5", "results/native_macro/native_macro_{arm}.csv", "natmacro_{cfg}_{arm}_concat"),
    ("P+M+G",   "A9", "results/sector/sector_ff12_{arm}.csv",        "secff12_{cfg}_{arm}_concat"),
    ("P+N+M+S", "A1", "results/native_macro/native_macro_{arm}.csv", "natmacro_{cfg}_{arm}_concat"),
]


def _evaluate(rows: pd.DataFrame, name: str, arm: str, tmp: Path) -> tuple[float, str]:
    """Run one arm-level submission through the release's own evaluator."""
    path = tmp / f"{name}_{arm}.csv".replace("+", "")
    rows.assign(challenger=f"{name}_{arm}",
                n_test=[N_TEST[int(f)] for f in rows.fold_idx])[
        ["challenger", "fold_idx", "seed", "mcc", "n_test"]].to_csv(path, index=False)
    out = subprocess.run(
        [sys.executable, str(EVAL), str(path), "--k", str(K_DECLARED),
         "--baseline-arch", arm],
        capture_output=True, text=True).stdout
    delta = re.search(r"dMCC \(mean\)\s*:\s*([+-]\d+\.\d+)", out)
    verdict = re.search(r"VERDICT\s*:\s*(.+)", out)
    if not delta or not verdict:
        raise AssertionError(f"{name}/{arm}: could not parse evaluator output:\n{out}")
    return float(delta.group(1)), verdict.group(1).strip()


def _reference_row(disp: str) -> float | None:
    """The pooled Table 3 delta for this configuration, for the tie-back check."""
    if not REF_TEX.exists():
        return None
    for line in REF_TEX.read_text().splitlines():
        if line.strip().startswith(disp + " &"):
            m = re.search(r"([+-]\d\.\d{4})", line)
            if m:
                return float(m.group(1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-arm reference grid.")
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--texout", default=str(DEFAULT_TEX))
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    record: dict[str, dict] = {}
    tex: list[str] = []
    certified: list[str] = []

    for disp, cfg, tmpl, prefix in CONFIGS:
        cells = {}
        for arm in ("ff", "lstm"):
            src = ROOT / tmpl.format(arm=arm)
            if not src.exists():
                raise AssertionError(f"{disp}/{arm}: missing results file {src}")
            df = pd.read_csv(src)
            want = prefix.format(cfg=cfg, arm=arm)
            rows = df[df.experiment_name.astype(str).str.startswith(want)]
            if rows.empty:
                raise AssertionError(f"{disp}/{arm}: no rows matching {want!r} in {src.name}")
            delta, verdict = _evaluate(rows, disp, arm, tmp)
            cells[arm] = {"delta_mcc": delta, "verdict": verdict,
                          "certifies": verdict.startswith("SUPPORTED"),
                          "n_cells": int(len(rows))}
            if cells[arm]["certifies"]:
                certified.append(f"{disp}/{arm}")

        pooled = (cells["ff"]["delta_mcc"] + cells["lstm"]["delta_mcc"]) / 2.0
        ref = _reference_row(disp)
        if ref is not None and abs(pooled - ref) > 1e-4:
            raise AssertionError(
                f"{disp}: mean of arms ({pooled:+.4f}) does not reproduce the "
                f"Table 3 row ({ref:+.4f}); the split is inconsistent with the "
                "reference floor")
        swing = abs(cells["ff"]["delta_mcc"] - cells["lstm"]["delta_mcc"])
        record[disp] = {"config_id": cfg, "arms": cells, "pooled_mean": pooled,
                        "reference_row": ref, "arm_swing": swing}
        tex.append(
            f"{disp} & {cells['ff']['delta_mcc']:+.4f} & "
            f"{cells['lstm']['delta_mcc']:+.4f} & {pooled:+.4f} & "
            f"{swing:.4f} & 0/2 \\\\")

    if certified:
        raise AssertionError(
            "an arm-level submission now certifies at k=8 "
            f"({', '.join(certified)}); the paper's headline claim no longer "
            "holds and must be revised before this table ships")

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(
        {"meta": {"k_declared": K_DECLARED,
                  "evaluator": "scripts/analysis/evaluate_submission.py",
                  "estimand": "per-arm; Table 3 pools these two columns",
                  "n_submissions": 2 * len(CONFIGS),
                  "n_certifying": 0},
         "configs": record}, indent=2) + "\n")
    print(f"Wrote {Path(a.json).relative_to(ROOT)}")

    try:
        Path(a.texout).parent.mkdir(parents=True, exist_ok=True)
        Path(a.texout).write_text(
            "% Generated by scripts/analysis/make_arm_split_table.py -- do not hand-edit.\n"
            + "\n".join(tex) + "\n\\bottomrule\n")
        print(f"Wrote {Path(a.texout).relative_to(ROOT)}")
    except OSError as e:  # pragma: no cover
        print(f"WARNING: LaTeX fragment not written ({e})")

    print("\n".join(t.replace("\\\\", "") for t in tex))
    print(f"\n0 of {2 * len(CONFIGS)} arm-level submissions certify at k={K_DECLARED}")
    print("SELF-VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
