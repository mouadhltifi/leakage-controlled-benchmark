#!/usr/bin/env python3
"""Pre-tag release gate. NO TAG IS CUT unless this exits 0 on the tree
being tagged (run it on a fresh clone to match what reviewers get):

    python3 scripts/release_gate.py [--expect-version X.Y.Z]

Checks, in order:
  1. MANIFEST integrity: verify_integrity.py exits 0 with zero
     modified/missing/untracked (the class of failure that shipped three
     times before this gate existed).
  2. The evaluator regression battery (certification gates + the two
     adversarially discovered exploits).
  3. The paper's Section-6 demo command, byte-compared against the
     committed claim block.
  4. Metadata: croissant.json loads (if mlcroissant is installed),
     CITATION.cff parses with a scalar license (Zenodo loader
     constraint), and CITATION/croissant version strings agree (and
     match --expect-version when given).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS: list[str] = []


def step(label: str, ok: bool, detail: str = ""):
    print(("PASS  " if ok else "FAIL  ") + label + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-version", default=None)
    a = ap.parse_args()

    # 1. integrity
    r = subprocess.run([sys.executable, str(ROOT / "scripts/verify_integrity.py")],
                       capture_output=True, text=True)
    clean = r.returncode == 0 and ", 0 modified/missing, 0 untracked" in r.stdout
    step("MANIFEST integrity (fresh-tree verify)", clean,
         r.stdout.strip().splitlines()[-1] if r.stdout else "no output")

    # 2. evaluator regression battery
    r = subprocess.run([sys.executable,
                        str(ROOT / "scripts/analysis/test_evaluate_submission.py")],
                       capture_output=True, text=True)
    step("evaluator regression battery", r.returncode == 0,
         r.stdout.strip().splitlines()[-1] if r.stdout else "no output")

    # 3. the Section-6 demo, byte-compared
    demo = ROOT / "examples/demo_submission/submission.csv"
    committed = ROOT / "examples/demo_submission/claim.txt"
    r = subprocess.run([sys.executable,
                        str(ROOT / "scripts/analysis/evaluate_submission.py"),
                        str(demo), "--k", "1", "--baseline-arch", "ff"],
                       capture_output=True, text=True)
    match = committed.exists() and r.stdout == committed.read_text()
    step("S6 demo == committed claim block (byte-identical)", match)

    # 4. metadata
    try:
        import yaml
        cff = yaml.safe_load((ROOT / "CITATION.cff").read_text())
        scalar = isinstance(cff.get("license"), str)
        step("CITATION.cff parses, scalar license", scalar,
             f"license={cff.get('license')!r} version={cff.get('version')}")
        cro = json.loads((ROOT / "croissant.json").read_text())
        agree = str(cff.get("version")) == str(cro.get("version"))
        step("CITATION/croissant versions agree", agree,
             f"{cff.get('version')} vs {cro.get('version')}")
        if a.expect_version:
            step("version matches --expect-version",
                 str(cff.get("version")) == a.expect_version,
                 f"{cff.get('version')} vs {a.expect_version}")
    except Exception as e:  # pragma: no cover
        step("metadata parse", False, str(e))
    try:
        import mlcroissant as mlc
        mlc.Dataset(str(ROOT / "croissant.json"))
        step("croissant loads under mlcroissant", True)
    except ImportError:
        print("SKIP  croissant load (mlcroissant not installed)")
    except Exception as e:
        step("croissant loads under mlcroissant", False, str(e)[:80])

    print(f"\n{'RELEASE GATE: PASS' if not FAILS else 'RELEASE GATE: FAIL -- do not tag'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
