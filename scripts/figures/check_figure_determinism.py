#!/usr/bin/env python3
"""Verify that the paper's figures regenerate byte-identically.

The release claims its figures regenerate from the shipped generators. That
claim was not checkable until v1.0.15: matplotlib stamps ``/CreationDate``
into every PDF from the wall clock, so two runs of the same generator on the
same data produced different bytes, and "regenerate and compare" could not
distinguish a real change from the timestamp. The generators now pin
``SOURCE_DATE_EPOCH``; this script proves it holds.

Method: run each generator twice into two separate output directories and
compare the resulting PDFs byte-for-byte. Any difference means either the
determinism pin regressed or a generator has a genuine nondeterminism
(dict/set ordering, an unseeded sample), both of which break the
regenerate-and-compare workflow reviewers are invited to use.

Exit 0 iff every figure is byte-identical across the two runs.
"""
from __future__ import annotations

import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATORS = [
    ROOT / "scripts" / "figures" / "make_overview_figure.py",
    ROOT / "scripts" / "figures" / "make_benchmark_figures.py",
]
FIGDIR = ROOT / "figures"
# The paper's six figures. Named explicitly so a generator that silently stops
# emitting one is a FAILURE rather than a smaller, still-green run.
EXPECTED = {"fig-overview.pdf", "fig-selection.pdf", "fig-split.pdf",
            "fig-forest.pdf", "fig-timeline.pdf", "fig-sectorband.pdf"}


def _run_once(dest: Path) -> list[str]:
    """Generate the figures, copying out ONLY the ones actually (re)written.

    ``figures/`` also holds output from other generators in the repo. Copying
    everything would report those as "identical" without regenerating them --
    a false pass of exactly the kind this release's gates exist to prevent --
    so we take an mtime snapshot first and keep only files the run touched.
    """
    before = {p.name: p.stat().st_mtime_ns for p in FIGDIR.glob("*.pdf")} \
        if FIGDIR.exists() else {}
    for gen in GENERATORS:
        r = subprocess.run([sys.executable, str(gen)], capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"{gen.name} failed:\n{r.stdout}\n{r.stderr}")
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for pdf in sorted(FIGDIR.glob("*.pdf")):
        if before.get(pdf.name) == pdf.stat().st_mtime_ns:
            continue          # untouched by this run: not ours to certify
        shutil.copy2(pdf, dest / pdf.name)
        names.append(pdf.name)
    if not names:
        raise AssertionError("no figures were regenerated -- the generators "
                             "wrote nothing, so determinism is unverified")
    return names


def main() -> int:
    if not FIGDIR.parent.exists():
        print("FAIL  artifact root not found")
        return 1
    tmp = Path(tempfile.mkdtemp())
    a, b = tmp / "run_a", tmp / "run_b"
    names_a = _run_once(a)
    names_b = _run_once(b)

    if set(names_a) != set(names_b):
        print(f"FAIL  the two runs produced different figure sets: "
              f"{sorted(set(names_a) ^ set(names_b))}")
        return 1

    missing = sorted(EXPECTED - set(names_a))
    if missing:
        print(f"FAIL  generators did not emit {len(missing)} expected "
              f"figure(s): {', '.join(missing)}")
        return 1
    bad = [n for n in names_a if not filecmp.cmp(a / n, b / n, shallow=False)]
    for n in names_a:
        print(("FAIL  " if n in bad else "OK    ") + n)
    if bad:
        print(f"\nFAIL: {len(bad)} figure(s) are not byte-reproducible "
              f"({', '.join(bad)}). Check that SOURCE_DATE_EPOCH is pinned in "
              "the generators and that nothing samples without a fixed seed.")
        return 1
    print(f"\nAll {len(names_a)} figures regenerate byte-identically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
