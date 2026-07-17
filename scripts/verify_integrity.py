#!/usr/bin/env python3
"""Harness-integrity check for Level-3 audits (see the paper's audit ladder).

Generate mode (maintainers, at release/tag time):
    python3 scripts/verify_integrity.py --generate
writes MANIFEST.sha256 — one line per tracked file under src/, scripts/,
configs/, tables/: "<sha256>  <path>".

Verify mode (auditors; the default):
    python3 scripts/verify_integrity.py
recomputes every hash and reports OK / MODIFIED / MISSING / UNTRACKED.
A Level-3 audit runs this first: regeneration under a modified harness
establishes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
COVERED = ("src", "scripts", "configs", "tables",
           "results", "audits", "experiments", "examples",
           "data/raw/macro", "data/processed")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store"}
SCOPE_HEADER = (
    "# Scope: the reproduction tree (code, configs, tables, results, audits,"
    " experiments, examples) plus the materialized raw inputs"
    " (data/raw/macro) and the derived-feature deposit (data/processed)."
    " Excluded: top-level docs, LICENSE, croissant.json,"
    " and this manifest itself. Verify: python3 scripts/verify_integrity.py"
)


def covered_files() -> list[Path]:
    files = []
    for top in COVERED:
        for p in sorted((ROOT / top).rglob("*")):
            if p.is_file() and not (set(p.parts) & SKIP_PARTS):
                files.append(p)
    return files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true",
                    help="(re)write MANIFEST.sha256 from the current tree")
    a = ap.parse_args()

    if a.generate:
        lines = [f"{sha256(p)}  {p.relative_to(ROOT)}" for p in covered_files()]
        MANIFEST.write_text(SCOPE_HEADER + "\n" + "\n".join(lines) + "\n")
        print(f"wrote {MANIFEST.name}: {len(lines)} files")
        return 0

    if not MANIFEST.exists():
        sys.exit("MANIFEST.sha256 missing — run --generate at release time")
    recorded = {}
    for line in MANIFEST.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        recorded[rel] = digest
    ok = modified = 0
    for rel, digest in recorded.items():
        p = ROOT / rel
        if not p.exists():
            print(f"MISSING   {rel}")
            modified += 1
        elif sha256(p) != digest:
            print(f"MODIFIED  {rel}")
            modified += 1
        else:
            ok += 1
    untracked = [str(p.relative_to(ROOT)) for p in covered_files()
                 if str(p.relative_to(ROOT)) not in recorded]
    for rel in untracked:
        print(f"UNTRACKED {rel}")
    print(f"{ok} OK, {modified} modified/missing, {len(untracked)} untracked")
    return 1 if (modified or untracked) else 0


if __name__ == "__main__":
    sys.exit(main())
