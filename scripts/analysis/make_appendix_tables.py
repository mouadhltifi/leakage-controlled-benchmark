#!/usr/bin/env python3
r"""Appendix table fragments for Paper B (KDD D&B) — two \input tables:

  appendix_perfold.tex  per-fold mean DeltaMCC for the 8 primary contrasts,
                        with the single-state pairing IMPORTED from
                        make_reference_table_v2 (same A7 baseline, same cells),
                        so each row's mean column is identical to
                        ref_direction_v2's DeltaMCC.
  appendix_sector.tex   FF12-vs-GICS sector-taxonomy paired comparison from
                        results/sector/, per (config, arch), paired within
                        (fold, seed, fusion) — the 45-cell equivalence campaign
                        behind audits/sector/SECTOR-EQUIVALENCE.md.

Both fragments are booktabs body rows only, terminated by \bottomrule INSIDE
the fragment (\input-in-tabular breaks if \bottomrule follows the \input); the
paper supplies the table environment, column spec, header row and caption.
Each fragment self-validates against a pinned anchor before being written;
failure exits non-zero. Prints one LEDGER line per row.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

# Reuse the reference table's exact pairing/loading so the per-fold means are
# the identical single-state estimand (same harness-native A7, same 30 cells).
from make_reference_table_v2 import CONFIGS, R, ROOT, load

SECTOR_DISPLAY = {"A8": "P+G", "A9": "P+M+G"}  # match ref_direction_v2 names


# --------------------------------------------------------------------------- #
# 1. per-fold mean DeltaMCC (8 primary contrasts)
# --------------------------------------------------------------------------- #
def _published_ref_dmcc(tables_dir: Path) -> dict[str, float]:
    """Parse ref_direction_v2.tex -> {display: DeltaMCC} for the self-check."""
    tex = tables_dir / "ref_direction_v2.tex"
    if not tex.exists():
        sys.exit(f"ref_direction_v2.tex missing at {tex}; run "
                 "make_reference_table_v2.py first (the per-fold means are "
                 "checked against its published DeltaMCC column).")
    out: dict[str, float] = {}
    for line in tex.read_text().splitlines():
        if "&" not in line or line.lstrip().startswith("%"):
            continue
        cells = [c.strip() for c in line.split("&")]
        m = re.match(r"[+-]?\d*\.\d+", cells[1])
        if m:
            out[cells[0]] = float(m.group())
    return out


def build_perfold(tables_dir: Path) -> list[str]:
    a7, _ = load("A7", "native_core")
    assert len(a7) == 30, f"A7 cells {len(a7)} != 30"
    ref = _published_ref_dmcc(tables_dir)

    lines, ledger = [], []
    for disp, cfg, block in CONFIGS:
        c, _ = load(cfg, block)
        idx = c.index.intersection(a7.index)
        assert len(idx) == 30, f"{cfg}: matched {len(idx)} != 30"
        deltas = (c.loc[idx] - a7.loc[idx]).sort_index()
        per_fold = deltas.groupby("fold_idx").mean()
        folds = [float(per_fold.loc[f]) for f in range(5)]
        mean = float(deltas.mean())  # pooled == mean of fold means (balanced 6/fold)

        if disp not in ref or abs(mean - ref[disp]) > 1e-4:
            sys.exit(f"SELF-CHECK FAIL (per-fold): {disp} mean {mean:+.4f} != "
                     f"ref_direction_v2 DeltaMCC "
                     f"{ref.get(disp, float('nan')):+.4f} (tol 1e-4)")

        cells = " & ".join(f"{v:+.4f}" for v in folds)
        lines.append(f"{disp} & {cells} & {mean:+.4f} \\\\")
        ledger.append(f"LEDGER perfold: {disp:9s} "
                      + " ".join(f"F{i}={v:+.4f}" for i, v in enumerate(folds))
                      + f" mean={mean:+.4f} (ref {ref[disp]:+.4f})")
    for line in ledger:
        print(line)
    return lines


# --------------------------------------------------------------------------- #
# 2. FF12-vs-GICS sector-taxonomy paired comparison
# --------------------------------------------------------------------------- #
def sector_cells(taxonomy: str, cfg: str, arch: str) -> pd.Series:
    """MCC cells for one (taxonomy, config, arch), indexed by (fold,seed,fusion)."""
    d = pd.read_csv(R / f"sector/sector_{taxonomy}_{arch}.csv")
    d = d[d.experiment_name.str.startswith(f"sec{taxonomy}_{cfg}_{arch}_")]
    return d.set_index(["fold_idx", "seed", "fusion_strategy"]).mcc.sort_index()


def build_sector() -> list[str]:
    lines, ledger = [], []
    anchor_dm = anchor_p = None
    for cfg in ("A8", "A9"):
        disp = SECTOR_DISPLAY[cfg]
        for arch in ("ff", "lstm"):
            ff12 = sector_cells("ff12", cfg, arch)
            gics = sector_cells("gics", cfg, arch)
            idx = ff12.index.intersection(gics.index)
            assert len(idx) == 45, f"{cfg}-{arch}: matched {len(idx)} != 45"
            f12 = ff12.loc[idx].sort_index()
            gic = gics.loc[idx].sort_index()
            delta = f12 - gic
            dm = float(delta.mean())
            d_eff = dm / float(delta.std(ddof=1))
            _, p = stats.ttest_rel(f12.to_numpy(), gic.to_numpy())
            p = float(p)
            arch_disp = arch.upper()

            lines.append(f"{disp} & {arch_disp} & {float(f12.mean()):+.4f} & "
                         f"{float(gic.mean()):+.4f} & {dm:+.4f} & {d_eff:+.3f} & "
                         f"{p:.3f} \\\\")
            ledger.append(f"LEDGER sector: {disp:7s} {arch_disp:4s} "
                          f"FF12={float(f12.mean()):+.4f} GICS={float(gic.mean()):+.4f} "
                          f"delta(FF12-GICS)={dm:+.4f} d={d_eff:+.3f} p={p:.3f} "
                          f"n={len(idx)}")
            if cfg == "A8" and arch == "ff":
                anchor_dm, anchor_p = dm, p

    # canon anchor: price+graph FF, FF12 minus GICS (SECTOR-EQUIVALENCE.md)
    anchor_msg = (f"A8-ff delta={anchor_dm:+.4f} (want +0.0102 +-0.0005), "
                  f"p={anchor_p:.4f} (want ~0.001)")
    if anchor_dm is None or abs(anchor_dm - 0.0102) > 5e-4 or anchor_p >= 0.005:
        sys.exit(f"SELF-CHECK FAIL (sector anchor): {anchor_msg}")
    for line in ledger:
        print(line)
    print(f"  sector anchor OK: {anchor_msg}")
    return lines


# --------------------------------------------------------------------------- #
def _write(path: Path, banner: str, rows: list[str]) -> None:
    with path.open("w") as f:
        f.write(banner)
        for r in rows:
            f.write(r + "\n")
        f.write("\\bottomrule\n")
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(ROOT / "tables"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    perfold_rows = build_perfold(outdir)
    _write(outdir / "appendix_perfold.tex",
           "% Generated by scripts/analysis/make_appendix_tables.py -- per-fold "
           "mean DeltaMCC for the 8\n% primary contrasts; single-state pairing "
           "imported from make_reference_table_v2\n% (mean column == "
           "ref_direction_v2 DeltaMCC). Do not hand-edit.\n",
           perfold_rows)

    sector_rows = build_sector()
    _write(outdir / "appendix_sector.tex",
           "% Generated by scripts/analysis/make_appendix_tables.py -- FF12-vs-"
           "GICS sector taxonomy,\n% paired within (fold,seed,fusion) per "
           "(config,arch); see\n% audits/sector/SECTOR-EQUIVALENCE.md. Do not "
           "hand-edit.\n",
           sector_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
