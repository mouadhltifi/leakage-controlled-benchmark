#!/usr/bin/env python3
"""Generate the benchmark's reference-baseline tables (LaTeX fragments) from
the committed result CSVs — self-verifying against the published canon.

Composition rule (the paper's): macro-containing configurations use the
leakage-free mmfp re-run (results/macrolag/, publication-lagged macro, paired
against the re-run A7); the remaining configurations use the core grid
(v1-runner CSVs; A8/A9 graph rows are the LayerNorm re-run), paired against
the core A7. The provenance column carries this per row; the two blocks are
bridged by the validated v1<->mmfp equivalence gate (see experiments/).

Statistics: each block reproduces its published recipe (verified empirically
against the canon). Core block: concat-fusion cells per (fold, seed, arch),
paired vs the concat A7 cells (fusion was a separate analysis axis in the
core grid; the primary RQ1 comparison is concat). Macrolag block: fusion-
averaged cells per (fold, seed, arch), paired vs the re-run A7 (identical to
scripts/analysis/analyze_macrolag.py). Both give n=30; d = mean/sd(ddof=1)
of diffs; p from a one-sample t-test; p_bonf = min(1, 8p). Verification band
vs the published values: 5e-5 on deltas, 5e-3 on d (the band absorbs one
ddof-convention rounding: P+N regenerates d=-0.188 vs -0.191 in print).

The script HARD-FAILS if any emitted number deviates from the published
canonical values (tolerance 5e-5 on deltas, 5e-3 on d) — regeneration and
verification are the same act.

Usage:
    python scripts/analysis/make_reference_tables.py [--outdir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# display name, config id, source block
CONFIGS = [
    ("P+N",     "A3", "core"),
    ("P+M",     "A4", "macrolag"),
    ("P+S",     "A6", "core"),
    ("P+G",     "A8", "core"),
    ("P+N+M",   "A2", "macrolag"),
    ("P+M+S",   "A5", "macrolag"),
    ("P+M+G",   "A9", "macrolag"),
    ("P+N+M+S", "A1", "macrolag"),
]
PROV = {"core": "core grid (v1 runner)", "macrolag": "leakage-free re-run (mmfp)"}

# Published canon (ICAIF paper Table 5 / rq1 forest; verified twice in the
# v2 gauntlet). The generator must reproduce these or die.
CANON = {  # display -> (delta, d, p_bonf_str)
    "P+N":     (-0.0034, -0.191, "1.000"),
    "P+M":     (-0.0063, -0.251, "1.000"),
    "P+S":     (+0.0005, +0.030, "1.000"),
    "P+G":     (+0.0000, +0.001, "1.000"),
    "P+N+M":   (-0.0097, -0.458, "0.143"),
    "P+M+S":   (-0.0052, -0.248, "1.000"),
    "P+M+G":   (-0.0057, -0.260, "1.000"),
    "P+N+M+S": (-0.0048, -0.241, "1.000"),
}


def load_core() -> pd.DataFrame:
    """Core direction grid: base + news files, A8/A9 replaced by graph-LN."""
    frames = []
    for arch, files in (
        ("ff",   ["ablation_ff.csv", "ablation_news_ff.csv"]),
        ("lstm", ["ablation_lstm.csv", "ablation_news_lstm.csv"]),
    ):
        for f in files:
            d = pd.read_csv(RESULTS / "ablation" / f)
            d["arch"] = arch
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[~df.config_id.isin(["A8", "A9"])]
    for arch, f in (("ff", "ablation_graph_ln_ff.csv"),
                    ("lstm", "ablation_graph_ln_lstm.csv")):
        d = pd.read_csv(RESULTS / "ablation" / f)
        d["arch"] = arch
        frames = [df, d]
        df = pd.concat(frames, ignore_index=True)
    return df[["config_id", "fusion_type", "fold_idx", "seed", "arch", "mcc"]]


def load_macrolag() -> pd.DataFrame:
    frames = []
    for arch, f in (("ff", "ablation_macrolag_ff.csv"),
                    ("lstm", "ablation_macrolag_lstm.csv")):
        d = pd.read_csv(RESULTS / "macrolag" / f)
        d["arch"] = arch
        d["config_id"] = d.experiment_name.str.extract(r"macrolag_(A\d+)_")
        d["fusion_type"] = d.experiment_name.str.extract(r"_(concat|gated|mha)")
        d = d.rename(columns={"fold_idx": "fold_idx"})
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df[["config_id", "fusion_type", "fold_idx", "seed", "arch", "mcc"]]


def cells(df: pd.DataFrame, cfg: str, block: str) -> pd.Series:
    """Per-(fold, seed, arch) MCC cells under the block's published recipe."""
    sub = df[df.config_id == cfg]
    if block == "core":
        sub = sub[sub.fusion_type == "concat"]
        return sub.set_index(["fold_idx", "seed", "arch"]).mcc
    return sub.groupby(["fold_idx", "seed", "arch"]).mcc.mean()


def paired(diff: np.ndarray) -> tuple[float, float, float]:
    m = float(diff.mean())
    s = float(diff.std(ddof=1))
    _, p = stats.ttest_1samp(diff, 0.0)
    return m, (m / s if s > 0 else 0.0), float(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(ROOT / "tables"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    core, mlag = load_core(), load_macrolag()
    a7 = {"core": cells(core, "A7", "core"),
          "macrolag": cells(mlag, "A7", "macrolag")}

    rows, failures = [], []
    for disp, cfg, block in CONFIGS:
        df = core if block == "core" else mlag
        c = cells(df, cfg, block)
        idx = c.index.intersection(a7[block].index)
        m, d, p = paired((c.loc[idx] - a7[block].loc[idx]).values)
        pb = min(1.0, 8 * p)
        rows.append((disp, m, d, p, pb, len(idx), PROV[block]))
        cd, cdd, cpb = CANON[disp]
        if abs(m - cd) > 5e-5 or abs(d - cdd) > 5e-3:
            failures.append(f"{disp}: got delta={m:+.4f} d={d:+.3f}, "
                            f"canon delta={cd:+.4f} d={cdd:+.3f}")
        got_pb = f"{pb:.3f}" if pb < 0.9995 else "1.000"
        if got_pb != cpb:
            failures.append(f"{disp}: got p_bonf={got_pb}, canon {cpb}")

    tex = outdir / "ref_direction.tex"
    with tex.open("w") as f:
        f.write("% Generated by scripts/analysis/make_reference_tables.py -- "
                "self-verified against the published canon. Do not hand-edit.\n")
        for disp, m, d, p, pb, n, prov in rows:
            pbs = f"{pb:.3f}" if pb < 0.9995 else "1.000"
            f.write(f"{disp} & {m:+.4f} & {d:+.3f} & {p:.3f} & {pbs} & {n} & "
                    f"{prov} \\\\\n")

    print(f"{'config':9s} {'dMCC':>8s} {'d':>7s} {'p':>6s} {'p_bonf':>7s} "
          f"{'n':>3s}  provenance")
    for disp, m, d, p, pb, n, prov in rows:
        pbs = f"{pb:.3f}" if pb < 0.9995 else "1.000"
        print(f"{disp:9s} {m:+8.4f} {d:+7.3f} {p:6.3f} {pbs:>7s} {n:3d}  {prov}")
    if failures:
        print("\nCANON VERIFICATION FAILED:")
        for x in failures:
            print(" ", x)
        return 1
    print(f"\nCANON VERIFICATION PASSED -- wrote {tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
