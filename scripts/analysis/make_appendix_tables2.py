#!/usr/bin/env python3
r"""Appendix table fragments for Paper B (KDD D&B) -- two more \input tables:

  appendix_volatility.tex  per-configuration paired Delta-RMSE vs the price-only
                           (A7) baseline on the VOLATILITY-target task. Source is
                           the leakage-free re-run
                           results/macrolag/ablation_macrolag_volatility_{ff,lstm}.csv
                           ("macrolagvol_" runs) -- the same harness and pinned
                           env as reference Table 3; the older
                           results/volatility/ campaign (all 9 configs, but
                           pre-leakage-fix) is deliberately NOT used so the
                           appendix stays inside the paper's leakage-free frame.
                           The vol re-run covered the macro-family configs
                           (P+M, P+N+M, P+M+S, P+M+G, P+N+M+S -- every config that
                           carries the publication-lagged macro block) plus the
                           A7 baseline, so k=5. Concat estimand, paired within
                           (fold, seed, arch): 30 cells = 5 folds x 3 seeds x 2
                           arch, exactly as make_reference_table_v2. Delta-RMSE =
                           config - A7; NEGATIVE = lower error = better. Columns:
                           config | Delta-RMSE | fold-block 95% CI | p_fold (n=5)
                           | p_bonf (k=5). Statistics identical to Table 3:
                           fold-block bootstrap CI (fold_block_ci, imported),
                           t-test on the 5 per-fold mean deltas, Bonferroni k=5.

  appendix_datastats.tex   per-fold dataset statistics: expanding-window test
                           window | n_train | n_val | n_test | decisive fraction
                           | up-share. The test window is the exact YYYY-MM span
                           from mmfp.data.assemble.FOLD_BOUNDARIES (imported, one
                           source of truth) -- month-bounded expanding folds, NOT
                           calendar years; F0-F3 are 12-month windows and F4 is
                           the 2023-07->2023-12 six-month stub (self-checked).
                           n_train/n_val/n_test are the median over
                           the A7 native_core rows (constant per fold). n_test and
                           the up-share are ALSO recomputed live from the harness
                           assembly (mmfp.data.assemble.assemble_fold on A7) so
                           they are byte-identical to the rows the models were
                           scored on and cross-check the CSV; the decisive
                           fraction = n_test / (55 tickers x trading days in the
                           fold's test window) (the price grid is complete, so the
                           denominator is the price-parquet row count in-window).
                           REGENERATING the up-share / decisive columns needs the
                           local feature deposit (data/processed/), the same
                           requirement as make_naive_anchors -- from CSVs alone
                           only the n_* and period columns are reproducible.

Both fragments are booktabs body rows only, terminated by \bottomrule INSIDE the
fragment (an \input inside a tabular breaks if \bottomrule follows the \input);
the paper supplies the table environment, column spec, header row and caption.
Each fragment self-validates before it is written; failure exits non-zero and
prints the contradiction. Prints one LEDGER line per row, plus a compute ledger
(runs + CPU-hours summed across every results/*.csv). Do not hand-edit outputs.
"""
from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Reuse the reference table's roots + the exact fold-block bootstrap so the
# volatility CI is the identical honest-unit interval used for Table 3.
from make_reference_table_v2 import R, ROOT, fold_block_ci

# --------------------------------------------------------------------------- #
# 1. volatility-target paired contrasts vs price-only (leakage-free re-run)
# --------------------------------------------------------------------------- #
VOL_BASE = "A7"
VOL_CONFIGS = [  # display, config id -- macro-family order (matches ref table)
    ("P+M", "A4"),
    ("P+N+M", "A2"),
    ("P+M+S", "A5"),
    ("P+M+G", "A9"),
    ("P+N+M+S", "A1"),
]


def _vol_cells(cfg: str) -> pd.DataFrame:
    """Concat vol cells for one config, indexed by (fold, seed, arch)."""
    frames = []
    for arch in ("ff", "lstm"):
        d = pd.read_csv(R / f"macrolag/ablation_macrolag_volatility_{arch}.csv")
        d = d[d.experiment_name.str.match(rf"macrolagvol_{cfg}_(?:ff|lstm)_concat_")]
        d = d.copy()
        d["arch"] = arch
        frames.append(d[["fold_idx", "seed", "arch", "vol_rmse", "vol_r2"]])
    df = pd.concat(frames, ignore_index=True)
    return df.set_index(["fold_idx", "seed", "arch"]).sort_index()


def build_volatility() -> list[str]:
    base = _vol_cells(VOL_BASE)
    assert len(base) == 30, f"A7 vol cells {len(base)} != 30"
    k = len(VOL_CONFIGS)
    base_r2 = float(base.vol_r2.mean())

    lines, ledger, contradictions = [], [], []
    all_mean_r2_neg = base_r2 < 0
    for disp, cfg in VOL_CONFIGS:
        c = _vol_cells(cfg)
        idx = c.index.intersection(base.index)
        assert len(idx) == 30, f"{cfg}: matched {len(idx)} != 30"
        deltas = (c.vol_rmse.loc[idx] - base.vol_rmse.loc[idx]).sort_index()
        m = float(deltas.mean())
        sd = float(deltas.std(ddof=1))
        d_eff = m / sd if sd > 0 else 0.0
        lo, hi = fold_block_ci(deltas)
        fold_means = deltas.groupby("fold_idx").mean().to_numpy()
        _, p_fold = stats.ttest_1samp(fold_means, 0.0)
        p_fold = float(p_fold)
        pb = min(1.0, k * p_fold)
        cfg_r2 = float(c.vol_r2.mean())
        all_mean_r2_neg = all_mean_r2_neg and (cfg_r2 < 0)

        # "beats price-only" on RMSE = LOWER error = negative delta, made
        # significant at the corrected fold level OR a fold-block CI fully < 0.
        beats = (m < 0 and pb < 0.05) or (hi < 0)
        if beats:
            contradictions.append(
                f"  CRITICAL: {disp} (cfg {cfg}) dRMSE={m:+.5f} p_bonf={pb:.3f} "
                f"CI=[{lo:+.5f}, {hi:+.5f}] -> SIGNIFICANTLY BEATS price-only "
                f"on the volatility task")

        pbs = f"{pb:.3f}" if pb < 0.9995 else "1.000"
        lines.append(f"{disp} & {m:+.5f} & [{lo:+.5f}, {hi:+.5f}] & "
                     f"{p_fold:.3f} & {pbs} \\\\")
        ledger.append(f"LEDGER vol: {disp:9s} dRMSE={m:+.5f} "
                      f"CI=[{lo:+.5f}, {hi:+.5f}] d={d_eff:+.3f} "
                      f"p_fold={p_fold:.3f} p_bonf={pbs} vol_r2={cfg_r2:+.4f} "
                      f"n={len(idx)}")

    for line in ledger:
        print(line)
    print(f"LEDGER vol-R2: baseline A7 mean vol_r2={base_r2:+.4f} "
          f"({'negative' if base_r2 < 0 else 'NON-negative'}); "
          f"every per-config MEAN vol_r2 negative={all_mean_r2_neg} "
          f"(no config explains volatility variance -- R2<0 everywhere on the mean)")

    if contradictions:
        print("\n".join(contradictions))
        sys.exit("SELF-CHECK FAIL (volatility): a configuration significantly "
                 "beats price-only on the volatility task -- the paper's null is "
                 "CONTRADICTED. See the CRITICAL line(s) above; do NOT suppress.")
    print(f"  volatility self-check OK: none of k={k} macro-family configs beats "
          f"price-only on RMSE (all p_bonf reported above; no fold-block CI < 0).")
    return lines


# --------------------------------------------------------------------------- #
# 2. per-fold dataset statistics
# --------------------------------------------------------------------------- #
N_TICKERS = 55
#: Expected inclusive test-window length per fold (months): F0-F3 full year,
#: F4 the 2023-07->2023-12 six-month stub. Self-checked against FOLD_BOUNDARIES.
EXPECTED_MONTHS = {0: 12, 1: 12, 2: 12, 3: 12, 4: 6}


def _thousands(n: int) -> str:
    """LaTeX-safe thousands separator (text-mode {,} renders a plain comma)."""
    return f"{int(n):,}".replace(",", "{,}")


def _fold_boundaries() -> list[dict]:
    """The harness's exact fold windows (mmfp.data.assemble.FOLD_BOUNDARIES) --
    the one source of truth for the test-window column; no data deposit needed."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from mmfp.data.assemble import FOLD_BOUNDARIES
    except Exception as exc:  # pragma: no cover - environment guard
        sys.exit(f"SELF-CHECK FAIL (datastats): cannot import FOLD_BOUNDARIES "
                 f"from mmfp.data.assemble ({exc}).")
    return FOLD_BOUNDARIES


def _month_span(test_start: str, test_end: str) -> tuple[str, int]:
    """('YYYY-MM--YYYY-MM' label, inclusive month count) for a test window."""
    ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
    label = f"{ts.strftime('%Y-%m')}--{te.strftime('%Y-%m')}"
    months = (te.year - ts.year) * 12 + (te.month - ts.month) + 1
    return label, months


def _harness_perfold() -> dict[int, dict]:
    """Per-fold (n_test, up-share, decisive-grid) straight from the A7 harness
    assembly -- byte-identical to the scored rows. Needs data/processed/ locally
    (same requirement as make_naive_anchors)."""
    warnings.filterwarnings("ignore")
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from mmfp.config.load import load_config
        from mmfp.data.assemble import assemble_fold
        from mmfp.data.paths import FEATURES_DIR
    except Exception as exc:  # pragma: no cover - environment guard
        sys.exit(f"SELF-CHECK FAIL (datastats): cannot import the mmfp harness "
                 f"({exc}). The up-share / decisive columns need src/mmfp + the "
                 f"local feature deposit; regenerate from the data deposit.")

    price_path = FEATURES_DIR / "price_features.parquet"
    if not price_path.exists():
        sys.exit(f"SELF-CHECK FAIL (datastats): {price_path} absent. The "
                 f"up-share / decisive columns need the local data deposit "
                 f"(data/processed/); regenerate there, as make_naive_anchors does.")
    price_d = pd.to_datetime(pd.read_parquet(price_path)["Date"])

    out: dict[int, dict] = {}
    for f in range(5):
        art = assemble_fold(load_config(overrides=[f"data.fold_idx={f}"]))
        y = np.asarray(art.test_labels_cls).astype(int)
        dts = pd.to_datetime(pd.Series(art.test_dates))
        distinct = int(dts.dt.normalize().nunique())
        grid = int(((price_d >= dts.min()) & (price_d <= dts.max())).sum())
        out[f] = {"n_test": int(len(y)), "up": float(y.mean()),
                  "distinct": distinct, "grid": grid,
                  "dmin": dts.min(), "dmax": dts.max()}
    return out


def build_datastats() -> list[str]:
    nc = pd.concat(
        [pd.read_csv(R / f"native_core/native_core_{a}.csv") for a in ("ff", "lstm")],
        ignore_index=True)
    a7 = nc[nc.experiment_name.str.match(r"natcore_A7_")]
    assert len(a7) == 30, f"A7 native_core rows {len(a7)} != 30"
    med = a7.groupby("fold_idx")[["n_train", "n_val", "n_test"]].median().round().astype(int)

    fb = _fold_boundaries()
    if len(fb) != 5:
        sys.exit(f"SELF-CHECK FAIL (datastats): FOLD_BOUNDARIES has {len(fb)} "
                 f"folds, expected 5")

    hp = _harness_perfold()
    lines, ledger = [], []
    tot_ntest, up_weighted = 0, 0.0
    for f in range(5):
        ntr, nva, nte = (int(med.loc[f, c]) for c in ("n_train", "n_val", "n_test"))
        h = hp[f]
        if nte != h["n_test"]:
            sys.exit(f"SELF-CHECK FAIL (datastats): fold {f} CSV n_test={nte} != "
                     f"harness-assembled n_test={h['n_test']}")

        ts, te = fb[f]["test_start"], fb[f]["test_end"]
        window, months = _month_span(ts, te)
        # the fold's declared window must be the length we expect (F4 = 6-mo stub)
        if months != EXPECTED_MONTHS[f]:
            sys.exit(f"SELF-CHECK FAIL (datastats): fold {f} window {window} spans "
                     f"{months} months, expected {EXPECTED_MONTHS[f]}")
        # and the harness-observed test dates must sit inside that declared window
        if not (pd.Timestamp(ts) <= h["dmin"] and h["dmax"] <= pd.Timestamp(te)):
            sys.exit(f"SELF-CHECK FAIL (datastats): fold {f} observed test dates "
                     f"[{h['dmin'].date()}, {h['dmax'].date()}] fall outside the "
                     f"declared window {window}")

        dec = h["n_test"] / h["grid"]
        up = h["up"]
        tot_ntest += nte
        up_weighted += up * nte
        lines.append(f"F{f} & {window} & {_thousands(ntr)} & "
                     f"{_thousands(nva)} & {_thousands(nte)} & {dec:.3f} & "
                     f"{up:.3f} \\\\")
        ledger.append(f"LEDGER datastats: F{f} {window:16s} ({months:>2}mo) "
                      f"n_train={ntr:>6} n_val={nva:>6} n_test={nte:>6} "
                      f"decisive={dec:.3f} up={up:.3f} "
                      f"(grid={h['grid']} = {N_TICKERS}x{h['distinct']} days)")

    # exactly one fold (F4) is the short stub
    n_stub = sum(1 for f in range(5) if EXPECTED_MONTHS[f] != 12)
    if n_stub != 1 or EXPECTED_MONTHS[4] != 6:
        sys.exit(f"SELF-CHECK FAIL (datastats): expected exactly one 6-month stub "
                 f"(F4); got {n_stub} short fold(s)")

    pooled_up = up_weighted / tot_ntest
    for line in ledger:
        print(line)
    print(f"LEDGER datastats: sum n_test={tot_ntest} (~44k), "
          f"pooled up-share={pooled_up:.4f}")

    if not (40_000 <= tot_ntest <= 48_000):
        sys.exit(f"SELF-CHECK FAIL (datastats): sum n_test={tot_ntest} is off the "
                 f"~44k order of magnitude")
    if not (0.52 <= pooled_up <= 0.54):
        sys.exit(f"SELF-CHECK FAIL (datastats): pooled up-share={pooled_up:.4f} "
                 f"not in [0.52, 0.54] (always-up anchor is 0.528)")
    print(f"  datastats self-check OK: sum n_test={tot_ntest}, pooled "
          f"up-share={pooled_up:.4f} in [0.52, 0.54] (matches the 0.528 anchor).")
    return lines


# --------------------------------------------------------------------------- #
# 3. total compute across every shipped results CSV (LEDGER only, no file)
# --------------------------------------------------------------------------- #
def compute_ledger() -> None:
    total_s, total_runs, skipped = 0.0, 0, []
    for p in sorted(glob.glob(str(R / "**" / "*.csv"), recursive=True)):
        d = pd.read_csv(p)
        col = next((c for c in ("elapsed_seconds", "elapsed_s") if c in d.columns), None)
        if col is None:
            skipped.append(Path(p).relative_to(ROOT))
            continue
        s = pd.to_numeric(d[col], errors="coerce")
        total_s += float(s.sum())
        total_runs += int(s.notna().sum())
    print(f"LEDGER compute: {total_runs} runs across every results/*.csv, "
          f"{total_s / 3600:.1f} total CPU-hours "
          f"(sum of elapsed_seconds/elapsed_s)")
    if skipped:
        print(f"  NOTE: {len(skipped)} CSV(s) had no elapsed column and were "
              f"excluded: {', '.join(str(s) for s in skipped)}")


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

    print("== volatility-target paired contrasts (leakage-free re-run) ==")
    vol_rows = build_volatility()
    _write(outdir / "appendix_volatility.tex",
           "% Generated by scripts/analysis/make_appendix_tables2.py -- "
           "volatility-target\n% paired Delta-RMSE vs price-only (A7), "
           "leakage-free re-run (macrolagvol_), concat\n% estimand paired within "
           "(fold,seed,arch); Delta-RMSE<0 = lower error = better;\n% p_bonf "
           "Bonferroni k=5 (macro-family configs). Do not hand-edit.\n",
           vol_rows)

    print("\n== per-fold dataset statistics ==")
    ds_rows = build_datastats()
    _write(outdir / "appendix_datastats.tex",
           "% Generated by scripts/analysis/make_appendix_tables2.py -- per-fold "
           "dataset stats:\n% test window (YYYY-MM span, from "
           "mmfp.data.assemble.FOLD_BOUNDARIES; F4 is the\n% 2023-07--2023-12 "
           "6-month stub) | n_train | n_val | n_test (median over A7\n% "
           "native_core) | decisive fraction (n_test / 55 tickers x in-window "
           "trading\n% days) | up-share among decisive test days. n_test + "
           "up-share recomputed from\n% the mmfp harness assembly on A7 (needs "
           "the local data deposit to regenerate).\n% Do not hand-edit.\n",
           ds_rows)

    print("\n== compute ledger ==")
    compute_ledger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
