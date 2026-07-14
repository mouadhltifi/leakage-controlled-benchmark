#!/usr/bin/env python3
"""Compute the benchmark's naive / classical anchor baselines (reference floor).

Six anchor predictors are scored on the benchmark's primary task -- next-day
direction with a 0.5% dead-zone -- over the five expanding-window folds. The
task's exact labels and fold splits are obtained from the harness itself
(``mmfp.data.assemble.assemble_fold`` on the price-only A7 config), so the test
sets here are byte-for-byte the same rows the ablation models were scored on.

Anchors
-------
1. always-up      -- predict 1 (up) for every test sample.
2. always-down    -- predict 0 (down) for every test sample.
3. random         -- Bernoulli(0.5) predictions; reported as the mean MCC /
                     accuracy over ``RANDOM_DRAWS`` draws from a seed-42 stream.
4. momentum       -- predict the sign of the last realised daily return for the
                     same stock (up iff ``return_1d > 0``); the raw (un-scaled)
                     ``return_1d`` price column, joined by (Ticker, Date).
5. reversal       -- the opposite of momentum (up iff ``return_1d <= 0``).
6. logistic-price -- ``sklearn`` logistic regression fit on the fold's train
                     split over the ten ``*_norm`` price features (the harness's
                     train-fold-fit price block), predicting the test split.

Metrics use ``sklearn.metrics`` (``matthews_corrcoef`` returns 0.0 for a
single-class prediction, which is the benchmark's convention for the always-*
rows). Per-anchor figures are the unweighted mean of the five per-fold values.

Self-verification (hard assertions; the script exits non-zero on any failure)
-----------------------------------------------------------------------------
* every fold's assembled ``n_test`` matches the harness counts in
  :data:`HARNESS_NTEST` (10,100 / 9,851 / 9,979 / 9,994 / 4,514);
* always-up and always-down MCC are exactly 0.0 on every fold;
* per-fold always-up + always-down accuracy sums to 1.0 (label cross-check);
* the random anchor's mean MCC is within +/- 0.01 of 0;
* the raw ``return_1d`` join has zero missing values on every test set;
* the logistic anchor's per-fold MCC is finite.

Outputs
-------
* ``results/analysis/naive_anchors.json`` -- committed machine-readable record
  (per-fold arrays + means + provenance + the verified fold counts).
* a LaTeX fragment (default: the KDD paper's ``tables/naive_anchors.tex``) --
  one row per anchor: ``name & mean-MCC & mean-accuracy & provenance``. Written
  best-effort; skipped with a warning if the target directory is absent.

The harness (``src/mmfp``) and the bundled feature parquets are resolved from
this repository, so the script is self-contained -- no external checkout needed.

Usage
-----
    python scripts/analysis/make_naive_anchors.py [--json PATH] [--texout PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]  # kdd-artifact/
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmfp.config.load import load_config  # noqa: E402
from mmfp.data.assemble import assemble_fold  # noqa: E402
from mmfp.data.paths import FEATURES_DIR  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

N_FOLDS = 5
RANDOM_SEED = 42
RANDOM_DRAWS = 100

#: Harness per-fold test-set sizes (committed ablation CSV ``n_test``; the
#: mmfp assembly reproduces these exactly -- asserted below).
HARNESS_NTEST: dict[int, int] = {0: 10100, 1: 9851, 2: 9979, 3: 9994, 4: 4514}

#: Emission order.
ANCHOR_ORDER = [
    "always-up",
    "always-down",
    "random",
    "momentum",
    "reversal",
    "logistic-price",
]

DEFAULT_JSON = ROOT / "results" / "analysis" / "naive_anchors.json"
DEFAULT_TEX = Path(
    "/Users/mouadh/Thesis/thesis/kdd/tables/naive_anchors.tex"
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def _return_1d_lookup() -> dict[tuple[str, str], float]:
    """Map ``(ticker, ISO-date) -> raw return_1d`` from the price parquet.

    ``return_1d`` is the last realised daily return (t-1 -> t) carried on the
    row for day t; momentum/reversal key on its raw sign, so this is the raw
    (un-standardised) column, not ``return_1d_norm``.
    """
    price = pd.read_parquet(FEATURES_DIR / "price_features.parquet")
    ds = pd.to_datetime(price["Date"]).dt.strftime("%Y-%m-%d").to_numpy()
    tk = price["Ticker"].to_numpy()
    r = price["return_1d"].to_numpy()
    return {(str(tk[i]), str(ds[i])): float(r[i]) for i in range(len(r))}


def _momentum_return(fold_art, lut: dict[tuple[str, str], float]) -> np.ndarray:
    """Per-test-sample raw ``return_1d`` aligned to the fold's test rows."""
    tickers = fold_art.tickers
    return np.array(
        [
            lut.get((tickers[si], d), np.nan)
            for si, d in zip(fold_art.test_stock_idx, fold_art.test_dates)
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Per-fold scoring
# --------------------------------------------------------------------------- #


def score_fold(
    fold: int,
    lut: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    """Return ``{anchor: (mcc, accuracy)}`` for one fold.

    The ``rng`` is shared across folds so the whole run is deterministic under
    :data:`RANDOM_SEED`.
    """
    art = assemble_fold(load_config(overrides=[f"data.fold_idx={fold}"]))
    y = art.test_labels_cls.astype(np.int64)
    n = y.shape[0]

    if n != HARNESS_NTEST[fold]:
        raise AssertionError(
            f"fold {fold}: assembled n_test={n} != harness {HARNESS_NTEST[fold]}"
        )
    if art.test_features.shape[1] != 10:
        raise AssertionError(
            f"fold {fold}: expected a 10-dim price block, got "
            f"{art.test_features.shape[1]} features"
        )

    out: dict[str, tuple[float, float]] = {}

    # always-up / always-down
    up = np.ones(n, dtype=np.int64)
    dn = np.zeros(n, dtype=np.int64)
    out["always-up"] = (matthews_corrcoef(y, up), accuracy_score(y, up))
    out["always-down"] = (matthews_corrcoef(y, dn), accuracy_score(y, dn))

    # random: mean over RANDOM_DRAWS Bernoulli(0.5) draws
    rnd_mcc, rnd_acc = [], []
    for _ in range(RANDOM_DRAWS):
        pred = rng.integers(0, 2, size=n)
        rnd_mcc.append(matthews_corrcoef(y, pred))
        rnd_acc.append(accuracy_score(y, pred))
    out["random"] = (float(np.mean(rnd_mcc)), float(np.mean(rnd_acc)))

    # momentum / reversal from the raw last daily return
    r1d = _momentum_return(art, lut)
    if np.isnan(r1d).any():
        raise AssertionError(
            f"fold {fold}: {int(np.isnan(r1d).sum())} test rows have no "
            "return_1d in the price parquet (join incomplete)"
        )
    mom = (r1d > 0).astype(np.int64)
    rev = 1 - mom  # opposite of momentum (up iff return_1d <= 0)
    out["momentum"] = (matthews_corrcoef(y, mom), accuracy_score(y, mom))
    out["reversal"] = (matthews_corrcoef(y, rev), accuracy_score(y, rev))

    # logistic regression on the harness price block (train-fold fit)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(art.train_features, art.train_labels_cls.astype(np.int64))
    lr_pred = clf.predict(art.test_features)
    lr_mcc = matthews_corrcoef(y, lr_pred)
    if not np.isfinite(lr_mcc):
        raise AssertionError(f"fold {fold}: logistic MCC is not finite ({lr_mcc})")
    out["logistic-price"] = (float(lr_mcc), float(accuracy_score(y, lr_pred)))

    # label cross-check: P(up) + P(down) == 1
    if abs(out["always-up"][1] + out["always-down"][1] - 1.0) > 1e-9:
        raise AssertionError(
            f"fold {fold}: always-up + always-down accuracy != 1 "
            f"({out['always-up'][1]} + {out['always-down'][1]})"
        )
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Naive/classical anchor baselines.")
    ap.add_argument("--json", default=str(DEFAULT_JSON), help="JSON output path.")
    ap.add_argument(
        "--texout", default=str(DEFAULT_TEX), help="LaTeX fragment output path."
    )
    args = ap.parse_args()

    lut = _return_1d_lookup()
    rng = np.random.default_rng(RANDOM_SEED)

    # anchor -> per-fold lists
    mcc: dict[str, list[float]] = {a: [] for a in ANCHOR_ORDER}
    acc: dict[str, list[float]] = {a: [] for a in ANCHOR_ORDER}
    ntest: dict[int, int] = {}

    for fold in range(N_FOLDS):
        res = score_fold(fold, lut, rng)
        ntest[fold] = HARNESS_NTEST[fold]
        for a in ANCHOR_ORDER:
            m, ac = res[a]
            mcc[a].append(round(float(m), 6))
            acc[a].append(round(float(ac), 6))

    # cross-fold means (unweighted) + top-level assertions
    mcc_mean = {a: round(float(np.mean(mcc[a])), 6) for a in ANCHOR_ORDER}
    acc_mean = {a: round(float(np.mean(acc[a])), 6) for a in ANCHOR_ORDER}

    assert all(v == 0.0 for v in mcc["always-up"]), mcc["always-up"]
    assert all(v == 0.0 for v in mcc["always-down"]), mcc["always-down"]
    assert abs(mcc_mean["random"]) <= 0.01, mcc_mean["random"]

    # ---- JSON ----
    payload = {
        "meta": {
            "task": "next-day direction, 0.5% dead-zone",
            "deadzone": 0.005,
            "n_folds": N_FOLDS,
            "assembly": "mmfp.data.assemble.assemble_fold (price-only A7 config)",
            "logistic_features": "10 *_norm price block (harness train-fold scaler)",
            "momentum_feature": "raw return_1d (sign), joined by (Ticker, Date)",
            "random_seed": RANDOM_SEED,
            "random_draws": RANDOM_DRAWS,
            "mcc": "sklearn.metrics.matthews_corrcoef (0.0 for single-class)",
            "fold_mean": "unweighted mean over the five folds",
            "generated_by": "scripts/analysis/make_naive_anchors.py",
        },
        "harness_ntest": {str(k): v for k, v in HARNESS_NTEST.items()},
        "per_fold_ntest": {str(k): v for k, v in ntest.items()},
        "anchors": {
            a: {
                "mcc_per_fold": mcc[a],
                "accuracy_per_fold": acc[a],
                "mcc_mean": mcc_mean[a],
                "accuracy_mean": acc_mean[a],
                "provenance": "computed",
            }
            for a in ANCHOR_ORDER
        },
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    # ---- LaTeX fragment ----
    tex_lines = [
        "% Generated by scripts/analysis/make_naive_anchors.py -- do not hand-edit.\n"
    ]
    for a in ANCHOR_ORDER:
        tex_lines.append(
            f"{a} & {mcc_mean[a]:+.3f} & {acc_mean[a]:.3f} & computed \\\\\n"
        )
    tex_path = Path(args.texout)
    wrote_tex = False
    if tex_path.parent.is_dir():
        tex_path.write_text("".join(tex_lines))
        wrote_tex = True

    # ---- stdout ----
    print("Per-fold test counts (assembled vs harness):")
    for fold in range(N_FOLDS):
        print(f"  fold {fold}: n_test={ntest[fold]:>6d}  harness={HARNESS_NTEST[fold]:>6d}  OK")
    print()
    print(f"{'anchor':15s} {'MCC':>8s} {'acc':>7s}   per-fold MCC")
    for a in ANCHOR_ORDER:
        pf = " ".join(f"{v:+.3f}" for v in mcc[a])
        print(f"{a:15s} {mcc_mean[a]:+8.3f} {acc_mean[a]:7.3f}   [{pf}]")
    print()
    print("LaTeX fragment:")
    print("".join(tex_lines), end="")
    print()
    print(f"Wrote {json_path}")
    if wrote_tex:
        print(f"Wrote {tex_path}")
    else:
        print(f"Skipped LaTeX (parent dir absent): {tex_path}")
    print("SELF-VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
