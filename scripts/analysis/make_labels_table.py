#!/usr/bin/env python3
"""Emit the frozen test-set labels table the predictions-mode evaluator scores against.

Reads the committed deposit (`data/processed/multimodal_dataset_v2.h5`) and writes
`data/processed/labels_direction.parquet` with one row per scoreable test example:

    date (str, YYYY-MM-DD) | ticker (str) | fold_idx (int) | y_true (int, -1 or +1)

Scoreable = test-window row with |next-day return| >= the 0.5% dead-zone
(`|label_reg| >= 0.005`); dead-zone days are excluded from scoring, exactly as in
the harness. The script self-checks the per-fold row counts against the frozen
`n_test` values carried by every shipped baseline row (10100/9851/9979/9994/4514)
and refuses to write on any mismatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import json

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
H5 = ROOT / "data" / "processed" / "multimodal_dataset_v2.h5"
OUT = ROOT / "data" / "processed" / "labels_direction.parquet"

DEADZONE = 0.005
FOLDS = [
    ("2019-07-01", "2020-06-30"),
    ("2020-07-01", "2021-06-30"),
    ("2021-07-01", "2022-06-30"),
    ("2022-07-01", "2023-06-30"),
    ("2023-07-01", "2023-12-31"),
]
EXPECTED_N_TEST = {0: 10100, 1: 9851, 2: 9979, 3: 9994, 4: 4514}


def main() -> int:
    with h5py.File(H5, "r") as h:
        reg = h["label_reg"][:]
        dates = h["dates"][:].astype(str)
        stock_idx = h["stock_idx"][:]
        raw = h.attrs["tickers"]
        if isinstance(raw, bytes):
            raw = raw.decode()
        # the H5 stores list-valued attrs as JSON strings
        tickers = np.array(json.loads(raw) if isinstance(raw, str)
                           else [t.decode() if isinstance(t, bytes) else str(t)
                                 for t in raw])
        if len(tickers) != 55:
            sys.exit(f"FATAL: decoded {len(tickers)} tickers, expected 55")
    frames = []
    for i, (a, b) in enumerate(FOLDS):
        m = (dates >= a) & (dates <= b) & (np.abs(reg) >= DEADZONE)
        n = int(m.sum())
        if n != EXPECTED_N_TEST[i]:
            sys.exit(f"FATAL: fold {i} scoreable rows {n} != frozen n_test "
                     f"{EXPECTED_N_TEST[i]} — refusing to write")
        frames.append(pd.DataFrame({
            "date": dates[m],
            "ticker": tickers[stock_idx[m]],
            "fold_idx": i,
            "y_true": np.where(reg[m] > 0, 1, -1).astype("int8"),
        }))
        print(f"F{i}: {n} rows (matches frozen n_test)")
    df = pd.concat(frames, ignore_index=True)
    dup = df.duplicated(subset=["date", "ticker"]).sum()
    if dup:
        sys.exit(f"FATAL: {dup} duplicate (date, ticker) rows")
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df)} rows, "
          f"{df['y_true'].eq(1).mean():.3f} up-share")

    # All-day three-class labels (planned-extension task): every test-window
    # row, y_true in {-1, 0, +1} with 0 = the dead-zone class. Reference
    # results for this task are a planned extension; the labels ship so the
    # task is runnable today.
    frames3 = []
    with h5py.File(H5, "r") as h:
        reg = h["label_reg"][:]
        dates = h["dates"][:].astype(str)
        stock_idx = h["stock_idx"][:]
    for i, (a, b) in enumerate(FOLDS):
        m = (dates >= a) & (dates <= b)
        y = np.zeros(int(m.sum()), dtype="int8")
        y[reg[m] >= DEADZONE] = 1
        y[reg[m] <= -DEADZONE] = -1
        frames3.append(pd.DataFrame({
            "date": dates[m], "ticker": tickers[stock_idx[m]],
            "fold_idx": i, "y_true": y}))
    df3 = pd.concat(frames3, ignore_index=True)
    out3 = OUT.with_name("labels_direction_allday.parquet")
    df3.to_parquet(out3, index=False)
    print(f"wrote {out3.relative_to(ROOT)}: {len(df3)} rows "
          f"(three-class; {int((df3['y_true'] == 0).sum())} dead-zone)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
