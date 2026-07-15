#!/usr/bin/env python3
"""Ticker -> SIC (SEC EDGAR) -> Fama-French-12 sector; emit the 55x55 same-sector
adjacency for the KDD benchmark release (GICS labels cannot ship - S&P/MSCI proprietary).

Self-check: rebuilds the GICS adjacency from mmfp's universe module in ALL_TICKERS
order and asserts it matches data/processed/graphs/sector_adjacency.npy, so the
FF12 comparison is guaranteed to be in the harness's row order.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/mouadh/Thesis/work")
from mmfp.data.universe import ALL_TICKERS, STOCK_UNIVERSE  # noqa: E402

GRAPHS = Path("/Users/mouadh/Thesis/work/data/processed/graphs")
UA = {"User-Agent": "PoliMi thesis research (contact: mouadh.ltifi@hotmail.com)"}

# Fama-French 12-industry definition by SIC range (Ken French data library).
FF12_RANGES = [
    ((100, 999), 1), ((2000, 2399), 1), ((2700, 2749), 1), ((2770, 2799), 1),
    ((3100, 3199), 1), ((3940, 3989), 1),
    ((2500, 2519), 2), ((2590, 2599), 2), ((3630, 3659), 2), ((3710, 3711), 2),
    ((3714, 3714), 2), ((3716, 3716), 2), ((3750, 3751), 2), ((3792, 3792), 2),
    ((3900, 3939), 2), ((3990, 3999), 2),
    ((2520, 2589), 3), ((2600, 2699), 3), ((2750, 2769), 3), ((3000, 3099), 3),
    ((3200, 3569), 3), ((3580, 3629), 3), ((3700, 3709), 3), ((3712, 3713), 3),
    ((3715, 3715), 3), ((3717, 3749), 3), ((3752, 3791), 3), ((3793, 3799), 3),
    ((3830, 3839), 3), ((3860, 3899), 3),
    ((1200, 1399), 4), ((2900, 2999), 4),
    ((2800, 2829), 5), ((2840, 2899), 5),
    ((3570, 3579), 6), ((3660, 3692), 6), ((3694, 3699), 6), ((3810, 3829), 6),
    ((7370, 7379), 6),
    ((4800, 4899), 7),
    ((4900, 4949), 8),
    ((5000, 5999), 9), ((7200, 7299), 9), ((7600, 7699), 9),
    ((2830, 2839), 10), ((3693, 3693), 10), ((3840, 3859), 10), ((8000, 8099), 10),
    ((6000, 6999), 11),
]
FF12_NAMES = {1: "NoDur", 2: "Durbl", 3: "Manuf", 4: "Enrgy", 5: "Chems", 6: "BusEq",
              7: "Telcm", 8: "Utils", 9: "Shops", 10: "Hlth", 11: "Money", 12: "Other"}


def ff12(sic: int) -> int:
    for (lo, hi), k in FF12_RANGES:
        if lo <= sic <= hi:
            return k
    return 12


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def same_sector_adjacency(sector_of: dict) -> np.ndarray:
    n = len(ALL_TICKERS)
    a = np.zeros((n, n), dtype=np.float32)
    for i, ti in enumerate(ALL_TICKERS):
        for j, tj in enumerate(ALL_TICKERS):
            if i != j and sector_of[ti] == sector_of[tj]:
                a[i, j] = 1.0
    return a


def main() -> None:
    # --- self-check: GICS adjacency from universe.py must equal the .npy on disk
    gics_of = {t: s for s, ts in STOCK_UNIVERSE.items() for t in ts}
    gics_adj = same_sector_adjacency(gics_of)
    on_disk = np.load(GRAPHS / "sector_adjacency.npy")
    if on_disk.shape != gics_adj.shape:
        sys.exit(f"FATAL: shape mismatch disk {on_disk.shape} vs rebuilt {gics_adj.shape}")
    if not np.array_equal((on_disk > 0).astype(np.float32) * (1 - np.eye(len(ALL_TICKERS), dtype=np.float32)),
                          gics_adj):
        # tolerate self-loops / dtype differences but not structural ones
        diff = int(np.abs((on_disk > 0).astype(int) - (gics_adj > 0).astype(int)).sum())
        off_diag = int(np.abs(((on_disk > 0).astype(int) - (gics_adj > 0).astype(int))[~np.eye(len(ALL_TICKERS), dtype=bool)].sum()))
        print(f"WARNING: disk vs rebuilt GICS differ (total cells {diff}, off-diagonal {off_diag})")
        if off_diag != 0:
            sys.exit("FATAL: off-diagonal mismatch - row order assumption broken; stop.")
        print("(diagonal-only difference - self-loops; proceeding)")
    print(f"self-check OK: disk adjacency == GICS(universe.py) in ALL_TICKERS order (n={len(ALL_TICKERS)})")

    # --- ticker -> CIK -> SIC -> FF12
    cik_map = {e["ticker"]: f'{e["cik_str"]:010d}' for e in get("https://www.sec.gov/files/company_tickers.json").values()}
    missing = [t for t in ALL_TICKERS if t not in cik_map]
    if missing:
        sys.exit(f"FATAL: no CIK for {missing}")
    ff12_of, sic_of = {}, {}
    for t in ALL_TICKERS:
        sub = get(f"https://data.sec.gov/submissions/CIK{cik_map[t]}.json")
        sic_of[t] = int(sub["sic"])
        ff12_of[t] = ff12(sic_of[t])
        time.sleep(0.15)
    ff12_adj = same_sector_adjacency(ff12_of)
    np.save(GRAPHS / "sector_adjacency_ff12.npy", ff12_adj)

    # --- report
    print("\nticker  SIC   FF12        GICS")
    for t in ALL_TICKERS:
        print(f"{t:6s}  {sic_of[t]:4d}  {FF12_NAMES[ff12_of[t]]:10s}  {gics_of[t]}")
    ge, fe = int(gics_adj.sum() // 2), int(ff12_adj.sum() // 2)
    both = np.logical_and(gics_adj > 0, ff12_adj > 0)
    either = np.logical_or(gics_adj > 0, ff12_adj > 0)
    print(f"\nGICS edges {ge} | FF12 edges {fe} | Jaccard {both.sum() / either.sum():.3f}")
    print(f"saved: {GRAPHS / 'sector_adjacency_ff12.npy'}")


if __name__ == "__main__":
    main()
