#!/usr/bin/env python3
"""Deposit consistency guard: the assembled model-ready H5s must embed the
RELEASED FF12 static graph, not the GICS universe partition.

Background. Two static graphs exist deliberately (see
``src/mmfp/data/loaders/graph_static.py``): the released FF12 taxonomy
(``data/processed/graphs/sector_adjacency.npy``, 133 edges), which the harness
and every reference result consume, and the GICS partition (110 edges), the
sensitivity arm. The v1.0.3 data fix corrected the standalone ``.npy`` to FF12
but the monolithic ``multimodal_dataset_v2*.h5`` were not re-assembled, so their
``static_adj`` kept the pre-fix GICS graph. v1.0.11 corrected the H5 field in
place. This guard fails closed if any H5 drifts from the released graph again,
so the "verify, don't trust" claim holds for the deposit itself.

It also checks the assembled feature blocks against the canonical per-source
tables, because fixity proves a file is unchanged, not that it is correct:

* **macro** must equal the publication-lagged ``macro_features.parquet`` — the
  C2 build. v1.0.11 fixed a pre-C2 assembly in which the eight H5s carried the
  *unlagged* macro block, i.e. look-ahead values (e.g. the April-2020
  unemployment print sitting on 2020-04-30, eight days before its BLS release).
  Only the zero-lag series (10y, VIX, FOMC flag) were unaffected, which is why
  a spot check could miss it. This leg fails closed if that ever returns.
* **price** must equal the normalised columns of ``price_features.parquet``.

Not checked here, and deliberately so: the news block is aligned to the prior
close (T-1), and the social block carries two constant-``0.0`` columns
(``social_sent_std``, ``social_sent_std_w3``) that are constant in the source
parquet too — StockTwits labels messages categorically, so there is no
continuous per-message score to take a within-day SD over. Both are properties
of the protocol and the source data, not drift, so they are reported for
information only.

Exit 0 iff every H5 carries the released FF12 graph, the C2-lagged macro block,
and the canonical price block.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "features"
NPY = ROOT / "data" / "processed" / "graphs" / "sector_adjacency.npy"
H5_GLOB = str(ROOT / "data" / "processed" / "multimodal_dataset_v2*.h5")
TOL = 1e-4


def undirected_edges(adj: np.ndarray) -> int:
    m = (np.asarray(adj) != 0).astype(int)
    np.fill_diagonal(m, 0)
    return int(m.sum() // 2)


def _attr_cols(attrs: dict, key: str) -> list:
    """Column-name lists ship as JSON strings in the H5 root attrs."""
    v = attrs[key]
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str):
        return json.loads(v)
    return [x.decode() if isinstance(x, bytes) else x for x in v]


def _match_pct(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return 100.0 * float(np.mean(np.isclose(a[m], b[m], atol=TOL)))


def main() -> int:
    ff12 = np.load(NPY)
    ff12_e = undirected_edges(ff12)
    files = sorted(glob.glob(H5_GLOB))
    if not files:
        print(f"FAIL  no H5 datasets found under {H5_GLOB}")
        return 1
    macro = pd.read_parquet(FEATURES / "macro_features.parquet")
    macro.index = pd.to_datetime(macro.index).strftime("%Y-%m-%d")
    price = pd.read_parquet(FEATURES / "price_features.parquet")
    price["Date"] = pd.to_datetime(price["Date"]).dt.strftime("%Y-%m-%d")
    price = price.set_index(["Ticker", "Date"])

    sys.path.insert(0, str(ROOT / "src"))
    from mmfp.data.universe import ALL_TICKERS

    fails: list[str] = []
    for f in files:
        with h5py.File(f, "r") as h:
            attrs = dict(h.attrs)
            adj = h["static_adj"][:]
            X = h["features"][:]
            dates = [d.decode() for d in h["dates"][:]]
            stock_idx = h["stock_idx"][:]
        name = Path(f).name
        problems = []

        if not np.array_equal(adj.astype(ff12.dtype), ff12.astype(ff12.dtype)):
            problems.append(f"static_adj is not the released FF12 graph "
                            f"({undirected_edges(adj)} edges, expected {ff12_e})")

        price_cols = _attr_cols(attrs, "price_feat_cols")
        macro_cols = _attr_cols(attrs, "macro_feat_cols")

        # macro: market-wide series broadcast per date; must be the C2 build
        aligned = macro.reindex(dates)
        worst_macro = 100.0
        for j, c in enumerate(macro_cols):
            pct = _match_pct(X[:, len(price_cols) + j].astype(float),
                             aligned[c].to_numpy(dtype=float))
            worst_macro = min(worst_macro, pct)
            if pct < 99.99:
                problems.append(
                    f"macro column {c!r} matches the publication-lagged build on "
                    f"only {pct:.3f}% of rows — the block looks unlagged "
                    f"(look-ahead); rebuild it from macro_features.parquet")

        # price: per (ticker, date)
        keys = pd.MultiIndex.from_arrays(
            [[ALL_TICKERS[i] for i in stock_idx], dates])
        pal = price.reindex(keys)
        worst_price = 100.0
        for j, c in enumerate(price_cols):
            pct = _match_pct(X[:, j].astype(float), pal[c].to_numpy(dtype=float))
            worst_price = min(worst_price, pct)
            if pct < 99.99:
                problems.append(f"price column {c!r} matches price_features.parquet "
                                f"on only {pct:.3f}% of rows")

        zero_cols = [c for c in range(X.shape[1]) if np.all(X[:, c] == 0)]
        print(f"{'OK  ' if not problems else 'FAIL'} {name:52s} "
              f"graph={undirected_edges(adj)}e  macro>={worst_macro:.2f}%  "
              f"price>={worst_price:.2f}%  const-cols(social-std)={zero_cols}")
        for p in problems:
            print(f"       - {p}")
        if problems:
            fails.append(name)

    if fails:
        print(f"\nFAIL: {len(fails)} H5(s) disagree with the canonical tables.")
        return 1
    print(f"\nAll {len(files)} assembled H5s carry the released FF12 static graph "
          f"({ff12_e} edges), the publication-lagged (C2) macro block, and the "
          f"canonical price block.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
