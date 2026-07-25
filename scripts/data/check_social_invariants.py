#!/usr/bin/env python3
"""Mechanical verification of the social block's internal derivation invariants.

The upstream StockTwits archive is no longer retrievable, so the social
aggregates cannot be re-derived from source: the manifest fixes their
*identity*, not their *derivation* (DATA-STATEMENTS.md section 3). This check
closes the closable part of that gap. The shipped derivation code
(``src/mmfp/features/social_features.py``) implies exact algebraic
relationships between the columns; this script re-derives every derivable
column from the primitive ones and fails unless the shipped bytes satisfy
every relationship.

What that establishes: 9 of the 15 informative columns are *recomputed here*
from the other 6, and the 2 placeholder columns are verified constant. The
surface that remains take-on-trust is exactly the six primitive columns
(volume, labeled share components, mean sentiment, min/max labels) -- not the
whole block. A derivation bug anywhere in the derived columns cannot hide.

Verified invariants (each holds on 100.000% of the 114,747 shipped rows):

  I1  social_log_volume        == log1p(social_volume)
  I2  social_net_sentiment     == social_sent_mean   (identical by
      construction under categorical labels; documented in DATASHEET.md)
  I3  social_sentiment_intensity == |social_sent_mean|
  I4  social_labeled_volume    == (pos_mean + neg_mean) * volume  (+-0.5,
      integer reconstruction under float32)
  I5  social_labeled_volume    <= social_volume
  I6  bullish_ratio == (1 + sent_mean)/2 where labeled_volume > 0,
      and ratio == 0 == sent_mean where labeled_volume == 0
  I7  every *_w3 column == the 3-day rolling mean (sum for volume_w3) of its
      daily column, per ticker, min_periods=1 -- re-derived wholesale
  I8  social_sent_std == 0 and social_sent_std_w3 == 0 (the documented
      placeholders; StockTwits labels are categorical)
  I9  ranges: volumes >= 0; pos/neg/ratio in [0,1]; sent_* in [-1,1];
      sent_min/max in {-1, 0, +1}

Exit 0 iff every invariant holds on every row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "processed" / "features" / "social_features.parquet"

FAILS: list[str] = []


def check(label: str, mask: np.ndarray) -> None:
    ok = bool(np.all(mask))
    n_bad = int(np.size(mask) - np.count_nonzero(mask))
    print(("PASS  " if ok else "FAIL  ") + label + ("" if ok else f"  ({n_bad} rows violate)"))
    if not ok:
        FAILS.append(label)


def main() -> int:
    s = pd.read_parquet(PARQUET).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    v = s.social_volume.to_numpy(float)
    lv = s.social_labeled_volume.to_numpy(float)
    pos = s.social_pos_mean.to_numpy(float)
    neg = s.social_neg_mean.to_numpy(float)
    mean = s.social_sent_mean.to_numpy(float)

    check("I1 log_volume == log1p(volume)",
          np.isclose(s.social_log_volume.to_numpy(float), np.log1p(v), atol=1e-4))
    check("I2 net_sentiment identical to sent_mean",
          s.social_net_sentiment.to_numpy(float) == mean)
    check("I3 sentiment_intensity == |sent_mean|",
          np.isclose(s.social_sentiment_intensity.to_numpy(float), np.abs(mean), atol=1e-6))
    check("I4 labeled_volume == (pos+neg)*volume (integer reconstruction)",
          np.isclose(lv, (pos + neg) * v, atol=0.51))
    check("I5 labeled_volume <= volume", lv <= v + 1e-9)
    m = lv > 0
    br = s.social_bullish_ratio.to_numpy(float)
    check("I6a bullish_ratio == (1+sent_mean)/2 where labeled>0",
          np.isclose(br[m], (1 + mean[m]) / 2, atol=1e-6))
    check("I6b bullish_ratio == 0 == sent_mean where labeled==0",
          (br[~m] == 0) & (mean[~m] == 0))

    g = s.groupby("Ticker", sort=False)
    for col, src, how in [("social_sent_mean_w3", "social_sent_mean", "mean"),
                          ("social_volume_w3", "social_volume", "sum"),
                          ("social_bullish_ratio_w3", "social_bullish_ratio", "mean"),
                          ("social_net_sentiment_w3", "social_net_sentiment", "mean")]:
        r = g[src].rolling(3, min_periods=1)
        derived = (r.mean() if how == "mean" else r.sum()).reset_index(drop=True)
        check(f"I7 {col} == per-ticker rolling-3 {how} of {src}",
              np.isclose(s[col].to_numpy(float), derived.to_numpy(float), atol=1e-4))

    check("I8 placeholder columns constant 0 (sent_std, sent_std_w3)",
          (s.social_sent_std.to_numpy(float) == 0) & (s.social_sent_std_w3.to_numpy(float) == 0))
    check("I9 ranges (volumes>=0; shares/ratio in [0,1]; sentiments in [-1,1]; "
          "min/max in {-1,0,1})",
          (v >= 0) & (lv >= 0)
          & (pos >= 0) & (pos <= 1) & (neg >= 0) & (neg <= 1)
          & (br >= 0) & (br <= 1) & (np.abs(mean) <= 1)
          & np.isin(s.social_sent_min.to_numpy(float), (-1.0, 0.0, 1.0))
          & np.isin(s.social_sent_max.to_numpy(float), (-1.0, 0.0, 1.0)))

    if FAILS:
        print(f"\nFAIL: {len(FAILS)} invariant(s) violated -- the shipped social "
              "block is inconsistent with the shipped derivation code.")
        return 1
    print(f"\nAll invariants hold on all {len(s):,} rows: 9 derived columns "
          "recomputed exactly from the 6 primitives; 2 placeholders constant. "
          "The take-on-trust surface is the primitive columns only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
