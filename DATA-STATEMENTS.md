# Data statements — per-source release terms

> One statement per source family: what ships, what does not, the licensing
> basis, and how to regenerate what we cannot redistribute. The release is
> **feature-level by construction**: derived numeric features and aggregates
> ship; raw third-party text and proprietary classifications do not. Code is
> MIT; the derived-feature tables ship under CC-BY-4.0 (see `LICENSE`).
> Precedent for this pattern: VideoConviction (KDD '25 D&B; annotations and
> transcripts, not raw video), FinRL-Meta (delegate-to-source connectors),
> Qlib (fetch scripts, not corpora), StockEmotions (processed text +
> removal-on-request).

## 1. Prices (Yahoo Finance daily OHLCV, 55 US large-caps, 2015–2023)

- **Ships:** all derived price features (returns, 10 technical indicators,
  rolling-normalized variants) and labels; the ticker list and date ranges;
  a `fetch_prices.py` script that re-downloads raw OHLCV from the user's own
  access (yfinance; Stooq documented as the redistribution-friendly
  alternative).
- **Does not ship:** raw OHLCV bars as a bulk corpus.
- **Basis:** daily prices are uncopyrightable facts (*Feist*), but Yahoo's
  terms cast its service as personal/non-commercial; the
  tickers+dates+script pattern (FinRL-Meta, Qlib) avoids any redistribution
  question while losing nothing — regeneration is exact.

## 2. News (FNSPID-derived features)

- **Ships:** per-stock-day sentiment feature blocks (FinBERT 3-class
  statistics) and sentence-embedding features (PCA-reduced), plus the full
  extraction pipeline (code) and a pointer to the upstream dataset.
- **Does not ship:** any article text, headline, or per-article record.
- **Basis:** FNSPID is released CC-BY-NC-4.0, and the underlying article
  text carries the original publishers' copyright, which the FNSPID authors
  cannot relicense. Our derived numeric features are non-reconstructive;
  the NC clause is honored by shipping extraction code + pointer so users
  obtain FNSPID under its own terms. Attribution: Dong et al. (2024).
  The sentence-embedding features are day-mean-pooled, PCA-reduced, and
  non-invertible — the same aggregate class as the sentiment statistics
  (no article is recoverable from them), which is the basis for the
  CC-BY-4.0 grant on the shipped feature values as distinct from the
  NC-governed corpus.

## 3. Social (StockTwits-derived aggregates)

- **Ships:** per-stock-day aggregate features only — message counts,
  bullish/bearish ratios, mean sentiment, and a has-data indicator
  (17 features + flag), 2015–2022 coverage with the 2023 gap explicitly
  flagged.
- **Does not ship:** any message text, message ID, user ID, or per-message
  record.
- **Basis:** StockTwits' terms prohibit redistribution/extraction and there
  is no rehydration endpoint; messages carry author copyright. Day-level
  aggregates over ≥dozens of messages are non-reconstructive statistics.
- **Regeneration:** none — unlike price, macro, and graph, the social
  features are use-as-shipped: the upstream corpus cannot be re-pulled, so
  independent regeneration from raw messages is not possible. Audits of the
  social source verify the shipped aggregates, not their derivation.
- **Removal on request:** following the StockEmotions precedent — if
  StockTwits or a rights-holder objects to any aggregate, we will remove or
  further coarsen the affected columns; contact the maintainers (README).

## 4. Macroeconomic series (FRED)

- **Ships:** the small business-day-resampled CSV/parquet extracts of the
  five federal series (Fed Funds, 10Y Treasury, CPI, unemployment, GDP)
  with their publication-lag-corrected variants, plus FOMC meeting dates.
  "Citation requested" per FRED terms (U.S.-government works, public
  domain).
- **VIX:** the released volatility features are **price-derived realized
  volatility computed from the benchmark's own return series** — the
  CBOE-proprietary VIX series (FRED VIXCLS, "Copyrighted: Citation
  Required") is not redistributed. The loader accepts a user-supplied
  VIXCLS extract for exact replication of any VIX-dependent variant.
- **Basis:** federal series are public domain; VIX is CBOE property —
  substitution severs the dependency entirely.

## 5. Sector structure (public classification)

- **Ships:** per-ticker sector labels derived from **SEC EDGAR SIC codes
  mapped to the Fama-French-12 industry scheme** (both public), the mapping
  script (`kdd_sector_map.py` lineage), and the resulting 55×55 same-sector
  adjacency; plus the dynamic rolling-correlation graphs' build code
  (price-derived, no third-party dependency).
- **Does not ship:** GICS sector assignments in any form — the ticker→GICS
  mapping is S&P/MSCI proprietary licensed content, and a same-sector
  adjacency derived from it still encodes the mapping.
- **Equivalence:** the benchmark's reference results for graph
  configurations are reported under the public (FF12) sector graph, with a
  paired GICS-vs-FF12 equivalence table in the appendix (the two partitions
  differ materially — edge Jaccard 0.365 — so the equivalence is
  established empirically, not assumed).

## Provenance and reproducibility claims (exact scope)

1. Every reported reference number **regenerates from the committed result
   files** by the analysis scripts (seconds, no GPU, no data download).
2. Re-training within the shipped harness is **deterministic (CPU
   bit-identical) under fixed seeds and the pinned environment, including
   its BLAS/torch thread configuration** (`requirements.txt`; the reference
   results pin `OMP_NUM_THREADS=2` — thread count alters floating-point
   summation order, shifting single runs while ensemble means hold).
3. The historical core grid was produced by the study's first-generation
   codebase and bridged to the shipped harness by a **mean-level
   equivalence gate**: means agree within 0.0013 MCC (inside the ±0.005
   band), but per-fold agreement is coarser and the shipped gate report is
   marked `Result: FAILED` on the stricter ±0.015 per-fold band (F1:
   −0.0160) — which is exactly why no reference row mixes codebases (every
   reference row is harness-native in a single pinned state). Reference-table
   provenance is stated in each table's caption (and the anchors table's
   Source column).
