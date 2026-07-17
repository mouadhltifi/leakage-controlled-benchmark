# Data — layout, what ships, and the from-scratch build order

This is the landing page `REPRODUCE.md` §3a and §3d point at. It tells you what
is in `data/`, where each input comes from, and in what order the derived
tables are built. Per-source licensing terms and the exact scope of every
redistribution decision live in [`DATA-STATEMENTS.md`](../DATA-STATEMENTS.md);
the field-level schema of every table is in the datasheet
([`DATASHEET.md`](../DATASHEET.md)).

## Layout

```
data/
  raw/
    macro/                     COMMITTED (public-domain FRED + FOMC dates)
      macro_indicators.parquet          five federal series, as published
      macro_indicators.lagged.parquet   the availability-timed (C2) variant
      fomc_dates.csv
    prices/                    NOT SHIPPED (Yahoo terms: link, don't redistribute)
  processed/                   THE DERIVED-FEATURE DEPOSIT (licensing-clean layer)
    features/                  per-family per-stock-day feature tables (parquet)
      price_features.parquet             10 indicators + rolling normalization
      news_features*.parquet             FinBERT/encoder sentiment statistics
                                         (7 pooling/encoder variants) + PCA-32
      news_per_article_sentiments.parquet
      news_pca_projector.pkl             the frozen PCA used by the news variants
      social_features.parquet            17 StockTwits day-level aggregates + flag
      macro_features.parquet             model-ready macro block
    graphs/
      sector_adjacency.npy               FF12 sector graph (released taxonomy)
      dynamic/                           rolling-correlation graphs
    multimodal_dataset_v2*.h5            assembled model-ready datasets
                                         (main + encoder/pooling variants)
```

The harness and every driver in `REPRODUCE.md` §3b–§3d consume
`data/processed/`. Every committed result CSV in `results/` was produced from
exactly these tables; `MANIFEST.sha256` fixes their hashes.

## What ships vs. what you fetch

| Layer | Ships? | Why / how to get it |
|---|---|---|
| Derived features (`data/processed/`) | **Yes — committed in-repo** (hashes in `MANIFEST.sha256`; DOI-archived on acceptance) | Feature-level, non-invertible derivations; the licensing-clean layer (DATA-STATEMENTS.md) |
| FRED macro (`data/raw/macro/`) | Yes, committed | US-government public domain |
| Raw prices | No | `python scripts/data/fetch_prices.py` (yfinance, ticker list inside) |
| Raw news text (FNSPID) | No — cannot be redistributed | Obtain from the upstream FNSPID release; see DATA-STATEMENTS.md §2 |
| Raw StockTwits messages | No — cannot be redistributed | Upstream terms; see DATA-STATEMENTS.md §3. Use the shipped aggregates as-is |
| GICS sector labels | No — proprietary (S&P/MSCI) | The released graph uses the public Fama–French-12 mapping (`scripts/data/kdd_sector_map.py`); the GICS-vs-FF12 equivalence check ships in `results/` and Appendix B of the paper |

## From-scratch build order (deep verification only)

The supported path is the shipped deposit — the committed CSVs plus
`data/processed/` regenerate every number. Rebuilding the deposit itself from
raw sources requires the two non-redistributable corpora above and runs:

1. **Prices** — `scripts/data/fetch_prices.py` → raw OHLCV →
   `src/mmfp/features/price_ta.py` (indicators, rolling normalization) →
   `features/price_features.parquet`.
2. **Macro** — `scripts/data/fetch_fred.py` (or use the committed parquets) →
   `src/mmfp/features/macro_events.py`; the availability-timed variant is
   produced by `scripts/audits/macro_lag/apply_macro_publication_lag.py`
   (REPRODUCE.md §3d).
3. **News** — FNSPID corpus → `src/mmfp/features/news_encode.py` (frozen
   encoder embeddings) → `news_aggregate.py` / `news_stats.py` (pooling
   variants, PCA-32 with the shipped projector) → `features/news_features*.parquet`.
4. **Social** — StockTwits dump → `src/mmfp/features/social_features.py` →
   `features/social_features.parquet`.
5. **Graphs** — `scripts/data/kdd_sector_map.py` (FF12 static) and
   `src/forecast/datasets/graph_precompute.py` (rolling correlation) →
   `graphs/`.
6. **Assembly** — the dataset builder in `src/mmfp/datasets/` joins the family
   tables into `multimodal_dataset_v2*.h5` on the frozen universe, window, and
   folds (`configs/`).

Rebuilds are bit-stable under the pinned environment for steps 1–2 and 5–6;
steps 3–4 depend on upstream corpus snapshots (see the exact-scope notes in
DATA-STATEMENTS.md, "Provenance and reproducibility claims").
