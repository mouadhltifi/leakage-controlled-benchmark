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
  processed/                   THE DERIVED-FEATURE DEPOSIT (raw-text-free layer)
    features/                  per-family per-stock-day feature tables (parquet)
      price_features.parquet             10 indicators + rolling normalization
      news_features*.parquet             FinBERT/encoder sentiment statistics
                                         (7 pooling/encoder variants) + PCA-32
      news_per_article_sentiments.parquet
      news_pca_projector.pkl             the frozen PCA used by the news variants
      social_features.parquet            17 StockTwits day-level aggregate columns
                                         (15 informative + 2 placeholders) + flag
      macro_features.parquet             model-ready macro block
    graphs/
      sector_adjacency.npy               FF12 sector graph (released taxonomy)
      dynamic/                           rolling-correlation graphs
    multimodal_dataset_v2*.h5            assembled convenience datasets
                                         (main + encoder/pooling variants; see note)
```

The harness and every driver in `REPRODUCE.md` §3b–§3d consume the **per-source
parquets + `graphs/`** under `data/processed/` (via `assemble.py`), never the
monolithic `.h5`. Every committed result CSV in `results/` was produced from
exactly these tables; `MANIFEST.sha256` fixes their hashes.

**Note on the monolithic `multimodal_dataset_v2*.h5`.** These are the original
assembled convenience datasets, retained for re-users who want one file. They
were a **pre-C2 assembly** until v1.0.11; three points about them, all now
enforced by the release gate or superseded by the canonical sources:

- Their **macro block was unlagged** — macroeconomic values sitting on dates
  before their public release, i.e. the look-ahead safeguard C2 exists to
  remove. (`multimodal_dataset_v2.h5` carried the April-2020 unemployment print
  on 2020-04-30, eight days before its 2020-05-08 BLS release; only the zero-lag
  series matched the shipped build.) All seven macro columns were rewritten in
  v1.0.11 to the publication-lagged `features/macro_features.parquet`. No
  reported number was affected — the harness reads that parquet through
  `assemble.py` and never the H5 — but anyone who trained on the H5 macro block
  before v1.0.11 trained on look-ahead data.
- Their `static_adj` is the released **FF12** graph (133 edges), corrected in
  place in v1.0.11 to match `graphs/sector_adjacency.npy`.
  (Earlier deposits carried the GICS 110-edge partition here, from before the
  v1.0.3 graph fix — never read on any results path, but inconsistent with the
  datasheet.)
- Their flat feature block carries the same two constant-zero social columns
  described below — this is a property of the social feature set itself, not of
  the H5 assembly.

`scripts/data/check_h5_consistency.py` (a release-gate step) now verifies the
macro, price, and graph content of every H5 against the canonical tables on
every tag, so this class of drift fails closed rather than shipping. Audited at
the same time and found correct: the price block matches `price_features.parquet`
exactly, and the news block is correctly aligned to the prior close (T−1).

Re-users should build from the per-source parquets + `sector_adjacency.npy`
(the `assemble.py` path), not the monolithic block.

**Label encodings differ between the two shipped label surfaces** — check
which one you are reading:

- `multimodal_dataset_v2*.h5` → `label_cls`: **+1 = up, 0 = down,
  −1 = dead-zone** (verified against `label_reg`: every `−1` row has
  |log return| < 0.005; the `0` rows average −0.018).
- `labels_direction_allday.parquet`: the same {−1, 0, +1} alphabet with
  **−1 = down, 0 = dead-zone**.

The two are permutations of each other, so silently swapping them inverts
the direction label. `labels_direction.parquet` (decisive days only, the
scoring surface the evaluator's predictions mode uses) is unaffected.

**Note on the social aggregate count.** `social_features.parquet` ships 17
aggregate columns plus the `has_social` coverage flag, but two of the 17 —
`social_sent_std` and its rolling mean `social_sent_std_w3` — are constant
`0.0` on every row, by construction: StockTwits labels messages categorically
(bullish / bearish / unlabeled), so there is no continuous per-message score to
take a within-day standard deviation over. The placeholder is stated in the
derivation code (`src/mmfp/features/social_features.py`, module docstring and
the assignment at `social_sent_std = 0.0`) and is a v1-compatibility artifact of
the column schema. **The social block is therefore 15 informative aggregates +
2 inert columns + the coverage flag.** Two consequences worth stating plainly:

- The two inert columns are dead input dimensions in every social run — they add
  no signal and no noise, so no reported result changes, but the social feature
  count should be read as 15, not 17.
- Message-level *dispersion* is not measured by these two columns. Disagreement
  is still represented in the block by `social_bullish_ratio` (and its `_w3`
  roll), which is near 0.5 under disagreement and near 0/1 under consensus.

## What ships vs. what you fetch

| Layer | Ships? | Why / how to get it |
|---|---|---|
| Derived features (`data/processed/`) | **Yes — committed in-repo** (hashes in `MANIFEST.sha256`; DOI-archived on Zenodo) | Feature-level, non-invertible derivations; raw-text-free, per-source terms preserved (DATA-STATEMENTS.md) |
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
