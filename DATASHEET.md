# Datasheet

> Condensed datasheet (after Gebru et al., "Datasheets for Datasets") for the
> benchmark's released feature datasets. Licensing detail and per-source
> exclusions live in `DATA-STATEMENTS.md`; upkeep in `MAINTENANCE.md`.

## Motivation

- **Why created:** published multi-source stock-prediction results are
  decided by evaluation conditions rather than by data. The benchmark exists
  so that the five-condition evaluation protocol is cheap to adopt: aligned
  point-in-time features for five source families, a harness that enforces
  the protocol, and reference baselines to measure new claims against.
- **Who created / funded:** the authors (Politecnico di Milano); no external
  funding specific to the dataset.

## Composition

- **Instances:** per-stock-day feature rows for **55 US large-cap equities**
  (11 sectors × 5, most-liquid selection), business days
  with features spanning **2016-01-04 → 2023-12-28** (2,011 feature days,
  ≈111k stock-days; prices are fetched from 2015-01-02 (design start 2015-02-03; the shipped fetch over-covers it) to warm up the
  trailing technical indicators), plus graph structures.
- **Universe (55 tickers, by GICS-era sector; the released sector labels are
  FF12 — see DATA-STATEMENTS):** IT: AAPL MSFT NVDA AVGO ORCL · Health:
  UNH JNJ LLY ABBV MRK · Financials: BRK-B JPM V MA BAC · Cons. Discr.:
  AMZN TSLA HD MCD NKE · Comm.: GOOGL META NFLX DIS CMCSA · Industrials:
  GE CAT UNP RTX HON · Staples: PG KO PEP COST WMT · Energy: XOM CVX COP
  SLB EOG · Utilities: NEE SO DUK SRE AEP · Materials: LIN APD SHW FCX
  NEM · Real Estate: PLD AMT CCI EQIX SPG. (Canonical list:
  `src/mmfp/data/universe.py`.)
- **Folds (expanding train; July-to-June test windows;
  `FOLD_BOUNDARIES` in `src/mmfp/data/assemble.py`):**
  F0 test 2019-07-01→2020-06-30 · F1 2020-07-01→2021-06-30 ·
  F2 2021-07-01→2022-06-30 · F3 2022-07-01→2023-06-30 ·
  F4 2023-07-01→2023-12-31 (six-month stub). Validation = the last 20% of
  each fold's training window by calendar.
- **Feature families:**
  1. **Price** — returns (1/5/20d), 10 technical indicators, rolling-normalized
     variants, next-day label components.
  2. **News** — per-stock-day FinBERT 3-class sentiment statistics (11-dim
     block) and PCA-reduced embedding features, derived from FNSPID-tagged
     articles (53/55 names have >=1 tagged article; HON and RTX are
     uncovered and zero-filled).
  3. **Social** — per-stock-day StockTwits aggregates: counts, bull/bear
     ratios, mean sentiment (17 aggregates) + a has-data flag; coverage
     ends 2022-12-30, the 2023 gap is explicit.
  4. **Macro** — five federal FRED series resampled to business days, in
     original and **publication-lag-corrected** variants (+ FOMC dates);
     volatility features are price-derived (no CBOE VIX redistribution).
  5. **Graph** — 55×55 same-sector adjacency from **public SIC→Fama-French-12
     labels** and rolling-correlation dynamic graphs (20-day, price-derived).
- **Labels:** next-day direction (sign of log return, symmetric 0.5%
  dead-zone) and next-day realized volatility.
- **No personal data:** no message text, user IDs, or article text ship —
  only day-level numeric aggregates and derived features (see
  DATA-STATEMENTS §2–3).

## Collection & preprocessing

- Prices via Yahoo Finance (regeneration script ships); news from FNSPID;
  social from the historical StockTwits API (2008–2022 archive, filtered to
  the universe/window); macro from FRED; sectors from SEC EDGAR SIC codes.
- All normalization is **rolling and train-slice-fitted** in the harness
  (nothing is normalized against future statistics at training time).
- Timing discipline: text features aligned T−1; macro series shifted to
  publication dates (CPI +30, unemployment +27, Fed Funds +23, GDP +88
  business days).
- Preprocessing code ships in `src/` and `scripts/`; every derived table is
  regenerable from upstream sources via documented scripts.

## Uses

- **Intended:** evaluating multi-source daily equity prediction claims under
  the protocol (`README.md`, the paper's task definitions); reproducibility
  studies; methodology teaching.
- **Not intended / out of scope:** live trading (the study's own economic
  analysis shows the effects are below friction); redistribution of the
  underlying raw third-party content; person-level inference of any kind
  (impossible by construction — day-level aggregates only).
- **Known limitations:** single universe (liquid US large-caps — adversarial
  to the multi-source claim by design); daily horizon; social coverage gap
  in 2023; news coverage 53/55 names (HON, RTX uncovered); sector labels are FF12-derived (the
  GICS-vs-FF12 equivalence table ships in the appendix/repo).

## Distribution

- GitHub (code + committed results + small public-domain inputs) and, on
  acceptance, an archival DOI deposit (Zenodo) of the derived feature
  tables. Code MIT; derived feature tables CC-BY-4.0; per-source terms in
  `DATA-STATEMENTS.md`.

## Maintenance

See `MAINTENANCE.md` (versioning, corrections, removal-on-request, contact).
