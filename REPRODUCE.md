# Reproduction guide

Two tiers: the **headline path** (no GPU, no large data, minutes) and the
**full from-scratch path** (GPU, data fetch, days). Most reviewers want the
headline path.

Before either path, fixity: `python3 scripts/verify_integrity.py` checks the
300-file `MANIFEST.sha256` covering the reproduction tree (code, configs,
tables, results, audits, experiments, examples) and the materialized raw
inputs (`data/raw/macro/`); top-level docs, LICENSE, and `croissant.json`
sit outside its scope (see the manifest's header line).

## 0. Environment

- **Python 3.11+** (the config system uses the stdlib `tomllib`). Verified on
  CPython 3.14.
- Create an environment and install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Headline path only (lighter): pip install numpy pandas scipy matplotlib
```

All commands below are run **from the artifact root**, with **no manual
`PYTHONPATH`** unless a command shows one. The analysis/figure scripts are pure
pandas/matplotlib over `results/` (which they auto-locate); the `run_*` and audit
driver scripts put `src/` on `sys.path` themselves before importing `mmfp` /
`forecast`. Only the test suites (§4) and the upstream-clone audits (§2) take an
explicit `PYTHONPATH`, as shown.

---

## 1. Headline path — reproduce the paper's numbers from committed results

These read only the CSVs/JSON under `results/` and finish in seconds.

### 1a. The ablation null (RQ1)

```bash
python scripts/analysis/analyze_ablation.py --section rq1     # the null table
python scripts/analysis/analyze_ablation.py                   # full analysis
```

Expect every multi-source configuration to be non-significant against the
price-only baseline (A7) after Bonferroni (p_bonf = 1.0). Sections: `rq1`,
`rq2`, `rq3`, `arch`, `overview`, `financial`, `fin_metrics`.

### 1b. All paper figures

```bash
python scripts/figures/make_paper_figures.py   # writes vector PDFs to figures/
```

Produces the per-config forest plot, the MSGCA selection-inflation diagnostic
(reported/reproduced vs validation-based, plus the per-budget within-run
inflation), the news-by-fold panel, the cross-family scatter, the parameter
budget, the volatility heatmap, Sharpe, and the encoder boxplot.

### 1c. Leakage-free macro re-run

```bash
python scripts/analysis/analyze_macrolag.py
```

Recomputes the macro-config deltas on the publication-lagged FRED features. The
null strengthens and the previous A4 near-miss disappears (A4 delta vs A7
becomes negative, d ≈ -0.25).

### 1d. Volatility-target ablation

```bash
python scripts/analysis/analyze_volatility.py
```

---

## 2. The two reproducibility audits

### 2a. MSGCA selection-inflation (+0.04 to +0.07 MCC)

The finished re-run output the figures consume is committed at
`results/analysis/msgca_diagnostic_rerun.json` (+ `_history.json`). To re-run it
yourself you need the MSGCA authors' released code and their BigData22 caches:

```bash
git clone https://github.com/changzong/MSGCA.git    # @ commit 253925d
cd MSGCA
# Run from INSIDE the clone so the script can import the authors' `model` and
# `diagnostic_bias` modules and read data/bd22_stock/cache/. PYTHONPATH=. makes
# the clone's own modules importable.
PYTHONPATH=. python /ABS/PATH/TO/scripts/audits/msgca_diagnostic_rerun.py            # full (GPU-light)
PYTHONPATH=. python /ABS/PATH/TO/scripts/audits/msgca_diagnostic_rerun.py --canary   # 5-epoch smoke
```

It compares, within a single run, selecting the reported epoch by the TEST
metric (test-selected, as released) vs by a held-out validation metric (validation-based), at epoch
budgets {200, 300, 400}. No OpenAI call is made when the shipped caches are
present.

### 2b. CAMEF non-temporal split

The CAMEF audit was run on a free Kaggle T4. The evidence pack
(`audits/camef/`) contains the master log (`PROGRESS.md`), the training harness
(`camef-train/`), the released-checkpoint evaluation (`camef-eval/`, whose
full console log including the `Evaluation results: ... MSE=0.000431...` line
is `camef-eval/camef_eval_console_full.log`), and the
split-leakage probe (`camef-splitcheck/splitcheck.py`). The headline finding:
the code's default **positional** split (file order, not time order) places
~99.7% of training events *after* the earliest test event; forcing a
**chronological** split collapses the result (MSE rises by orders of
magnitude). To reproduce:

```bash
git clone https://github.com/lakebodhi/CAMEF.git    # @ commit bd31b0c
# dataset via the authors' Google Drive link in their README (~4 GB)
# then push camef-train/ and camef-splitcheck/ to a Kaggle T4 (see PROGRESS.md)
```

A lightweight, local variant of the split probe (synthetic interleaved events
over their sample series) is `scripts/audits/camef_split_audit.py`:

```bash
CAMEF_REPO=/path/to/CAMEF python scripts/audits/camef_split_audit.py
```

---

## 3. Full from-scratch experiment re-runs (GPU, days)

> The published numbers are already in `results/`. Re-running from scratch is for
> deep verification only. The committed CSVs are the canonical record.

### 3a. Data

The derived feature parquets, graphs, and assembled HDF5 datasets the drivers
consume are **committed** under `data/processed/` (layout and provenance:
`data/README.md`; hashes: `MANIFEST.sha256`), so the re-runs below work from a
plain clone. Rebuilding that deposit itself from raw sources is the
deep-verification path: fetch/regenerate per `data/README.md` (thin fetch stubs
in `scripts/data/`; raw prices and the news/social corpora are not
redistributable, and the news configs need the FNSPID embeddings regenerated
from the upstream corpus).

### 3b. The classifier (v2) ablation

```bash
# one cell (config x fold x seed); the full grid sweeps fusion x encoder x 5 folds x 3 seeds
python scripts/run_platform.py run \
    --config configs/mmfp/experiments/A4.toml \
    --set data.fold_idx=0 --set seed=42 \
    --set fusion.strategy=concat --set model.price_encoder=lstm \
    --output /tmp/a4.csv
```

Configs `A1`-`A10` are materialized in `configs/mmfp/experiments/`
(regenerate with `python configs/mmfp/_generate_ablation_configs.py`). The
(config -> modality) map matches the published tables.

### 3c. The forecaster (v3) TFT ablation

```bash
python scripts/run_forecast.py run \
    --config configs/forecast/experiments/A7_price_only_tft.toml
python scripts/run_v3_m8_ablation.py        # full 9-config (135 runs; ~10-15 h CPU)
python scripts/run_v3_a4_seed_retest.py     # A4 8-seed vetting (demotes A4)
```

### 3d. Macro publication-lag re-run

The leakage-free macro results are **already committed** under `results/macrolag/`
(and `analyze_macrolag.py` reads them — see §1c). The steps below regenerate them
from scratch. Because this artifact ships the already-corrected macro feature
cache, `apply_macro_publication_lag.py` detects that and exits cleanly with a note;
to actually rebuild the lagged cache, start from the *original unlagged* feature
cache per `data/README.md`.

```bash
# Build the lagged macro features. On the shipped (already-corrected) cache this
# prints a note and exits 0; from the unlagged cache it writes the corrected one.
python scripts/audits/macro_lag/apply_macro_publication_lag.py

# Sanity-check the row reconstruction is faithful (replays a few recorded rows
# against the active cache and compares MCC bit-for-bit):
python scripts/audits/macro_lag/rerun_macro_lag.py --verify

# Re-run the macro configs. Writes fresh CSVs to a TEMP dir by default so it never
# clobbers the committed results; set MACROLAG_OUT to materialise them elsewhere.
python scripts/audits/macro_lag/rerun_macro_lag.py --run
MACROLAG_OUT=/tmp/macrolag python scripts/audits/macro_lag/rerun_macro_lag.py --run

# Analyse the committed leakage-free deltas (same as §1c):
python scripts/analysis/analyze_macrolag.py
```

---

## 4. Tests

```bash
PYTHONPATH=src pytest src/mmfp/tests      # config / model / leakage / reproducibility
PYTHONPATH=src pytest src/forecast/tests
```

Most tests are data-independent; integration tests skip gracefully when the
cached feature parquets are absent. `src/mmfp/tests/reproducibility/` asserts
CPU bit-identical results under a fixed seed.

## Determinism

Seeds are fixed via `mmfp/utils/seeding.py::set_all_seeds` (Python, NumPy,
PyTorch CPU/CUDA/MPS, `PYTHONHASHSEED`, deterministic cuDNN). The published
grids use 3 seeds x 5 folds per config (the v3 A4 retest adds 5 more seeds).
Results are bit-identical on CPU; GPU/MPS carry a small documented tolerance.
