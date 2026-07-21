# Auditing Multi-Source Stock Prediction: An Evaluation Protocol and Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21431362.svg)](https://doi.org/10.5281/zenodo.21431362)

What would it take to believe that news, social media, macroeconomic series,
or an inter-stock graph improves next-day stock prediction? This repository
ships the answer as an instrument: an **evaluation protocol** (four
safeguards — a comparably tuned price-only baseline; chronological splits
with every
input lagged to availability; test-insulated model selection; corrected
paired statistics at the honest unit — run on a fixed universe adversarial
to the claim),
the **harness** that enforces it end to end, **availability-timed
(release-lag aligned) feature datasets** for five source families (price, news, social, macro, inter-stock
graphs; 55 liquid US large-caps, 2015–2023), and **reference baselines**
distilled from 2,468 runs (2,151 controlled). Under the protocol, no source
combination beats the tuned price-only baseline — a reference null any future
multi-source claim can be measured against. Two reproducibility audits price
individual safeguards: test-set model selection is worth +0.04 to +0.07 MCC
within a run, and a released macro-event model's from-scratch retrain at its
shipped default budget is stable under its non-temporal split but diverges
under a chronological one (its published checkpoint reproduces).

Each safeguard is textbook; the contribution is the enforced conjunction,
made cheap to adopt.

## Quick start (seconds, no GPU, no data download)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The reference null (paired stats vs the tuned price-only baseline):
python scripts/analysis/analyze_ablation.py --section rq1

# The availability-timed macro re-run:
python scripts/analysis/analyze_macrolag.py
```

Every reference-table and audit number regenerates from the committed
result files in `results/`, and the naive-anchor table from the committed
feature tables in `data/processed/` (its committed JSON carries the values). `REPRODUCE.md`
documents the full from-scratch path (environment, data regeneration,
re-runs, and the two audits).

## Submitting a model

The I/O contract — frozen folds, seeds, label rule, the submission CSV
schema, and the claim computation — is **`SUBMITTING.md`**. The claim block
comes from `scripts/analysis/evaluate_submission.py`; a worked end-to-end
example (a shipped configuration replayed as an external challenger, with
its committed claim) is `examples/demo_submission/`. Pre-registration for
the pristine post-2023 window: `MAINTENANCE.md`.

## What ships, what does not

The release is **feature-level by construction** — derived numeric features
and aggregates ship; raw third-party text and proprietary classifications do
not. Per-source terms, exclusions, regeneration paths, and the
removal-on-request policy: **`DATA-STATEMENTS.md`**. Code is MIT; derived
feature tables are CC-BY-4.0 (`LICENSE`). The paper's economic anchors
(gross Sharpe, the 50.15% balanced-accuracy floor, the cost bound) derive
step by step in **`ECONOMIC-CONTEXT.md`**.

## Provenance (exact reproducibility scope)

1. Reference numbers regenerate from committed results (verified).
2. Re-training in the shipped harness is deterministic (CPU bit-identical)
   under fixed seeds and the pinned environment, *including its BLAS/torch
   thread configuration* (verified: a run reproduces bit-exactly under the
   same thread count; changing `OMP_NUM_THREADS` shifts single runs by up
   to ~0.04 MCC through summation order while ensemble means hold — the
   reference results pin `OMP_NUM_THREADS=2`).
3. The historical core grid was produced by the study's first-generation
   codebase and bridged to this harness by an equivalence gate whose
   report ships as generated in `experiments/equivalence/`: mean-level
   agreement passes (−0.0013 MCC against a ±0.005 band); per-fold
   agreement is coarser (up to ≈0.016, with one fold marginally outside
   the gate's ±0.015 per-fold band). That coarseness is exactly why no
   reference number mixes codebases: every reference row is
   harness-native in a single pinned state, and the historical grid is
   provenance, not a source of reference rows.

## Citing

See `CITATION.cff`. The accompanying paper, *Auditing Multi-Source Stock
Prediction: An Evaluation Protocol and Benchmark* (Ltifi, Puoti,
Pittorino), is under review at the KDD 2027 Datasets & Benchmarks track
(cycle 1). The derived-feature deposit is DOI-archived on Zenodo — concept
DOI [10.5281/zenodo.21431362](https://doi.org/10.5281/zenodo.21431362),
always resolving to the latest version.

## Contact

Corresponding author: Mouadh Ltifi (mouadh.ltifi@hotmail.com), or GitHub
issues — including removal-on-request per `DATA-STATEMENTS.md`.
