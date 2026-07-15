# A Leakage-Controlled Evaluation Protocol and Benchmark for Multi-Source Stock Prediction

What would it take to believe that news, social media, macroeconomic series,
or an inter-stock graph improves next-day stock prediction? This repository
ships the answer as an instrument: an **evaluation protocol** (five
conditions — a tuned price-only baseline; chronological splits with every
input lagged to availability; test-insulated model selection; corrected
paired statistics at the honest unit; a universe adversarial to the claim),
the **harness** that enforces it end to end, **point-in-time feature
datasets** for five source families (price, news, social, macro, inter-stock
graphs; 55 liquid US large-caps, 2015–2023), and **reference baselines**
distilled from 2,468 controlled runs. Under the protocol, no source
combination beats the tuned price-only baseline — a reference null any future
multi-source claim can be measured against. Two reproducibility audits price
individual conditions: test-set model selection is worth +0.04 to +0.07 MCC
within a run, and a non-temporal split that trains on the test period's
future collapses when made chronological.

Each condition is textbook; the contribution is the enforced conjunction,
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
result files in `results/` (the naive-anchor table additionally needs the
feature deposit; its committed JSON carries the values). `REPRODUCE.md`
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
   codebase and bridged to this harness by a validated equivalence gate
   (±0.005 MCC at the mean level; the gate report ships in
   `experiments/`). Reference tables carry a per-block provenance column.

## Citing

See `CITATION.cff`. The accompanying paper is under review at the KDD 2027
Datasets & Benchmarks track (cycle 1).
