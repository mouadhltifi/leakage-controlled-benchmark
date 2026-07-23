# Phase 3-R Experiment Matrix

**Author**: Analyst
**Date**: 2026-02-13
**Status**: Pending RQG review

## Overview

Phase 3-R screens 129 experiments across 6 studies to identify which hyperparameters
and data processing choices meaningfully affect performance before committing to a
full ablation re-run. All studies use **concat fusion** to isolate effects from
fusion strategy interactions.

**Baseline reference**: A7/concat on v1 dataset = MCC 0.014 +/- 0.022 (15 runs, 3 seeds x 5 folds)

**Dataset versions**:
- `v2_nonews`: Assembled with Phase 2-R fixes (deadzone, macro split, no news forward-fill artifacts). Excludes news.
- `v2_full`: Same fixes plus lagged news (t-1), corrected social features. Includes all modalities.

**Fold selection for screening (folds 0, 2, 4)**:
- F0 (2019-07 to 2020-06): Pre-COVID + early COVID — median difficulty
- F2 (2021-07 to 2022-06): Bull-to-Bear — hardest fold (MCC=-0.028 baseline)
- F4 (2023-07 to 2023-12): AI Rally — short fold, mild difficulty
- Rationale: covers easy/hard/short regimes. F1 (easiest, MCC=0.050) and F3 (moderate) excluded to save budget.
  If a change doesn't help on the hardest fold (F2), it's unlikely to help overall.

**Success criteria**: An improvement is "promising" if mean delta MCC > +0.005 across
the 3 screening folds. It is "significant" if paired t-test p < 0.10 (relaxed for
screening, will tighten to p < 0.05 for full batch).

---

## Study 1: Movement Threshold (HIGHEST PRIORITY)

### Hypothesis
Filtering out near-zero-return days (where movement direction is essentially random
noise) will increase signal-to-noise ratio and improve MCC. StockNet uses
[-0.5%, +0.55%] threshold, filtering 38.7% of samples.

### Rationale
- Current deadzone = 0.0: ALL samples included, including |return| < 0.001 where
  direction is meaningless
- Phase 1-R threshold analysis (Task #3) quantified sample loss at each threshold
- This is the highest-priority experiment because it changes what the model is
  TRYING to predict (the label definition), not just how it learns

### Design
| Factor | Values |
|--------|--------|
| Config | A7 (price only) |
| Fusion | concat |
| Dataset | v2_nonews |
| Threshold | {0, 0.003, 0.005, 0.01} |
| Seeds | {42, 123, 456} |
| Folds | {0, 1, 2, 3, 4} |
| **Total** | **4 x 3 x 5 = 60 runs** |

### Why all 5 folds and 3 seeds
This is the highest-priority study and determines the label definition for all
subsequent experiments. It needs full statistical power (15 paired observations per
threshold comparison) to make a confident decision.

### Baseline
A7/concat/threshold=0 on v2_nonews (15 runs). This also serves as the regression test
for v2 dataset assembly — results should be close to v1 MCC=0.014.

### Success criteria
- Primary: threshold with highest mean MCC across all 15 runs
- Secondary: threshold must not reduce sample count below 50% (too few training examples)
- Significance: paired t-test vs threshold=0, p < 0.05
- Decision: winning threshold becomes the default for all subsequent studies

### What to watch
- MCC improvement vs sample size trade-off: if MCC doubles but we lose 50% of data, net effect on model training may be negative
- Per-fold variation: threshold may help more in sideways markets (F0, F3) where movements are smaller
- Regression check: threshold=0 on v2 should match v1 baseline within +/- 0.005

### CLI template
```bash
python -m src.training.experiment_runner_v2 \
    --h5-path data/processed/v2_nonews.h5 \
    --config-id A7 --fusion-type concat \
    --deadzone {THRESHOLD} --seed {SEED} --fold-idx {FOLD} \
    --output-csv experiments/phase3r/study1_threshold.csv
```

---

## Study 2: Training Dynamics

### Hypothesis
Batch size and early stopping patience interact: larger batches need more patience
to converge because gradient estimates are smoother (less noisy). The current
batch=64 with patience=20 may be stopping too early for larger batches, or batch=64
itself may produce too noisy gradients for stable training.

### Rationale
- MSGCA uses batch=1024, 200 epochs. Our models stop at ~41 epochs.
- Round 1 tried batch=256 + patience=40 bundled with capacity increase — confounded.
- Need to isolate batch size and patience effects independently.
- Literature suggests batch size affects generalization gap (Keskar et al., 2017).

### Design
| Factor | Values |
|--------|--------|
| Config | A4 (price + macro) |
| Fusion | concat |
| Dataset | v2_nonews (with best threshold from Study 1) |
| Batch size | {64, 128, 256, 512} |
| Patience | {10, 20, 40} |
| Seed | 42 |
| Folds | {0, 2, 4} |
| **Total** | **4 x 3 x 1 x 3 = 36 runs** |

### Why A4 (not A7)
A7 (price-only) doesn't use fusion at all. A4 (price+macro) is the simplest
multimodal config and lets us see how training dynamics affect the fusion mechanism.
If batch size changes improve A4, they'll likely improve A7 too (but not vice versa).

### Baseline
A4/concat/batch=64/patience=20 on v2_nonews with best threshold (3 runs: folds 0,2,4).

### Success criteria
- Best (batch, patience) combination has mean MCC > baseline by +0.005
- Val-test gap < baseline (less overfitting)
- Training should not hit max_epochs (150) — if it does, patience may be too high

### What to watch
- Batch=512 may underfit (too few gradient updates per epoch with ~2K training samples per fold)
- Patience=40 may overfit (too many epochs past best validation)
- Check epochs_trained: is the model actually training longer with more patience?

### CLI template
```bash
python -m src.training.experiment_runner_v2 \
    --h5-path data/processed/v2_nonews.h5 \
    --config-id A4 --fusion-type concat \
    --batch-size {BATCH} --patience {PATIENCE} \
    --deadzone {BEST_THRESHOLD} --seed 42 --fold-idx {FOLD} \
    --output-csv experiments/phase3r/study2_training_dynamics.csv
```

---

## Study 3: Gradient Clipping

### Hypothesis
The current gradient clipping at max_norm=1.0 may be too aggressive, limiting the
model's ability to learn from informative gradients. Relaxing the clip to 5.0 or 10.0
may allow faster and more complete learning.

### Rationale
- Default grad_clip=1.0 was set without justification
- Literature varies widely (PyTorch default is no clipping; many papers use 5.0 or 10.0)
- If gradients are rarely >1.0, clipping has no effect and this study is moot
- If gradients frequently exceed 1.0, clipping may be preventing useful updates

### Design
| Factor | Values |
|--------|--------|
| Config | A4 |
| Fusion | concat |
| Dataset | v2_nonews |
| Grad clip | {1.0, 5.0, 10.0} |
| Seed | 42 |
| Folds | {0, 2, 4} |
| **Total** | **3 x 1 x 3 = 9 runs** |

### Baseline
A4/concat/clip=1.0 on v2_nonews (3 runs: same as Study 2 baseline at batch=64, patience=20).

### Success criteria
- Mean MCC improvement > +0.003
- No instability (check for NaN losses, wild MCC variance)

### What to watch
- If clip=5.0 and clip=10.0 produce identical results to clip=1.0, gradients never exceed 1.0 and this parameter doesn't matter
- If clip=10.0 produces unstable training (high variance), keep clip=1.0

### CLI template
```bash
python -m src.training.experiment_runner_v2 \
    --h5-path data/processed/v2_nonews.h5 \
    --config-id A4 --fusion-type concat \
    --grad-clip {CLIP} --deadzone {BEST_THRESHOLD} --seed 42 --fold-idx {FOLD} \
    --output-csv experiments/phase3r/study3_grad_clip.csv
```

---

## Study 4: Architecture Sensitivity (Hidden Dimension)

### Hypothesis
With the prediction head bug fixed (now scales with hidden_dim instead of hardcoded
32), the relationship between hidden_dim and performance may differ from Round 1.
Smaller hidden_dim (32) may prevent overfitting; larger (128) may capture more
complex patterns IF the prediction head bottleneck is removed.

### Rationale
- Round 1 tested hidden_dim=128 but with hardcoded prediction head dim=32 — bottleneck
- Phase 2-R fix: mid_dim = max(32, hidden_dim // 2). Now dim=32 maps to 32, dim=64 maps to 32, dim=128 maps to 64
- Need to re-test with the fix in place
- Round 1 also bundled capacity with batch/patience changes — confounded

### Design
| Factor | Values |
|--------|--------|
| Config | A4 |
| Fusion | concat |
| Dataset | v2_nonews |
| Hidden dim | {32, 64, 128} |
| Seed | 42 |
| Folds | {0, 2, 4} |
| **Total** | **3 x 1 x 3 = 9 runs** |

### Baseline
A4/concat/hidden=64 on v2_nonews (3 runs). Same as Study 2/3 baselines.

### Success criteria
- hidden=32 should have lower val-test gap (less overfitting)
- hidden=128 should either improve MCC or show clear overfitting (larger gap)
- If hidden=128 overfits again (like Round 1), confirm that capacity is not the path forward

### What to watch
- Compare n_params for each: dim=32 (~7K), dim=64 (~14K), dim=128 (~55K)
- Val-test gap: does it scale with hidden_dim?
- Interaction with threshold: does filtering noisy labels help larger models more?

### CLI template
```bash
python -m src.training.experiment_runner_v2 \
    --h5-path data/processed/v2_nonews.h5 \
    --config-id A4 --fusion-type concat \
    --hidden-dim {DIM} --deadzone {BEST_THRESHOLD} --seed 42 --fold-idx {FOLD} \
    --output-csv experiments/phase3r/study4_hidden_dim.csv
```

---

## Study 5: News Processing (Lagged + Fixed)

### Hypothesis
Using day-(t-1) news features instead of day-t eliminates look-ahead bias from
post-market commentary. The 10-dim sentiment features with the forward-fill bug
fixed (no more stale data propagation) may produce a cleaner but still weak signal.
768-dim embeddings may capture more nuanced information beyond pos/neg/neu.

### Rationale
- Phase 1-R (Task #4) found: same-day correlation r=0.143 (reacting to prices),
  next-day r=0.005 (no predictive signal). Forward-fill inflates data by 31%.
- After fixing: lagged-1 day, no forward-fill, true coverage ~60% median
- If news STILL doesn't help after these fixes, the conclusion is definitive:
  financial news sentiment at daily frequency lacks predictive power for our universe.
  This is a strong RQ3 finding.
- 768-dim worth re-testing ONLY with the lag fix, since prior 768-dim test was
  confounded by look-ahead bias

### Design
| Factor | Values |
|--------|--------|
| Config | A3 (price + news) |
| Fusion | concat |
| Dataset | v2_full (with lag=1 news) |
| News dim | {10-dim sentiment, 768-dim embedding} |
| Seed | 42 |
| Folds | {0, 2, 4} |
| **Total** | **2 x 1 x 3 = 6 runs** |

### Baseline
A7/concat on v2_nonews at best threshold (3 runs from Study 1). Adding news should
improve over price-only if it has signal.

### Success criteria
- A3 with lagged news > A7 price-only: MCC delta > 0
- If both 10-dim and 768-dim fail (delta <= 0), news modality is confirmed as not
  useful for daily prediction — strong RQ3 result
- If 768-dim helps but 10-dim doesn't: the compression is the issue, not the signal

### What to watch
- 768-dim will have many more parameters — check val-test gap for overfitting
- Coverage: with forward-fill removed, ~40% of stock-days have no news.
  The model sees all-zeros for those rows. Is this better or worse than stale data?

### CLI template
```bash
python -m src.training.experiment_runner_v2 \
    --h5-path data/processed/v2_full.h5 \
    --config-id A3 --fusion-type concat \
    --news-lag 1 --news-dim {10|768} \
    --deadzone {BEST_THRESHOLD} --seed 42 --fold-idx {FOLD} \
    --output-csv experiments/phase3r/study5_news.csv
```

---

## Study 6: Graph Threshold

### Hypothesis
The current dynamic graph threshold (0.001) produces near-complete graphs
(density ~0.33), which are too dense to be informative. Increasing the threshold
to 0.3-0.5 will create sparser, more meaningful edges and improve the graph
encoder's ability to capture inter-stock relationships.

### Rationale
- Phase 1-R (Task #6, RQG) found dynamic graphs are 4x denser than static
- At threshold=0.001, almost all stock pairs are connected — the graph provides
  no structural information beyond "all stocks are related"
- Literature (THGNN, DASF-Net) uses higher thresholds for meaningful sparsity
- A8 baseline with current graphs: MCC=0.011 +/- 0.018 (13 runs)

### Design
| Factor | Values |
|--------|--------|
| Config | A8 (price + graph) |
| Fusion | concat |
| Dataset | v2_nonews (with rebuilt graphs at each threshold) |
| Graph threshold | {0.1, 0.3, 0.5} |
| Seed | 42 |
| Folds | {0, 2, 4} |
| **Total** | **3 x 1 x 3 = 9 runs** |

### Prerequisite
Graphs must be rebuilt at each threshold before running experiments. This requires
modifying `build_graphs.py` to accept a configurable threshold and regenerating
the dynamic graph snapshots.

### Baseline
A8/concat/threshold=0.001 on v2_nonews (3 runs at folds 0,2,4).
Also compare with A7 (no graph): if A8 at best threshold < A7, graphs don't help.

### Success criteria
- A8 at optimal threshold > A7 (price-only): graph adds value
- Optimal threshold produces graphs with density 0.05-0.15 (literature range)
- If all thresholds produce A8 <= A7, graph modality is not useful with current
  correlation-based construction

### What to watch
- Graph density at each threshold: too sparse (0.5) may disconnect the graph entirely
- Is the graph encoder actually learning? Check gradient norms on graph encoder weights
- Consider: static graph (GICS sectors) may be more informative than dense dynamic graphs

### CLI template
```bash
# First: rebuild graphs
python -m src.features.build_graphs --threshold {THRESH}

# Then: reassemble dataset
python -m src.data.assemble_dataset --skip-news --output data/processed/v2_nonews_graph{THRESH}.h5

# Then: run experiment
python -m src.training.experiment_runner_v2 \
    --h5-path data/processed/v2_nonews_graph{THRESH}.h5 \
    --config-id A8 --fusion-type concat \
    --deadzone {BEST_THRESHOLD} --seed 42 --fold-idx {FOLD} \
    --output-csv experiments/phase3r/study6_graph.csv
```

---

## Execution Order

Studies are ordered by priority and dependency:

```
Study 1 (threshold)      [60 runs, ~2-3 hrs]
    |
    v  (best threshold feeds into all subsequent studies)
Study 2 (batch/patience) [36 runs, ~1.5 hrs]
Study 3 (grad clip)      [9 runs, ~20 min]      } can run in parallel
Study 4 (hidden dim)     [9 runs, ~20 min]      }
    |
    v  (best hyperparams from 2-4 feed into Studies 5-6)
Study 5 (news fix)       [6 runs, ~15 min]      } can run in parallel
Study 6 (graph thresh)   [9 runs, ~20 min]      }
```

**Total**: 129 runs
**Estimated time**: ~5-6 hours on MPS (Mac M-series)
**Output directory**: `experiments/phase3r/`

---

## Decision Tree After Phase 3-R

```
Study 1 result:
  - Best threshold identified → set as default for all configs

Studies 2-4 results:
  - Best (batch, patience, clip, hidden_dim) → set as new defaults
  - If val-test gap decreases → architecture is better regularized
  - If nothing helps → current defaults are already near-optimal

Study 5 result:
  - Lagged news helps → include news in final ablation with lag fix
  - Lagged news doesn't help → exclude news, strong RQ3 negative result

Study 6 result:
  - Sparse graphs help → include graphs in final ablation
  - Dense/sparse both fail → exclude graphs, note in thesis

THEN: Phase 4-R full ablation re-run with optimized settings
  - Only include modalities that showed promise in screening
  - Use best threshold + hyperparams
  - Full matrix: configs x fusions x 3 seeds x 5 folds
```

---

## Statistical Power Notes

- Studies 2-6 use single seed (42) x 3 folds = 3 observations per condition.
  This is a SCREENING design, not a confirmatory test. Power is low (cannot detect
  effects < Cohen's d ~1.5). The purpose is to eliminate clearly unhelpful changes
  and identify promising directions for full-batch testing.

- Study 1 uses 3 seeds x 5 folds = 15 observations per threshold. This provides
  adequate power to detect medium effects (d >= 0.8) at alpha=0.05.

- Any promising result from Studies 2-6 MUST be confirmed in the full Phase 4-R
  ablation with 3 seeds x 5 folds before being reported in the thesis.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| v2 dataset assembly fails | Run threshold=0 on v2 first; compare with v1 baseline as regression test |
| Study 1 shows no threshold improvement | Proceed with threshold=0; focus RQ3 on regime analysis instead |
| All screening studies show no improvement | Conclude that architecture/hyperparams are not the bottleneck; the signal is genuinely weak at daily frequency |
| Overfitting increases with threshold (fewer samples) | Monitor val-test gap; if gap > 0.05, threshold is too aggressive |
| Runtime exceeds estimates | Studies 3-6 are parallelizable; can run overnight |
