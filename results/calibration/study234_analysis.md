# Studies 3.2-3.4: Training Dynamics, Gradient Clipping, Hidden Dimension

## Context

These studies use **proper train/val/test splits** (unlike Study 3.1 which had
test-set early stopping). The trainer now evaluates on a held-out validation set
for early stopping, with the test set used only for final evaluation.

---
## Study 3.2: Training Dynamics (Batch Size x Patience)

- **Config**: A4 (price + macro), concat fusion, seed=42, dz=0.005
- **Variables**: batch_size x patience
- **Folds**: [np.int64(0), np.int64(2), np.int64(4)]
- **Total experiments**: 36

- **Combos tested**: 12

- **Unique results**: 36 (duplicates: 0)

### 3.2.1 Summary Table

| Batch | Patience | F0 MCC | F2 MCC | F4 MCC | Mean MCC | Mean Val MCC | Val-Test Gap |
|-------|----------|--------|--------|--------|----------|-------------|-------------|
| 128 | 10 | -0.0132 | 0.0104 | -0.0051 | -0.0026 | 0.0145 | 0.0171 |
| 128 | 20 | -0.0132 | 0.0104 | -0.0051 | -0.0026 | 0.0145 | 0.0171 |
| 128 | 40 | -0.0125 | -0.0012 | -0.0163 | -0.0100 | 0.0372 | 0.0472 |
| 256 | 10 | -0.0112 | 0.0029 | -0.0084 | -0.0056 | 0.0170 | 0.0226 |
| 256 | 20 | -0.0112 | 0.0029 | -0.0084 | -0.0056 | 0.0170 | 0.0226 |
| 256 | 40 | -0.0112 | -0.0081 | -0.0243 | -0.0146 | 0.0295 | 0.0441 |
| 512 | 10 | -0.0215 | 0.0061 | -0.0121 | -0.0092 | 0.0194 | 0.0285 |
| 512 | 20 | -0.0215 | 0.0061 | -0.0121 | -0.0092 | 0.0194 | 0.0285 |
| 512 | 40 | -0.0215 | -0.0105 | -0.0121 | -0.0147 | 0.0296 | 0.0443 |
| 64 | 10 | -0.0176 | -0.0290 | -0.0257 | -0.0241 | 0.0161 | 0.0402 |
| 64 | 20 | -0.0176 | -0.0185 | -0.0136 | -0.0166 | 0.0250 | 0.0415 |
| 64 | 40 | -0.0157 | -0.0185 | 0.0166 | -0.0059 | 0.0426 | 0.0485 |

### 3.2.2 Per-Fold Breakdown

**Fold 0**: MCC range [-0.0215, -0.0112], Best: bs=256/p=10
**Fold 2**: MCC range [-0.0290, 0.0104], Best: bs=128/p=10
**Fold 4**: MCC range [-0.0257, 0.0166], Best: bs=64/p=40

### 3.2.3 Val-Test Gap Distribution

- Mean gap: 0.0335
- Std gap: 0.0186
- Min gap: 0.0020
- Max gap: 0.0694
- All gaps positive: True
- Mean val MCC: 0.0235 (positive)
- Mean test MCC: -0.0100 (negative)

---
## Study 3.3: Gradient Clipping

- **Config**: A4 (price + macro), concat, seed=42, dz=0.005, batch=64, patience=20
- **Variable**: grad_clip = [np.float64(1.0), np.float64(5.0), np.float64(10.0)]
- **Folds**: [np.int64(0), np.int64(2), np.int64(4)]
- **Total experiments**: 9

- **Unique results**: 9 (duplicates: 0)

### 3.3.1 Summary Table

| Grad Clip | F0 MCC | F2 MCC | F4 MCC | Mean MCC | Mean Val MCC | Val-Test Gap |
|-----------|--------|--------|--------|----------|-------------|-------------|
| 1.0 | -0.0176 | -0.0185 | -0.0136 | -0.0166 | 0.0250 | 0.0415 |
| 5.0 | -0.0176 | -0.0185 | -0.0136 | -0.0166 | 0.0250 | 0.0415 |
| 10.0 | -0.0176 | -0.0185 | -0.0136 | -0.0166 | 0.0250 | 0.0415 |

**NOTE**: All grad_clip values produce IDENTICAL results per fold.
This means gradient norms never exceed the smallest clip value (1.0).
Gradient clipping is a non-factor for this model/data combination.

---
## Study 3.4: Hidden Dimension

- **Config**: A4 (price + macro), concat, seed=42, dz=0.005, batch=64, patience=20
- **Variable**: hidden_dim = [np.int64(32), np.int64(64), np.int64(128)]
- **Folds**: [np.int64(0), np.int64(2), np.int64(4)]
- **Total experiments**: 9

- **Unique results**: 9 (duplicates: 0)

### 3.4.1 Summary Table

| Hidden Dim | Params | F0 MCC | F2 MCC | F4 MCC | Mean MCC | Mean Val MCC | Val-Test Gap |
|------------|--------|--------|--------|--------|----------|-------------|-------------|
| 32 | 5,027 | -0.0049 | 0.0092 | -0.0059 | -0.0005 | 0.0531 | 0.0536 |
| 64 | 13,987 | -0.0176 | -0.0185 | -0.0136 | -0.0166 | 0.0250 | 0.0415 |
| 128 | 52,547 | -0.0228 | -0.0119 | -0.0020 | -0.0122 | 0.0621 | 0.0743 |

---
## 4. Cross-Study Comparison

### 4.1 Overall Distribution

- Total unique experiments: 54
- Mean test MCC: -0.0111
- Median test MCC: -0.0132
- % negative test MCC: 85.2%
- Best test MCC: 0.0166
- Worst test MCC: -0.0290

### 4.2 Val-Test Gap Distribution

- Mean gap: 0.0387
- Std gap: 0.0197
- Min gap: 0.0020
- Max gap: 0.0893
- All gaps positive: True

| Gap Range | Count | % |
|-----------|-------|---|
| 0-0.02 | 11 | 20.4% |
| 0.02-0.04 | 14 | 25.9% |
| 0.04-0.06 | 23 | 42.6% |
| 0.06-0.10 | 6 | 11.1% |
| 0.10+ | 0 | 0.0% |

### 4.3 Best Hyperparameter Combinations (Top 5 by Test MCC)

| Rank | Study | Fold | Batch | Pat | Clip | HidDim | Test MCC | Val MCC | Gap | Epochs |
|------|-------|------|-------|-----|------|--------|----------|---------|-----|--------|
| 1 | S2-training | 4 | 64 | 40 | 1.0 | 64 | 0.0166 | 0.0358 | 0.0192 | 150 |
| 2 | S2-training | 2 | 128 | 10 | 1.0 | 64 | 0.0104 | 0.0125 | 0.0020 | 15 |
| 3 | S2-training | 2 | 128 | 20 | 1.0 | 64 | 0.0104 | 0.0125 | 0.0020 | 25 |
| 4 | S4-hidden | 2 | 64 | 20 | 1.0 | 32 | 0.0092 | 0.0685 | 0.0593 | 31 |
| 5 | S2-training | 2 | 512 | 10 | 1.0 | 64 | 0.0061 | 0.0208 | 0.0147 | 23 |

### 4.4 Per-Fold Pattern

| Fold | Mean Test MCC | Mean Val MCC | Mean Gap | n |
|------|---------------|-------------|----------|---|
| 0 | -0.0159 | 0.0323 | 0.0482 | 18 |
| 2 | -0.0069 | 0.0362 | 0.0430 | 18 |
| 4 | -0.0105 | 0.0143 | 0.0248 | 18 |

---
## 5. Comparison: Study 3.1 (Test-Set Leakage) vs Studies 3.2-3.4 (Proper Splits)

| Metric | Study 3.1 (leaked) | Studies 3.2-3.4 (proper) | Delta |
|--------|-------------------|-----------------------|-------|
| Mean test MCC | 0.0232 | -0.0111 | -0.0342 |
| Std test MCC | 0.0261 | 0.0100 | |
| % negative | 20.0% | 85.2% | |
| Mean val MCC | 0.0232 | 0.0276 | |
| Note | val==test (leaked) | val!=test (proper) | |

**Key insight**: The ~0.03 positive MCC seen in Study 3.1 was entirely an artifact
of test-set early stopping. With proper train/val/test splits, the model shows
**negative** mean test MCC, meaning it performs worse than random on unseen data.

---
## 6. Conclusions

### Critical Finding

With proper validation (no test-set leakage), the model produces **negative test MCC**
across nearly all hyperparameter configurations. This is not a hyperparameter issue --
it is a fundamental signal issue. The model learns patterns on train/val that do not
generalize to the test period.

### Specific Findings

1. **Training dynamics (Study 3.2)**: No batch/patience combo achieves positive mean test MCC.
   Best combo achieves MCC=0.0166 (single fold).
2. **Gradient clipping (Study 3.3)**: All clip values produce IDENTICAL results.
   Gradients never exceed 1.0 norm -- clipping is irrelevant for this model.
3. **Hidden dimension (Study 3.4)**: Smaller model (32-dim) slightly outperforms larger
   (128-dim), but all are negative mean MCC. More capacity = more overfitting.
4. **Val-test gap**: Mean 0.0387 across all experiments. The model reliably overfits.
5. **Previous results were inflated**: Study 3.1 MCC of ~0.023 was an artifact of
   test-set early stopping, not genuine predictive ability.

### Implications for Project

The bottleneck is NOT hyperparameters. Before proceeding with Phase 4-R (750-run
ablation), we need to address the fundamental signal/generalization issue:
- The model overfits to train/val temporal patterns that do not persist into test periods
- Adding more modalities on top of this broken baseline will not help
- This may require: (a) different temporal split strategy, (b) regularization,
  (c) fundamentally different model, or (d) acceptance that daily prediction at this
  granularity is near-random with these features

---
*Analysis generated: 2026-02-14*
*Data: study2_training.csv (36 rows), study3_gradclip.csv (9 rows), study4_hidden.csv (9 rows)*