Loading data...
  Loaded 240 rows from ablation_ff.csv
  Loaded 135 rows from ablation_news_ff.csv
  FF total: 375 experiments
  Loaded 240 rows from ablation_lstm.csv
  Loaded 135 rows from ablation_news_lstm.csv
  LSTM total: 375 experiments
  Loaded 90 supplementary rows from ablation_social_lstm.csv
  Loaded 90 supplementary rows from ablation_graph_ln_ff.csv
  Loaded 90 supplementary rows from ablation_graph_ln_lstm.csv
  FF: replaced 90 original A8/A9 rows with 90 Graph LayerNorm rows
  LSTM: replaced 90 original A8/A9 rows with 90 Graph LayerNorm rows
Combined: 750 total experiments
  Configs: ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
  Fusions: ['concat', 'gated_cross_attention', 'mha']
  Seeds: [np.int64(42), np.int64(123), np.int64(456)]
  Folds: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]
  Architectures: ['FF', 'LSTM']

# Phase 4-R / Phase 5 Ablation Analysis

## Overview

- Total experiments: 750
- Configs: ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
- Fusion types: ['concat', 'gated_cross_attention', 'mha']
- Seeds: [np.int64(42), np.int64(123), np.int64(456)]
- Folds: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]
- Architectures: 2

  FF: 375/420 (89% complete)
  LSTM: 375/420 (89% complete)

### MCC Distribution

- Mean: 0.0078
- Median: 0.0065
- Std: 0.0240
- Min: -0.0718
- Max: 0.0791
- % negative: 41.6%

### Val-Test Gap

- Mean gap: 0.0246
- Std gap: 0.0352
- All positive: False

### MCC Summary: Config x Fusion

| Config | Description | concat | gated_cross_attention | mha | Mean |
|---|---|---|---|---|---|
| A1 | Full model (4 modalities) | 0.0044 | 0.0096 | 0.0043 | 0.0061 |
| A2 | Remove social | 0.0027 | -0.0002 | 0.0058 | 0.0028 |
| A3 | Price + news | 0.0061 | 0.0089 | 0.0090 | 0.0080 |
| A4 | Price + macro | 0.0081 | 0.0140 | 0.0156 | 0.0126 |
| A5 | Price + macro + social | 0.0142 | 0.0030 | 0.0042 | 0.0072 |
| A6 | Price + social | 0.0100 | 0.0111 | 0.0083 | 0.0098 |
| A7 | Price only (baseline) | 0.0095 | N/A | N/A | 0.0095 |
| A8 | Price + graph | 0.0096 | 0.0014 | 0.0080 | 0.0063 |
| A9 | Price + macro + graph | 0.0110 | 0.0067 | 0.0100 | 0.0092 |

### Diagnostic Check: A7 vs Phase 3-R Reference

- A7 FF: Phase 4-R MCC=0.0071 vs Phase 3-R ref=0.009 (delta=-0.0019) [OK]
  F0: 0.0263 vs ref 0.043 (delta=-0.0167) [OK]
  F1: 0.0459 vs ref 0.045 (delta=+0.0009) [OK]
  F2: -0.0085 vs ref -0.015 (delta=+0.0065) [OK]
  F3: -0.0093 vs ref -0.002 (delta=-0.0073) [OK]
  F4: -0.0191 vs ref -0.013 (delta=-0.0061) [OK]
- A7 LSTM: Phase 4-R MCC=0.0120 vs Phase 3-R ref=0.019 (delta=-0.0070) [OK]


---
## RQ1: Multi-Source vs Single-Source

Baseline: A7 (price only). Each multi-modal config compared via paired t-tests.

### RQ1.1: Fold-Averaged Comparison (vs A7)

Paired t-tests match on (fold, seed, fusion_type, architecture). Since A7 only runs
concat fusion, all pairs are concat-vs-concat. Mean MCC columns show the paired subset
means (concat only) for consistency with the statistical test.

| Config | Description | Paired MCC | A7 MCC | Delta | p-value | p (Bonf) | Cohen d | n | Sig | Wins |
|--------|-------------|------------|--------|-------|---------|----------|---------|---|-----|------|
| A1 | Full model (4 modalities) | 0.0044 | 0.0095 | -0.0051 | 0.3490 | 1.0000 | -0.177 | 30 | ns | 12/30 |
| A2 | Remove social | 0.0027 | 0.0095 | -0.0068 | 0.2215 | 1.0000 | -0.232 | 30 | ns | 11/30 |
| A3 | Price + news | 0.0061 | 0.0095 | -0.0034 | 0.3127 | 1.0000 | -0.191 | 30 | ns | 14/30 |
| A4 | Price + macro | 0.0081 | 0.0095 | -0.0014 | 0.7855 | 1.0000 | -0.051 | 30 | ns | 12/30 |
| A5 | Price + macro + social | 0.0142 | 0.0095 | +0.0047 | 0.4603 | 1.0000 | 0.139 | 30 | ns | 19/30 |
| A6 | Price + social | 0.0100 | 0.0095 | +0.0005 | 0.8724 | 1.0000 | 0.030 | 30 | ns | 19/30 |
| A8 | Price + graph | 0.0096 | 0.0095 | +0.0000 | 0.9962 | 1.0000 | 0.001 | 30 | ns | 15/30 |
| A9 | Price + macro + graph | 0.0110 | 0.0095 | +0.0015 | 0.8072 | 1.0000 | 0.046 | 30 | ns | 19/30 |

### RQ1.2: Per-Fold Breakdown (Multi-Source Delta vs A7)

| Config | F0 | F1 | F2 | F3 | F4 | Mean |
|---|---|---|---|---|---|---|
| A1 | -0.0366 | -0.0026 | +0.0121 | +0.0150 | -0.0134 | -0.0051 |
| A2 | -0.0345 | -0.0076 | +0.0096 | +0.0127 | -0.0141 | -0.0068 |
| A3 | -0.0013 | -0.0119 | +0.0048 | -0.0089 | +0.0004 | -0.0034 |
| A4 | -0.0350 | -0.0001 | +0.0017 | +0.0183 | +0.0080 | -0.0014 |
| A5 | -0.0401 | +0.0038 | +0.0187 | +0.0296 | +0.0116 | +0.0047 |
| A6 | -0.0022 | -0.0078 | +0.0085 | +0.0028 | +0.0011 | +0.0005 |
| A8 | -0.0118 | -0.0272 | +0.0200 | -0.0056 | +0.0248 | +0.0000 |
| A9 | -0.0333 | +0.0058 | +0.0189 | +0.0073 | +0.0087 | +0.0015 |

### RQ1.3: Per-Fusion Breakdown (Multi-Source Delta vs A7)

Note: A7 only runs concat. For non-concat fusions, delta is computed as
config-fusion MCC minus A7-concat MCC (matched on fold, seed, architecture).

| Config | concat | gated_cross_attention | mha |
|---|---|---|---|
| A1 | -0.0051 | +0.0001 | -0.0053 |
| A2 | -0.0068 | -0.0097 | -0.0037 |
| A3 | -0.0034 | -0.0007 | -0.0005 |
| A4 | -0.0014 | +0.0044 | +0.0061 |
| A5 | +0.0047 | -0.0065 | -0.0053 |
| A6 | +0.0005 | +0.0016 | -0.0012 |
| A8 | +0.0000 | -0.0081 | -0.0015 |
| A9 | +0.0015 | -0.0029 | +0.0004 |

### RQ1 Summary

- Any significant improvement over A7: NO
- Any positive delta: YES
- Best config vs A7: A5 (delta=+0.0047, p=0.4603)
- Worst config vs A7: A2 (delta=-0.0068, p=0.2215)


---
## RQ2: Marginal Contribution per Modality

### RQ2.1: Leave-One-Out from Full Model (A1)

Remove one modality from the full model and measure MCC change.

| Removed | Full (A1) | Without | Delta | p-value | Cohen d | Sig | Interpretation |
|---------|-----------|---------|-------|---------|---------|-----|----------------|
| news | 0.0061 | 0.0072 | -0.0011 | 0.6976 | -0.041 | ns | hurts |
| social | 0.0061 | 0.0028 | +0.0033 | 0.1851 | 0.142 | ns | helps |

### RQ2.2: Add-One-In from Baseline (A7)

Add one modality to price-only baseline and measure MCC change.

| Added | A7 (base) | With modality | Config | Delta | p-value | Cohen d | Sig |
|-------|-----------|---------------|--------|-------|---------|---------|-----|
| macro | 0.0095 | 0.0126 | A4 | -0.0014 | 0.7855 | -0.051 | ns |
| news | 0.0095 | 0.0080 | A3 | -0.0034 | 0.3127 | -0.191 | ns |
| social | 0.0095 | 0.0098 | A6 | +0.0005 | 0.8724 | 0.030 | ns |
| graph | 0.0095 | 0.0063 | A8 | +0.0000 | 0.9962 | 0.001 | ns |

### RQ2.3: Modality Value Ranking

Ranked by add-one-in delta MCC (positive = helps, negative = hurts):

1. **social**: delta=+0.0005, d=0.030 (negligible), p=0.8724 ns
2. **graph**: delta=+0.0000, d=0.001 (negligible), p=0.9962 ns
3. **macro**: delta=-0.0014, d=-0.051 (negligible), p=0.7855 ns
4. **news**: delta=-0.0034, d=-0.191 (negligible), p=0.3127 ns

### RQ2.4: Per-Fold Marginal Contribution (Add-One-In Delta)

| Modality | F0 | F1 | F2 | F3 | F4 | Mean |
|---|---|---|---|---|---|---|
| macro | -0.0350 | -0.0001 | +0.0017 | +0.0183 | +0.0080 | -0.0014 |
| news | -0.0013 | -0.0119 | +0.0048 | -0.0089 | +0.0004 | -0.0034 |
| social | -0.0022 | -0.0078 | +0.0085 | +0.0028 | +0.0011 | +0.0005 |
| graph | -0.0118 | -0.0272 | +0.0200 | -0.0056 | +0.0248 | +0.0000 |


---
## RQ3: Conditional Analysis (When Does Multi-Source Help?)

### RQ3.1: MCC by Config x Fold (Market Regime)

| Config | F0 | F1 | F2 | F3 | F4 | Mean | Std |
|---|---|---|---|---|---|---|---|
| A1 | -0.0009 | 0.0275 | -0.0049 | 0.0276 | -0.0187 | 0.0061 | 0.0185 |
| A2 | -0.0034 | 0.0338 | -0.0092 | 0.0155 | -0.0229 | 0.0028 | 0.0198 |
| A3 | 0.0264 | 0.0284 | -0.0142 | 0.0060 | -0.0067 | 0.0080 | 0.0171 |
| A4 | -0.0036 | 0.0340 | -0.0036 | 0.0296 | 0.0064 | 0.0126 | 0.0162 |
| A5 | -0.0062 | 0.0260 | -0.0024 | 0.0223 | -0.0038 | 0.0072 | 0.0140 |
| A6 | 0.0317 | 0.0259 | -0.0150 | 0.0125 | -0.0060 | 0.0098 | 0.0180 |
| A7 | 0.0283 | 0.0360 | -0.0162 | 0.0083 | -0.0088 | 0.0095 | 0.0203 |
| A8 | 0.0118 | 0.0177 | -0.0033 | 0.0006 | 0.0048 | 0.0063 | 0.0076 |
| A9 | -0.0069 | 0.0331 | 0.0030 | 0.0164 | 0.0005 | 0.0092 | 0.0141 |

**Fold regime labels:**
- F0: Pre-COVID (2019H2-2020H1)
- F1: Recovery (2020H2-2021H1)
- F2: Bull-to-Bear (2021H2-2022H1)
- F3: Bear Market (2022H2-2023H1)
- F4: AI Rally (2023H2)

### RQ3.2: Best Config per Fold

| Fold | Regime | Best Config | MCC | 2nd Best | MCC | A7 MCC | Delta vs A7 |
|------|--------|-------------|-----|----------|-----|--------|-------------|
| F0 | Pre-COVID (2019H2-2020H1) | A6 | 0.0317 | A7 | 0.0283 | 0.0283 | +0.0035 |
| F1 | Recovery (2020H2-2021H1) | A7 | 0.0360 | A4 | 0.0340 | 0.0360 | +0.0000 |
| F2 | Bull-to-Bear (2021H2-2022H1) | A9 | 0.0030 | A5 | -0.0024 | -0.0162 | +0.0192 |
| F3 | Bear Market (2022H2-2023H1) | A4 | 0.0296 | A1 | 0.0276 | 0.0083 | +0.0213 |
| F4 | AI Rally (2023H2) | A4 | 0.0064 | A8 | 0.0048 | -0.0088 | +0.0152 |

### RQ3.3: Multi-Source Advantage by Regime

For each fold, does ANY multi-modal config significantly beat A7?

**Note**: Per-fold tests are exploratory (not Bonferroni-corrected across folds).
Sample sizes per fold are small (n reported); interpret with caution.

| Fold | Regime | Best Multi | Delta vs A7 | p-value | n | Sig | Verdict |
|------|--------|------------|-------------|---------|---|-----|---------|
| F0 | Pre-COVID (2019H2-2020H1) | A3 | -0.0013 | 0.9053 | 6 | ns | NO DIFF |
| F1 | Recovery (2020H2-2021H1) | A9 | +0.0058 | 0.7493 | 6 | ns | NO DIFF |
| F2 | Bull-to-Bear (2021H2-2022H1) | A8 | +0.0200 | 0.0445 | 6 | * | MULTI WINS |
| F3 | Bear Market (2022H2-2023H1) | A5 | +0.0296 | 0.0298 | 6 | * | MULTI WINS |
| F4 | AI Rally (2023H2) | A8 | +0.0248 | 0.0300 | 6 | * | MULTI WINS |

### RQ3.4: Fusion Strategy Effectiveness by Fold

| Fold | Regime | concat | gated_cross_attention | mha | Best |
|---|---|---|---|---|---|
| F0 | Pre-COVID (2019H2-2020H1) | 0.0066 | 0.0064 | 0.0081 | mha |
| F1 | Recovery (2020H2-2021H1) | 0.0307 | 0.0261 | 0.0288 | concat |
| F2 | Bull-to-Bear (2021H2-2022H1) | -0.0057 | -0.0089 | -0.0053 | mha |
| F3 | Bear Market (2022H2-2023H1) | 0.0163 | 0.0181 | 0.0136 | gated_cross_attention |
| F4 | AI Rally (2023H2) | -0.0058 | -0.0076 | -0.0043 | mha |

### RQ3.5: Cross-Fold Consistency

How consistent is each config's ranking across folds?

| Config | F0 Rank | F1 Rank | F2 Rank | F3 Rank | F4 Rank | Mean Rank | Std Rank |
|---|---|---|---|---|---|---|---|
| A1 | 5 | 6 | 5 | 2 | 8 | 5.2 | 1.9 |
| A2 | 6 | 3 | 6 | 5 | 9 | 5.8 | 1.9 |
| A3 | 3 | 5 | 7 | 8 | 6 | 5.8 | 1.7 |
| A4 | 7 | 2 | 4 | 1 | 1 | 3.0 | 2.3 |
| A5 | 8 | 7 | 2 | 3 | 4 | 4.8 | 2.3 |
| A6 | 1 | 8 | 8 | 6 | 5 | 5.6 | 2.6 |
| A7 | 2 | 1 | 9 | 7 | 7 | 5.2 | 3.1 |
| A8 | 4 | 9 | 3 | 9 | 2 | 5.4 | 3.0 |
| A9 | 9 | 4 | 1 | 4 | 3 | 4.2 | 2.6 |


---
## Architecture Comparison: FF vs LSTM

### Overall

- FF: mean MCC = 0.0042 (std=0.0242, n=375)
- LSTM: mean MCC = 0.0114 (std=0.0234, n=375)
- Delta (LSTM - FF): +0.0071
- Paired t-test: t=5.342, p=0.0000 ***
- Cohen's d: 0.276 (small)
- LSTM wins: 221/375 (59%)

### Per-Config Architecture Comparison

| Config | FF MCC | LSTM MCC | Delta | p-value | Cohen d | Sig | LSTM Win% |
|--------|--------|----------|-------|---------|---------|-----|-----------|
| A1 | 0.0014 | 0.0108 | +0.0095 | 0.0226 | 0.356 | * | 53% |
| A2 | -0.0014 | 0.0069 | +0.0083 | 0.0407 | 0.318 | * | 58% |
| A3 | 0.0059 | 0.0101 | +0.0042 | 0.1733 | 0.209 | ns | 53% |
| A4 | 0.0102 | 0.0149 | +0.0047 | 0.3288 | 0.149 | ns | 51% |
| A5 | 0.0016 | 0.0127 | +0.0111 | 0.0056 | 0.439 | ** | 73% |
| A6 | 0.0061 | 0.0135 | +0.0074 | 0.0080 | 0.419 | ** | 67% |
| A7 | 0.0071 | 0.0120 | +0.0049 | 0.5035 | 0.184 | ns | 53% |
| A8 | 0.0062 | 0.0065 | +0.0004 | 0.8952 | 0.020 | ns | 53% |
| A9 | 0.0030 | 0.0154 | +0.0124 | 0.0202 | 0.363 | * | 64% |

### Per-Fold Architecture Effect

**Note**: Per-fold tests are exploratory (not Bonferroni-corrected across folds).
Cohen's d and n reported for transparency.

| Fold | Regime | FF MCC | LSTM MCC | Delta | p-value | Cohen d | n | Sig |
|------|--------|--------|----------|-------|---------|---------|---|-----|
| F0 | Pre-COVID (2019H2-2020H1) | 0.0012 | 0.0128 | +0.0116 | 0.0000 | 0.565 | 75 | *** |
| F1 | Recovery (2020H2-2021H1) | 0.0306 | 0.0266 | -0.0039 | 0.2363 | -0.139 | 75 | ns |
| F2 | Bull-to-Bear (2021H2-2022H1) | -0.0072 | -0.0060 | +0.0013 | 0.5331 | 0.073 | 75 | ns |
| F3 | Bear Market (2022H2-2023H1) | 0.0063 | 0.0257 | +0.0194 | 0.0000 | 0.830 | 75 | *** |
| F4 | AI Rally (2023H2) | -0.0096 | -0.0023 | +0.0073 | 0.0424 | 0.240 | 75 | * |

### Regime-Architecture Interaction

Key question from the classifier findings: Does LSTM excel in volatile/transitional markets
while FF works better in calm/trending markets?
*(Exploratory analysis — per-fold tests not corrected for multiple comparisons)*

- F0 (Pre-COVID (2019H2-2020H1)): **LSTM** *** (delta=+0.0116, d=0.565, n=75)
- F1 (Recovery (2020H2-2021H1)): **FF** ns (delta=-0.0039, d=-0.139, n=75)
- F2 (Bull-to-Bear (2021H2-2022H1)): **LSTM** ns (delta=+0.0013, d=0.073, n=75)
- F3 (Bear Market (2022H2-2023H1)): **LSTM** *** (delta=+0.0194, d=0.830, n=75)
- F4 (AI Rally (2023H2)): **LSTM** * (delta=+0.0073, d=0.240, n=75)


---
## Fusion Strategy Analysis

### Overall Fusion Comparison

| Fusion | Mean MCC | Std | Median | % Negative | N |
|--------|----------|-----|--------|------------|---|
| concat | 0.0084 | 0.0244 | 0.0039 | 43.3% | 270 |
| gated_cross_attention | 0.0068 | 0.0236 | 0.0062 | 41.7% | 240 |
| mha | 0.0082 | 0.0242 | 0.0084 | 39.6% | 240 |

### Pairwise Fusion Comparisons

| Comparison | Delta | p-value | p (Bonf) | Cohen d | Sig |
|------------|-------|---------|----------|---------|-----|
| concat vs gated_cross_attention | +0.0015 | 0.3489 | 1.0000 | 0.061 | ns |
| concat vs mha | +0.0001 | 0.9409 | 1.0000 | 0.005 | ns |
| gated_cross_attention vs mha | -0.0014 | 0.3499 | 1.0000 | -0.061 | ns |

### Fusion Effectiveness by Config (Only multi-modal)

| Config | concat | gated_cross_attention | mha | Best |
|---|---|---|---|---|
| A1 | 0.0044 | 0.0096 | 0.0043 | gated_cross_attention |
| A2 | 0.0027 | -0.0002 | 0.0058 | mha |
| A3 | 0.0061 | 0.0089 | 0.0090 | mha |
| A4 | 0.0081 | 0.0140 | 0.0156 | mha |
| A5 | 0.0142 | 0.0030 | 0.0042 | concat |
| A6 | 0.0100 | 0.0111 | 0.0083 | gated_cross_attention |
| A8 | 0.0096 | 0.0014 | 0.0080 | concat |
| A9 | 0.0110 | 0.0067 | 0.0100 | concat |


---
## Financial Metrics Analysis

**Note**: cumulative_return and max_drawdown excluded (computed with incorrect formula in pre-fix experiment runs; see metrics.py fix).


### Financial Metrics by Config

| Config | Sharpe | PF | DirAcc |
|---|---|---|---|
| A1 | 0.1136 | 1.0217 | 0.5102 |
| A2 | -0.0385 | 0.9970 | 0.5072 |
| A3 | 0.2962 | 1.0536 | 0.5161 |
| A4 | 0.0120 | 1.0038 | 0.5062 |
| A5 | 0.0768 | 1.0147 | 0.5085 |
| A6 | 0.3902 | 1.0692 | 0.5201 |
| A7 | 0.4696 | 1.0824 | 0.5213 |
| A8 | 0.2713 | 1.0492 | 0.5127 |
| A9 | 0.1165 | 1.0207 | 0.5102 |

### Best Config by Each Financial Metric

- **sharpe_ratio**: A7 (0.4696)
- **profit_factor**: A7 (1.0824)
- **directional_accuracy**: A7 (0.5213)


---
## Executive Summary

### Overall Performance
- Mean MCC across all 750 experiments: 0.0078
- % negative MCC: 41.6%

### Config Ranking (by mean MCC)

1. A4 (Price + macro): 0.0126
2. A6 (Price + social): 0.0098
3. A7 (Price only (baseline)): 0.0095
4. A9 (Price + macro + graph): 0.0092
5. A3 (Price + news): 0.0080
6. A5 (Price + macro + social): 0.0072
7. A8 (Price + graph): 0.0063
8. A1 (Full model (4 modalities)): 0.0061
9. A2 (Remove social): 0.0028

### Fusion Ranking

1. concat: 0.0084
2. mha: 0.0082
3. gated_cross_attention: 0.0068

### Architecture
- FF mean MCC: 0.0042
- LSTM mean MCC: 0.0114
- Better overall: LSTM

### Research Question Answers (Preliminary)

**RQ1** (multi vs single): Best multi-modal (A4) MCC=0.0126 vs A7 MCC=0.0095 (delta=+0.0030)

**RQ2** (which sources help): See Section RQ2 for marginal contribution analysis.

**RQ3** (when does it help): See Section RQ3 for per-fold regime analysis.
