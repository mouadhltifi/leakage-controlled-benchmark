<!--
HISTORICAL PROVENANCE (thesis-era, 2026-02). This is a generated dump of
the study-grid tables. Its literature-comparison annotations are SUPERSEDED
by audits/census/CENSUS.md, audits/camef/, and
results/analysis/msgca_diagnostic_rerun.json. In particular: the CAMEF split
is positional/file-order, NOT "chronological (temporal, no leakage)" as an
annotation below states (see the paper, Demonstration 2); MSGCA's published
number is 0.1112 (any 0.1157 here is this file's own seed-42 rerun best);
and the "953% / cherry-picking" framing is thesis-era, not the paper's
non-adversarial register. Cited paths like experiments/phase4r/... are
thesis-tree, not artifact paths. Read the audit dirs for the current record.
-->
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

# Chapter 4 Analysis Functions
Running all 16 Chapter 4 analysis functions on 750 experiments...


---
## Table 4.3: A7 Baseline Results by Architecture and Fusion

| Architecture | Fusion | Mean MCC | 95% CI | Std | Median | Dir Acc | Sharpe | Val-Test Gap | n | MCC > 0? (p) |
|-------------|--------|----------|--------|-----|--------|---------|--------|-------------|---|--------------|
| FF | concat | 0.0071 | [-0.0086, 0.0227] | 0.0273 | 0.0010 | 0.5225 | 0.4271 | 0.0188 | 15 | 0.3488 |
| LSTM | concat | 0.0120 | [-0.0006, 0.0246] | 0.0219 | 0.0224 | 0.5201 | 0.5122 | 0.0176 | 15 | 0.0604 |

**A7 overall**: mean=0.0095, t=2.061, p=0.0484 -> MCC significantly > 0: YES
  % positive MCC: 63.3%

## Table 4.4: A7 Baseline Per-Fold Breakdown

| Fold | Regime | FF MCC | FF 95% CI | LSTM MCC | LSTM 95% CI | n (per arch) |
|------|--------|--------|-----------|----------|-------------|-------------|
| F0 | Pre-COVID (2019H2-2020H1) | 0.0263 | [-0.0283, 0.0809] | 0.0302 | [0.0020, 0.0585] | 3 |
| F1 | Recovery (2020H2-2021H1) | 0.0459 | [0.0198, 0.0721] | 0.0261 | [0.0191, 0.0332] | 3 |
| F2 | Bull-to-Bear (2021H2-2022H1) | -0.0085 | [-0.0499, 0.0330] | -0.0239 | [-0.0623, 0.0145] | 3 |
| F3 | Bear Market (2022H2-2023H1) | -0.0093 | [-0.0343, 0.0158] | 0.0260 | [0.0126, 0.0393] | 3 |
| F4 | AI Rally (2023H2) | -0.0191 | [-0.0252, -0.0131] | 0.0015 | [-0.0088, 0.0117] | 3 |

**Note**: A7 only runs concat fusion (single-modality, no cross-modal fusion to perform).


---
## Table 4.5: Multi-Source vs Baseline (All Configs, Bonferroni k=9)

| Config | Modalities | Mean MCC | Delta vs A7 | p-value | p (Bonf) | Cohen's d | Effect | Sig | n | Verdict |
|--------|-----------|----------|-------------|---------|----------|-----------|--------|-----|---|---------|
| A7 | Price only | 0.0095 | -- | -- | -- | -- | -- | -- | 30 | Baseline |
| A3 | Price + News | 0.0080 | -0.0034 | 0.3127 | 1.0000 | -0.191 | negligible | ns | 30 | NO DIFFERENCE |
| A4 | Price + Macro | 0.0126 | -0.0014 | 0.7855 | 1.0000 | -0.051 | negligible | ns | 30 | NO DIFFERENCE |
| A6 | Price + Social | 0.0098 | +0.0005 | 0.8724 | 1.0000 | 0.030 | negligible | ns | 30 | NO DIFFERENCE |
| A8 | Price + Graph | 0.0063 | +0.0000 | 0.9962 | 1.0000 | 0.001 | negligible | ns | 30 | NO DIFFERENCE |
| A2 | Price + News + Macro | 0.0028 | -0.0068 | 0.2215 | 1.0000 | -0.232 | small | ns | 30 | NO DIFFERENCE |
| A5 | Price + Macro + Social | 0.0072 | +0.0047 | 0.4603 | 1.0000 | 0.139 | negligible | ns | 30 | NO DIFFERENCE |
| A9 | Price + Macro + Graph | 0.0092 | +0.0015 | 0.8072 | 1.0000 | 0.046 | negligible | ns | 30 | NO DIFFERENCE |
| A10 | Price + News + Social | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 | *Data pending* |
| A1 | Price + News + Macro + Social | 0.0061 | -0.0051 | 0.3490 | 1.0000 | -0.177 | negligible | ns | 30 | NO DIFFERENCE |

**RQ1 Answer**: No multi-source configuration significantly outperforms the price-only baseline after Bonferroni correction (k=9).


---
## Table 4.6: Additive Marginal Contribution (Each Source Added to A7)

| Rank | Source Added | Config | A7 MCC | Config MCC | Delta MCC | p-value | p (Bonf) | Cohen's d | Effect | Sig | Wins |
|------|-------------|--------|--------|------------|-----------|---------|----------|-----------|--------|-----|------|
| 1 | Social Media | A6 | 0.0095 | 0.0098 | +0.0005 | 0.8724 | 1.0000 | 0.030 | negligible | ns | 19/30 |
| 2 | Graph | A8 | 0.0095 | 0.0063 | +0.0000 | 0.9962 | 1.0000 | 0.001 | negligible | ns | 15/30 |
| 3 | Macro | A4 | 0.0095 | 0.0126 | -0.0014 | 0.7855 | 1.0000 | -0.051 | negligible | ns | 12/30 |
| 4 | News | A3 | 0.0095 | 0.0080 | -0.0034 | 0.3127 | 1.0000 | -0.191 | negligible | ns | 14/30 |

**Source Value Ranking** (by additive delta MCC):
  1. Social Media: +0.0005 [NEUTRAL]
  2. Graph: +0.0000 [NEUTRAL]
  3. Macro: -0.0014 [NEUTRAL]
  4. News: -0.0034 [NEUTRAL]


---
## Table 4.7: Leave-One-Out from Full Model (A1)

| Source Removed | Full (A1) MCC | Reduced MCC | Reduced Config | Delta (drop) | p-value | Cohen's d | Sig | Importance |
|--------------|---------------|-------------|----------------|-------------|---------|-----------|-----|------------|
| News | 0.0061 | 0.0072 | A5 | -0.0011 | 0.6976 | -0.041 | ns | REDUNDANT |
| Social Media | 0.0061 | 0.0028 | A2 | +0.0033 | 0.1851 | 0.142 | ns | MARGINAL |
| Macro | 0.0061 | N/A | A10 | N/A | N/A | N/A | N/A | *Data pending* |

**Sign convention**: Delta = A1 (full) minus reduced config. Positive delta means the full model outperforms (removing the source hurts). Negative delta means the reduced model outperforms (removing the source helps).

**Note**: A10 (price+news+social) fills the previously missing leave-one-out slot for macro. All three non-price sources can now be evaluated.


---
## Table 4.8: Incremental Value Over Price+Macro Baseline (A4)

| Source Added | Config | A4 MCC | Config MCC | Delta | p-value | p (Bonf) | Cohen's d | Sig | n | Verdict |
|-------------|--------|--------|------------|-------|---------|----------|-----------|-----|---|---------|
| + News | A2 | 0.0126 | 0.0028 | -0.0098 | 0.0012 | 0.0048 | -0.355 | ** | 90 | DEGRADES |
| + Social Media | A5 | 0.0126 | 0.0072 | -0.0054 | 0.0610 | 0.2442 | -0.201 | ns | 90 | NO DIFFERENCE |
| + Graph | A9 | 0.0126 | 0.0092 | -0.0033 | 0.2782 | 1.0000 | -0.116 | ns | 90 | NO DIFFERENCE |
| + News + Social | A1 | 0.0126 | 0.0061 | -0.0065 | 0.0200 | 0.0800 | -0.251 | ns | 90 | NO DIFFERENCE |

**Finding**: No source addition significantly improves over A4 (price+macro). A4 may represent the performance ceiling for this prediction task.


---
## Table 4.9 / Figure 4.3: Multi-Source Delta by Config x Fold

| Config | F0 | F1 | F2 | F3 | F4 | Mean | Overall p |
|---|---|---|---|---|---|---|---|
| A1 | -0.0366*** | -0.0026ns | +0.0121ns | +0.0150ns | -0.0134ns | -0.0051 | 0.3490 |
| A2 | -0.0345* | -0.0076ns | +0.0096ns | +0.0127ns | -0.0141ns | -0.0068 | 0.2215 |
| A3 | -0.0013ns | -0.0119ns | +0.0048ns | -0.0089ns | +0.0004ns | -0.0034 | 0.3127 |
| A4 | -0.0350** | -0.0001ns | +0.0017ns | +0.0183ns | +0.0080ns | -0.0014 | 0.7855 |
| A5 | -0.0401* | +0.0038ns | +0.0187ns | +0.0296* | +0.0116ns | +0.0047 | 0.4603 |
| A6 | -0.0022ns | -0.0078ns | +0.0085ns | +0.0028ns | +0.0011ns | +0.0005 | 0.8724 |
| A8 | -0.0118ns | -0.0272** | +0.0200* | -0.0056ns | +0.0248* | +0.0000 | 0.9962 |
| A9 | -0.0333* | +0.0058ns | +0.0189ns | +0.0073ns | +0.0087ns | +0.0015 | 0.8072 |

**Reading guide**: Each cell shows delta MCC vs A7 baseline. Positive = multi-source helps. Significance: * p<0.05, ** p<0.01, *** p<0.001.

**Regime-specific patterns**:
  - F0 (Pre-COVID (2019H2-2020H1)): hurts — best A3 (-0.0013), worst A5 (-0.0401)
  - F1 (Recovery (2020H2-2021H1)): mixed — best A9 (+0.0058), worst A8 (-0.0272)
  - F2 (Bull-to-Bear (2021H2-2022H1)): helps — best A8 (+0.0200), worst A4 (+0.0017)
  - F3 (Bear Market (2022H2-2023H1)): mixed — best A5 (+0.0296), worst A3 (-0.0089)
  - F4 (AI Rally (2023H2)): mixed — best A8 (+0.0248), worst A2 (-0.0141)


---
## Table 4.10 / Figure 4.4: Architecture x Regime Interaction

| Fold | Regime | FF MCC | LSTM MCC | Delta (LSTM-FF) | p-value | Cohen's d | Effect | Sig | LSTM Wins |
|------|--------|--------|----------|-----------------|---------|-----------|--------|-----|-----------|
| F0 | Pre-COVID (2019H2-2020H1) | 0.0012 | 0.0128 | +0.0116 | 0.0000 | 0.565 | medium | *** | 51/75 |
| F1 | Recovery (2020H2-2021H1) | 0.0306 | 0.0266 | -0.0039 | 0.2363 | -0.139 | negligible | ns | 31/75 |
| F2 | Bull-to-Bear (2021H2-2022H1) | -0.0072 | -0.0060 | +0.0013 | 0.5331 | 0.073 | negligible | ns | 35/75 |
| F3 | Bear Market (2022H2-2023H1) | 0.0063 | 0.0257 | +0.0194 | 0.0000 | 0.830 | large | *** | 60/75 |
| F4 | AI Rally (2023H2) | -0.0096 | -0.0023 | +0.0073 | 0.0424 | 0.240 | small | * | 44/75 |

**Overall**: LSTM delta = +0.0071, p=0.0000 ***, d=0.276 (small)

**Regime-dependent architecture effect**:
  - LSTM significantly better in: F0 (Pre-COVID (2019H2-2020H1)), F3 (Bear Market (2022H2-2023H1)), F4 (AI Rally (2023H2))

**Hypothesis check**: LSTM excels in volatile/transitional, FF in calm/trending
  - F0 (Pre-COVID (2019H2-2020H1)): LSTM wins (d=0.565, medium)
  - F3 (Bear Market (2022H2-2023H1)): LSTM wins (d=0.830, large)
  - F4 (AI Rally (2023H2)): LSTM wins (d=0.240, small)


---
## Section 4.5.4: News Degradation Analysis

### News over price-only
*Does news help when added to price alone?*

**Overall**: A3 MCC=0.0080 vs A7 MCC=0.0095
  Delta: -0.0034, p_raw=0.3127, p_bonf=0.9382 (k=3), d=-0.191 -> **NO EFFECT**

**Per-architecture split** (A3 vs A7):

| Architecture | With News MCC | Without News MCC | Delta | p-value | Cohen's d | Effect | Sig | n |
|-------------|--------------|-----------------|-------|---------|-----------|--------|-----|---|
| FF | 0.0059 | 0.0071 | +0.0000 | 0.9966 | 0.001 | negligible | ns | 15 |
| LSTM | 0.0101 | 0.0120 | -0.0068 | 0.1323 | -0.427 | small | ns | 15 |

| Fold | Regime | A3 MCC | A7 MCC | Delta | p-value | Sig | News Effect |
|------|--------|------------|------------|-------|---------|-----|-------------|
| F0 | Pre-COVID (2019H2-2020H1) | 0.0264 | 0.0283 | -0.0013 | 0.9053 | ns | NEUTRAL |
| F1 | Recovery (2020H2-2021H1) | 0.0284 | 0.0360 | -0.0119 | 0.0678 | ns | HURTS |
| F2 | Bull-to-Bear (2021H2-2022H1) | -0.0142 | -0.0162 | +0.0048 | 0.4834 | ns | NEUTRAL |
| F3 | Bear Market (2022H2-2023H1) | 0.0060 | 0.0083 | -0.0089 | 0.3637 | ns | HURTS |
| F4 | AI Rally (2023H2) | -0.0067 | -0.0088 | +0.0004 | 0.9321 | ns | NEUTRAL |

  News hurts in: F1, F3

### News over price+macro
*Does news add value beyond macro?*

**Overall**: A2 MCC=0.0028 vs A4 MCC=0.0126
  Delta: -0.0098, p_raw=0.0012, p_bonf=0.0036 (k=3), d=-0.355 -> **NEWS HURTS**

**Per-architecture split** (A2 vs A4):

| Architecture | With News MCC | Without News MCC | Delta | p-value | Cohen's d | Effect | Sig | n |
|-------------|--------------|-----------------|-------|---------|-----------|--------|-----|---|
| FF | -0.0014 | 0.0102 | -0.0116 | 0.0049 | -0.447 | small | ** | 45 |
| LSTM | 0.0069 | 0.0149 | -0.0080 | 0.0748 | -0.275 | small | ns | 45 |

| Fold | Regime | A2 MCC | A4 MCC | Delta | p-value | Sig | News Effect |
|------|--------|------------|------------|-------|---------|-----|-------------|
| F0 | Pre-COVID (2019H2-2020H1) | -0.0034 | -0.0036 | +0.0002 | 0.9784 | ns | NEUTRAL |
| F1 | Recovery (2020H2-2021H1) | 0.0338 | 0.0340 | -0.0002 | 0.9750 | ns | NEUTRAL |
| F2 | Bull-to-Bear (2021H2-2022H1) | -0.0092 | -0.0036 | -0.0056 | 0.2026 | ns | HURTS |
| F3 | Bear Market (2022H2-2023H1) | 0.0155 | 0.0296 | -0.0141 | 0.0432 | * | HURTS |
| F4 | AI Rally (2023H2) | -0.0229 | 0.0064 | -0.0293 | 0.0014 | ** | HURTS |

  News hurts in: F2, F3, F4

### News over price+macro+social
*Does news help in the full model?*

**Overall**: A1 MCC=0.0061 vs A5 MCC=0.0072
  Delta: -0.0011, p_raw=0.6976, p_bonf=1.0000 (k=3), d=-0.041 -> **NO EFFECT**

**Per-architecture split** (A1 vs A5):

| Architecture | With News MCC | Without News MCC | Delta | p-value | Cohen's d | Effect | Sig | n |
|-------------|--------------|-----------------|-------|---------|-----------|--------|-----|---|
| FF | 0.0014 | 0.0016 | -0.0002 | 0.9515 | -0.009 | negligible | ns | 45 |
| LSTM | 0.0108 | 0.0127 | -0.0019 | 0.6358 | -0.072 | negligible | ns | 45 |

| Fold | Regime | A1 MCC | A5 MCC | Delta | p-value | Sig | News Effect |
|------|--------|------------|------------|-------|---------|-----|-------------|
| F0 | Pre-COVID (2019H2-2020H1) | -0.0009 | -0.0062 | +0.0053 | 0.2193 | ns | HELPS |
| F1 | Recovery (2020H2-2021H1) | 0.0275 | 0.0260 | +0.0015 | 0.7747 | ns | NEUTRAL |
| F2 | Bull-to-Bear (2021H2-2022H1) | -0.0049 | -0.0024 | -0.0025 | 0.4154 | ns | NEUTRAL |
| F3 | Bear Market (2022H2-2023H1) | 0.0276 | 0.0223 | +0.0053 | 0.4767 | ns | HELPS |
| F4 | AI Rally (2023H2) | -0.0187 | -0.0038 | -0.0149 | 0.0963 | ns | HURTS |

  News helps in: F0, F3
  News hurts in: F4

### News Degradation Summary

- **News over price-only**: NO EFFECT (p_bonf=0.9382)
  - FF: delta=+0.0000, p=0.9966, d=0.001 (helps)
  - LSTM: delta=-0.0068, p=0.1323, d=-0.427 (hurts)
- **News over price+macro**: NEWS HURTS (p_bonf=0.0036)
  - FF: delta=-0.0116, p=0.0049, d=-0.447 (hurts)
  - LSTM: delta=-0.0080, p=0.0748, d=-0.275 (hurts)
- **News over price+macro+social**: NO EFFECT (p_bonf=1.0000)
  - FF: delta=-0.0002, p=0.9515, d=-0.009 (hurts)
  - LSTM: delta=-0.0019, p=0.6358, d=-0.072 (hurts)


---
## Table 4.13: Fusion Strategy Comparison (Friedman Test)

Subjects with all 3 fusions: 240

### Fusion Descriptive Statistics (Multi-Source Only)

| Fusion | Mean MCC | Std | Median | N |
|--------|----------|-----|--------|---|
| concat | 0.0083 | 0.0243 | 0.0038 | 240 |
| gated_cross_attention | 0.0068 | 0.0236 | 0.0062 | 240 |
| mha | 0.0082 | 0.0242 | 0.0084 | 240 |

### By Architecture

| Fusion | FF Mean | LSTM Mean |
|--------|---------|-----------|
| concat | 0.0051 | 0.0114 |
| gated_cross_attention | 0.0031 | 0.0105 |
| mha | 0.0042 | 0.0122 |

### Friedman Test
- Chi-square statistic: 1.3000
- p-value: 0.522046
- Significance: ns

Friedman test not significant — no evidence that fusion strategies differ.
All fusion strategies perform comparably on multi-source configs.


---
## Table 4.11: Best Fusion by Fold and Architecture

| Fold | Regime | Architecture | concat | gated_cross_attention | mha | Best | Friedman p | Sig |
|------|--------|-------------|-------|-------|-------|------|-----------|-----|
| F0 | Pre-COVID (2019H2-2020H1) | FF | -0.0006 | 0.0002 | 0.0009 | mha | 0.6873 | ns |
| F0 | Pre-COVID (2019H2-2020H1) | LSTM | 0.0084 | 0.0126 | 0.0152 | mha | 0.1667 | ns |
| F1 | Recovery (2020H2-2021H1) | FF | 0.0279 | 0.0332 | 0.0287 | gated_cross_attention | 0.4531 | ns |
| F1 | Recovery (2020H2-2021H1) | LSTM | 0.0322 | 0.0189 | 0.0289 | concat | 0.0302 | * |
| F2 | Bull-to-Bear (2021H2-2022H1) | FF | -0.0049 | -0.0102 | -0.0064 | concat | 0.4531 | ns |
| F2 | Bull-to-Bear (2021H2-2022H1) | LSTM | -0.0038 | -0.0077 | -0.0042 | concat | 0.4531 | ns |
| F3 | Bear Market (2022H2-2023H1) | FF | 0.0091 | 0.0073 | 0.0044 | concat | 0.7470 | ns |
| F3 | Bear Market (2022H2-2023H1) | LSTM | 0.0254 | 0.0289 | 0.0228 | gated_cross_attention | 0.6873 | ns |
| F4 | AI Rally (2023H2) | FF | -0.0058 | -0.0151 | -0.0067 | concat | 0.1969 | ns |
| F4 | AI Rally (2023H2) | LSTM | -0.0051 | -0.0002 | -0.0019 | gated_cross_attention | 0.5134 | ns |

**Note**: Per-fold Friedman tests are exploratory (small n per cell).
No fold-level fusion difference reaches significance in prior analyses.


---
## Table 4.12: Architecture Comparison (All Configs Pooled)

| Architecture | Mean MCC | 95% CI | Std | Median | n |
|-------------|----------|--------|-----|--------|---|
| FF | 0.0042 | [0.0018, 0.0067] | 0.0242 | 0.0013 | 375 |
| LSTM | 0.0114 | [0.0090, 0.0138] | 0.0233 | 0.0120 | 375 |

### Paired Comparison (LSTM vs FF)

| Metric | Value |
|--------|-------|
| Paired observations | 375 |
| Mean delta (LSTM - FF) | +0.0071 |
| 95% CI of delta | [0.0045, 0.0098] |
| t-statistic | 5.342 |
| p-value | 0.000000 |
| Cohen's d | 0.276 (small) |
| LSTM win rate | 221/375 (58.9%) |

### Per-Config Architecture Delta

| Config | FF MCC | LSTM MCC | Delta | p-value | Cohen's d | Effect | Sig | n (pairs) |
|--------|--------|----------|-------|---------|-----------|--------|-----|-----------|
| A1 | 0.0014 | 0.0108 | +0.0095 | 0.0226 | 0.356 | small | * | 45 |
| A2 | -0.0014 | 0.0069 | +0.0083 | 0.0407 | 0.318 | small | * | 45 |
| A3 | 0.0059 | 0.0101 | +0.0042 | 0.1733 | 0.209 | small | ns | 45 |
| A4 | 0.0102 | 0.0149 | +0.0047 | 0.3288 | 0.149 | negligible | ns | 45 |
| A5 | 0.0016 | 0.0127 | +0.0111 | 0.0056 | 0.439 | small | ** | 45 |
| A6 | 0.0061 | 0.0135 | +0.0074 | 0.0080 | 0.419 | small | ** | 45 |
| A7 | 0.0071 | 0.0120 | +0.0049 | 0.5035 | 0.184 | negligible | ns | 15 |
| A8 | 0.0062 | 0.0065 | +0.0004 | 0.8952 | 0.020 | negligible | ns | 45 |
| A9 | 0.0030 | 0.0154 | +0.0124 | 0.0202 | 0.363 | small | * | 45 |


---
## Table 4.14: Training Dynamics Summary

| Architecture | Mean Epochs | Std Epochs | Median Epochs | Val-Test Gap | Gap Std | Mean Params | Param Range |
|-------------|-------------|------------|---------------|-------------|---------|-------------|-------------|
| FF | 60.2 | 30.1 | 52 | 0.0276 | 0.0349 | 24542 | 9251-36003 |
| LSTM | 67.9 | 33.3 | 61 | 0.0216 | 0.0352 | 76574 | 61283-88035 |

### Training Dynamics by Config

| Config | Mean Epochs | Early Stop Rate | Val-Test Gap | Mean Params |
|--------|-------------|-----------------|-------------|-------------|
| A1 | 71.6 | 97% | 0.0287 | 53400 |
| A2 | 66.9 | 93% | 0.0294 | 50691 |
| A3 | 54.9 | 100% | 0.0198 | 48686 |
| A4 | 67.8 | 96% | 0.0227 | 48430 |
| A5 | 63.9 | 98% | 0.0291 | 51139 |
| A6 | 62.0 | 99% | 0.0203 | 49134 |
| A7 | 54.1 | 100% | 0.0182 | 35267 |
| A8 | 62.7 | 99% | 0.0207 | 53038 |
| A9 | 66.0 | 98% | 0.0280 | 55043 |


---
## Table 4.15: Computational Cost

| Architecture | Mean Runtime (s) | Std (s) | Total Runtime (h) | Mean Params | Experiments |
|-------------|-----------------|---------|-------------------|-------------|-------------|
| FF | 971.0 | 2960.3 | 101.1 | 24542 | 375 |
| LSTM | 1110.8 | 3328.3 | 115.7 | 76574 | 375 |
| **Total** | 1040.9 | 3152.6 | 216.9 | 50558 | 750 |

### Runtime by Config

| Config | Mean (s) | Total (h) | n |
|--------|----------|-----------|---|
| A1 | 1090.2 | 27.3 | 90 |
| A2 | 1886.6 | 47.2 | 90 |
| A3 | 1203.3 | 30.1 | 90 |
| A4 | 686.4 | 17.2 | 90 |
| A5 | 610.7 | 15.3 | 90 |
| A6 | 661.9 | 16.5 | 90 |
| A7 | 448.6 | 3.7 | 30 |
| A8 | 1174.1 | 29.4 | 90 |
| A9 | 1211.7 | 30.3 | 90 |


---
## Financial Metrics Summary

**Note**: cumulative_return and max_drawdown excluded (computed with incorrect formula in pre-fix experiment runs).

| Config | Architecture | Mean Sharpe | Sharpe Std | Profit Factor | Dir Accuracy | n |
|--------|-------------|------------|-----------|--------------|-------------|---|
| A1 | FF | 0.006 | 0.507 | 1.004 | 0.5057 | 45 |
| A1 | LSTM | 0.221 | 0.497 | 1.039 | 0.5147 | 45 |
| A2 | FF | 0.038 | 0.538 | 1.009 | 0.5101 | 45 |
| A2 | LSTM | -0.115 | 0.517 | 0.985 | 0.5044 | 45 |
| A3 | FF | 0.333 | 0.578 | 1.060 | 0.5178 | 45 |
| A3 | LSTM | 0.259 | 0.589 | 1.047 | 0.5144 | 45 |
| A4 | FF | -0.007 | 0.469 | 1.001 | 0.5053 | 45 |
| A4 | LSTM | 0.031 | 0.433 | 1.007 | 0.5072 | 45 |
| A5 | FF | 0.105 | 0.484 | 1.020 | 0.5088 | 45 |
| A5 | LSTM | 0.049 | 0.450 | 1.010 | 0.5083 | 45 |
| A6 | FF | 0.377 | 0.510 | 1.067 | 0.5195 | 45 |
| A6 | LSTM | 0.403 | 0.530 | 1.072 | 0.5207 | 45 |
| A7 | FF | 0.427 | 0.446 | 1.075 | 0.5225 | 15 |
| A7 | LSTM | 0.512 | 0.470 | 1.090 | 0.5201 | 15 |
| A8 | FF | 0.303 | 0.494 | 1.056 | 0.5143 | 45 |
| A8 | LSTM | 0.240 | 0.477 | 1.043 | 0.5112 | 45 |
| A9 | FF | 0.117 | 0.381 | 1.021 | 0.5092 | 45 |
| A9 | LSTM | 0.116 | 0.450 | 1.021 | 0.5111 | 45 |

**Overall Sharpe ratio**: mean=0.167, range=[-1.378, 1.467]

**Note**: A statistically positive MCC does not imply profitable trading without proper position sizing, transaction costs, and risk management.


---
## Tables 4.16-4.17: Literature Performance and Methodology Comparison

### Table 4.16: Performance Comparison with Published Literature

## Table 3: Our Results vs Literature

**Our best mean MCC**: 0.0126 (config A4, averaged over seeds x folds)
**Our best single-run MCC**: 0.0847
**Total experiments**: 1110
**Statistical rigor**: 3 seeds x 5 folds with paired significance tests

| Paper | Their MCC | Our Mean MCC | Gap | Key Methodological Difference |
|-------|-----------|-------------|-----|-------------------------------|
| MSGCA (BigData22) | 0.1112 | 0.0126 | -0.0986 | **Reported MCC inflated ~953% by test-set epoch cherry-picking (honest MCC: -0.008, see Table 6)** |
| MSGCA (CIKM18) | 0.0807 | 0.0126 | -0.0681 | Same evaluation bias likely applies; honest MCC unknown for this dataset |
| MSGCA (ACL18) | 0.0593 | 0.0126 | -0.0467 | Same evaluation bias likely applies; honest MCC unknown for this dataset |
| MSGCA (InnoStock) | 0.0550 | 0.0126 | -0.0424 | Same evaluation bias likely applies; honest MCC unknown for this dataset |
| StockNet | 0.0808 | 0.0126 | -0.0682 | GloVe+BiGRU, 88 stocks, 38.72% filtered by threshold, 2-year period |
| STONK (best) | 0.3300 | 0.0126 | -0.3174 | DeBERTa fine-tuned, S&P500 index-level, FinSen 160K articles, logistic regression |
| STONK (concat) | 0.3100 | 0.0126 | -0.2974 | MiniLM fine-tuned, S&P500 index-level, binary on index |

### Our Top 5 Configurations (by mean MCC)

| Config | Fusion | Architecture | Mean MCC | Std | n |
|--------|--------|-------------|----------|-----|---|
| A4 | mha | LSTM | 0.0205 | 0.0217 | 15 |
| A9 | concat | LSTM | 0.0182 | 0.0320 | 30 |
| A4 | gated_cross_attention | LSTM | 0.0170 | 0.0235 | 15 |
| A9 | mha | LSTM | 0.0168 | 0.0257 | 30 |
| A5 | concat | LSTM | 0.0160 | 0.0262 | 30 |

### Table 4.17: Statistical Rigor Comparison

## Table 4: Statistical Rigor Comparison

| Paper | Seeds | Folds | Total Runs | Significance Tests | Regime Analysis | Multi-Task |
|-------|-------|-------|------------|-------------------|----------------|------------|
| **Ours** | 3 | 5 | 750+ | Yes (paired t, Bonferroni) | Yes (5 regimes) | Yes |
| MSGCA | 5 | 1 | ~20 | No | No | No |
| StockNet | 1 | 1 | 1 | No | No | No |
| STONK | ? | 5 | ~50 | No | No | No |
| CAMEF | ? | 1 | ~30 | No | No | No |
| CMTF | ? | 1 | ~5 | No | No | No |
| MSMF | ? | ? | ? | No | No | Yes |

### Evaluation Methodology Differences (13 dimensions)

## Table 5: Evaluation Methodology Comparison

This table documents the evaluation methodology of each paper in detail. It supports the argument that our lower absolute MCC values reflect more honest evaluation, not inferior models.

| Dimension | **Ours** | **MSGCA** | **StockNet** | **STONK** | **CAMEF** | **CMTF** |
|-----------|---------|----------|------------|---------|---------|--------|
| Train/test split method | Expanding temporal window (5 folds, strictly chronological, no future leakage) | Random stock-level sampling (`random.sample`, main.py:50); temporal date sliding within each split | Chronological 19/2/3 months (temporal, no leakage) | 5-fold TimeSeriesSplit (temporal, no leakage) | 60/20/20 chronological (temporal, no leakage) | 60/20/20 chronological (temporal, no leakage) |
| Validation set | Yes — early stopping on validation MCC, test set never seen during training | **None** — no validation set used; model evaluated directly on test set every epoch (main.py:132-138) | Yes — 2-month dev set for hyperparameter tuning | Not specified — unclear if separate validation exists within each fold | Yes — 20% validation split | Yes — 20% validation split (Optuna hyperparameter search) |
| Model selection criterion | Best validation MCC (patience-based early stopping); test set used only for final evaluation | **Best test-set MCC across all epochs** (main.py:109,137: `if mcc > best_mcc: best_mcc = mcc`). This is test-set selection bias. | Not fully specified; standard practice would use dev set | Not specified | Not specified (10 epochs, likely uses validation loss) | Optuna TPE on validation set |
| Test leakage risk | **None** — test set isolated behind validation-based early stopping | **High** — model is selected based on best test-epoch performance. Reported MCC is the maximum over ~200-400 epochs evaluated on the test set, not a single held-out evaluation. | **Low** — proper temporal split with dev set, but only 1 seed | **Low** — temporal splits, but unclear if validation is separate | **Low** — proper chronological split with validation | **Low** — Optuna searches on validation, final eval on test |
| Number of seeds | 3 (42, 123, 456) | 5 (average + variance reported) | 1 (single run) | Not specified | Not specified | Not specified (Optuna trials may vary) |
| Number of temporal folds | 5 (covering 5 distinct market regimes: pre-COVID, recovery, bull-to-bear, bear market, AI rally) | 1 (single chronological split per dataset) | 1 (fixed 19/2/3 month split) | 5 (TimeSeriesSplit) | 1 (fixed 60/20/20 split) | 1 (fixed 60/20/20 split) |
| Total evaluation runs | 750+ (9 configs x 3 fusions x 2 archs x 3 seeds x 5 folds) | ~20 (1 config x 5 seeds x 4 datasets) | 1 per configuration | ~50 (5 folds x ~10 encoder configs) | ~30 (multiple ablations x 1 split) | ~5 (5 stocks x 1 split, Optuna internal trials separate) |
| Reported metric | Mean +/- std over all seeds and folds | Best single test-epoch result averaged over 5 seeds. Note: 'best' is selected per-epoch on test, inflating the reported value. | Single test-set result | Mean over 5 folds (but single seed unclear) | Single test-set result | Single test-set result |
| Significance tests | Yes — paired t-tests with Bonferroni correction, Cohen's d effect sizes | No | No | No | No | No |
| Movement threshold / class definition | 0.5% deadzone excluded from dataset (binary: up/down on remaining samples) | Dataset-specific thresholds (0.3-1.0%); 3-class (up/flat/down) | 0.5%/0.55% threshold; **38.72% of samples filtered** — only predicts clear movements | No threshold; binary (up/down) on all samples | N/A (regression task, not classification) | No threshold; binary (up/down) on all samples |
| Text encoder adaptation | Frozen FinBERT, mean-pooled to 10-dim features | Frozen OpenAI ada-002 (1536-dim), zero-filled for missing timestamps | GloVe + BiGRU (end-to-end trained, pre-transformer era) | DeBERTa/MiniLM fine-tuned on FiQA + FinancialPhraseBank (5.8K samples) | RoBERTa (last layer trainable), MOMENT (pre-trained TS model) | Llama-3.1-8B (frozen, reports) + CatBoost (news sentiment) |
| Dataset size (test period) | ~2,016 trading days across 55 stocks (8 years, 2016-2023) | ~242-504 days per dataset (1-2 years) | ~504 days, 88 stocks (2 years, 2014-2016) | ~4,032 days, index-level (16 years, 2007-2023) | ~4,032 days, 5 assets at 5-min freq (16 years, 2008-2024) | ~1,360 days, 5 stocks (5 years, 2019-2024) |
| Regime analysis | Yes — 5 distinct market regimes analyzed separately | No | No | No | No (but event-type analysis provided) | No |

### MSGCA Test-Set Selection Bias: Code Evidence

The MSGCA codebase (publicly available at `github.com/changzong/MSGCA`, file `main.py`) reveals a critical evaluation methodology issue:

```python
# main.py lines 49-51: Random (not temporal) stock split
idxs = list(range(len(input_data[0])))
train_idxs = random.sample(idxs, int(len(input_data[0])*args.sample_ratio))
test_idxs = list(set(idxs)^set(train_idxs))

# main.py lines 109, 134-138: Best-test-epoch selection (no validation set)
best_acc = 0.0
best_mcc = 0.0
for epoch in range(args.epoch_num):  # 200-400 epochs
    # ... training ...
    acc, mcc = model(test_set, ...)  # evaluate on TEST every epoch
    if acc > best_acc:
        best_acc = acc
    if mcc > best_mcc:
        best_mcc = mcc
print('Best result: ACC: ' + str(best_acc) + ' MCC: ' + str(best_mcc))
```

**Impact**: The reported MCC is the maximum over all ~200-400 training epochs evaluated on the test set. This is equivalent to using the test set for model selection (choosing the best-performing epoch), which inflates reported metrics. Additionally, `best_acc` and `best_mcc` are tracked independently — they may come from different epochs, reporting a combination that never existed simultaneously.

With proper validation-based early stopping (as in our approach), the model stops at the epoch with best validation performance, and the test set is evaluated only once. The gap between 'best test-epoch MCC' and 'single test evaluation MCC' can be substantial, especially over hundreds of epochs where random fluctuations in test performance are captured as 'best results'.

### Methodological note: Honest Evaluation vs. Inflated Numbers

Our evaluation methodology is deliberately conservative across every dimension:

1. **No test-set peeking**: We use validation-based early stopping. The test set is evaluated exactly once per run, after training completes. MSGCA evaluates on the test set every epoch and reports the best. Our empirical diagnostic (Table 6) shows this inflates their MCC from -0.008 (honest) to 0.116 (reported) — a **953% inflation**.

2. **Mean over 15 runs, not best single run**: We report mean MCC across 3 seeds x 5 folds = 15 independent runs. Most papers report their single best configuration or average over seeds but with a single temporal split. Our means are lower by construction but more representative of expected real-world performance.

3. **Five market regimes, not one favorable period**: Our 5-fold expanding window covers pre-COVID stability, COVID recovery, bull-to-bear transition, bear market, and AI rally. Papers testing on a single 1-2 year period may inadvertently select a favorable regime. Our worst fold (F2, bull-to-bear) has negative MCC, which honestly reflects model limitations during regime transitions.

4. **Individual stocks, not indices**: We predict direction for 55 individual stocks across 11 sectors. STONK (MCC=0.33) predicts S&P 500 index direction, which is inherently easier due to diversification smoothing. StockNet filters 38.72% of ambiguous samples, predicting only clear movements.

5. **Statistical significance testing**: We apply paired t-tests with Bonferroni correction and report Cohen's d effect sizes. No comparable paper in our comparison set performs significance testing on their results. This means their reported improvements over baselines may not be statistically significant.

**Bottom line**: A direct MCC comparison between our results and the literature is methodologically invalid. The contribution is not beating SOTA on a single metric, but providing the most rigorous empirical analysis of when and why multi-source integration helps, with evaluation methodology that would survive scrutiny from a statistics reviewer.

### MSGCA Evaluation Bias Summary

- Reported MCC: 0.1157
- Honest MCC (val-selected): -0.0077
- Inflation: 953%
- Full diagnostic in `experiments/phase4r/analysis/msgca_diagnostic.json`



---
## Table 4.18: Summary of Key Findings

| # | RQ | Finding | Evidence | Effect Size | Significance |
|---|-----|---------|----------|-------------|--------------|
| 1 | RQ1 | No modality significantly beats price-only baseline | Best: A5 vs A7, delta=+0.0047 | d=0.139 (negligible) | p=0.4603 (all p_bonf > 0.05) |
| 2 | RQ1 | More sources does not mean better (4/8 configs > A7) | 8 configs tested vs A7 | negligible deltas | all ns after Bonferroni |
| 3 | RQ2 | Source ranking: Social (+0.0005) > Graph (+0.0000) > Macro (-0.0014) > News (-0.0034) | Additive delta vs A7 (Table 4.6) | all negligible | all ns |
| 4 | RQ2 | A4 (price+macro) is performance ceiling | A5 vs A4: -0.0054; A9 vs A4: -0.0033 | d=-0.201, -0.116 | p=0.0610, 0.2782 |
| 5 | RQ3 | LSTM > FF, especially in volatile regimes | Overall LSTM-FF; strongest: F3 (Bear Market (2022H2-2023H1)) | overall d=0.276; best fold d=0.830 | p=0.000000 *** |
| 6 | RQ3 | Social + LSTM significant interaction (A5: Price + macro + social) | A5 LSTM vs FF | d=0.439 (small) | p=0.0056 ** |
| 7 | RQ3 | Social + LSTM significant interaction (A6: Price + social) | A6 LSTM vs FF | d=0.419 (small) | p=0.0080 ** |
| 8 | RQ3 | News over macro: DEGRADES | A2 vs A4, delta=-0.0098 | d=-0.355 (small) | p_raw=0.0012, p_bonf=0.0048 |
| 9 | Arch/Fusion | All fusion strategies equivalent | Friedman test on 240 matched subjects | chi2=1.300 | p=0.5220 ns |
| 10 | RQ3 | Graph architecture-dependent (exploratory): FF -0.0009, LSTM -0.0055 | A8 vs A7 split by architecture (ns both directions) | opposite signs, neither significant | ns (exploratory observation) |

**Total findings**: 10 (3 statistically significant)

### Narrative Summary

**RQ1**: No modality significantly beats price-only baseline; More sources does not mean better (4/8 configs > A7)
**RQ2**: Source ranking: Social (+0.0005) > Graph (+0.0000) > Macro (-0.0014) > News (-0.0034); A4 (price+macro) is performance ceiling
**RQ3**: LSTM > FF, especially in volatile regimes; Social + LSTM significant interaction (A5: Price + macro + social); Social + LSTM significant interaction (A6: Price + social); News over macro: DEGRADES; Graph architecture-dependent (exploratory): FF -0.0009, LSTM -0.0055
**Architecture/Fusion**: All fusion strategies equivalent
