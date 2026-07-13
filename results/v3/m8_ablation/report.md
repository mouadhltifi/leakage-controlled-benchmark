# v3 M8 Report — Full 9-Config Ablation

**v3 runs**: 135  ·  **v2 FF+concat matched subset**: 135
**Bonferroni k**: 9  ·  **Publishable threshold**: p_bonf < 0.05 AND |d| ≥ 0.3

## Overall

- **v3 mean MCC (all 135)**: `+0.0099`
- **v2 mean MCC (matched subset)**: `+0.0054`
- **v3 mean coverage_80**: `0.749` (target ≈ 0.8)
- **v3 mean interval_width**: `2.070`

## Per-config paired comparison (v3 vs v2, matched on fold × seed)

| Config | Modalities | n | v2 MCC | v3 MCC | diff | p_raw | p_bonf | d | d 95% CI | Pub? | cov_80 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | price + news + social + macro (full 4-mod) | 15 | `-0.0015` | `+0.0155` | `+0.0170` | 0.0376 | 0.3387 | `+0.593` | [+0.161, +1.142] | no | 0.783 |
| A2 | price + news + macro (A1−social) | 15 | `+0.0005` | `+0.0127` | `+0.0122` | 0.0924 | 0.8314 | `+0.466` | [-0.013, +1.092] | no | 0.746 |
| A3 | price + news | 15 | `+0.0071` | `+0.0055` | `-0.0016` | 0.7708 | 1.0000 | `-0.077` | [-0.571, +0.544] | no | 0.779 |
| A4 | price + macro | 15 | `+0.0090` | `+0.0211` | `+0.0121` | 0.2426 | 1.0000 | `+0.315` | [-0.211, +0.872] | no | 0.719 |
| A5 | price + macro + social | 15 | `+0.0127` | `+0.0145` | `+0.0018` | 0.8248 | 1.0000 | `+0.058` | [-0.613, +0.532] | no | 0.741 |
| A6 | price + social | 15 | `+0.0035` | `-0.0042` | `-0.0076` | 0.1378 | 1.0000 | `-0.406` | [-0.924, +0.061] | no | 0.708 |
| A7 | price only (baseline) | 15 | `+0.0071` | `+0.0027` | `-0.0044` | 0.5502 | 1.0000 | `-0.158` | [-0.772, +0.367] | no | 0.786 |
| A8 | price + graph | 15 | `+0.0070` | `+0.0138` | `+0.0068` | 0.2320 | 1.0000 | `+0.323` | [-0.183, +0.961] | no | 0.716 |
| A9 | price + macro + graph | 15 | `+0.0028` | `+0.0074` | `+0.0046` | 0.6024 | 1.0000 | `+0.138` | [-0.473, +0.602] | no | 0.759 |

## Per-fold breakdown (v3 − v2 diff per config × fold)

| Config | F0 | F1 | F2 | F3 | F4 |
|---|---|---|---|---|---|
| A1 | `+0.0200` | `-0.0087` | `+0.0279` | `+0.0236` | `+0.0223` |
| A2 | `+0.0109` | `-0.0085` | `+0.0218` | `+0.0392` | `-0.0023` |
| A3 | `+0.0023` | `-0.0138` | `-0.0026` | `-0.0037` | `+0.0100` |
| A4 | `+0.0050` | `-0.0053` | `+0.0327` | `+0.0141` | `+0.0139` |
| A5 | `+0.0080` | `-0.0158` | `+0.0067` | `-0.0012` | `+0.0111` |
| A6 | `-0.0057` | `-0.0266` | `+0.0063` | `-0.0136` | `+0.0013` |
| A7 | `-0.0353` | `-0.0262` | `+0.0220` | `+0.0063` | `+0.0111` |
| A8 | `+0.0219` | `+0.0034` | `+0.0023` | `+0.0002` | `+0.0061` |
| A9 | `+0.0060` | `-0.0105` | `+0.0021` | `+0.0270` | `-0.0014` |

## Verdict

**No significant architecture-level difference between v3 TFT and v2 FF classifier.** Null result is ARCHITECTURE-ROBUST across the full 9-config ablation. This STRENGTHENS v2's C2 (robust null): the finding that no modality beats price-only holds under both classifier-shaped (the classifier) AND forecaster-shaped (v3) architectures, across 5 encoders (v2 Phase 2), across 9 modality combinations. Result of interest: v3 confirms the null is signal-limited, not architecture-limited.

**Quantile calibration ACCEPTABLE** (mean cov_80 = 0.749, in hard bounds [0.5, 0.95]).
