# v3 M6 Report — A7 Price-Only TFT Baseline

**Data**: `experiments/v3/m6_smoke/a7_price_only.csv`
**Architecture**: Temporal Fusion Transformer (v3)
**Reference**: v2 archived A7 FF baseline (M8 equivalence gate)

## Overall (across 5 folds × 3 seeds)

- **n_runs**: 15
- **v3 mean MCC**: `+0.0027` (std 0.0187)
- **v2 archived MCC**: `+0.0071`
- **diff (v3 − v2)**: `-0.0044`
- **Sanity tolerance**: ±0.02
- **Sanity verdict (mean)**: PASS

## Per-fold MCC (v2 vs v3)

| Fold | v2 archived | v3 (mean ± std) | diff | within ±0.03 |
|---|---|---|---|---|
| F0 | `+0.0263` | `-0.0090` ± 0.0279 | `-0.0353` | NO |
| F1 | `+0.0459` | `+0.0198` ± 0.0074 | `-0.0261` | yes |
| F2 | `-0.0085` | `+0.0136` ± 0.0169 | `+0.0221` | yes |
| F3 | `-0.0093` | `-0.0030` ± 0.0143 | `+0.0063` | yes |
| F4 | `-0.0191` | `-0.0080` ± 0.0088 | `+0.0111` | yes |

**Per-fold sanity verdict**: FAIL

## Quantile calibration

- **Mean coverage_80**: `0.786` (target ≈ 0.8)
- **Mean interval_width**: `2.219` (normalised units)
- **Hard calibration pass** (coverage ∈ [0.5, 0.95]): PASS
- **Soft calibration target** (coverage ∈ [0.75, 0.85]): YES

## Verdict

**v3 A7 baseline shows DEVIATION.** Investigate before M8:
- Per-fold deviation in: F0.

## Notes

- v3 uses the same seeds as v2 ([42, 123, 456]) for direct comparison.
- Exact MCC equivalence is NOT expected — v2 is a FF+LSTM classifier, v3 is a TFT quantile forecaster. Order-of-magnitude sanity per the pre-registered protocol.
- v3 MCC derives from `sign(q_median)`, not a trained classification head.