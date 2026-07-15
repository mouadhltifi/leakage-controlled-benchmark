# Equivalence Gate Report — A7 Price-Only

**Result**: FAILED

## Summary

- Runs in v2 CSV: 15 (expected 15: 5 folds x 3 seeds)
- Mean tolerance:     +/- 0.0050
- Per-fold tolerance: +/- 0.0150

## Mean MCC

| Source | Mean MCC |
| --- | --- |
| v1 archived | +0.0071 |
| v2 (this sweep) | +0.0058 |
| diff (v2 - v1) | -0.0013 |

## Per-Fold MCC

| Fold | v1 archived | v2 this sweep | diff | within +/- tol? |
| --- | --- | --- | --- | --- |
| F0 | +0.0263 | +0.0123 | -0.0140 | yes |
| F1 | +0.0459 | +0.0299 | -0.0160 | NO |
| F2 | -0.0085 | -0.0066 | +0.0019 | yes |
| F3 | -0.0093 | +0.0053 | +0.0146 | yes |
| F4 | -0.0191 | -0.0117 | +0.0074 | yes |

## Notes

- equivalence gate FAILED: see breakdown below.
- per-fold drift:
-   F1: v2=+0.0299, v1=+0.0459, diff=-0.0160, tolerance=+/-0.0150.

## Wall-time

- Sweep elapsed: 645.2 seconds
