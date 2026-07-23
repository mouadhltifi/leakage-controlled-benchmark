# Synthetic positive control — the instrument can say yes

**This challenger is SYNTHETIC**: the shipped ff price-only baseline plus a
uniform +0.05 MCC per (fold, seed) — a literature-magnitude effect, not a
real model. It exists to demonstrate certification reachability: a
consistent effect of the size the multi-source literature claims clears
both bars (tuned arm + classical anchor) and is SUPPORTED. Regenerate:

    python3 scripts/analysis/evaluate_submission.py \
        examples/positive_control/submission.csv --k 1 --baseline-arch ff

The regression battery (`scripts/analysis/test_evaluate_submission.py`)
asserts this case certifies and the adversarial cases refuse.
