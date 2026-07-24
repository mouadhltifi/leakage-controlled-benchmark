# Economic context — derivation note

The paper's economic-anchor sentence (§3.2, "Tasks, harness, and
reference results") makes three quantitative
claims. Each derives from released quantities; this note is the trail.

## 1. "Near 52% directional accuracy at a gross annualized Sharpe of ≈0.4–0.5"

Source: the study's historical 750-run grid,
`results/analysis/phase5_tables_750core.md` (provenance: v1 codebase —
see the provenance section of the README; these numbers set *scale*, they
are not reference-floor rows). Price-only baseline (A7):

- Directional accuracy: 0.522 (FF) – 0.520 (LSTM).
- Sharpe: 0.427 (FF), 0.512 (LSTM) → "≈0.4–0.5".

Definition: sign-of-prediction long/short next-day return, annualized by
√252, **gross** — no transaction costs, no slippage, no capacity or
borrow constraints. Reported for scale, not as a strategy.

## 2. "The smallest delta the design resolves under even its most liberal pooling (≈+0.004 MCC, balanced accuracy 50.19%)"

The most liberal (explicitly anti-conservative) pooling in the design
treats every fold × seed × architecture cell of a paired comparison as an
observation. At its largest (an architecture contrast) n = 375 paired
cells. Minimum detectable effect at α = 0.05 (two-sided), power 0.80:

    d_MDE = (1.96 + 0.84) / √375 ≈ 0.145

The scale factor must be the SD of the PAIRED DIFFERENCES, not of either
arm: the contrast's measured paired-delta SD is 0.0259 MCC (the per-arm SDs
are 0.0242 FF / 0.0234 LSTM, which is what "≈0.02" referred to before this
was corrected). So the smallest resolvable delta is
≈ 0.145 × 0.0259 ≈ **+0.0037 MCC**. Every stricter unit
resolves only *larger* deltas, so +0.0037 is the design's floor of
visibility: nothing smaller is detectable under any reading the paper
entertains.

The stricter units, made explicit — and because certification applies the
C4 Bonferroni correction (k = 8 primary configurations), the MDE relevant to
a *certifying* claim is the corrected one:

    per-configuration pooled, n = 30:
        (1.96 + 0.84) / √30 ≈ 0.51   uncorrected
        (2.73 + 0.84) / √30 ≈ 0.65   k = 8 (normal); ≈ 0.70 with the t(29) quantile
    honest fold unit,         n = 5:
        ≈ 1.7   uncorrected (t(4));   ≈ 3   k = 8-corrected (t(4))

So the reference floor's power statement — "80% power near |d| ≈ 0.7 at the
anti-conservative n = 30 pooling, the honest fold unit coarser (|d| ≈ 1.7
uncorrected, ≈ 3 corrected)" — quotes the **k = 8-corrected** MDE at n = 30,
consistent with the corrected fold-unit figure. The uncorrected n = 30 value
(≈ 0.51) is the same formula without the multiplicity penalty; certification
carries the penalty, so 0.7 is the operative number.

Accuracy translation: on the dead-zoned labels, for a predictor whose
predicted class rates match the actual class rates, MCC ≈ 2·(balanced
accuracy) − 1, so

    balanced accuracy = (1 + MCC) / 2 = (1 + 0.0037) / 2 = 50.19%.

This is **balanced** accuracy — a +0.19pp edge over the random anchor's
0.500 — *not* raw accuracy. (Raw accuracy is inflated by class imbalance:
the always-up anchor scores 0.528 raw with MCC 0.0; see
`tables/naive_anchors.tex` and `results/analysis/naive_anchors.json`.)

## 3. "Sits below any plausible round-trip cost"

A sign-of-prediction long/short bet with P(correct) = q on a decisive day
of magnitude m earns expected P&L = (2q − 1)·m per round trip. For a
balanced predictor MCC = 2q − 1 (section 2's identity), so

    E[gross edge per round trip] = MCC × E[|next-day move| given a decisive day]
                                 = 0.0037 × E[|move|].

(The earlier `(balanced accuracy − 0.5)` form double-discounts: it equals
MCC/2, half the correct coefficient.) E[|move|] must be the DECISIVE-day
conditional the formula names: measured on this benchmark's own decisive
test days it is **1.90%** (the 1–1.5% range quoted before this correction is
the ALL-day figure, 1.42% in-window). At 1.90% the floor-sized edge is
≈0.7 bp. For it to reach even **one basis point**, the mean decisive-day
move would need to exceed 2.7% (0.0037 × 0.027 = 1.0 × 10⁻⁴) — well above
liquid large-cap daily norms. Realistic round-trip costs for liquid US large-caps are ≈1–5 bps.
The floor-sized edge is therefore economically invisible before costs are
even estimated precisely, which is why the benchmark's claims are
informational, never trading advice.

## Scope

All three anchors are context, not results: no reference-floor row, test,
or interval in the paper depends on them.
