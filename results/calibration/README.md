# Calibration grid (the 117-run phase behind C1's frozen defaults)

The disclosed calibration phase that froze the shared training defaults
before the reference grids ran: four validation-metric studies
(dead-zone threshold, training regime, gradient clipping, hidden size),
116 runs + 1 regression check. Every configuration — baseline and
challengers alike — inherits the same frozen defaults from this grid
(C1's comparable-tuning basis: equal budgets, shared search, validation
metrics only). `phase3r_experiment_matrix.md` is the design;
`study*_analysis.md` are the readouts.

**Recording caveat (Study 1, dead-zone threshold).** In this Phase-3R
calibration harness `best_val_mcc == mcc` for all 60 threshold runs
(`study1_analysis.md` §8): the trainer logged the test MCC into the
`best_val_mcc` field, so the calibration-era validation/test gap is
uninformative. This is a logging quirk, not a leak — early stopping ran
on training loss — and it does not touch the reference grid, whose
selection reads the calendar-tail validation split only (control C3, the
paper's §3.2). The frozen 0.5% dead-zone rests on the literature
convention plus the sweep's null threshold effect (a no-filter control
included; ANOVA across widths p≈0.9997), not on this gap.
