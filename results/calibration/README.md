# Calibration grid (the 117-run phase behind C1's frozen defaults)

The disclosed calibration phase that froze the shared training defaults
before the reference grids ran: four validation-metric studies
(dead-zone threshold, training regime, gradient clipping, hidden size),
116 runs + 1 regression check. Every configuration — baseline and
challengers alike — inherits the same frozen defaults from this grid
(C1's comparable-tuning basis: equal budgets, shared search, validation
metrics only). `phase3r_experiment_matrix.md` is the design;
`study*_analysis.md` are the readouts.
