# Submitting a model to the benchmark

This is the I/O contract. A submission is: (1) test-set predictions produced
under the frozen design below, (2) the claim block computed by
`scripts/analysis/evaluate_submission.py`, and (3) a release that lets an
independent party regenerate both (audit ladder Level 2; Level 3 when someone
does). Nothing else counts as "beating the benchmark".

## The frozen design (challengers vary only the model)

| Fixed | Value |
|---|---|
| Universe | 55 liquid US large-caps (11 sectors × 5 names; the full ticker list is in `DATASHEET.md` and `src/mmfp/data/universe.py`) |
| Window | Feature rows 2016-01-04 → 2023-12-28 (prices from 2015-01-02 serve only as indicator warm-up) |
| Folds | 5 expanding chronological folds (`fold_idx` 0–4), July-to-June test windows: F0 2019-07-01→2020-06-30 · F1 2020-07-01→2021-06-30 · F2 2021-07-01→2022-06-30 · F3 2022-07-01→2023-06-30 · F4 2023-07-01→2023-12-31 (six-month stub). Train expands from 2016-01-04 (the 2015 price history is indicator warm-up only, never training rows) to each fold's `train_end` = the day before `test_start`. Source of truth: `FOLD_BOUNDARIES` and `TRAIN_START` in `src/mmfp/data/assemble.py` |
| Validation | the last 20% of each fold's training window **by calendar** — model and epoch selection read only this; the test set is read once per run |
| Seeds | 42, 123, 456 (≥3 required; these three pair exactly with the shipped baseline) |
| Labels | next-day direction under the symmetric 0.5% dead-zone (applied to train/val/test alike); secondary: next-day realized volatility |
| Features | availability-timed five-source tables (schema: `DATASHEET.md`, `croissant.json`); text is T−1 aligned, macro enters at publication dates. Data via the DOI deposit (on acceptance) or regeneration per `DATA-STATEMENTS.md` |
| Metric | MCC on the primary task |

Every shipped result row records its `n_train`/`n_val`/`n_test`, so the
assembly is checkable; if your submission includes `n_test`,
`evaluate_submission.py` verifies it against the frozen folds and flags any
mismatch (a misaligned split otherwise produces a normal-looking but
meaningless claim). A challenger may run inside the harness (implement a model against
`src/mmfp`'s config interface — see `configs/mmfp/`) or outside it, provided
every row of the design above is honored and declared.

## What you must not do (and must declare if you deviate)

The reporting rules, condensed (they mirror the paper's §2–§3):

1. **Paired, not pooled-vs-pooled** — report per-fold paired deltas vs the
   shipped price-only baseline under identical folds and seeds.
2. **Honest unit** — significance at the fold level (n=5 independent market
   regimes); seeds within a fold are replicates, not evidence. State n.
3. **Multiplicity** — declare the family size k of configurations you
   compared (everything you ran, not everything you report); Bonferroni.
4. **Effect size** — paired Cohen's d plus the fold-block bootstrap CI.
5. **Selection** — model/epoch selection on the calendar-tail validation
   split only; the test set is read once.
6. **Timing** — no feature enters before it is knowable; trailing-window
   normalization fit on the training slice only.
7. **Power context** — at n=5, only large effects certify; smaller claimed
   effects are estimation reports, not certifications.
8. **Partial coverage** — configurations consuming the social source (ends
   2022-12-30) report the primary contrast on all folds AND restricted to
   folds whose test windows begin within coverage
   (`--restrict-folds 0,1,2,3`; note F3's window runs to 2023-06 so its
   second half is uncovered, F4 lies entirely past coverage. The genuinely
   fully covered restriction is F0–F2; report it too if the verdict is
   close).

## Submission file format

A CSV with one row per (fold, seed) test evaluation:

```
challenger,fold_idx,seed,mcc
my_model,0,42,0.0113
my_model,0,123,0.0071
...
```

Required columns: `challenger`, `fold_idx` (0–4), `seed`, `mcc`. All 5 folds,
≥3 seeds. Optional: `accuracy`, `f1`, `n_test` (recommended — lets auditors
sanity-check the assembly).

## Computing the claim

```
python3 scripts/analysis/evaluate_submission.py path/to/submission.csv --k 1
```

By default the challenger is paired per (fold, seed) against the **stronger
shipped baseline arm** (conservative). If your model belongs to a declared
architecture family, pass `--baseline-arch ff` or `--baseline-arch lstm` for
like-for-like. The script prints the claim block: per-fold deltas, ΔMCC,
fold-block bootstrap 95% CI, p_fold (n=5), Bonferroni-corrected p at your
declared k, descriptive pooled d, and the verdict.

Also read your absolute per-fold MCC against the shipped anchors
(`results/analysis/naive_anchors.json`): the bar C1 enforces is a *tuned*
price-only model, not the strongest possible one, and a simple untuned
logistic-price anchor sits descriptively above it (+0.014 pooled). If your
model does not clear that anchor, say so. The verdict:

- **SUPPORTED** — positive delta with corrected fold-level significance
  (rules 1–6 honored). This is a Level-1 claim until released (Level 2) and
  independently audited (Level 3); the benchmark reports it as
  "claimed-unaudited" until then.
- **WITHIN THE REFERENCE NULL** — everything else. This is the reference
  outcome: no combination of news, social, macro, or graph features has
  cleared it (2,468 runs; 2,151 controlled).

A worked example (a shipped price+news configuration replayed as if it were
an external challenger, evaluated like-for-like against the ff baseline
arm via `--baseline-arch ff`) lives in `examples/demo_submission/` with its
committed claim block. Note the committed claim uses that like-for-like
mode; the bare default pairs against the stronger shipped arm and reads
lower.

## Releasing (Levels 2–3)

Release runnable code, configuration, and seeds such that your claimed number
regenerates; state your family k and any deviation from rules 1–8. To take a
claim to Level 3, an independent party re-runs it AND reads the code against
the controls (selection touches only the validation split; no data path
reaches the test period; timing honored). The audits in the paper's §4 are
the template; `audits/` and `results/analysis/` show what that looks like in
practice.
