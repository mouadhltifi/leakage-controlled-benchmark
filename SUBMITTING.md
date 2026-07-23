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
| Features | availability-timed five-source tables (schema: `DATASHEET.md`, `croissant.json`); text is T−1 aligned, macro enters at publication dates. Data ships committed under `data/processed/` (hashes in `MANIFEST.sha256`; also DOI-archived on Zenodo); regeneration per `DATA-STATEMENTS.md` |
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
   close). A `--restrict-folds` claim can reach SUPPORTED on the declared
   subset when it spans at least four folds; the verdict is tagged
   RESTRICTED COVERAGE so its scope is always visible.

## Submission file formats

**Predictions mode (preferred — the evaluator recomputes the metric
itself):** a CSV with one row per (seed, test example):

```
seed,date,ticker,y_pred
42,2019-07-01,AAPL,1
42,2019-07-01,MSFT,-1
...
```

`y_pred` in {-1, +1} (or {0, 1}). Every seed must cover the frozen test set
exactly (the 44,438 scoreable rows of `data/processed/labels_direction.parquet`;
dead-zone days are not scoreable) — any missing or extra row is
NOT COMPARABLE. MCC is recomputed against the frozen labels, so assembly
conformance is verified by construction.

**Score mode:** a CSV with one row per (fold, seed) test evaluation:

```
challenger,fold_idx,seed,mcc,n_test
my_model,0,42,0.0113,10100
my_model,0,123,0.0071,10100
...
```

Required: `challenger`, `fold_idx` (0–4), `seed`, `mcc`, and `n_test` —
without per-row `n_test` matching the frozen folds the claim is
**UNCERTIFIED** (assembly unverified); a mismatching `n_test` is
NOT COMPARABLE. All 5 folds, ≥3 seeds. Optional: `accuracy`, `f1`.

## Computing the claim

```
# predictions mode (preferred):
python3 scripts/analysis/evaluate_submission.py --predictions preds.csv \
    --k K --baseline-arch ff
# score mode:
python3 scripts/analysis/evaluate_submission.py submission.csv \
    --k K --baseline-arch ff
```

Two arguments are **required and fail closed**:

- `--baseline-arch ff|lstm` declares the runnable baseline arm your model
  is paired against, like for like. `--baseline-arch envelope` (the
  per-cell max over both arms) exists as an explicit conservative
  **sensitivity read only** — it is test-selected and not a runnable
  model, so an envelope-referenced claim can never be SUPPORTED.
- `--k` declares your comparison-family size (rule 3: everything you ran,
  not everything you report). Without it the verdict is
  **UNCERTIFIED — comparison family undeclared**.

The script prints the claim block: per-fold deltas, ΔMCC, fold-block
bootstrap 95% CI (descriptive), the certifying fold-t CI, p_fold (n=5),
Bonferroni-corrected p at your declared k, descriptive pooled d, the
per-fold contrast against the untuned logistic-price anchor (the second
leg of the dual bar: the anchor sits descriptively above the tuned
neural baseline at +0.014 pooled, and certification requires clearing
it), and the verdict:

- **SUPPORTED** — positive delta with corrected fold-level significance
  against the declared tuned arm, **and** a positive fold-level contrast
  against the untuned logistic-price anchor (the dual bar: C1 exists to
  kill weak-baseline inflation, so a claim can never certify below a
  simple runnable price model — the anchor leg is a hard floor whose $p$
  is reported), with full coverage, the three contract seeds per fold,
  verified assembly, and a runnable baseline arm (rules 1–6 honored).
  This is a Level-1 claim until released (Level 2) and independently
  audited (Level 3).
- **WITHIN THE REFERENCE NULL** — the reference outcome: no combination
  of news, social, macro, or graph features has cleared it (2,468 runs;
  2,151 controlled).
- **UNCERTIFIED / NOT COMPARABLE** — the claim cannot be certified as
  submitted (undeclared family, unverified or mismatched assembly).

A worked example (a shipped price+news configuration replayed as if it
were an external challenger) lives in `examples/demo_submission/` with its
committed claim block, generated by the canonical command
(`--k 1 --baseline-arch ff`; `--k 1` is honest only because the replay is
a single configuration -- a submitter who swept configurations must
declare the full family, rule 3): ΔMCC $+0.0033$, within the reference null —
the same block the paper's Section 6 prints. The explicit envelope
sensitivity read is $-0.0066$.

## Releasing (Levels 2–3)

Release runnable code, configuration, and seeds such that your claimed number
regenerates; state your family k and any deviation from rules 1–8. To take a
claim to Level 3, an independent party re-runs it AND reads the code against
the controls (selection touches only the validation split; no data path
reaches the test period; timing honored). The audits in the paper's §4 are
the template; `audits/` and `results/analysis/` show what that looks like in
practice.
