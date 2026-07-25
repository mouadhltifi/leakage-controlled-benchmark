# Changelog

All tags, newest first. Result CSVs under `results/` are unchanged across
every entry below — every reference number regenerates identically at every
tag; the changes are to data-file correctness, tooling, and documentation.
"Archived" = published as a GitHub Release and ingested by Zenodo under the
concept DOI 10.5281/zenodo.21431362.

## v1.0.12 — 2026-07-25

Adds a second classical price-only anchor and states one auditability limit
plainly. Both come from an external pre-submission review. No existing
reference number changes: the six prior anchors, the reference tables, the
labels tables and the Section-6 demo claim block all regenerate identically.

- **New anchor: `gbm-price`.** Reviewers reasonably ask whether the tuned
  *neural* price-only baseline is simply weak, which would make the
  multi-source null an artifact of a soft comparator. The anchors now bracket
  it with a non-linear learner on the identical train-fold price block —
  gradient-boosted trees (`HistGradientBoostingClassifier`, 200 iters, lr
  0.06, depth 4, L2 1.0), same harness assembly, same test rows as the
  logistic anchor. It reads **+0.002** mean test MCC, against the logistic
  anchor's +0.014 and the neural baseline's +0.005. The neural bar therefore
  sits *between* two classical price-only models rather than below them, and
  the logistic anchor remains the binding classical leg of the dual bar.
  `scripts/analysis/make_naive_anchors.py` computes it; self-validation passes.
- **Social auditability limit stated (`DATA-STATEMENTS.md`).** The upstream
  StockTwits archive is no longer retrievable, so the aggregates cannot be
  re-derived from source and the manifest proves file identity, not derivation
  correctness. The statement now says so plainly and names what *is* auditable
  (the derivation code ships; schema, coverage, ranges and the two constant
  placeholder columns are checkable), and notes the direction of the residual
  risk: a derivation error that inflated the social signal would push toward a
  false positive we do not report.

## v1.0.11 — 2026-07-24

Fix release. Everything below was found by adversarial review — two external
model reviews and a seven-lens internal one — and every item was reproduced
before it was fixed. **No reference number moves:** the labels tables, the
reference tables, and the Section-6 demo claim block all regenerate
byte-identically.

- **Certification harness — fabrication tripwire hardened, fails closed.**
  Surfaced by two post-v1.0.10 external adversarial reviews that executed the
  harness. (a) An in-range oracle (predictions echoing the frozen labels:
  finite, within `[-1,1]`, clearing both bars) used to print
  `SUPPORTED  [FABRICATION CHECK ...]`, so a consumer grepping the verdict for
  `SUPPORTED` could read a pass; certification now WITHHOLDS — the verdict token
  becomes `NOT CERTIFIED`, `supported` is set false, and the JSON gains an
  explicit `certified` boolean. (b) The tripwire threshold was 0.5, roughly 6x
  above anything this task produces, so a partial label-echo scoring ~0.42
  certified clean; tightened to 0.15. Calibration, ascending: reference cells
  ~0.01, highest shipped test MCC 0.087, highest validation-selection MCC 0.104,
  the literature's leakage-inflated headline for this task shape 0.116 — 0.15
  clears all four, so it cannot fire on a defensible result. The battery gains
  two cases (in-range oracle, 0.30 partial-echo); the +0.05 positive controls
  (~0.06 max cell) still certify. No reference number changes.
- **Certification harness — two seed-contract bypasses closed.** Both found by
  an external adversarial review and reproduced here. (a) **Seed padding:** the
  gate tested only that the labels 42/123/456 were *present* per fold, never
  that the three rows were distinct runs — so a best-seed-per-fold test-set
  selection, written three times under the three labels, certified SUPPORTED,
  defeating the very gate whose stated purpose is policing that selection (and
  reducing the >=3-seed rule to a formality). Certification now also requires
  the seeds to be genuine replicates: an identical MCC across all three seeds in
  *every* fold is one result relabelled, not three runs. The test is all-folds,
  so a legitimate tie inside a single fold still certifies (measured against the
  shipped grids: zero honest submissions affected). (b) **Restricted-coverage
  seed prune:** the seed contract was evaluated on the post-`--restrict-folds`
  frame while the classical-anchor floor scored the full submitted grid, so a
  fold outside the restriction sat inside the hard floor yet escaped the
  contract — letting a submitter prune that fold to its best seed and push the
  floor over zero. The contract is now evaluated on the full submitted grid.
  Battery gains three cases (seed padding, restricted seed-prune, and a
  false-positive guard).
- **Certification harness — one claim, one model.** The required `challenger`
  column was read once, for a print statement, and never validated, so a file
  assembled by taking the best cell across a family of models (one row per
  cell, eleven different model names) certified SUPPORTED and was reported
  under the first row's name. That is the challenger-side twin of the
  `envelope` baseline the tool already refuses as "test-selected, not a
  runnable model" — the guard was applied to one side of the comparison only.
  A submission mixing challenger names is now NOT COMPARABLE. Both shipped
  examples are unaffected (single name each).
- **Certification harness — boundary validation extended beyond `mcc`.** The
  v1.0.11 hardening covered only the `mcc` column, so malformed `n_test`, a
  non-integer `fold_idx`, a `fold_idx` outside the frozen 0–4 grid, or an
  empty file answered with a raw traceback instead of a verdict — and the
  conformance line printed OK for folds it had never checked. All of these now
  fail closed with a verdict. The JSON claim block also no longer emits bare
  `NaN` tokens, which are not valid JSON (RFC 8259) and made a degenerate
  run's claim unparseable by a strict reader.
- **Regression battery — mutation-tested and the gaps closed.** Mutating each
  certification gate in turn showed five whose deletion changed no assertion:
  the envelope gate, `n_test` conformance, the duplicate-row check, the
  `k`-declared gate, and the fabrication fail-closed behaviour — all because
  those cases asserted against the already-null demo, i.e. they tested that a
  *message* appears, not that a *gate* bites. Each now runs on input that
  would certify if the gate were removed. The battery also never passed
  `--json` (its `import json` was dead), so the printed verdict and the JSON
  `certified` field could disagree unobserved; three cases now assert on the
  JSON directly. **31 cases total**, demo claim block unchanged.
- **Detectability-floor derivation corrected (`ECONOMIC-CONTEXT.md`, and the
  paper's §3.2 + Appendix D).** The floor scaled the n=375 MDE by a per-arm
  MCC SD (≈0.02) where a *paired* MDE requires the SD of the paired
  differences (0.0259 measured). The floor is therefore ≈+0.0037 MCC, not
  +0.003 (balanced accuracy 50.19%, not 50.15%). Separately, the basis-point
  conversion used E[|move|] ≈ 1–1.5%, which is the ALL-day mean (1.42%
  in-window) rather than the decisive-day conditional the formula names
  (1.90% measured); the floor-sized edge is ≈0.7 bp, and one basis point
  would need a 2.7% mean decisive-day move. Both corrections make the floor
  *higher*, i.e. the null more conservative; the conclusion (the floor sits
  below realistic 1–5 bp round-trip costs) is unchanged. The note also now
  ships the corrected-MDE derivation for the stricter units, so the paper's
  “80% power near |d|≈0.7 at n=30” reconciles to a formula (it is the
  k=8-corrected value; the uncorrected one is ≈0.51).
- **Documentation corrections found by the same review:** the news RecordSet in
  `croissant.json` claimed one row per (ticker, day) *with* a tagged article,
  but 34.2% of its rows are zero-filled no-news days; `REPRODUCE.md` said the
  manifest was the only file outside its own scope (`.gitignore` is too) and
  described the H5s as what the drivers consume (they are not); the seven-day
  terminal news blackout (2023-12-19 to 2023-12-28, inside fold 4's test
  window) is now stated in the datasheet and the paper; and the two shipped
  label surfaces use *permuted* {−1,0,+1} encodings, now documented in
  `data/README.md` (the H5's `label_cls` is +1 up / 0 down / −1 dead-zone).
- **Release gate hardened.** Its Croissant leg used to SKIP when
  `mlcroissant` was missing — failing open on the only automated check of a
  headline D&B deliverable, with the validator absent from `requirements.txt`.
  It now fails closed and the validator is pinned. The gate also gained the
  deposit-consistency step above, which was negative-tested (injecting drift
  into one H5 makes it exit non-zero).
- **Social aggregate count corrected: 17 columns, 15 informative.** Two of the
  seventeen social aggregate columns — `social_sent_std` and its rolling mean
  `social_sent_std_w3` — are constant `0.0` on all 114,747 rows of
  `social_features.parquet`, by construction: StockTwits labels messages
  categorically (bullish / bearish / unlabeled), so there is no continuous
  per-message score to take a within-day SD over. The derivation code documents
  the placeholder (`src/mmfp/features/social_features.py`), but the user-facing
  count did not, so the datasheet, `data/README.md`, and the Figure 1 source tile
  advertised "17 aggregates" as though all were informative. Now stated as 15
  informative + 2 inert placeholders + the coverage flag. **No result changes** —
  the two columns are dead input dimensions carrying neither signal nor noise —
  and message-level disagreement remains represented via
  `social_bullish_ratio`(`_w3`). Surfaced by an external review re-audit.
- **Deposit — model-ready H5 static graph corrected to FF12.** The eight
  monolithic `data/processed/multimodal_dataset_v2*.h5` carried the GICS
  110-edge partition in `static_adj`, contradicting the datasheet (which
  documents the released FF12 133-edge graph) — a stale field from before the
  v1.0.3 `.npy` fix, **never read on any results path** (the harness builds
  `static_adj` from `sector_adjacency.npy` via `assemble.py`), so no reference
  number is affected. Corrected in place to FF12; every other dataset in each
  H5 verified byte-identical. A new release-gate step
  (`scripts/data/check_h5_consistency.py`) fails closed if any H5's graph drifts
  from the released `.npy` again.
- **Deposit — assembled H5 macro block was pre-C2 (look-ahead); corrected.**
  Found by an external adversarial review and reproduced independently. The same
  eight H5s carried the **unlagged** macro block — macroeconomic values sitting
  on dates before their public release, the exact look-ahead safeguard C2 exists
  to remove, in a benchmark whose contribution is leakage control. Only the
  zero-lag series (10y, VIX, FOMC flag) matched the shipped build, which is why
  a spot check missed it; `fed_funds_rate_norm` and `cpi_norm` matched the C2
  build on **0.000%** of rows, `unemployment_norm` on 0.796%, `gdp_norm` on
  4.923%. Concretely, `multimodal_dataset_v2.h5` carried the April-2020
  unemployment print on **2020-04-30**, eight days before its 2020-05-08 BLS
  release. All seven macro columns in all eight files are now rewritten to the
  publication-lagged `macro_features.parquet`; every byte outside the macro slice
  is verified identical. **No reported number is affected** — the harness reads
  `macro_features.parquet` through `assemble.py`, never the H5 (verified: the
  labels tables regenerate byte-identically). The guard now also verifies the
  macro and price blocks against the canonical tables on every release, and was
  negative-tested: injecting drift into one file makes it exit non-zero.
  Audited at the same time and found correct: the price block matches
  `price_features.parquet` exactly, and the news block is correctly aligned to
  the prior close (T-1).
- **Correction to an earlier note in this section:** an interim draft of this
  entry said the two constant social columns had their "complete values in
  `social_features.parquet`". That is false — they are constant `0.0` in the
  source parquet too, for the reason given above. See `data/README.md`.

## v1.0.10 — 2026-07-24 (archived; the docs-only release accompanying the submitted PDF)

- **Figure 1 legibility + four-controls framing** (paper schematic,
  regenerated from `scripts/figures/make_overview_figure.py`): the
  "Level 3 · independently audited" rung heading no longer clips its box;
  C5 is drawn as the scope condition (outline badge, set apart) rather
  than a fifth identical control tile, matching the paper's "four controls
  + a fixed universe"; source-family tiles show their flags (Social
  "17 aggregates + flag", Macro "6 series + FOMC") to match Table 2; a
  labeled rule ("the scope, not a control") separates C5 from the four
  controls so the distinction is explicit; the "→ established" chip is
  centered on the two-line Level-3 body; and the claim label ("paired
  per-fold Δ vs the baseline") is sized to stay inside the gutter, clear
  of the harness and the ladder boxes.
- **ECONOMIC-CONTEXT.md correction:** the FF/LSTM directional-accuracy
  values were swapped (had 0.520 FF / 0.522 LSTM; source phase5 Table 4.3
  is FF 0.5225 / LSTM 0.5201). Corrected; the Sharpe line was already
  right, and the paper's "near 52%" is unaffected. Surfaced by the
  pre-publish numbers-verification sweep (7 auditors, ~250 numbers
  recomputed from shipped data, 0 material mismatches).
- Schematic + doc only — no data, harness, or reference-results change.

## v1.0.9 — 2026-07-24 (archived; the release first accompanying the submitted PDF)

- **Certification input-validation closed (two blocking bypasses,
  adversarially discovered by a pre-publish decorrelated review round,
  both reproduced, fixed, regression-tested):** (a) `--k` below 1 is
  rejected at the boundary --- `--k 0` zeroed and negative `--k` inverted
  the Bonferroni multiplicity gate, certifying noise or a swept challenger
  as SUPPORTED; (b) a non-finite or out-of-`[-1,1]` `mcc` is rejected as
  malformed --- NaN-poisoning two of three seed rows (rows present) let a
  bad model's good seed carry the fold mean while the seed contract still
  saw three seeds, flipping WITHIN-NULL to SUPPORTED. The battery gains
  four cases (`--k 0`, `--k -5`, NaN-poison, out-of-range).
- **Restricted-coverage entitlement made explicit:** a `--restrict-folds
  0,1,2,3` claim now certifies only with `--social-coverage-justified`
  (the submitter's attestation that the model consumes the social source,
  audit-verified at Level 3, not machine-checked --- the tool cannot see
  which sources a model consumes). Without it the rule-8 subset is
  NON-CERTIFYING; a model weak on the F4 stub could otherwise drop it and
  certify. Every restricted verdict now prints the full five-fold mean
  delta alongside.
- **Predictions-mode fabrication check:** recomputing MCC verifies
  assembly, not provenance --- echoing the shipped frozen labels scores a
  perfect MCC. Any per-fold MCC above 0.5 (far above the task's ~0.01
  ceiling) prints a FABRICATION CHECK caveat routing to the Level-3 audit.
- **MANIFEST fixity scope extended** over `requirements.txt` (the
  environment pin the determinism claim rests on), `Makefile`, and
  `data/README.md`; `.gitignore` named as an explicit exclusion. The
  paper's "verify every shipped file" is now true as printed (479 files).
- Doc reconciliations: SECTOR-EQUIVALENCE.md data-path reference repaired;
  calibration README notes Study 1's val==test logging quirk (not a leak;
  the reference grid selects on the calendar-tail split); SUBMITTING
  documents the assembly-not-provenance scope of predictions mode.
- **CAMEF ten-epoch retrain ships** (`audits/camef/camef-train-10ep/`,
  both arms at the budget the CAMEF paper documents, T4, EPOCHS 5→10 the
  only change): index MSE 0.000596 (1.22× the published 0.00048860 — the
  budget explains the five-epoch positional shortfall), time MSE 15.50
  (vs 15.27 at five epochs — the chronological collapse persists;
  best-validation checkpoint saved in epoch 1, final-epoch loss steps
  climb to 7.40 vs epoch mean 2.20). The paper's §4.3 now prices the
  split at either documented budget; `REPRODUCE.md` §2b and
  `audits/camef/PROGRESS.md` carry the full record.
- **Certification loophole closed (adversarially discovered, reproduced,
  fixed, regression-tested):** the dual bar's classical-anchor floor is
  now computed over the FULL five-fold grid regardless of
  `--restrict-folds`, and a certifying restriction is hard-gated to the
  one documented rule-8 subset {0,1,2,3} (every other subset is analysis,
  never certification). Previously, restricting to folds {1,2,3,4}
  dropped the anchor's strongest fold and let a challenger genuinely
  below the full-universe logistic reference certify.
- **Regression battery ships** (`scripts/analysis/test_evaluate_submission.py`,
  10 cases incl. both discovered exploits and a non-oracle positive
  control) and a **pre-tag release gate**
  (`scripts/release_gate.py`: fresh-tree integrity, the battery, the
  byte-compared S6 demo, metadata consistency). No tag is cut without it
  — MANIFEST staleness shipped three times before this gate existed.
- **Positive-control example ships** (`examples/positive_control/`,
  clearly synthetic: baseline +0.05/fold certifies SUPPORTED —
  certification is demonstrably reachable without an oracle).
- **The 117-run calibration grid ships** (`results/calibration/`: the
  four validation-metric studies behind C1's frozen shared defaults).
- Thread pins (`OMP/MKL/OPENBLAS_NUM_THREADS=2`) added to every
  documented entry point; REPRODUCE's Determinism section documents the
  thread scope explicitly.
- Front door rewired: README/Makefile/REPRODUCE now lead with
  `make_reference_table_v2.py` (the paper's Table 3); the historical
  grid analysis is labeled provenance.
- MANIFEST coverage extended over the normative top-level docs
  (SUBMITTING, MAINTENANCE, CHANGELOG, DATA-STATEMENTS, DATASHEET,
  REPRODUCE, README, ECONOMIC-CONTEXT, LICENSE, CITATION.cff,
  croissant.json) so the rule text submitters are bound by is inside the
  tamper-evidence scope.
- Doc hygiene: sector-equivalence provenance repointed at released
  paths; deprecated `--baseline-arch stronger` alias now prints a
  warning; demo `--k 1` annotated (honest only for a single-configuration
  replay); CHANGELOG tag dates corrected to actual tag dates.

## v1.0.8 — 2026-07-23 (tag only; superseded before archiving)

- Tagged and pushed, but never published as a GitHub Release and never
  archived to Zenodo: a pre-publish decorrelated review round found two
  blocking certification input-validation bypasses (see v1.0.9). v1.0.9
  supersedes it as the archival tag; nothing external references v1.0.8.

## v1.0.7 — 2026-07-23 (dual-bar certification; all-day labels)

- **Dual-bar certification**: a SUPPORTED claim must now clear BOTH the
  declared tuned baseline arm (corrected fold-level significance, as
  before) AND the untuned logistic-price anchor (a hard positive-contrast
  floor whose p is reported). C1's own rationale — killing weak-baseline
  inflation — forbids certifying below a simple runnable price model;
  previously a challenger could in principle certify against the tuned
  neural arm while sitting under the logistic anchor. Refusals carry an
  explicit verdict tag. Validated: a crafted challenger significant vs
  the arm (p_bonf≈0) but below the anchor is refused; the oracle
  predictions case still certifies.
- **All-day three-class labels ship**
  (`data/processed/labels_direction_allday.parquet`: 62,315 test-window
  rows, y_true ∈ {−1, 0, +1} with 0 = dead-zone) so the all-day task is
  runnable today; its reference results remain a planned extension.
- README null phrasing aligned with the paper ("establishes a
  multiplicity-corrected gain"); SUBMITTING documents the dual bar.

## v1.0.6 — 2026-07-23 (evaluator hardening: fail-closed certification)

- **`evaluate_submission.py` no longer has a test-selected default
  reference.** `--baseline-arch` is REQUIRED: `ff`/`lstm` pair
  like-for-like against a runnable arm; `envelope` (per-cell max over both
  arms) remains available as an explicit conservative sensitivity read and
  **can never certify** (its verdict is tagged not-certifiable). This
  closes the protocol inconsistency where the tool whose benchmark forbids
  test-set selection (C3) defaulted to a test-selected reference.
- **Certification fails closed** on: undeclared comparison family
  (no `--k` ⇒ "UNCERTIFIED — comparison family undeclared"), unverified
  assembly (score mode without per-row `n_test` ⇒ "UNCERTIFIED — assembly
  unverified"), mismatched `n_test` (⇒ NOT COMPARABLE), missing contract
  seeds, or incomplete fold coverage.
- **Predictions mode** (`--predictions preds.csv`): the evaluator now
  recomputes MCC itself from per-example predictions against the new
  frozen labels table `data/processed/labels_direction.parquet`
  (44,438 scoreable test rows; generated + self-checked against the frozen
  per-fold `n_test` by `scripts/analysis/make_labels_table.py`), with
  exact-coverage enforcement — assembly conformance by construction.
- The claim block now prints a descriptive per-fold read against the
  untuned logistic-price anchor alongside the tuned-baseline contrast.
- Committed demo claim regenerated with the canonical command
  (`--k 1 --baseline-arch ff`): ΔMCC +0.0033, within the reference null
  (the envelope sensitivity read, −0.0066, remains documented).
- SUBMITTING.md rewritten for the two input modes and the fail-closed
  rules; README updated.

## v1.0.5 — 2026-07-21 (the archived release accompanying the KDD submission)

- README/DATASHEET retitled and reframed to the paper's registered framing
  (title *Auditing Multi-Source Stock Prediction: An Evaluation Protocol and
  Benchmark*; four safeguards run on a fixed liquid-universe scope); DOI
  badge, citation, and contact added to the README.
- `CHANGELOG.md` introduced (retroactive to v1.0.0); maintenance plan points
  here for the paper-accompanying tag.
- `CITATION.cff`: co-author ORCIDs added (Puoti 0009-0006-4661-3613,
  Pittorino 0000-0002-1919-6141), corresponding-author email added.
- Figure-1 generator artwork matches the four-controls + scope framing;
  ladder rung renamed "independently audited".
- Doc sweep: economic-context pointer §4.1→§3.2; "licensing-clean" wording
  replaced by "raw-text-free, per-source terms preserved" (croissant,
  .gitignore); census heading scoped (five reporting dimensions);
  sector-equivalence note's long-resolved "Pending" block replaced with the
  recorded verdict and repo-relative paths (the census copy is a pointer);
  `verify_integrity.py` docstring synced to the real covered tree and the
  documented `--emit-gics` output whitelisted so the fixity check stays
  clean after REPRODUCE §3e; REPRODUCE §4 notes the two test suites run as
  separate pytest invocations.

## v1.0.4 — 2026-07-20 (tag only; superseded by v1.0.5 before archiving)

- `CITATION.cff` license reverted to the single scalar `MIT` (+ the dual
  grant spelled out in the abstract text): Zenodo's CFF loader rejects the
  CFF-1.2.0 license *list* introduced in v1.0.2, which is what broke
  v1.0.3's archival ("Citation metadata load failed").

## v1.0.3 — 2026-07-20 (released; Zenodo ingestion failed on the CFF license list)

- **Data fix:** `data/processed/graphs/sector_adjacency.npy` corrected to
  the Fama–French-12 same-sector graph (133 edges) — the graph the
  reference and sector-arm results were produced with. Earlier deposits
  mistakenly carried the GICS-partition file (110 edges). The GICS
  sensitivity arm is materialized on demand via
  `scripts/data/kdd_sector_map.py --emit-gics` (builder de-hardcoded, now
  repo-relative, with an EDGAR release check).
- `scripts/audits/diagnostic_bias.py` ships (the MSGCA audit's data-prep
  shim; the from-scratch rerun previously died on a missing import).
- Submission harness: certification requires the three contract seeds per
  covered fold (best-seed-per-fold selections refuse with SEED CONTRACT NOT
  MET); rule-8 `--restrict-folds` submissions can reach SUPPORTED tagged
  RESTRICTED COVERAGE (previously structurally impossible); undeclared
  `--k` warns; assembly mismatch returns NOT COMPARABLE; duplicate
  (fold, seed) rows rejected; headline ΔMCC fold-weighted.
- Demo claim regenerated with the bare default command (envelope pairing,
  ΔMCC −0.0066 — matches the paper's §6 block byte-for-byte).
- Title synced across CITATION.cff/croissant; "point-in-time" corrected to
  "availability-timed (release-lag aligned)".

## v1.0.2 — 2026-07-20 (tag only; never released)

- Licensing made precise: LICENSE resolves the VIX self-contradiction (the
  Cboe VIX close ships as a small cited extract under FRED
  "Copyrighted: Citation Required" terms, carved out of the CC-BY-4.0
  grant); `CITATION.cff` carried the dual license list (later reverted to
  scalar for Zenodo, v1.0.4); DATASHEET regenerability qualified for the
  non-re-pullable StockTwits corpus; REPRODUCE stale manifest count fixed.

## v1.0.1 — 2026-07-18 (archived)

- `CITATION.cff` title corrected to the de-scoped paper title of the time
  and the concept DOI added. (Metadata-only; v1.0.0 had shipped the interim
  "Leakage-Controlled" title.)

## v1.0.0 — 2026-07-18 (archived)

- Initial public deposit: evaluation protocol + harness, availability-timed
  derived-feature tables for five source families (committed in-repo,
  219 MB), reference results (4,681 result rows / 3,931 distinct runs),
  naive and classical anchors, audits, census, datasheet, croissant
  metadata, MANIFEST integrity, demo submission.
