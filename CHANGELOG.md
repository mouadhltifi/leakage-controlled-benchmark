# Changelog

All tags, newest first. Result CSVs under `results/` are unchanged across
every entry below — every reference number regenerates identically at every
tag; the changes are to data-file correctness, tooling, and documentation.
"Archived" = published as a GitHub Release and ingested by Zenodo under the
concept DOI 10.5281/zenodo.21431362.

## Unreleased (v1.0.9, the archival tag accompanying the submitted PDF)

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
