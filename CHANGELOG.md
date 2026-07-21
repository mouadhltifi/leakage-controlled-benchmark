# Changelog

All tags, newest first. Result CSVs under `results/` are unchanged across
every entry below — every reference number regenerates identically at every
tag; the changes are to data-file correctness, tooling, and documentation.
"Archived" = published as a GitHub Release and ingested by Zenodo under the
concept DOI 10.5281/zenodo.21431362.

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
