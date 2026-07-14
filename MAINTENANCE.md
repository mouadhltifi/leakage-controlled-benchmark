# Maintenance plan

- **Hosting.** Code, committed reference results, and small public-domain
  inputs: this GitHub repository. Derived feature tables: archival deposit
  with a DOI (Zenodo) upon paper acceptance; the repository README links the
  DOI. The repository is the canonical index; the DOI deposit is the
  canonical data copy.
- **Versioning.** Tagged releases (`v1.0` = the version evaluated in the
  paper). Any change to data, labels, task definitions, or reference numbers
  bumps the version and is CHANGELOG'd; reference results are always tied to
  a tag. The 2015–2023 window is **frozen** — maintenance means corrections
  and documentation, not silent extension. A window extension, if ever
  released, would be a new major version with its own reference baselines.
- **Corrections.** Errors in features, labels, or documentation: GitHub
  issues; fixes land with a version bump and a note in the CHANGELOG and, if
  they affect any reference number, an updated reference table with the
  change highlighted.
- **Removal on request.** Per `DATA-STATEMENTS.md` §3: if a rights-holder
  objects to any released aggregate or derived feature, we will remove or
  further coarsen the affected columns in the next tagged release and mark
  the change in the CHANGELOG. Contact: GitHub issues or the corresponding
  author's email (see `CITATION.cff` / the paper).
- **Environment.** `requirements.txt` is pinned; the determinism claim
  (CPU bit-identical re-training under fixed seeds) is scoped to the pinned
  environment. Dependency refreshes, if needed for installability, are
  releases with the determinism claim re-verified.
- **Longevity.** The DOI deposit guarantees data availability independent of
  the repository; the repository remains maintained on a best-effort basis
  by the authors.
