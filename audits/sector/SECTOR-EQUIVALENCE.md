# Sector-graph equivalence verdict — GICS vs public SIC/FF12 (2026-07-14)

> Paper B work item (plan Task 2). Campaign: 360 runs, both arms fresh under
> the same code+cache state (mmfp, thread-capped, byte-exact GICS restore
> verified post-campaign). Analysis: paired per (fold, seed, fusion, arch),
> 180 matched cells. Script: `work/scripts/kdd_sector_equivalence.py`;
> data: `work/experiments/phase4r/sector_ff12/sector_{gics,ff12}_{ff,lstm}.csv`.

## Result: NOT equivalent at the ±0.005 bar — the fallback is invoked

| Config | Arch | GICS mean | FF12 mean | Δ (FF12−GICS) | paired d | p | n |
|---|---|---|---|---|---|---|---|
| A8 (P+G) | ff | +0.0029 | +0.0131 | **+0.0102** | +0.527 | 0.001 | 45 |
| A8 (P+G) | lstm | +0.0060 | +0.0111 | +0.0051 | +0.278 | 0.069 | 45 |
| A9 (P+M+G) | ff | −0.0046 | −0.0031 | +0.0015 | +0.085 | 0.573 | 45 |
| A9 (P+M+G) | lstm | +0.0046 | +0.0013 | −0.0033 | −0.140 | 0.352 | 45 |

Concat-only (the reference-table recipe): A8 Δ=+0.0106 (n=30), A9 Δ=−0.0000.
Max |Δ mean| = 0.0102 > 0.005 → **fail**, per the pre-registered bar.

## Decision (the plan's pre-registered fallback)

The release ships the **FF12/SIC sector graph as the benchmark's reference**
(licensing-clean; SEC-EDGAR SIC → Fama–French-12, mapping script included),
plus the **GICS derivation script** for users with licensed access, with the
measured delta **disclosed, not hidden**. The reference table's A8/A9 rows
use the FF12 arm (harness-native, from this campaign).

## Interpretation for the paper (exhibit, not embarrassment)

The two partitions differ materially (edge Jaccard 0.365; GICS 110 edges vs
FF12 133 — REITs merge into Money, GOOGL/META join tech, V/MA leave
Financials). The sector taxonomy is an evaluation-relevant choice that the
multi-source literature does not even report — and swapping it moves this
grid by up to +0.0102 mean MCC, more than most claimed multi-source effects.
That is the paper's thesis (the verdict turns on evaluation choices) showing
up in a place nobody looks. Framing caution: FF12 is "better" only in this
grid/architecture; the claim is sensitivity, not superiority.

## Pending (native A7, running now)

Whether FF12-A8 stays within the reference null vs price requires the
harness-native A7 (90-run A3/A6/A7 concat re-run in flight,
`work/experiments/phase4r/native_core/`). If FF12-A8-ff vs native-A7-ff were
to clear the fold-level bar, that would be a *finding requiring its own
scrutiny* (single-arch, single-config — the multiplicity and honest-unit
rules apply to us first). Verdict to be appended here.

## Drift diagnostic (context, not a gate)

The GICS arm's mmfp-native means (A8: ff +0.0029 / lstm +0.0060) sit within
the documented v1↔mmfp cross-codebase band of the recorded graph-LN canon
(v1: ff ≈0.006 / lstm ≈0.007) — consistent with the M8 equivalence gate;
no drift.
