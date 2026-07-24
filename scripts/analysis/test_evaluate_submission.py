#!/usr/bin/env python3
"""Regression battery for evaluate_submission.py's certification gates.

Run:  python3 scripts/analysis/test_evaluate_submission.py
Exits nonzero on any regression. The release gate runs this on a fresh
clone before any tag.

Covers, among others, the two adversarially discovered exploits:
  * best-seed-per-fold selection certifying (seed-contract gate), and
  * --restrict-folds dropping the anchor's strongest fold to duck the
    dual bar's classical-anchor floor (anchor leg is full-grid; only the
    documented rule-8 subset {0,1,2,3} may certify).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "scripts" / "analysis" / "evaluate_submission.py"
DEMO = ROOT / "examples" / "demo_submission" / "submission.csv"
N_TEST = {0: 10100, 1: 9851, 2: 9979, 3: 9994, 4: 4514}
FAILS: list[str] = []


def run(*args: str) -> str:
    r = subprocess.run([sys.executable, str(EVAL), *args],
                       capture_output=True, text=True)
    return r.stdout + r.stderr


def baseline_ff() -> pd.DataFrame:
    import glob
    b = pd.concat([pd.read_csv(p) for p in
                   glob.glob(str(ROOT / "results/native_core/native_core_*.csv"))])
    b = b[b.experiment_name.str.startswith("natcore_A7_")
          & (b.price_encoder == "feedforward")]
    return b[["fold_idx", "seed", "mcc"]]


def craft(name: str, bump) -> Path:
    rows = [{"challenger": name, "fold_idx": int(r.fold_idx),
             "seed": int(r.seed), "mcc": float(r.mcc) + bump(int(r.fold_idx)),
             "n_test": N_TEST[int(r.fold_idx)]}
            for _, r in baseline_ff().iterrows()]
    p = Path(tempfile.mkdtemp()) / f"{name}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def check(label: str, out: str, must: list[str], must_not: list[str] = []):
    ok = all(m in out for m in must) and not any(m in out for m in must_not)
    print(("PASS  " if ok else "FAIL  ") + label)
    if not ok:
        FAILS.append(label)
        for m in must:
            if m not in out:
                print(f"      missing: {m!r}")
        for m in must_not:
            if m in out:
                print(f"      present: {m!r}")


def main() -> int:
    # 1. demo canonical: in-null, anchor floor printed and not cleared
    out = run(str(DEMO), "--k", "1", "--baseline-arch", "ff")
    check("demo canonical (+0.0033, in-null)", out,
          ["+0.0033", "WITHIN THE REFERENCE NULL", "hard floor"],
          ["SUPPORTED"])

    # 2. EXPLOIT (adjudicator-reproduced): below-anchor challenger must
    # not certify via a restriction that drops the anchor's best fold
    exploit = craft("drop_f0_exploit", lambda f: 0.008)
    out = run(str(exploit), "--k", "1", "--baseline-arch", "ff",
              "--restrict-folds", "1,2,3,4")
    check("drop-worst-fold exploit refused", out,
          ["NON-CERTIFYING RESTRICTION"], ["SUPPORTED  ["])
    out_full = run(str(exploit), "--k", "1", "--baseline-arch", "ff")
    check("same challenger, full grid: anchor floor refuses", out_full,
          ["DOES NOT CLEAR THE CLASSICAL ANCHOR"], [])

    # 3. rule-8 legitimate path still certifies (full 5-fold file,
    # documented subset, strong effect clears both bars)
    strong = craft("strong_social", lambda f: 0.05)
    out = run(str(strong), "--k", "1", "--baseline-arch", "ff",
              "--restrict-folds", "0,1,2,3", "--social-coverage-justified")
    check("rule-8 {0,1,2,3} certifies when attested and both bars clear", out,
          ["SUPPORTED", "RESTRICTED COVERAGE"], ["NON-CERTIFYING"])
    # rule-8 subset WITHOUT the entitlement attestation must not certify
    # (a model weak on the F4 stub could otherwise drop it and certify)
    out = run(str(strong), "--k", "1", "--baseline-arch", "ff",
              "--restrict-folds", "0,1,2,3")
    check("rule-8 subset unattested: NON-CERTIFYING", out,
          ["NON-CERTIFYING RESTRICTION", "entitlement was not"],
          ["SUPPORTED  ["])

    # 4. positive control: a plausible strong challenger certifies on the
    # full grid (the instrument can say yes without an oracle)
    out = run(str(strong), "--k", "1", "--baseline-arch", "ff")
    check("non-oracle positive control certifies", out, ["SUPPORTED"],
          ["NOT cleared"])

    # 5. sig-vs-arm but below anchor: refused with the explicit tag
    below = craft("sig_below_anchor", lambda f: 0.004)
    out = run(str(below), "--k", "1", "--baseline-arch", "ff")
    check("significant vs arm, below anchor: refused", out,
          ["DOES NOT CLEAR THE CLASSICAL ANCHOR"], ["SUPPORTED  ["])

    # 6. fail-closed gates
    out = run(str(DEMO), "--baseline-arch", "ff")
    check("k undeclared: UNCERTIFIED", out,
          ["UNCERTIFIED -- comparison family undeclared"])
    no_ntest = Path(tempfile.mkdtemp()) / "no_ntest.csv"
    pd.read_csv(DEMO).drop(columns=["n_test"]).to_csv(no_ntest, index=False)
    out = run(str(no_ntest), "--k", "1", "--baseline-arch", "ff")
    check("n_test absent: UNCERTIFIED", out,
          ["UNCERTIFIED -- assembly unverified"])
    out = run(str(DEMO), "--k", "1", "--baseline-arch", "envelope")
    check("envelope: non-certifiable", out, ["ENVELOPE REFERENCE"],
          ["SUPPORTED  ["])
    cherry = Path(tempfile.mkdtemp()) / "cherry.csv"
    s = pd.read_csv(DEMO)
    s.loc[s.groupby("fold_idx")["mcc"].idxmax()].to_csv(cherry, index=False)
    out = run(str(cherry), "--k", "1", "--baseline-arch", "ff")
    check("best-seed-per-fold cherry-pick refused", out,
          ["SEED CONTRACT NOT MET"], ["SUPPORTED  ["])

    # 7. EXPLOIT (decorrelated-gauntlet-reproduced): a below-significance
    # challenger must not certify by declaring k<1, which zeros or inverts
    # the Bonferroni gate. k=0 and k=-5 are rejected at the boundary.
    kexploit = craft("k_zero_exploit", lambda f: 0.05)  # clears both bars but...
    out = run(str(kexploit), "--k", "0", "--baseline-arch", "ff")
    check("k=0 rejected (would zero the multiplicity gate)", out,
          ["--k must be a positive integer"], ["SUPPORTED", "p_bonf (k=0)"])
    out = run(str(kexploit), "--k", "-5", "--baseline-arch", "ff")
    check("negative k rejected (would invert the gate)", out,
          ["--k must be a positive integer"], ["SUPPORTED", "p_bonf (k=-5)"])

    # 8. EXPLOIT (decorrelated-gauntlet-reproduced): NaN-poisoning bad seeds.
    # Two of three seeds set to NaN (rows present) would flip WITHIN-NULL to
    # SUPPORTED -- the per-fold mean skips NaN while the seed still counts as
    # present. Non-finite mcc is rejected at the boundary.
    nan_poison = Path(tempfile.mkdtemp()) / "nan_poison.csv"
    rows = []
    for _, r in baseline_ff().iterrows():
        f, sd = int(r.fold_idx), int(r.seed)
        rows.append({"challenger": "nan_poison", "fold_idx": f, "seed": sd,
                     "mcc": (float(r.mcc) + 0.05 if sd == 42 else float("nan")),
                     "n_test": N_TEST[f]})
    pd.DataFrame(rows).to_csv(nan_poison, index=False)
    out = run(str(nan_poison), "--k", "1", "--baseline-arch", "ff")
    check("NaN-poisoned seeds rejected (not SUPPORTED)", out,
          ["NOT COMPARABLE", "finite value in"], ["SUPPORTED"])

    # 9. out-of-range mcc (fabrication via impossible value) rejected at the
    # boundary -- MCC is bounded to [-1, 1]
    oor = craft("out_of_range", lambda f: 0.0)
    s = pd.read_csv(oor); s.loc[0, "mcc"] = 5.0; s.to_csv(oor, index=False)
    out = run(str(oor), "--k", "1", "--baseline-arch", "ff")
    check("out-of-range mcc rejected", out,
          ["NOT COMPARABLE", "finite value in"], ["SUPPORTED"])

    # 10. in-range oracle (echoes the frozen labels): finite and within
    # [-1, 1], so it clears the boundary guards, both bars, and would read
    # SUPPORTED -- but a per-fold MCC far above the task ceiling must fail
    # closed at the token (not merely annotate) and route to the Level-3 audit.
    oracle = craft("oracle_echo", lambda f: 0.9)
    out = run(str(oracle), "--k", "1", "--baseline-arch", "ff")
    check("in-range oracle (echoed labels): withheld, not certified", out,
          ["FABRICATION CHECK", "NOT CERTIFIED"], ["SUPPORTED"])

    # 11. partial label-echo -- the blind band the old 0.5 tripwire missed.
    # ~0.30 MCC is far above the 0.087 honest ceiling but was below the old
    # 0.5 cutoff, so it used to certify clean; the tightened 0.15 threshold
    # now withholds it. (The +0.05 positive controls above, ~0.06 max cell,
    # stay well under 0.15 and still certify -- so the tightening is safe.)
    partial = craft("partial_echo", lambda f: 0.30)
    out = run(str(partial), "--k", "1", "--baseline-arch", "ff")
    check("partial echo (0.30, old 0.15-0.5 blind band): withheld", out,
          ["FABRICATION CHECK", "NOT CERTIFIED"], ["SUPPORTED"])

    # 12. EXPLOIT (external-review-reproduced): seed PADDING. The gate used to
    # test only that the labels 42/123/456 were present, so a best-seed-per-fold
    # selection written three times under the three labels certified — defeating
    # the very gate that polices that selection. Three identical rows per fold,
    # in every fold, is one result relabelled, not three runs.
    pad = Path(tempfile.mkdtemp()) / "seed_padded.csv"
    s = pd.read_csv(DEMO)
    win = s.loc[s.groupby("fold_idx")["mcc"].idxmax()]
    pd.DataFrame([{"challenger": "seed_padded", "fold_idx": int(r.fold_idx),
                   "seed": sd, "mcc": float(r.mcc), "n_test": N_TEST[int(r.fold_idx)]}
                  for _, r in win.iterrows() for sd in (42, 123, 456)]
                 ).to_csv(pad, index=False)
    out = run(str(pad), "--k", "1", "--baseline-arch", "ff")
    check("seed-padded cherry-pick refused (relabelled, not replicated)", out,
          ["SEED CONTRACT NOT MET", "one result relabelled"], ["SUPPORTED  ["])

    # 13. FALSE-POSITIVE GUARD: an honest submission whose seeds genuinely
    # differ must still certify — the degeneracy test is all-folds, so a
    # legitimate tie inside a single fold does not trip it.
    honest = craft("honest_strong", lambda f: 0.05)
    h = pd.read_csv(honest)
    h.loc[(h.fold_idx == 0), "mcc"] = float(h.loc[h.fold_idx == 0, "mcc"].iloc[0])
    h.to_csv(honest, index=False)   # fold 0 tied, folds 1-4 genuinely vary
    out = run(str(honest), "--k", "1", "--baseline-arch", "ff")
    check("honest submission with a single tied fold still certifies", out,
          ["SUPPORTED"], ["SEED CONTRACT NOT MET"])

    # 14. EXPLOIT (external-review-reproduced): under --restrict-folds the seed
    # contract used to be evaluated on the POST-restriction frame while the
    # anchor floor scored the FULL grid — so fold 4 sat inside the hard floor
    # but outside the contract, and pruning it to its best seed pushed the floor
    # over zero. The contract is now evaluated on the full submitted grid.
    prune = Path(tempfile.mkdtemp()) / "anchor_prune.csv"
    rows = []
    for _, r in baseline_ff().iterrows():
        f, sd = int(r.fold_idx), int(r.seed)
        if f == 4 and sd != 42:
            continue                      # prune F4 to its best seed only
        rows.append({"challenger": "anchor_prune", "fold_idx": f, "seed": sd,
                     "mcc": float(r.mcc) + (0.05 if f != 4 else 0.09),
                     "n_test": N_TEST[f]})
    pd.DataFrame(rows).to_csv(prune, index=False)
    out = run(str(prune), "--k", "1", "--baseline-arch", "ff",
              "--restrict-folds", "0,1,2,3", "--social-coverage-justified")
    check("restricted claim with a seed-pruned out-of-scope fold refused", out,
          ["SEED CONTRACT NOT MET"], ["SUPPORTED  ["])

    # 15. MUTATION-COVERAGE cases. Mutation-testing this battery showed five
    # gates whose deletion changed no assertion, because the cases above test
    # them against the in-null DEMO -- i.e. they assert a MESSAGE is present,
    # not that the GATE bites. Each case below is run on input that WOULD
    # certify if the gate were removed, and each also checks the JSON claim
    # block, which the battery previously never read.
    def run_json(*args):
        """Run the evaluator and return (stdout, parsed claim.json)."""
        jp = Path(tempfile.mkdtemp()) / "claim.json"
        out = run(*args, "--json", str(jp))
        try:
            return out, json.loads(jp.read_text())
        except Exception:
            return out, None

    strong_path = str(craft("mut_strong", lambda f: 0.05))

    # (M3) envelope must be non-certifiable even for a challenger that clears
    # both bars -- previously only asserted against the in-null demo.
    out, cj = run_json(strong_path, "--k", "1", "--baseline-arch", "envelope")
    check("envelope refuses a both-bars-clearing challenger", out,
          ["ENVELOPE REFERENCE"], ["SUPPORTED  ["])
    check("envelope: JSON certified=false", "" if cj and cj.get("certified") is False else "MISSING",
          [""])

    # (M12) undeclared k must not certify, and the JSON must agree with the
    # printed verdict -- the mutation that broke this was invisible before.
    out, cj = run_json(strong_path, "--baseline-arch", "ff")
    check("k undeclared on a strong challenger: UNCERTIFIED + JSON false", out,
          ["UNCERTIFIED -- comparison family undeclared"], ["VERDICT           : SUPPORTED"])
    check("k-undeclared JSON certified=false",
          "" if cj and cj.get("certified") is False else "MISSING", [""])

    # (M13) fabrication must FAIL CLOSED in the JSON too, not just annotate.
    out, cj = run_json(str(craft("mut_fab", lambda f: 0.9)), "--k", "1",
                       "--baseline-arch", "ff")
    check("fabrication: JSON certified=false (not annotate-only)",
          "" if cj and cj.get("certified") is False else "MISSING", [""])

    # (M7) n_test conformance must bite on a NON-first row.
    nt = Path(tempfile.mkdtemp()) / "bad_ntest.csv"
    s = pd.read_csv(strong_path)
    _i = s.index[s.fold_idx == 0][-1]
    s.loc[_i, "n_test"] = N_TEST[0] + 7
    s.to_csv(nt, index=False)
    out = run(str(nt), "--k", "1", "--baseline-arch", "ff")
    check("n_test wrong on a non-first row: NOT COMPARABLE", out,
          ["NOT COMPARABLE"], ["VERDICT           : SUPPORTED"])

    # (M10) duplicate (fold, seed) rows must be refused -- the exploit was
    # duplicating a fold's best seed to swing the anchor mean.
    dup = Path(tempfile.mkdtemp()) / "dup.csv"
    s = pd.read_csv(strong_path)
    pd.concat([s, s[s.fold_idx == 4]]).to_csv(dup, index=False)
    out = run(str(dup), "--k", "1", "--baseline-arch", "ff")
    check("duplicate (fold, seed) rows refused", out,
          ["duplicate (fold_idx, seed)"], ["SUPPORTED"])

    # 16. mixed-challenger per-cell selection (external-review-reproduced):
    # one row per cell, but each row the best model for that cell.
    mix = Path(tempfile.mkdtemp()) / "mixed.csv"
    s = pd.read_csv(strong_path)
    s["challenger"] = ["model_" + str(i % 3) for i in range(len(s))]
    s.to_csv(mix, index=False)
    out = run(str(mix), "--k", "1", "--baseline-arch", "ff")
    check("mixed challenger names refused (per-cell model selection)", out,
          ["NOT COMPARABLE", "mixes"], ["SUPPORTED"])

    # 17. malformed non-mcc input must produce a verdict, not a traceback.
    for label, mutate in (
        ("NaN n_test", lambda df: df.assign(n_test=[float("nan")] * len(df))),
        ("fold_idx 5", lambda df: df.assign(
            fold_idx=[5] + list(df.fold_idx[1:]))),
    ):
        p = Path(tempfile.mkdtemp()) / "malformed.csv"
        mutate(pd.read_csv(strong_path)).to_csv(p, index=False)
        out = run(str(p), "--k", "1", "--baseline-arch", "ff")
        check(f"malformed input ({label}) rejected cleanly", out,
              ["NOT COMPARABLE"], ["Traceback"])
    empty = Path(tempfile.mkdtemp()) / "empty.csv"
    pd.read_csv(strong_path).head(0).to_csv(empty, index=False)
    out = run(str(empty), "--k", "1", "--baseline-arch", "ff")
    check("empty submission rejected cleanly", out,
          ["NOT COMPARABLE"], ["Traceback"])

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
