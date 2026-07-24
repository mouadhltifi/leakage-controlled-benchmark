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

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
