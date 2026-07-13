"""Apply realistic publication lags to the FRED macro series and regenerate
the cached macro feature parquet.

WHY
---
The original pipeline downloaded the *current revised* FRED series indexed by
their **reference** date and forward-filled onto trading days with no
publication delay. For the four released-with-lag series (CPI, unemployment,
the monthly fed-funds average, and quarterly GDP) this means the model saw, on
a given trading day, a value that was not public until weeks later — a
look-ahead bias. (The two genuinely daily series, the 10-year Treasury yield
and the VIX, and the FOMC-window flag built from a pre-announced calendar, are
already point-in-time and need no lag.)

FIX
---
Each released-with-lag series is shifted forward by a business-day count that
is at least its real publication delay, so its value only becomes observable on
or after its true release date. The shift is applied to the raw series; the
rolling z-score is then recomputed on the lagged series, so the normalisation
is itself leakage-free. The lags are deliberately on the conservative side of
the real delay: over-shifting only makes macro *staler*, which can weaken the
macro configurations but can never reintroduce look-ahead — the safe direction
for a study whose macro result is a null.

Validated: the 27-business-day unemployment shift lands the April-2020 print
(14.8%) on exactly 2020-05-08, the real BLS release date; rebuilding features
from the *unshifted* raw reproduces the original cache to max-abs-diff 0.0, so
the only change here is the publication lag.

Usage (from the artifact root)::
    python scripts/audits/macro_lag/apply_macro_publication_lag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# The packages live under ``<root>/src``; this script is at
# ``<root>/scripts/audits/macro_lag/`` (three levels down).
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mmfp.data.loaders.macro import load_macro_raw  # noqa: E402
from mmfp.data.paths import FEATURES_DIR, MACRO_PARQUET  # noqa: E402
from mmfp.features.macro_events import compute_macro_features  # noqa: E402

# Publication lag in BUSINESS DAYS, each >= the documented release delay from
# the FRED reference date to the public release. Daily series = 0.
LAGS_BD: dict[str, int] = {
    "cpi": 30,             # CPI for month M released ~12th of M+1   (~42 cal d)
    "unemployment": 27,    # UNRATE for M released ~1st Fri of M+1   (~37 cal d)
    "fed_funds_rate": 23,  # FEDFUNDS monthly avg known ~start of M+1 (~31 cal d)
    "gdp": 88,             # GDP quarter advance est. ~+30d post-quarter (~120 cal d)
    "treasury_10y": 0,     # daily — known same day
    "vix": 0,              # daily — known same day
}

MACRO_FEATURES = FEATURES_DIR / "macro_features.parquet"
LEAKED_FEATURES = FEATURES_DIR / "macro_features.leaked.parquet"
LAGGED_RAW = MACRO_PARQUET.with_name("macro_indicators.lagged.parquet")


def apply_lag(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    for col, n in LAGS_BD.items():
        if n and col in out.columns:
            out[col] = raw[col].shift(n)
    return out


def _max_abs_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Max absolute numeric difference over the shared index/columns of a, b."""
    idx = a.index.intersection(b.index)
    cols = [c for c in a.columns if c in b.columns]
    return float(
        (
            a.loc[idx, cols].select_dtypes("number")
            - b.loc[idx, cols].select_dtypes("number")
        ).abs().max().max()
    )


def main() -> int:
    raw = load_macro_raw()

    # Guard: confirm the unshifted rebuild reproduces the existing cache, so the
    # shifted rebuild differs ONLY by the publication lag.
    if MACRO_FEATURES.exists():
        cached = pd.read_parquet(MACRO_FEATURES)

        # This artifact SHIPS the already-corrected (publication-lagged) macro
        # cache, so the active cache will equal the *lagged* rebuild, not the
        # unlagged one. Detect that state and exit cleanly instead of asserting:
        # there is nothing to correct from scratch here.
        diff_lagged = _max_abs_diff(cached, compute_macro_features(apply_lag(raw)))
        if diff_lagged < 1e-9:
            print(
                "[guard] active macro cache already equals the publication-lagged "
                f"build (max diff {diff_lagged:.1e})."
            )
            print(
                "this artifact ships the already-corrected macro cache; "
                "leakage-free deltas are recorded in results/macrolag/*.csv; "
                "to reproduce the correction from scratch, start from the original "
                "unlagged feature cache per data/README.md"
            )
            return 0

        rebuilt0 = compute_macro_features(raw)
        diff = _max_abs_diff(cached, rebuilt0)
        assert diff < 1e-9, (
            f"unshifted rebuild does not match cache (max diff {diff}); "
            "build params drifted — aborting to avoid an apples-to-oranges cache."
        )
        print(f"[guard] unshifted rebuild matches cache (max diff {diff:.1e})")
        if not LEAKED_FEATURES.exists():
            cached.to_parquet(LEAKED_FEATURES)
            print(f"[backup] original cache -> {LEAKED_FEATURES.name}")

    lagged_raw = apply_lag(raw)
    lagged_raw.to_parquet(LAGGED_RAW)
    print(f"[write]  lagged raw -> {LAGGED_RAW.name}")

    corrected = compute_macro_features(lagged_raw)
    corrected.to_parquet(MACRO_FEATURES)
    print(f"[write]  corrected features -> {MACRO_FEATURES.name}")

    # Report what changed.
    if LEAKED_FEATURES.exists():
        old = pd.read_parquet(LEAKED_FEATURES)
        for c in corrected.columns:
            if c not in old.columns:
                continue
            d = (corrected[c].fillna(0) - old[c].fillna(0)).abs().max()
            tag = "CHANGED" if d > 1e-9 else "same"
            print(f"  {c:24s} max|Δ|={d:.4g}  [{tag}]")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
