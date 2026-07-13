"""A4 seed sensitivity retest — vetting step for the one positive candidate.

The 9-config ablation found A4 (price+macro) apparently publishable under the
TFT with d=+0.861, p_bonf=0.039 (k=8). Before promoting it to a contribution,
we verify robustness across additional seeds.

Runs A4 and A7 (baseline) × 5 folds × 5 new seeds:
  seeds = [789, 1000, 1234, 5678, 9999]

Total: 2 configs × 5 folds × 5 seeds = 50 runs.
ETA: ~7-9h on CPU, parallelism=4.

If A4's delta vs A7 holds at d > 0.3 after combining the original 3 seeds with
these 5 (8 total), the finding is robust. It did NOT — the combined estimate
fell to d=+0.397, p_bonf=0.131, so A4 was demoted (the fragile result was caught
before publication).

Output: results/v3/m8_ablation/a4_seed_retest.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from forecast.experiments.sweep import run_forecast_sweep  # noqa: E402
from mmfp.utils.logging_ import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

BASE_CFG = _ROOT / "configs" / "forecast" / "experiments" / "A7_price_only_tft.toml"
OUT_CSV = _ROOT / "results" / "v3" / "m8_ablation" / "a4_seed_retest.csv"

# Extension seeds declared up front to prevent post-hoc selection
EXTENSION_SEEDS = [789, 1000, 1234, 5678, 9999]
FOLDS = [0, 1, 2, 3, 4]

CONFIGS: dict[str, dict[str, bool]] = {
    "A7": {"news": False, "social": False, "macro": False, "graph": False},
    "A4": {"news": False, "social": False, "macro": True,  "graph": False},
}


def build_overrides() -> list[dict]:
    overrides: list[dict] = []
    for fold in FOLDS:
        for seed in EXTENSION_SEEDS:
            for label, flags in CONFIGS.items():
                ov = {"name": f"v3_a4_retest_{label}", "seed": seed, "data.fold_idx": fold}
                for mod, enabled in flags.items():
                    ov[f"{mod}.enabled"] = enabled
                overrides.append(ov)
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--device-strategy", choices=["cpu", "mps", "auto"], default="cpu")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_logging()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    overrides = build_overrides()

    log.info("A4 seed retest — %d runs (A4 + A7 × 5 folds × 5 seeds)", len(overrides))
    log.info("  seeds (extension): %s", EXTENSION_SEEDS)
    log.info("  base=%s  out=%s", BASE_CFG, OUT_CSV)

    if args.dry_run:
        for o in overrides[:10]:
            log.info("  %s", o)
        log.info("  ... (%d total)", len(overrides))
        return 0

    run_forecast_sweep(
        base_cfg_path=BASE_CFG,
        overrides=overrides,
        output_csv=OUT_CSV,
        resume=args.resume,
        parallelism=args.parallelism,
        device_strategy=args.device_strategy,
    )
    log.info("A4 retest complete. CSV at %s", OUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
