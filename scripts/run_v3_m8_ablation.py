"""v3 M8 driver — full 9-config ablation matching the v2 ablation grid.

Per the pre-registered plan. Runs 9 modality configs × 5 folds × 3 seeds
= 135 ForecastResultRecord rows on v3 TFT.

Config definitions match v2's A1-A9 (from
results/analysis/phase5_full_750core.md):
  A1: price + news + social + macro             (full model, 4 modalities — NO graph)
  A2: price + news + macro                      (A1 remove social)
  A3: price + news
  A4: price + macro
  A5: price + macro + social
  A6: price + social
  A7: price only                                 (baseline)
  A8: price + graph
  A9: price + macro + graph

News locked to Qwen3 + PCA128 per v3 scope. Graph mode: static_gics
(dynamic_corr ablation deferred). Architecture: TFT quantile forecaster.

Output: results/v3/m8_ablation/full_9config.csv
Expected wall-clock: ~10-15 hours on M1 Max CPU, parallelism=4.
"""
from __future__ import annotations

import os
# Pinned-state thread configuration (S3.2 determinism scope): thread count
# changes floating-point reduction order and can move a single run by up
# to ~0.04 MCC. The shipped grids ran under this pin; keep it for
# bit-identical reproduction (override only knowingly).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

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
OUT_CSV = _ROOT / "results" / "v3" / "m8_ablation" / "full_9config.csv"

SEEDS = [42, 123, 456]
FOLDS = [0, 1, 2, 3, 4]

# Config label -> modality flags. Price is always enabled.
# Definitions match the v2 ablation (see results/analysis/phase5_full_750core.md).
CONFIGS: dict[str, dict[str, bool]] = {
    "A1": {"news": True,  "social": True,  "macro": True,  "graph": False}, # full 4-mod (no graph)
    "A2": {"news": True,  "social": False, "macro": True,  "graph": False}, # price+news+macro (A1−social)
    "A3": {"news": True,  "social": False, "macro": False, "graph": False}, # price+news
    "A4": {"news": False, "social": False, "macro": True,  "graph": False}, # price+macro
    "A5": {"news": False, "social": True,  "macro": True,  "graph": False}, # price+macro+social
    "A6": {"news": False, "social": True,  "macro": False, "graph": False}, # price+social
    "A7": {"news": False, "social": False, "macro": False, "graph": False}, # price only
    "A8": {"news": False, "social": False, "macro": False, "graph": True},  # price+graph
    "A9": {"news": False, "social": False, "macro": True,  "graph": True},  # price+macro+graph
}


def build_overrides() -> list[dict]:
    """Construct 9 × 5 × 3 = 135 override dicts."""
    overrides: list[dict] = []
    for fold in FOLDS:
        for seed in SEEDS:
            for label, flags in CONFIGS.items():
                ov = {
                    "name": f"v3_m8_{label}",
                    "seed": seed,
                    "data.fold_idx": fold,
                }
                for mod, enabled in flags.items():
                    ov[f"{mod}.enabled"] = enabled
                overrides.append(ov)
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument(
        "--device-strategy", choices=["cpu", "mps", "auto"], default="cpu"
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_logging()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    overrides = build_overrides()

    log.info(
        "v3 M8 ablation — %d runs planned (9 configs × 5 folds × 3 seeds)",
        len(overrides),
    )
    log.info("  base=%s", BASE_CFG)
    log.info("  out=%s", OUT_CSV)
    log.info("  configs=%s", list(CONFIGS.keys()))

    if args.dry_run:
        for o in overrides[:12]:
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
    log.info("v3 M8 ablation sweep complete. CSV at %s", OUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
