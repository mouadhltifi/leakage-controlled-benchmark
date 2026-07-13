"""Faithfully re-run the macro-containing ablation configs with the
publication-lag-corrected FRED features.

Strategy
--------
The original phase4r ablation launcher is not committed, but every result row
records its full config (config_id, fusion_type, fold, seed, lookback,
deadzone, hidden_dim, batch_size, patience, grad_clip, learning_rate, alpha,
modalities). We reconstruct each macro-config run as a sweep override from its
recorded row and re-run it through the same validated ``run_sweep`` path.

Two modes:
  --verify : run a handful of rows against the CURRENTLY ACTIVE cache and check
             the replayed MCC is bit-identical to the recorded MCC. With the
             original (leaked) cache active this proves the reconstruction is
             faithful; only then is the corrected re-run trustworthy.
  --run    : run all macro-config rows (optionally filtered) to a NEW CSV under
             the output dir (a temp dir by default; set MACROLAG_OUT to override).
             Apply the lag first
             (scripts/audits/macro_lag/apply_macro_publication_lag.py) so the
             active cache is corrected.

Macro configs (contain the macro modality): A1, A2, A4, A5, A9.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# The packages live under ``<root>/src``; this script is at
# ``<root>/scripts/audits/macro_lag/`` (three levels down).
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mmfp.experiments.sweep import run_sweep  # noqa: E402
from mmfp.utils.logging_ import setup_logging  # noqa: E402

log = logging.getLogger(__name__)

# Source CSVs for the recorded rows we replay (the committed ablation results).
PHASE4R = _ROOT / "results" / "ablation"
# The re-run WRITES fresh CSVs; default to a temp dir so the documented command
# never clobbers the committed leakage-free results under results/macrolag/
# (override with MACROLAG_OUT to materialise them elsewhere).
OUT_DIR = Path(
    os.environ.get("MACROLAG_OUT", tempfile.gettempdir() + "/mmfp_macrolag_rerun")
)
BASE_CFG = _ROOT / "configs" / "mmfp" / "defaults.toml"

# Canonical modality set per config, RECOVERED from the recorded n_params
# (additive: price 9251 + news 4992 + macro 4736 + social 5440 + graph ~9300 at
# ff/concat/h64). Note A1 = price+news+macro+social with NO graph (n_params
# 24419 = 9251+4992+4736+5440 exactly); no config combines graph with the
# other sources. price is always on.
CONFIG_MODALITIES: dict[str, set[str]] = {
    "A7": set(),               # price-only matched baseline (re-run in mmfp so the
                               # macro deltas are within-codebase paired)
    "A1": {"news", "macro", "social"},
    "A2": {"news", "macro"},
    "A4": {"macro"},
    "A5": {"macro", "social"},
    "A9": {"macro", "graph"},
}
FUSION_MAP = {
    "concat": "concat",
    "gated_cross_attention": "gated_cross_attention",
    "mha": "multihead_attention",
}
# Which original CSV is authoritative for each config (A8/A9 use graph-LN
# re-runs; A1-A3 use the news files; A4-A7 use the base files).
SOURCES: dict[str, list[str]] = {
    "A7": ["ablation_ff.csv", "ablation_lstm.csv"],
    "A4": ["ablation_ff.csv", "ablation_lstm.csv"],
    "A5": ["ablation_ff.csv", "ablation_lstm.csv"],
    "A1": ["ablation_news_ff.csv", "ablation_news_lstm.csv"],
    "A2": ["ablation_news_ff.csv", "ablation_news_lstm.csv"],
    "A9": ["ablation_graph_ln_ff.csv", "ablation_graph_ln_lstm.csv"],
}


def row_to_override(row: pd.Series, cfg_id: str) -> dict:
    mods = CONFIG_MODALITIES[cfg_id]
    lookback = int(row["lookback"])
    arch = "lstm" if lookback > 1 else "ff"
    return {
        "name": f"macrolag_{cfg_id}_{arch}_{row['fusion_type']}"
                f"_f{int(row['fold_idx'])}_s{int(row['seed'])}",
        "seed": int(row["seed"]),
        "data.fold_idx": int(row["fold_idx"]),
        "data.lookback": lookback,
        "data.deadzone": float(row["deadzone"]),
        # ff = lookback 1 + feedforward price encoder; lstm = lookback 20 + lstm.
        "model.price_encoder": "lstm" if lookback > 1 else "feedforward",
        "news.enabled": "news" in mods,
        # The original A1-A3 used the 11-statistic FinBERT path (no PCA). The
        # validator requires pca_dims=None for finbert_11dim; set always so the
        # field is valid even when news is disabled.
        "news.encoder": "finbert_11dim",
        "news.pca_dims": None,
        "macro.enabled": "macro" in mods,
        "social.enabled": "social" in mods,
        "graph.enabled": "graph" in mods,
        "fusion.strategy": FUSION_MAP[row["fusion_type"]],
        "model.hidden_dim": int(row["hidden_dim"]),
        "training.batch_size": int(row["batch_size"]),
        "training.patience": int(row["patience"]),
        "training.grad_clip": float(row["grad_clip"]),
        "training.learning_rate": float(row["learning_rate"]),
        "head.mtl_alpha": float(row["alpha"]),
    }


def collect_rows(configs: list[str]) -> list[tuple[str, pd.Series]]:
    out: list[tuple[str, pd.Series]] = []
    for cfg_id in configs:
        for fname in SOURCES[cfg_id]:
            df = pd.read_csv(PHASE4R / fname)
            sub = df[df["config_id"] == cfg_id]
            for _, row in sub.iterrows():
                out.append((cfg_id, row))
    return out


def verify(targets: list[tuple[str, pd.Series]], parallelism: int) -> int:
    """Replay targets against the ACTIVE cache; compare to recorded MCC."""
    tmp = OUT_DIR / "_verify.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    overrides = [row_to_override(r, c) for c, r in targets]
    run_sweep(BASE_CFG, overrides, tmp, resume=False,
              parallelism=parallelism, device_strategy="cpu")
    res = pd.read_csv(tmp)
    ok = True
    print(f"\n{'config':24s} {'recorded mcc':>16s} {'replayed mcc':>16s}  match")
    for (cfg_id, row), ov in zip(targets, overrides):
        # The result schema stores the override's ``name`` in ``experiment_name``.
        rep = res[res["experiment_name"] == ov["name"]]
        if rep.empty:
            print(f"{ov['name']:24s} {'MISSING REPLAY':>34s}")
            ok = False
            continue
        rec_mcc = float(row["mcc"])
        rep_mcc = float(rep.iloc[0]["mcc"])
        match = abs(rec_mcc - rep_mcc) < 1e-9
        ok = ok and match
        print(f"{ov['name'][:24]:24s} {rec_mcc:16.10f} {rep_mcc:16.10f}  "
              f"{'OK' if match else 'MISMATCH (Δ=%.2e)' % abs(rec_mcc-rep_mcc)}")
    print("\nVERIFY:", "ALL BIT-IDENTICAL — reconstruction is faithful" if ok
          else "MISMATCH — do NOT trust the re-run until fixed")
    return 0 if ok else 1


def run_full(
    configs: list[str], parallelism: int, arch: str | None,
    target: str = "direction",
) -> int:
    rows = collect_rows(configs)
    if arch:
        want_lb = 20 if arch == "lstm" else 1
        rows = [(c, r) for c, r in rows if int(r["lookback"]) == want_lb]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # one CSV per arch to mirror the originals
    for a, want_lb in (("ff", 1), ("lstm", 20)):
        if arch and a != arch:
            continue
        sub = [(c, r) for c, r in rows if int(r["lookback"]) == want_lb]
        if not sub:
            continue
        suffix = "" if target == "direction" else f"_{target}"
        out_csv = OUT_DIR / f"ablation_macrolag{suffix}_{a}.csv"
        overrides = [row_to_override(r, c) for c, r in sub]
        if target == "volatility":
            # Mirror the v2 volatility grid: single-task volatility head.
            for ov in overrides:
                ov["name"] = ov["name"].replace("macrolag_", "macrolagvol_")
                ov["head.targets"] = ["volatility"]
                ov["training.class_weights"] = "none"
        log.info("macrolag %s/%s: %d runs -> %s", target, a, len(overrides), out_csv)
        run_sweep(BASE_CFG, overrides, out_csv, resume=True,
                  parallelism=parallelism, device_strategy="cpu")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument("--configs", nargs="+", default=["A1", "A2", "A4", "A5", "A9"])
    ap.add_argument("--arch", choices=["ff", "lstm"], default=None)
    ap.add_argument("--target", choices=["direction", "volatility"],
                    default="direction")
    args = ap.parse_args()
    setup_logging()

    if args.verify:
        # A fast, representative target set: A4 (macro-only) + A5 (macro+social)
        # + A9 (macro+graph, the graph-handling check), ff fold0 seed42 concat.
        targets: list[tuple[str, pd.Series]] = []
        for cfg_id in ["A4", "A5", "A9", "A2", "A1"]:
            df = pd.read_csv(PHASE4R / SOURCES[cfg_id][0])  # the *_ff.csv
            m = df[(df.config_id == cfg_id) & (df.fold_idx == 0)
                   & (df.seed == 42) & (df.fusion_type == "concat")]
            if not m.empty:
                targets.append((cfg_id, m.iloc[0]))
        return verify(targets, args.parallelism)

    if args.run:
        return run_full(args.configs, args.parallelism, args.arch,
                        target=args.target)

    ap.error("pass --verify or --run")


if __name__ == "__main__":
    raise SystemExit(main())
