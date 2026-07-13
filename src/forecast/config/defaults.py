"""In-Python default configuration dict for v3.

Paired with ``forecast/configs/defaults.toml``; the two must agree field-for-
field (enforced by ``test_defaults_toml_matches_default_config_dict``).

Shape: identical to v2's ``mmfp.config.defaults.DEFAULT_CONFIG`` plus a
``forecast`` section. v3 overrides a handful of training / data knobs per
the architecture spec:

* ``data.lookback`` 1 → 60
* ``training.min_epochs`` 0 → 20  (VSN needs ≥10 epochs to escape uniform init)
* ``training.learning_rate`` 1e-4 → 3e-4
* ``training.weight_decay`` 1e-5 → 1e-4
* ``training.optimizer`` "adam" → "adamw"
* ``news.encoder`` "finbert_11dim" → "qwen3_embedding"  (scope lock)
* ``news.pca_dims`` 32 → 128  (scope lock)

All other v2 defaults are preserved for continuity with the reused v2
trainer / dataset / runner machinery.
"""

from __future__ import annotations

from typing import Any

#: Canonical in-Python defaults. Keep key-for-key aligned with
#: ``forecast/configs/defaults.toml``. The round-trip test in
#: ``forecast/tests/unit/test_config_schema.py`` enforces parity.
DEFAULT_CONFIG: dict[str, Any] = {
    "name": "default_v3",
    "seed": 42,
    "device": "auto",
    "data": {
        "universe": "sp500_sector_55",
        "start_date": "2015-02-03",
        "end_date": "2023-12-31",
        "fold_idx": 0,
        "lookback": 60,  # v3: TFT window
        "deadzone": 0.005,
        "warmup_days": 252,
    },
    "price": {
        "enabled": True,
    },
    "news": {
        "enabled": False,
        "encoder": "qwen3_embedding",  # v3 scope lock
        "aggregation": "arithmetic_mean",
        "pca_dims": 128,  # v3 scope lock (up from v2's 32)
        "pca_train_end": "",
        "log1p_volume": True,
        "dispersion_feature": False,
        "empty_day_policy": "zero_fill_has_news_flag",
        "lag_days": 1,
    },
    "social": {
        "enabled": False,
        "aggregation_window": 3,
        "log1p_volume": True,
        "lag_days": 1,
    },
    "macro": {
        "enabled": False,
        "fomc_window": 1,
    },
    "graph": {
        "enabled": False,
        "source": "static_gics",
        "dynamic_window": 20,
        "dynamic_threshold": 0.3,
        "dynamic_refresh_every": 20,
    },
    "standardization": {
        "price": True,
        "macro": True,
        "news": True,
        "social": True,
        "graph_node": True,
        "fit_scope": "train_fold_only",
    },
    # v2 ``model`` block retained for runner-dispatch compatibility; ignored
    # by the v3 TFT predictor.
    "model": {
        "hidden_dim": 64,
        "dropout": 0.2,
        "num_heads": 4,
        "price_encoder": "lstm",
        "price_lstm_layers": 2,
        "tabular_hidden_layers": 1,
        "graph_gat_heads": 4,
        "graph_gat_layers": 2,
    },
    # v2 ``fusion`` block retained; unused under architecture="tft".
    "fusion": {
        "strategy": "concat",
        "primary_modality": "price",
    },
    # v2 ``head`` block retained; unused under architecture="tft".
    "head": {
        "architecture": "parallel_multitask",
        "targets": ["direction", "return"],
        "mtl_alpha": 0.5,
        "cascade_reg_target": "return",
        "detach_cascade": False,
    },
    "training": {
        "batch_size": 64,
        "max_epochs": 150,
        "min_epochs": 20,  # v3: TFT needs ≥10 epochs to escape VSN uniform init
        "patience": 30,
        "grad_clip": 1.0,
        "learning_rate": 3e-4,  # v3: up from 1e-4 (TFT paper default mid-band)
        "weight_decay": 1e-4,  # v3: up from 1e-5 (AdamW decoupled)
        "optimizer": "adamw",  # v3: AdamW with decoupled weight decay
        "scheduler": "cosine_warm_restarts",
        "class_weights": "inverse_frequency",
        "val_fraction": 0.2,
    },
    "intraday": {
        "enabled": False,
        "frequency": "daily",
        "source": "none",
    },
    "logging": {
        "level": "INFO",
        "per_epoch_log_every": 10,
    },
    "forecast": {
        "architecture": "tft",
        "hidden_dim": 64,
        "n_heads": 1,
        "n_lstm_layers": 1,
        "dropout": 0.3,
        "lookback": 60,
        "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
        "direction_aux_weight": 0.0,
        "volatility_aux_weight": 0.0,
        "graph_precompute": True,
        "graph_node_dim": 64,
        "n_tickers": 55,
        "n_sectors": 11,
        "warmup_epochs": 5,
    },
}


__all__ = ["DEFAULT_CONFIG"]
