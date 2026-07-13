"""In-Python default configuration dict.

Used as the lowest-precedence layer in :func:`mmfp.config.load.load_config`.
Kept in sync with ``mmfp/configs/defaults.toml``; a round-trip test
verifies the two files agree.

Mirrors the canonical TOML in spec Section 4.
"""

from __future__ import annotations

from typing import Any

#: Canonical in-Python defaults. Keep key-for-key aligned with
#: ``mmfp/configs/defaults.toml``.
DEFAULT_CONFIG: dict[str, Any] = {
    "name": "default",
    "seed": 42,
    "device": "auto",
    "data": {
        "universe": "sp500_sector_55",
        "start_date": "2015-02-03",
        "end_date": "2023-12-31",
        "fold_idx": 0,
        "lookback": 1,
        "deadzone": 0.005,
        "warmup_days": 252,
    },
    "price": {
        "enabled": True,
    },
    "news": {
        "enabled": False,
        "encoder": "finbert_11dim",
        "aggregation": "arithmetic_mean",
        "pca_dims": 32,
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
    "fusion": {
        "strategy": "concat",
        "primary_modality": "price",
    },
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
        "min_epochs": 0,
        "patience": 30,
        "grad_clip": 1.0,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "optimizer": "adam",
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
}


__all__ = ["DEFAULT_CONFIG"]
