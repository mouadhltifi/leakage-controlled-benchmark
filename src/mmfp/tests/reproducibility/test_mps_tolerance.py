"""MPS tolerance reproducibility test.

Per spec Section 3.11: on Apple Silicon MPS, the LSTM path is
documented as non-deterministic. The tolerance contract is 1e-4 on
per-epoch ``val_mcc``.

If MPS is unavailable (non-Apple hardware, CPU-only CI), the whole
module is skipped at collection time.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.config.validate import validate_experiment_config
from mmfp.data.assemble import FeatureSchema
from mmfp.models import Predictor
from mmfp.models.losses import build_loss
from mmfp.training import (
    BestModelCheckpoint,
    Trainer,
    build_scheduler,
)
from mmfp.utils.seeding import (
    make_dataloader_generator,
    make_worker_init_fn,
    set_all_seeds,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this machine",
)


class _DeterministicPriceDataset(Dataset):
    """Same tiny dataset used by the CPU determinism test."""

    def __init__(self, n_samples: int, price_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._price = rng.standard_normal(
            (n_samples, price_dim),
        ).astype(np.float32)
        self._cls = rng.integers(0, 2, size=n_samples).astype(np.int64)
        self._reg = rng.standard_normal(n_samples).astype(np.float32) * 0.01
        self._stock = rng.integers(0, 5, size=n_samples).astype(np.int64)

    def __len__(self) -> int:
        return len(self._price)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "price": torch.as_tensor(self._price[idx]),
            "cls_target": torch.as_tensor(self._cls[idx], dtype=torch.long),
            "reg_target": torch.as_tensor(self._reg[idx], dtype=torch.float32),
            "stock_idx": torch.as_tensor(self._stock[idx], dtype=torch.long),
        }


def _mps_cfg() -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test-mps-tolerance"
    d["seed"] = 42
    d["device"] = "mps"
    d["training"]["max_epochs"] = 3
    d["training"]["scheduler"] = "none"
    d["training"]["learning_rate"] = 1e-3
    d["training"]["batch_size"] = 8
    d["data"]["lookback"] = 1
    d["model"]["price_encoder"] = "feedforward"
    cfg = ExperimentConfig.model_validate(d)
    return validate_experiment_config(cfg)


def _run_once(cfg: ExperimentConfig) -> tuple[list[float], list[float]]:
    set_all_seeds(cfg.seed)

    schema = FeatureSchema(price=slice(0, 8))
    model = Predictor(cfg, schema)
    loss = build_loss(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = build_scheduler(cfg, optimizer)
    best_cb = BestModelCheckpoint(monitor="val_mcc", mode="max")

    train_ds = _DeterministicPriceDataset(n_samples=32, price_dim=8, seed=0)
    val_ds = _DeterministicPriceDataset(n_samples=16, price_dim=8, seed=1)

    gen = make_dataloader_generator(cfg.seed)
    worker_init_fn = make_worker_init_fn(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
        generator=gen,
        worker_init_fn=worker_init_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
    )

    trainer = Trainer(
        model=model, cfg=cfg, loss_fn=loss,
        optimizer=optimizer, scheduler=scheduler, callbacks=[best_cb],
    )
    history = trainer.fit(train_loader, val_loader)
    return history.train_loss, history.val_metrics["mcc"]


def test_mps_tolerance_on_val_mcc() -> None:
    """Two MPS runs agree on ``val_mcc`` within 1e-4 per epoch."""
    cfg = _mps_cfg()
    _, val_mcc_a = _run_once(cfg)
    _, val_mcc_b = _run_once(cfg)
    assert len(val_mcc_a) == len(val_mcc_b)
    for i, (a, b) in enumerate(zip(val_mcc_a, val_mcc_b)):
        assert abs(a - b) <= 1e-4, (
            f"MPS val_mcc[{i}] differs beyond tolerance: "
            f"{a!r} vs {b!r} (|delta|={abs(a - b)})"
        )
