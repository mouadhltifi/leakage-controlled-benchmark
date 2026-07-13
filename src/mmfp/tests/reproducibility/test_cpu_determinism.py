"""CPU bit-identical determinism test.

The spec (Section 3.11 / 5.3) demands that two invocations of a tiny
price-only trainer — same seed, same config, CPU device — produce
*bit-identical* ``train_loss`` and ``val_mcc`` histories.

Without this, later reproducibility claims in the study are hollow.

Strategy
--------

1. Construct a trivial synthetic price-only dataset and config.
2. Run trainer twice from scratch, each time re-seeding with
   :func:`set_all_seeds` before building model / optimizer / loaders.
3. Use ``DataLoader(generator=...)`` + ``worker_init_fn`` to freeze the
   sampler and worker RNG.
4. Assert per-epoch ``train_loss`` and ``val_mcc`` are equal bit-for-bit.

Shuffling
---------

We turn shuffle on so the test catches RNG issues in sampling order.
The seeded generator makes shuffling deterministic.
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


# ---------------------------------------------------------------------------
# Dataset + config helpers
# ---------------------------------------------------------------------------


class _DeterministicPriceDataset(Dataset):
    """Tiny dataset with controlled contents (no live RNG at __getitem__ time)."""

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


def _cpu_cfg() -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test-cpu-determinism"
    d["seed"] = 42
    d["device"] = "cpu"
    d["training"]["max_epochs"] = 3
    d["training"]["scheduler"] = "none"
    d["training"]["learning_rate"] = 1e-3
    d["training"]["batch_size"] = 8
    d["data"]["lookback"] = 1
    d["model"]["price_encoder"] = "feedforward"
    cfg = ExperimentConfig.model_validate(d)
    return validate_experiment_config(cfg)


def _run_once(cfg: ExperimentConfig) -> tuple[list[float], list[float]]:
    """One fresh training run; returns (train_loss_history, val_mcc_history)."""
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
        shuffle=True,  # shuffling ON — the seeded generator makes it deterministic
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
        optimizer=optimizer, scheduler=scheduler,
        callbacks=[best_cb],
    )
    history = trainer.fit(train_loader, val_loader)
    return history.train_loss, history.val_metrics["mcc"]


# ---------------------------------------------------------------------------
# The actual determinism test
# ---------------------------------------------------------------------------


def test_cpu_bit_identical_two_runs() -> None:
    """Two fresh runs of the same config+seed on CPU produce identical histories."""
    cfg = _cpu_cfg()

    train_loss_a, val_mcc_a = _run_once(cfg)
    train_loss_b, val_mcc_b = _run_once(cfg)

    assert len(train_loss_a) == len(train_loss_b), (
        f"Different history lengths: {len(train_loss_a)} vs {len(train_loss_b)}"
    )
    assert len(val_mcc_a) == len(val_mcc_b)

    # Bit-identical: compare float64 exactly.
    for i, (a, b) in enumerate(zip(train_loss_a, train_loss_b)):
        assert a == b, (
            f"train_loss[{i}] differs bit-identically: "
            f"{a!r} (run A) vs {b!r} (run B)"
        )
    for i, (a, b) in enumerate(zip(val_mcc_a, val_mcc_b)):
        assert a == b, (
            f"val_mcc[{i}] differs bit-identically: "
            f"{a!r} (run A) vs {b!r} (run B)"
        )


def test_different_seeds_diverge_on_cpu() -> None:
    """Sanity: swapping the seed produces a different history (not all flat)."""
    cfg_a = _cpu_cfg()
    cfg_b = copy.deepcopy(cfg_a.model_dump())
    cfg_b["seed"] = 43
    cfg_b = validate_experiment_config(
        ExperimentConfig.model_validate(cfg_b)
    )

    train_loss_a, _ = _run_once(cfg_a)
    train_loss_b, _ = _run_once(cfg_b)

    # With distinct seeds at least one epoch should differ.
    assert any(
        a != b for a, b in zip(train_loss_a, train_loss_b)
    ), (
        f"Expected different histories for different seeds; "
        f"got identical: {train_loss_a}"
    )


def test_cpu_bit_identical_with_no_shuffle() -> None:
    """Non-shuffling loaders — a simpler determinism path; also bit-identical."""
    cfg = _cpu_cfg()

    def _run_no_shuffle() -> list[float]:
        set_all_seeds(cfg.seed)
        schema = FeatureSchema(price=slice(0, 8))
        model = Predictor(cfg, schema)
        loss = build_loss(cfg)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
        )
        scheduler = build_scheduler(cfg, optimizer)

        train_ds = _DeterministicPriceDataset(32, 8, 0)
        val_ds = _DeterministicPriceDataset(16, 8, 1)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.training.batch_size, shuffle=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        )

        trainer = Trainer(
            model=model, cfg=cfg, loss_fn=loss,
            optimizer=optimizer, scheduler=scheduler, callbacks=[],
        )
        history = trainer.fit(train_loader, val_loader)
        return history.train_loss

    a = _run_no_shuffle()
    b = _run_no_shuffle()
    assert a == b, f"non-shuffle determinism failed: {a} vs {b}"
