"""Result-stability reproducibility test.

Spec Section 5.3 requires:

  "Two runs of the same experiment produce matching metrics to the
  documented tolerance."

On CPU the tolerance is bit-identity; on MPS it's 1e-4. This test
covers both, delegating the device-specific checks to the tolerance.

Scope
-----

This is a platform-level smoke test: we care about ``val_loss`` and the
primary validation metric (``val_mcc`` in the default direction+return
config) coming out the same across two identical runs. Full per-axis
coverage lives in the other reproducibility files.
"""

from __future__ import annotations

import copy
import math

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


class _DeterministicPriceDataset(Dataset):
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


def _cfg(device: str) -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test-result-stability"
    d["seed"] = 42
    d["device"] = device
    d["training"]["max_epochs"] = 3
    d["training"]["scheduler"] = "none"
    d["training"]["learning_rate"] = 1e-3
    d["training"]["batch_size"] = 8
    d["data"]["lookback"] = 1
    d["model"]["price_encoder"] = "feedforward"
    cfg = ExperimentConfig.model_validate(d)
    return validate_experiment_config(cfg)


def _run_once(cfg: ExperimentConfig) -> dict[str, float]:
    """Return the last-epoch metrics for a single run."""
    set_all_seeds(cfg.seed)
    schema = FeatureSchema(price=slice(0, 8))
    model = Predictor(cfg, schema)
    loss = build_loss(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.learning_rate,
    )
    scheduler = build_scheduler(cfg, optimizer)
    best_cb = BestModelCheckpoint(monitor="val_mcc", mode="max")

    train_ds = _DeterministicPriceDataset(32, 8, 0)
    val_ds = _DeterministicPriceDataset(16, 8, 1)

    gen = make_dataloader_generator(cfg.seed)
    worker_init_fn = make_worker_init_fn(cfg.seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, generator=gen,
        worker_init_fn=worker_init_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
    )

    trainer = Trainer(
        model=model, cfg=cfg, loss_fn=loss,
        optimizer=optimizer, scheduler=scheduler, callbacks=[best_cb],
    )
    history = trainer.fit(train_loader, val_loader)
    return {
        "train_loss_final": history.train_loss[-1],
        "val_loss_final": history.val_loss[-1],
        "val_mcc_final": history.val_metrics["mcc"][-1],
        "best_metric_value": history.best_metric_value,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResultStabilityCPU:
    def test_two_cpu_runs_bit_identical(self) -> None:
        cfg = _cfg("cpu")
        a = _run_once(cfg)
        b = _run_once(cfg)
        for key in a.keys():
            assert a[key] == b[key], (
                f"CPU stability failed on {key!r}: {a[key]!r} vs {b[key]!r}"
            )

    def test_different_seeds_differ_on_cpu(self) -> None:
        cfg_a = _cfg("cpu")
        cfg_b_dump = copy.deepcopy(cfg_a.model_dump())
        cfg_b_dump["seed"] = 43
        cfg_b = validate_experiment_config(
            ExperimentConfig.model_validate(cfg_b_dump)
        )
        a = _run_once(cfg_a)
        b = _run_once(cfg_b)
        assert a["train_loss_final"] != b["train_loss_final"], (
            f"Different seeds should produce different train_loss_final; "
            f"got {a['train_loss_final']} vs {b['train_loss_final']}"
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available",
)
class TestResultStabilityMPS:
    def test_two_mps_runs_within_tolerance(self) -> None:
        cfg = _cfg("mps")
        a = _run_once(cfg)
        b = _run_once(cfg)
        for key in a.keys():
            va, vb = a[key], b[key]
            # NaN tolerance: both NaN counts as stable.
            if (
                isinstance(va, float) and math.isnan(va)
                and isinstance(vb, float) and math.isnan(vb)
            ):
                continue
            assert abs(va - vb) <= 1e-4, (
                f"MPS stability failed on {key!r}: "
                f"{va!r} vs {vb!r} (|delta|={abs(va - vb)})"
            )
