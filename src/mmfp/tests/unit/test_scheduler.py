"""Unit tests for :mod:`mmfp.training.scheduler`.

The factory is tiny; the tests verify:

* Each supported choice maps to the expected concrete class.
* Cosine warm restarts actually changes the LR across epochs.
* Plateau schedulers are detected by :func:`is_plateau_scheduler`.
* ``"none"`` returns ``None``.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.training.scheduler import build_scheduler, is_plateau_scheduler


def _cfg(**overrides) -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test-scheduler"
    for dotted, value in overrides.items():
        cursor = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = value
    return ExperimentConfig.model_validate(d)


def _toy_optimizer(lr: float = 0.1) -> Adam:
    p = torch.nn.Parameter(torch.zeros(1))
    return Adam([p], lr=lr)


# ---------------------------------------------------------------------------
# Type dispatch
# ---------------------------------------------------------------------------


class TestBuildSchedulerDispatch:
    def test_cosine_warm_restarts_returns_cosine(self) -> None:
        cfg = _cfg(**{"training.scheduler": "cosine_warm_restarts"})
        opt = _toy_optimizer()
        sched = build_scheduler(cfg, opt)
        assert isinstance(sched, CosineAnnealingWarmRestarts)
        assert sched.T_0 == 10
        assert sched.T_mult == 2

    def test_reduce_on_plateau_returns_plateau(self) -> None:
        cfg = _cfg(**{"training.scheduler": "reduce_on_plateau"})
        opt = _toy_optimizer()
        sched = build_scheduler(cfg, opt)
        assert isinstance(sched, ReduceLROnPlateau)
        # Mode should be min per spec.
        assert sched.mode == "min"
        # patience=10, factor=0.5 per spec.
        assert sched.patience == 10
        assert sched.factor == 0.5

    def test_none_returns_none(self) -> None:
        cfg = _cfg(**{"training.scheduler": "none"})
        opt = _toy_optimizer()
        sched = build_scheduler(cfg, opt)
        assert sched is None


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class TestSchedulerBehaviour:
    def test_cosine_changes_lr_across_epochs(self) -> None:
        """Cosine warm restarts should drive LR away from the initial value."""
        cfg = _cfg(**{"training.scheduler": "cosine_warm_restarts"})
        opt = _toy_optimizer(lr=0.1)
        sched = build_scheduler(cfg, opt)
        assert sched is not None

        initial_lr = opt.param_groups[0]["lr"]
        seen_lrs = [initial_lr]
        for _ in range(6):
            sched.step()
            seen_lrs.append(opt.param_groups[0]["lr"])

        # At least one epoch-step changed the LR.
        assert any(abs(lr - initial_lr) > 1e-9 for lr in seen_lrs[1:]), (
            f"Cosine scheduler did not change LR across epochs: {seen_lrs}"
        )

    def test_cosine_restarts_at_T0(self) -> None:
        """After T_0=10 steps, cosine warm restart should pop the LR back up."""
        cfg = _cfg(**{"training.scheduler": "cosine_warm_restarts"})
        opt = _toy_optimizer(lr=0.1)
        sched = build_scheduler(cfg, opt)
        assert sched is not None

        # Step 9 times: close to the trough of the first cycle.
        for _ in range(9):
            sched.step()
        lr_at_trough = opt.param_groups[0]["lr"]

        # Step once more: this restarts the cycle, LR should jump up.
        sched.step()
        lr_after_restart = opt.param_groups[0]["lr"]

        assert lr_after_restart > lr_at_trough, (
            f"Expected LR to rise after T_0 restart; "
            f"got {lr_at_trough=} -> {lr_after_restart=}"
        )


# ---------------------------------------------------------------------------
# is_plateau_scheduler helper
# ---------------------------------------------------------------------------


class TestIsPlateauScheduler:
    def test_plateau_detected(self) -> None:
        cfg = _cfg(**{"training.scheduler": "reduce_on_plateau"})
        opt = _toy_optimizer()
        sched = build_scheduler(cfg, opt)
        assert is_plateau_scheduler(sched) is True

    def test_cosine_not_plateau(self) -> None:
        cfg = _cfg(**{"training.scheduler": "cosine_warm_restarts"})
        opt = _toy_optimizer()
        sched = build_scheduler(cfg, opt)
        assert is_plateau_scheduler(sched) is False

    def test_none_not_plateau(self) -> None:
        assert is_plateau_scheduler(None) is False


# ---------------------------------------------------------------------------
# Defensive
# ---------------------------------------------------------------------------


class TestDefensive:
    def test_unknown_scheduler_would_fail_schema_not_factory(self) -> None:
        """Invalid scheduler names never reach the factory — Pydantic rejects first."""
        # Construct a config that *is* valid and then monkey-mangle it.
        cfg = _cfg()
        # The Pydantic model has ``validate_assignment=True`` so direct
        # assignment of an invalid value raises.
        with pytest.raises(Exception):
            cfg.training.scheduler = "not_a_scheduler"  # type: ignore[assignment]
