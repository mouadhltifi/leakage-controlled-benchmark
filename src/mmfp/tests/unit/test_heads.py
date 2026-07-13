"""Unit tests for :mod:`mmfp.models.heads`.

Covers the contract for each head:

* Parallel head emits all requested targets.
* Sequential cascade reg_head receives gradient from cls loss alone
  when ``detach_cascade=False``. This is the single most important
  test of this milestone — it verifies the cascade is jointly trained.
* ``detach_cascade=True`` blocks that gradient flow.
* Single-task returns only the configured target.
* Target output dimensions match convention (direction=2, return=1,
  volatility=1).
"""

from __future__ import annotations

import copy

import pytest
import torch

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.models.heads import (
    ParallelMultiTaskHead,
    PredictionHead,
    SequentialCascadeHead,
    SingleTaskHead,
    build_head,
)


def _cfg(**overrides) -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test-head"
    for dotted, value in overrides.items():
        cursor = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = value
    return ExperimentConfig.model_validate(d)


# ---------------------------------------------------------------------------
# ParallelMultiTaskHead
# ---------------------------------------------------------------------------


class TestParallelMultiTaskHead:
    def test_direction_return(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction", "return"],
        })
        head = ParallelMultiTaskHead(cfg)
        out = head(torch.randn(4, cfg.model.hidden_dim))
        assert set(out.keys()) == {"direction", "return"}
        assert out["direction"].shape == (4, 2)
        assert out["return"].shape == (4, 1)

    def test_all_three_targets(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction", "return", "volatility"],
        })
        head = ParallelMultiTaskHead(cfg)
        out = head(torch.randn(3, cfg.model.hidden_dim))
        assert set(out.keys()) == {"direction", "return", "volatility"}
        assert out["volatility"].shape == (3, 1)

    def test_single_target_works(self) -> None:
        # Parallel head is a valid container for a single target too.
        cfg = _cfg(**{
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction"],
        })
        head = ParallelMultiTaskHead(cfg)
        out = head(torch.randn(2, cfg.model.hidden_dim))
        assert set(out.keys()) == {"direction"}

    def test_gradient_flow_reaches_all_heads(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction", "return"],
        })
        head = ParallelMultiTaskHead(cfg)
        out = head(torch.randn(4, cfg.model.hidden_dim))
        # Sum both branches: gradient reaches both.
        (out["direction"].sum() + out["return"].sum()).backward()
        for name, p in head.named_parameters():
            assert p.grad is not None, f"{name} no grad"

    def test_mid_dim_formula(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction"],
            "model.hidden_dim": 128,
        })
        head = ParallelMultiTaskHead(cfg)
        first_linear = head.heads["direction"][0]
        assert first_linear.out_features == max(32, 128 // 2)

    def test_mid_dim_minimum_32(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction"],
            "model.hidden_dim": 16,
        })
        head = ParallelMultiTaskHead(cfg)
        first_linear = head.heads["direction"][0]
        # max(32, 16 // 2) = max(32, 8) = 32
        assert first_linear.out_features == 32


# ---------------------------------------------------------------------------
# SequentialCascadeHead -- the critical gradient tests
# ---------------------------------------------------------------------------


class TestSequentialCascadeHead:
    def test_outputs_direction_and_reg(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "sequential_cascade",
            "head.targets": ["direction", "return"],
            "head.cascade_reg_target": "return",
        })
        head = SequentialCascadeHead(cfg)
        out = head(torch.randn(4, cfg.model.hidden_dim))
        assert set(out.keys()) == {"direction", "return"}
        assert out["direction"].shape == (4, 2)
        assert out["return"].shape == (4, 1)

    def test_cascade_to_volatility(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "sequential_cascade",
            "head.targets": ["direction", "volatility"],
            "head.cascade_reg_target": "volatility",
        })
        head = SequentialCascadeHead(cfg)
        out = head(torch.randn(4, cfg.model.hidden_dim))
        assert set(out.keys()) == {"direction", "volatility"}
        assert out["volatility"].shape == (4, 1)

    def test_cls_loss_flows_to_reg_head_when_not_detached(self) -> None:
        """CRITICAL: verify the cascade is jointly differentiable.

        When ``detach_cascade=False`` (the v2 default), backpropagating
        from the classification loss alone must still update the
        regression branch's parameters.
        """
        cfg = _cfg(**{
            "head.architecture": "sequential_cascade",
            "head.targets": ["direction", "return"],
            "head.cascade_reg_target": "return",
            "head.detach_cascade": False,
        })
        head = SequentialCascadeHead(cfg)
        x = torch.randn(8, cfg.model.hidden_dim)
        out = head(x)

        # Cross-entropy-style loss using the CLS output only.
        cls_target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        cls_loss = torch.nn.functional.cross_entropy(out["direction"], cls_target)
        cls_loss.backward()

        # Final linear of reg_head must receive non-zero gradient.
        final_reg_linear = head.reg_head[-1]
        assert final_reg_linear.weight.grad is not None
        assert final_reg_linear.weight.grad.abs().sum().item() > 0

    def test_detach_cascade_blocks_gradient_flow(self) -> None:
        """With ``detach_cascade=True`` the cls loss does NOT update reg_head."""
        cfg = _cfg(**{
            "head.architecture": "sequential_cascade",
            "head.targets": ["direction", "return"],
            "head.cascade_reg_target": "return",
            "head.detach_cascade": True,
        })
        head = SequentialCascadeHead(cfg)
        x = torch.randn(8, cfg.model.hidden_dim)
        out = head(x)

        cls_target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        cls_loss = torch.nn.functional.cross_entropy(out["direction"], cls_target)
        cls_loss.backward()

        # With detach, cls loss alone cannot update the reg branch.
        final_reg_linear = head.reg_head[-1]
        assert (
            final_reg_linear.weight.grad is None
            or final_reg_linear.weight.grad.abs().sum().item() == 0
        )

    def test_full_joint_loss_gradient(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "sequential_cascade",
            "head.targets": ["direction", "return"],
            "head.cascade_reg_target": "return",
        })
        head = SequentialCascadeHead(cfg)
        out = head(torch.randn(4, cfg.model.hidden_dim))
        (out["direction"].sum() + out["return"].sum()).backward()
        for name, p in head.named_parameters():
            assert p.grad is not None, f"{name} no grad"

    def test_missing_direction_target_raises(self) -> None:
        # Manually bypass validator by passing a crafted config; bypassing
        # validator means we still expect the head to fail loud.
        cfg = _cfg(**{
            "head.architecture": "sequential_cascade",
            "head.targets": ["return", "volatility"],
            "head.cascade_reg_target": "return",
        })
        with pytest.raises(ValueError, match="requires 'direction'"):
            SequentialCascadeHead(cfg)

    def test_missing_reg_target_in_targets_raises(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "sequential_cascade",
            "head.targets": ["direction", "return"],
            "head.cascade_reg_target": "volatility",
        })
        with pytest.raises(ValueError, match="to be in head.targets"):
            SequentialCascadeHead(cfg)


# ---------------------------------------------------------------------------
# SingleTaskHead
# ---------------------------------------------------------------------------


class TestSingleTaskHead:
    def test_direction_only(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "single_task",
            "head.targets": ["direction"],
        })
        head = SingleTaskHead(cfg)
        out = head(torch.randn(4, cfg.model.hidden_dim))
        assert set(out.keys()) == {"direction"}
        assert out["direction"].shape == (4, 2)

    def test_volatility_only(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "single_task",
            "head.targets": ["volatility"],
        })
        head = SingleTaskHead(cfg)
        out = head(torch.randn(3, cfg.model.hidden_dim))
        assert set(out.keys()) == {"volatility"}
        assert out["volatility"].shape == (3, 1)

    def test_return_only(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "single_task",
            "head.targets": ["return"],
        })
        head = SingleTaskHead(cfg)
        out = head(torch.randn(3, cfg.model.hidden_dim))
        assert out["return"].shape == (3, 1)

    def test_multi_target_raises(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "parallel_multitask",  # to pass validator
            "head.targets": ["direction", "return"],
        })
        # Then mutate the architecture to trick into a bad state.
        with pytest.raises(ValueError, match="exactly one target"):
            # Construct SingleTaskHead directly with a multi-target cfg.
            SingleTaskHead(cfg)


# ---------------------------------------------------------------------------
# build_head factory
# ---------------------------------------------------------------------------


class TestBuildHead:
    def test_parallel(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "parallel_multitask",
            "head.targets": ["direction", "return"],
        })
        assert isinstance(build_head(cfg), ParallelMultiTaskHead)

    def test_sequential(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "sequential_cascade",
            "head.targets": ["direction", "return"],
            "head.cascade_reg_target": "return",
        })
        assert isinstance(build_head(cfg), SequentialCascadeHead)

    def test_single_task(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "single_task",
            "head.targets": ["direction"],
        })
        assert isinstance(build_head(cfg), SingleTaskHead)

    def test_returns_prediction_head_subclass(self) -> None:
        cfg = _cfg()
        head = build_head(cfg)
        assert isinstance(head, PredictionHead)
