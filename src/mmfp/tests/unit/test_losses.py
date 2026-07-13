"""Unit tests for :mod:`mmfp.models.losses`.

Covers:

* ``MultiTaskLoss`` weight normalisation (sums to 1).
* ``ignore_index=-1`` handling: deadzone tokens contribute no CE loss.
* ``VolatilityLoss`` on single target, both dict and tensor inputs.
* ``compute_class_weights`` for both schemes.
* Factory dispatch between multi-task and single-target volatility.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig
from mmfp.models.losses import (
    MultiTaskLoss,
    VolatilityLoss,
    build_loss,
    compute_class_weights,
)


def _cfg(**overrides) -> ExperimentConfig:
    d = copy.deepcopy(DEFAULT_CONFIG)
    d["name"] = "test-loss"
    for dotted, value in overrides.items():
        cursor = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            cursor = cursor[p]
        cursor[parts[-1]] = value
    return ExperimentConfig.model_validate(d)


# ---------------------------------------------------------------------------
# MultiTaskLoss weight normalisation
# ---------------------------------------------------------------------------


class TestMultiTaskLossWeights:
    def test_two_targets_sums_to_one(self) -> None:
        cfg = _cfg(**{
            "head.targets": ["direction", "return"],
            "head.mtl_alpha": 0.7,
        })
        loss = MultiTaskLoss(cfg.head)
        s = sum(loss.weights.values())
        assert pytest.approx(s, rel=1e-6) == 1.0
        assert loss.weights["direction"] == pytest.approx(0.7)
        assert loss.weights["return"] == pytest.approx(0.3)

    def test_three_targets_with_explicit_beta(self) -> None:
        cfg = _cfg(**{
            "head.targets": ["direction", "return", "volatility"],
            "head.mtl_alpha": 0.5,
            "head.mtl_beta": 0.3,
        })
        loss = MultiTaskLoss(cfg.head)
        assert pytest.approx(sum(loss.weights.values())) == 1.0
        assert loss.weights["direction"] == pytest.approx(0.5)
        assert loss.weights["return"] == pytest.approx(0.3)
        assert loss.weights["volatility"] == pytest.approx(0.2)

    def test_three_targets_default_beta(self) -> None:
        cfg = _cfg(**{
            "head.targets": ["direction", "return", "volatility"],
            "head.mtl_alpha": 0.4,
        })
        # beta = (1 - alpha) / 2 = 0.3 for each of the two non-direction
        # targets, gamma = 1 - alpha - beta = 0.3.
        loss = MultiTaskLoss(cfg.head)
        assert pytest.approx(sum(loss.weights.values())) == 1.0
        assert loss.weights["direction"] == pytest.approx(0.4)
        assert loss.weights["return"] == pytest.approx(0.3)
        assert loss.weights["volatility"] == pytest.approx(0.3)

    def test_single_target_weight_is_one(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "single_task",
            "head.targets": ["direction"],
        })
        loss = MultiTaskLoss(cfg.head)
        assert loss.weights == {"direction": 1.0}


# ---------------------------------------------------------------------------
# MultiTaskLoss forward
# ---------------------------------------------------------------------------


class TestMultiTaskLossForward:
    def test_direction_plus_return(self) -> None:
        cfg = _cfg(**{
            "head.targets": ["direction", "return"],
            "head.mtl_alpha": 0.5,
        })
        loss = MultiTaskLoss(cfg.head)
        preds = {
            "direction": torch.randn(6, 2, requires_grad=True),
            "return": torch.randn(6, 1, requires_grad=True),
        }
        targets = {
            "direction": torch.tensor([0, 1, 0, 1, 0, 1]),
            "return": torch.randn(6),
        }
        total, details = loss(preds, targets)
        assert total.dim() == 0  # scalar
        assert "direction" in details and "return" in details and "total" in details
        total.backward()
        assert preds["direction"].grad is not None
        assert preds["return"].grad is not None

    def test_ignore_index_skips_deadzone_tokens(self) -> None:
        """Rows with ``cls_target == -1`` contribute no CE loss."""
        cfg = _cfg(**{
            "head.targets": ["direction"],
            "head.architecture": "single_task",
        })
        loss = MultiTaskLoss(cfg.head)
        # Valid sample + a deadzone sample; ensure the CE loss only sees
        # the valid one.
        preds = {"direction": torch.zeros(2, 2)}  # equal logits
        targets_valid = {"direction": torch.tensor([0, 1])}
        targets_half = {"direction": torch.tensor([0, -1])}

        loss_valid, _ = loss(preds, targets_valid)
        loss_half, _ = loss(preds, targets_half)
        # With uniform logits, CE loss = log(2) per valid sample.
        # Both should equal log(2) because CE is a per-sample mean over
        # valid samples.
        assert pytest.approx(loss_valid.item(), rel=1e-6) == np.log(2)
        assert pytest.approx(loss_half.item(), rel=1e-6) == np.log(2)

    def test_class_weights_buffer(self) -> None:
        cfg = _cfg(**{
            "head.targets": ["direction"],
            "head.architecture": "single_task",
        })
        cw = torch.tensor([1.0, 2.0])
        loss = MultiTaskLoss(cfg.head, class_weights=cw)
        # Buffer should be on the same dtype and shape.
        assert loss.class_weights.shape == (2,)
        assert loss.class_weights.dtype == torch.float32

    def test_class_weights_wrong_shape_raises(self) -> None:
        cfg = _cfg(**{
            "head.targets": ["direction"],
            "head.architecture": "single_task",
        })
        with pytest.raises(ValueError, match="class_weights"):
            MultiTaskLoss(cfg.head, class_weights=torch.tensor([1.0, 2.0, 3.0]))

    def test_missing_pred_key_raises(self) -> None:
        cfg = _cfg(**{"head.targets": ["direction", "return"]})
        loss = MultiTaskLoss(cfg.head)
        preds = {"direction": torch.randn(4, 2)}  # missing 'return'
        targets = {"direction": torch.tensor([0, 1, 0, 1]), "return": torch.randn(4)}
        with pytest.raises(KeyError, match="missing from preds"):
            loss(preds, targets)

    def test_missing_target_key_raises(self) -> None:
        cfg = _cfg(**{"head.targets": ["direction", "return"]})
        loss = MultiTaskLoss(cfg.head)
        preds = {"direction": torch.randn(4, 2), "return": torch.randn(4, 1)}
        targets = {"direction": torch.tensor([0, 1, 0, 1])}
        with pytest.raises(KeyError, match="missing from targets"):
            loss(preds, targets)


# ---------------------------------------------------------------------------
# VolatilityLoss
# ---------------------------------------------------------------------------


class TestVolatilityLoss:
    def test_accepts_dict_inputs(self) -> None:
        loss = VolatilityLoss()
        preds = {"volatility": torch.randn(5, 1)}
        targets = {"volatility": torch.randn(5)}
        total, details = loss(preds, targets)
        assert "volatility" in details and "total" in details

    def test_accepts_raw_tensors(self) -> None:
        loss = VolatilityLoss()
        total, _ = loss(torch.randn(4, 1), torch.randn(4))
        assert total.dim() == 0

    def test_gradient_flow(self) -> None:
        loss = VolatilityLoss()
        preds = torch.randn(4, 1, requires_grad=True)
        total, _ = loss(preds, torch.randn(4))
        total.backward()
        assert preds.grad is not None


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


class TestComputeClassWeights:
    def test_inverse_frequency_balanced(self) -> None:
        labels = np.array([0, 1])  # balanced
        cw = compute_class_weights(labels, scheme="inverse_frequency")
        assert torch.allclose(cw, torch.tensor([1.0, 1.0]))

    def test_inverse_frequency_imbalanced(self) -> None:
        labels = np.array([0, 0, 0, 1])  # 3:1
        cw = compute_class_weights(labels, scheme="inverse_frequency")
        # counts = [3, 1]. Raw weights = [1/3, 1]. Rescaled by min (=1/3).
        expected = torch.tensor([1.0, 3.0])
        assert torch.allclose(cw, expected)

    def test_balanced_sklearn_formula(self) -> None:
        labels = np.array([0, 0, 0, 1])  # n_samples=4, n_classes=2
        cw = compute_class_weights(labels, scheme="balanced")
        # weight[c] = n_samples / (n_classes * count_c)
        # = 4 / (2 * [3, 1]) = [0.667, 2.0]
        assert torch.allclose(cw, torch.tensor([4 / 6, 4 / 2]), atol=1e-5)

    def test_deadzone_ignored(self) -> None:
        labels = np.array([-1, 0, 1, 0])
        cw = compute_class_weights(labels, scheme="inverse_frequency")
        # Valid = [0, 1, 0] so counts = [2, 1], weights = [1/2, 1] -> rescaled [1, 2].
        assert torch.allclose(cw, torch.tensor([1.0, 2.0]))

    def test_unknown_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            compute_class_weights(np.array([0, 1]), scheme="custom")

    def test_empty_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="no non-deadzone"):
            compute_class_weights(np.array([-1, -1, -1]))

    def test_returns_float32_tensor(self) -> None:
        cw = compute_class_weights(np.array([0, 1]))
        assert cw.dtype == torch.float32


# ---------------------------------------------------------------------------
# build_loss factory
# ---------------------------------------------------------------------------


class TestBuildLoss:
    def test_volatility_only_gives_volatility_loss(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "single_task",
            "head.targets": ["volatility"],
        })
        assert isinstance(build_loss(cfg), VolatilityLoss)

    def test_direction_only_gives_multitask(self) -> None:
        cfg = _cfg(**{
            "head.architecture": "single_task",
            "head.targets": ["direction"],
        })
        # Single-target cases still go through MultiTaskLoss (one-weight).
        assert isinstance(build_loss(cfg), MultiTaskLoss)

    def test_direction_plus_return_multitask(self) -> None:
        cfg = _cfg(**{"head.targets": ["direction", "return"]})
        assert isinstance(build_loss(cfg), MultiTaskLoss)

    def test_class_weights_passed_through(self) -> None:
        cfg = _cfg(**{"head.targets": ["direction", "return"]})
        cw = torch.tensor([1.5, 0.5])
        loss = build_loss(cfg, class_weights=cw)
        assert isinstance(loss, MultiTaskLoss)
        assert torch.allclose(loss.class_weights, cw)
