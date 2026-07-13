"""Unit tests for :class:`forecast.models.tft.grn.GRN`.

Covers shape contract, optional context path, dropout behaviour in
train vs inference mode, gradient flow, and reproducibility under
``set_all_seeds``.
"""

from __future__ import annotations

import pytest
import torch

from forecast.models.tft.grn import GRN, _GatedLinearUnit
from mmfp.utils.seeding import set_all_seeds


# ---------------------------------------------------------------------------
# Shape contract.
# ---------------------------------------------------------------------------


def test_grn_shape_contract_same_in_out() -> None:
    """Input dim == output dim should preserve the last axis."""
    grn = GRN(input_dim=32, hidden_dim=32)
    x = torch.randn(4, 10, 32)
    y = grn(x)
    assert y.shape == x.shape


def test_grn_shape_contract_different_output_dim() -> None:
    """A different output dim triggers a skip-projection and resizes."""
    grn = GRN(input_dim=24, hidden_dim=32, output_dim=16)
    x = torch.randn(4, 10, 24)
    y = grn(x)
    assert y.shape == (4, 10, 16)


def test_grn_context_path_shape() -> None:
    """Passing a context should not change the output shape."""
    grn = GRN(input_dim=16, hidden_dim=32, output_dim=16, context_dim=8)
    x = torch.randn(3, 5, 16)
    c = torch.randn(3, 1, 8)  # broadcast across the sequence axis
    y = grn(x, c)
    assert y.shape == (3, 5, 16)


# ---------------------------------------------------------------------------
# Context path validation.
# ---------------------------------------------------------------------------


def test_grn_passing_context_without_context_dim_raises() -> None:
    grn = GRN(input_dim=8, hidden_dim=8)
    with pytest.raises(ValueError, match="without context_dim"):
        grn(torch.randn(2, 8), torch.randn(2, 4))


def test_grn_missing_context_when_configured_raises() -> None:
    grn = GRN(input_dim=8, hidden_dim=8, context_dim=4)
    with pytest.raises(ValueError, match="with context_dim but no context"):
        grn(torch.randn(2, 8))


# ---------------------------------------------------------------------------
# GLU gating sanity.
# ---------------------------------------------------------------------------


def test_glu_output_has_correct_shape() -> None:
    """Gated linear unit preserves the hidden dim."""
    glu = _GatedLinearUnit(hidden_dim=16)
    x = torch.randn(4, 16)
    y = glu(x)
    assert y.shape == x.shape


def test_glu_with_zero_input_nonzero_output_possible() -> None:
    """With zeroed biases and zero input, the GLU is identically zero."""
    glu = _GatedLinearUnit(hidden_dim=8)
    torch.nn.init.zeros_(glu.fc.bias)
    x = torch.zeros(3, 8)
    y = glu(x)
    assert torch.allclose(y, torch.zeros_like(y))


# ---------------------------------------------------------------------------
# Dropout train vs inference.
# ---------------------------------------------------------------------------


def test_grn_dropout_active_in_train_mode_only() -> None:
    """Train mode should produce different outputs on two forward passes
    (due to dropout stochasticity); inference mode should be deterministic.
    """
    set_all_seeds(0)
    grn = GRN(input_dim=16, hidden_dim=32, output_dim=16, dropout=0.5)
    x = torch.randn(8, 16)

    grn.train()
    set_all_seeds(1)
    y1 = grn(x)
    set_all_seeds(2)
    y2 = grn(x)
    assert not torch.allclose(y1, y2)

    grn.eval()
    y3 = grn(x)
    y4 = grn(x)
    assert torch.allclose(y3, y4)


def test_grn_with_zero_dropout_matches_across_modes() -> None:
    """Zero-dropout GRN should behave identically in train and inference."""
    set_all_seeds(0)
    grn = GRN(input_dim=8, hidden_dim=16, dropout=0.0)
    x = torch.randn(2, 8)
    grn.train()
    y_train = grn(x)
    grn.eval()
    y_infer = grn(x)
    assert torch.allclose(y_train, y_infer)


# ---------------------------------------------------------------------------
# Gradient flow.
# ---------------------------------------------------------------------------


def test_grn_gradient_flow_to_all_parameters() -> None:
    """Every parameter should receive a non-zero gradient after backward."""
    grn = GRN(input_dim=16, hidden_dim=32, output_dim=16, context_dim=8, dropout=0.0)
    x = torch.randn(4, 16, requires_grad=True)
    c = torch.randn(4, 8, requires_grad=True)
    y = grn(x, c)
    loss = y.pow(2).mean()
    loss.backward()
    for name, p in grn.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert p.grad.abs().sum().item() > 0, (
            f"{name} gradient is identically zero"
        )


# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------


def test_grn_reproducible_under_fixed_seed() -> None:
    """Same seed + same construction order should give identical output."""
    set_all_seeds(123)
    grn1 = GRN(input_dim=16, hidden_dim=32, dropout=0.0)
    x = torch.randn(4, 16)
    grn1.eval()
    y1 = grn1(x)

    set_all_seeds(123)
    grn2 = GRN(input_dim=16, hidden_dim=32, dropout=0.0)
    # Consume the same RNG state so x is regenerated identically.
    _ = torch.randn(4, 16)
    grn2.eval()
    y2 = grn2(x)

    assert torch.allclose(y1, y2)
