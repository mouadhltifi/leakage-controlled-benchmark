"""Unit tests for :class:`forecast.models.tft.vsn.VariableSelectionNetwork`.

Covers shape contract, softmax weight invariants (sum to 1), with/
without context, variable-count edge cases (1, 2, 5), gradient flow,
convex-combination property, and reproducibility.
"""

from __future__ import annotations

import pytest
import torch

from forecast.models.tft.vsn import VariableSelectionNetwork
from mmfp.utils.seeding import set_all_seeds


# ---------------------------------------------------------------------------
# Shape contract.
# ---------------------------------------------------------------------------


def test_vsn_shape_contract_two_variables() -> None:
    vsn = VariableSelectionNetwork(
        n_variables=2, input_dim=16, hidden_dim=16, context_dim=None, dropout=0.0
    )
    x1 = torch.randn(4, 10, 16)
    x2 = torch.randn(4, 10, 16)
    selected, weights = vsn([x1, x2])
    assert selected.shape == (4, 10, 16)
    assert weights.shape == (4, 10, 2)


def test_vsn_shape_contract_five_variables() -> None:
    vsn = VariableSelectionNetwork(
        n_variables=5, input_dim=32, hidden_dim=32, context_dim=8, dropout=0.0
    )
    xs = [torch.randn(3, 20, 32) for _ in range(5)]
    c = torch.randn(3, 1, 8)
    selected, weights = vsn(xs, c)
    assert selected.shape == (3, 20, 32)
    assert weights.shape == (3, 20, 5)


def test_vsn_single_variable_returns_placeholder_weight() -> None:
    """n=1 has nothing to select; weights should be identically 1."""
    vsn = VariableSelectionNetwork(
        n_variables=1, input_dim=8, hidden_dim=8, dropout=0.0
    )
    x = torch.randn(2, 5, 8)
    selected, weights = vsn([x])
    assert selected.shape == (2, 5, 8)
    assert weights.shape == (2, 5, 1)
    assert torch.allclose(weights, torch.ones_like(weights))


# ---------------------------------------------------------------------------
# Softmax invariants (the "must sum to 1" contract in the brief).
# ---------------------------------------------------------------------------


def test_vsn_weights_sum_to_one() -> None:
    """Softmax over variables must sum to 1 along the last axis (~1e-5)."""
    vsn = VariableSelectionNetwork(
        n_variables=3, input_dim=16, hidden_dim=16, dropout=0.0
    )
    xs = [torch.randn(4, 10, 16) for _ in range(3)]
    _, weights = vsn(xs)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_vsn_weights_are_nonnegative() -> None:
    """Softmax output is in (0, 1); each entry strictly nonnegative."""
    vsn = VariableSelectionNetwork(
        n_variables=4, input_dim=8, hidden_dim=8, dropout=0.0
    )
    xs = [torch.randn(2, 6, 8) for _ in range(4)]
    _, weights = vsn(xs)
    assert (weights >= 0).all()
    assert (weights <= 1).all()


# ---------------------------------------------------------------------------
# Context path.
# ---------------------------------------------------------------------------


def test_vsn_with_context_affects_weights() -> None:
    """Different contexts yield different weights under the same input."""
    set_all_seeds(0)
    vsn = VariableSelectionNetwork(
        n_variables=3, input_dim=16, hidden_dim=16, context_dim=8, dropout=0.0
    )
    vsn.train(False)
    xs = [torch.randn(4, 10, 16) for _ in range(3)]
    c1 = torch.randn(4, 1, 8)
    c2 = torch.randn(4, 1, 8)
    _, w1 = vsn(xs, c1)
    _, w2 = vsn(xs, c2)
    assert not torch.allclose(w1, w2)


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------


def test_vsn_wrong_number_of_inputs_raises() -> None:
    vsn = VariableSelectionNetwork(n_variables=3, input_dim=8, hidden_dim=8)
    xs = [torch.randn(2, 4, 8), torch.randn(2, 4, 8)]  # only 2 given
    with pytest.raises(ValueError, match="expected 3"):
        vsn(xs)


def test_vsn_shape_mismatch_among_inputs_raises() -> None:
    vsn = VariableSelectionNetwork(n_variables=2, input_dim=8, hidden_dim=8)
    xs = [torch.randn(2, 4, 8), torch.randn(2, 5, 8)]
    with pytest.raises(ValueError, match="shape"):
        vsn(xs)


# ---------------------------------------------------------------------------
# Dict input order determinism.
# ---------------------------------------------------------------------------


def test_vsn_dict_input_order_stable() -> None:
    """Dict inputs are processed in sorted-key order; should be deterministic."""
    set_all_seeds(0)
    vsn = VariableSelectionNetwork(
        n_variables=3, input_dim=8, hidden_dim=8, dropout=0.0
    )
    vsn.train(False)
    a = torch.randn(2, 4, 8)
    b = torch.randn(2, 4, 8)
    c = torch.randn(2, 4, 8)

    s1, w1 = vsn({"a": a, "b": b, "c": c})
    s2, w2 = vsn({"c": c, "a": a, "b": b})  # insertion order shuffled

    assert torch.allclose(s1, s2)
    assert torch.allclose(w1, w2)


# ---------------------------------------------------------------------------
# Gradient flow.
# ---------------------------------------------------------------------------


def test_vsn_gradient_flow_to_all_parameters() -> None:
    vsn = VariableSelectionNetwork(
        n_variables=3, input_dim=16, hidden_dim=16, context_dim=4, dropout=0.0
    )
    xs = [torch.randn(2, 5, 16, requires_grad=True) for _ in range(3)]
    c = torch.randn(2, 1, 4, requires_grad=True)
    selected, _ = vsn(xs, c)
    loss = selected.pow(2).mean()
    loss.backward()
    for name, p in vsn.named_parameters():
        assert p.grad is not None, f"{name} missing gradient"
        assert p.grad.abs().sum().item() > 0, f"{name} zero gradient"


# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------


def test_vsn_reproducible_under_fixed_seed() -> None:
    set_all_seeds(42)
    vsn1 = VariableSelectionNetwork(
        n_variables=2, input_dim=8, hidden_dim=8, dropout=0.0
    )
    vsn1.train(False)
    xs = [torch.randn(2, 4, 8), torch.randn(2, 4, 8)]
    s1, w1 = vsn1(xs)

    set_all_seeds(42)
    vsn2 = VariableSelectionNetwork(
        n_variables=2, input_dim=8, hidden_dim=8, dropout=0.0
    )
    vsn2.train(False)
    _ = torch.randn(2, 4, 8)
    _ = torch.randn(2, 4, 8)
    s2, w2 = vsn2(xs)

    assert torch.allclose(s1, s2)
    assert torch.allclose(w1, w2)
