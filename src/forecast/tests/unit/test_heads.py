"""Unit tests for :mod:`forecast.models.heads`.

Covers :class:`QuantileHead`, :func:`derive_direction`, and
:func:`derive_volatility`.
"""

from __future__ import annotations

import math

import pytest
import torch

from forecast.models.heads import (
    QuantileHead,
    derive_direction,
    derive_volatility,
)
from mmfp.utils.seeding import set_all_seeds


# ---------------------------------------------------------------------------
# QuantileHead.
# ---------------------------------------------------------------------------


def test_quantile_head_shape_contract() -> None:
    head = QuantileHead(hidden_dim=64, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9))
    h = torch.randn(4, 64)
    q = head(h)
    assert q.shape == (4, 5)


def test_quantile_head_wrong_input_shape_raises() -> None:
    head = QuantileHead(hidden_dim=16, quantiles=(0.1, 0.5, 0.9))
    with pytest.raises(ValueError, match=r"\(B, H=16\)"):
        head(torch.randn(4, 32))


# ---------------------------------------------------------------------------
# derive_direction.
# ---------------------------------------------------------------------------


def test_derive_direction_monotone_in_median() -> None:
    """Larger q_median should give a strictly larger up-class logit."""
    q_low = torch.tensor([[0.0, 0.0, -0.5, 0.0, 0.0]])
    q_high = torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0]])
    logits_low = derive_direction(q_low, median_idx=2)
    logits_high = derive_direction(q_high, median_idx=2)
    # Up-class logit == q_median; so up-logit(high) > up-logit(low).
    assert logits_high[0, 1].item() > logits_low[0, 1].item()
    # Down-class logit == -q_median, so down-logit(high) < down-logit(low).
    assert logits_high[0, 0].item() < logits_low[0, 0].item()


def test_derive_direction_gradient_flows_to_head() -> None:
    """The gradient of direction loss should flow back to QuantileHead weights."""
    head = QuantileHead(hidden_dim=16, quantiles=(0.1, 0.5, 0.9))
    h = torch.randn(4, 16)
    q = head(h)
    logits = derive_direction(q, median_idx=1)
    target = torch.tensor([0, 1, 1, 0])
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"{name} no grad"
        assert p.grad.abs().sum().item() > 0, f"{name} zero grad"


def test_derive_direction_invalid_median_idx_raises() -> None:
    q = torch.randn(2, 5)
    with pytest.raises(ValueError, match="median_idx"):
        derive_direction(q, median_idx=10)


# ---------------------------------------------------------------------------
# derive_volatility.
# ---------------------------------------------------------------------------


def test_derive_volatility_positive_when_sorted() -> None:
    q = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
    sigma = derive_volatility(q, lo_idx=0, hi_idx=4, coverage=0.8)
    assert sigma.shape == (1,)
    assert (sigma > 0).all()
    # (1.0 - (-1.0)) / (2 * 1.2816) ≈ 0.78
    expected = 2.0 / (2.0 * 1.2815515655446004)
    assert abs(sigma.item() - expected) < 1e-3


def test_derive_volatility_clamps_unsorted_quantiles() -> None:
    """An untrained net may produce q_hi < q_lo; must not crash."""
    q = torch.tensor([[0.5, 0.0, -0.5]])  # reversed
    sigma = derive_volatility(q, lo_idx=0, hi_idx=2, coverage=0.8)
    # The clamp floors the spread at 1e-6, so sigma is a tiny positive.
    assert sigma.shape == (1,)
    assert sigma.item() > 0
    assert sigma.item() < 1e-5


def test_derive_volatility_uses_erfinv_for_coverage() -> None:
    """Divisor at coverage=0.8 must match the Gaussian 80% band divisor."""
    # z_{0.9} = inv_cdf(0.9) ≈ 1.2816; divisor = 2 * 1.2816 ≈ 2.5631.
    q = torch.tensor([[0.0, 2.5631]])
    sigma = derive_volatility(q, lo_idx=0, hi_idx=1, coverage=0.8)
    # (2.5631 - 0) / 2.5631 = 1.0
    assert abs(sigma.item() - 1.0) < 1e-3


def test_derive_volatility_bad_coverage_raises() -> None:
    q = torch.randn(2, 3)
    with pytest.raises(ValueError, match="coverage"):
        derive_volatility(q, lo_idx=0, hi_idx=2, coverage=0.0)


# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------


def test_quantile_head_reproducible() -> None:
    set_all_seeds(5)
    h1 = QuantileHead(hidden_dim=16, quantiles=(0.1, 0.5, 0.9))
    h1.train(False)
    x = torch.randn(3, 16)
    q1 = h1(x)

    set_all_seeds(5)
    h2 = QuantileHead(hidden_dim=16, quantiles=(0.1, 0.5, 0.9))
    h2.train(False)
    _ = torch.randn(3, 16)
    q2 = h2(x)

    assert torch.allclose(q1, q2)
