"""Unit tests for :class:`forecast.models.losses.QuantileLoss` and
:class:`forecast.models.losses.ForecastLoss`.

Coverage:

* Hand-computed pinball against a fixed 3-sample toy fixture.
* Pinball is always non-negative.
* Pinball minimum at ``q_k == y`` for all k.
* ``ForecastLoss`` with all-zero auxiliary weights equals pure pinball.
* ``ForecastLoss`` with ``direction_aux_weight > 0`` adds a CE
  component reported in the per-component dict.
* CE with ``-1`` (deadzone) labels is ignored.
* Reproducibility of the component dict.
"""

from __future__ import annotations

import pytest
import torch

from forecast.config.defaults import DEFAULT_CONFIG
from forecast.config.schema import ForecastConfig, V3ExperimentConfig
from forecast.models.losses import ForecastLoss, QuantileLoss
from mmfp.utils.seeding import set_all_seeds


# ---------------------------------------------------------------------------
# Pinball hand-computation.
# ---------------------------------------------------------------------------


def test_pinball_hand_computed_fixture() -> None:
    """Verify pinball against a deliberately simple fixture.

    quantiles = (0.5,); y = [0]; q_pred = [[2.0]]
    err = y - q = -2.0
    alpha*err = 0.5 * -2 = -1.0
    (alpha-1)*err = -0.5 * -2 = 1.0
    max = 1.0 -> mean over (1, 1) = 1.0.
    """
    loss = QuantileLoss(quantiles=(0.5,))
    q = torch.tensor([[2.0]])
    y = torch.tensor([0.0])
    assert abs(loss(q, y).item() - 1.0) < 1e-6


def test_pinball_hand_computed_multi_quantile() -> None:
    """3 quantiles, 2 samples. Compute each cell by hand, average.

    quantiles = (0.1, 0.5, 0.9)
    y = [1.0, -1.0]
    q = [[0.0, 1.0, 2.0],
         [0.0, 0.0, 0.0]]

    Sample 0 (y=1, row=[0, 1, 2]):
      err = 1 - q = [1, 0, -1]
      alpha = [0.1, 0.5, 0.9]
      alpha*err      = [0.1, 0, -0.9]
      (alpha-1)*err  = [-0.9, 0, 0.1]
      max            = [0.1, 0, 0.1]
    Sample 1 (y=-1, row=[0, 0, 0]):
      err = -1 - q = [-1, -1, -1]
      alpha*err      = [-0.1, -0.5, -0.9]
      (alpha-1)*err  = [0.9, 0.5, 0.1]
      max            = [0.9, 0.5, 0.1]
    Mean over 6 cells = (0.1 + 0 + 0.1 + 0.9 + 0.5 + 0.1) / 6
                      = 1.7 / 6 ≈ 0.28333...
    """
    loss = QuantileLoss(quantiles=(0.1, 0.5, 0.9))
    q = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    y = torch.tensor([1.0, -1.0])
    expected = 1.7 / 6.0
    assert abs(loss(q, y).item() - expected) < 1e-6


# ---------------------------------------------------------------------------
# Basic invariants.
# ---------------------------------------------------------------------------


def test_pinball_always_non_negative() -> None:
    """Pinball loss is >= 0 for any prediction/target combo."""
    loss = QuantileLoss(quantiles=(0.1, 0.5, 0.9))
    for _ in range(20):
        q = torch.randn(8, 3) * 5.0
        y = torch.randn(8) * 5.0
        assert loss(q, y).item() >= 0.0


def test_pinball_zero_when_q_equals_y() -> None:
    """If every quantile equals y, the loss is zero."""
    loss = QuantileLoss(quantiles=(0.1, 0.5, 0.9))
    y = torch.tensor([0.3, -0.2, 1.5])
    # Broadcast y into a (3, 3) quantile prediction.
    q = y.unsqueeze(-1).expand(-1, 3).contiguous()
    assert loss(q, y).item() < 1e-7


def test_pinball_shape_mismatch_raises() -> None:
    loss = QuantileLoss(quantiles=(0.1, 0.5, 0.9))
    with pytest.raises(ValueError, match="quantiles"):
        loss(torch.randn(4, 5), torch.randn(4))  # 5 quantiles != 3


# ---------------------------------------------------------------------------
# ForecastLoss composite.
# ---------------------------------------------------------------------------


def _make_cfg(**forecast_overrides) -> ForecastConfig:
    base = dict(DEFAULT_CONFIG["forecast"])
    base.update(forecast_overrides)
    return ForecastConfig(**base)


def test_forecast_loss_zero_aux_equals_pure_pinball() -> None:
    """With both aux weights at 0, total == quantile loss alone."""
    cfg = _make_cfg(direction_aux_weight=0.0, volatility_aux_weight=0.0)
    composite = ForecastLoss(cfg)
    pure = QuantileLoss(cfg.quantiles)

    q = torch.randn(4, 5)
    y_return = torch.randn(4)
    total, components = composite(q, y_return)

    expected = pure(q, y_return).item()
    assert abs(total.item() - expected) < 1e-6
    assert components == {"quantile": pytest.approx(expected, rel=1e-5)}


def test_forecast_loss_with_direction_aux_adds_component() -> None:
    """Non-zero direction weight should surface a 'direction' component."""
    cfg = _make_cfg(direction_aux_weight=1.0)
    loss = ForecastLoss(cfg)
    q = torch.randn(4, 5, requires_grad=True)
    y_return = torch.randn(4)
    y_direction = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    total, components = loss(q, y_return, y_direction=y_direction)

    assert "quantile" in components
    assert "direction" in components
    assert components["direction"] >= 0.0
    # Gradient still flows.
    total.backward()
    assert q.grad is not None


def test_forecast_loss_missing_direction_target_raises() -> None:
    cfg = _make_cfg(direction_aux_weight=0.5)
    loss = ForecastLoss(cfg)
    with pytest.raises(ValueError, match="y_direction"):
        loss(torch.randn(2, 5), torch.randn(2))


def test_forecast_loss_direction_ignore_deadzone_labels() -> None:
    """CE on direction should ignore samples labelled ``-1`` (deadzone)."""
    cfg = _make_cfg(direction_aux_weight=1.0)
    loss = ForecastLoss(cfg)
    q = torch.randn(4, 5)
    y_return = torch.randn(4)
    # Two deadzone samples, two real.
    y_direction = torch.tensor([-1, 0, -1, 1], dtype=torch.long)
    total, components = loss(q, y_return, y_direction=y_direction)
    assert "direction" in components
    assert components["direction"] >= 0.0
    # Compare against CE on only the non-deadzone rows manually.
    from forecast.models.heads import derive_direction

    logits = derive_direction(q, median_idx=2)
    mask = y_direction != -1
    expected = torch.nn.functional.cross_entropy(
        logits[mask], y_direction[mask]
    ).item()
    assert abs(components["direction"] - expected) < 1e-5


def test_forecast_loss_with_volatility_aux_adds_component() -> None:
    cfg = _make_cfg(volatility_aux_weight=1.0)
    loss = ForecastLoss(cfg)
    q = torch.randn(4, 5, requires_grad=True)
    y_return = torch.randn(4)
    y_vol = torch.rand(4).abs()
    total, components = loss(q, y_return, y_volatility=y_vol)
    assert "volatility" in components
    assert components["volatility"] >= 0.0


def test_forecast_loss_reproducible() -> None:
    cfg = _make_cfg()
    loss = ForecastLoss(cfg)

    set_all_seeds(100)
    q1 = torch.randn(4, 5)
    y1 = torch.randn(4)
    total1, comps1 = loss(q1, y1)

    set_all_seeds(100)
    q2 = torch.randn(4, 5)
    y2 = torch.randn(4)
    total2, comps2 = loss(q2, y2)

    assert abs(total1.item() - total2.item()) < 1e-7
    assert comps1 == comps2
