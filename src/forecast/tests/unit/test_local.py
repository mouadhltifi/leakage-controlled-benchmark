"""Unit tests for :class:`forecast.models.tft.local.LocalProcessor`.

Covers:

* Output shape ``(B, L, H)``.
* Causal property: the output at timestep ``t`` depends only on inputs
  at timesteps ``<= t`` — verified by zeroing future timesteps and
  checking the output at ``t`` is unchanged.
* Static-context initialisation actually affects the output.
* Gradient flow to every parameter.
* Reproducibility under ``set_all_seeds``.
"""

from __future__ import annotations

import torch

from forecast.models.tft.local import LocalProcessor
from mmfp.utils.seeding import set_all_seeds


def test_local_processor_output_shape() -> None:
    lp = LocalProcessor(hidden_dim=16, n_lstm_layers=1, dropout=0.0)
    x = torch.randn(4, 20, 16)
    c_h = torch.randn(4, 16)
    c_c = torch.randn(4, 16)
    y = lp(x, c_h, c_c)
    assert y.shape == (4, 20, 16)


def test_local_processor_causal_property() -> None:
    """LSTM should not leak future information: output at time ``t``
    should be invariant to changes in inputs at times ``> t``.
    """
    lp = LocalProcessor(hidden_dim=16, n_lstm_layers=1, dropout=0.0)
    lp.train(False)
    B, L, H = 2, 8, 16
    x = torch.randn(B, L, H)
    c_h = torch.randn(B, H)
    c_c = torch.randn(B, H)

    y_full = lp(x, c_h, c_c)

    # Modify only timesteps > t and re-run.
    t = 3
    x_modified = x.clone()
    x_modified[:, t + 1 :, :] = torch.randn(B, L - t - 1, H)
    y_modified = lp(x_modified, c_h, c_c)

    # Prefix up to and including t must match bit-exactly.
    assert torch.allclose(y_full[:, : t + 1, :], y_modified[:, : t + 1, :])


def test_local_processor_static_init_used() -> None:
    """Different static init contexts should produce different outputs."""
    set_all_seeds(0)
    lp = LocalProcessor(hidden_dim=16, n_lstm_layers=1, dropout=0.0)
    lp.train(False)
    x = torch.randn(3, 10, 16)
    c_h_a = torch.randn(3, 16)
    c_c_a = torch.randn(3, 16)
    c_h_b = torch.randn(3, 16)
    c_c_b = torch.randn(3, 16)

    y_a = lp(x, c_h_a, c_c_a)
    y_b = lp(x, c_h_b, c_c_b)
    assert not torch.allclose(y_a, y_b), (
        "Different static inits should change LSTM output"
    )


def test_local_processor_gradient_flow() -> None:
    lp = LocalProcessor(hidden_dim=8, n_lstm_layers=1, dropout=0.0)
    x = torch.randn(2, 6, 8, requires_grad=True)
    c_h = torch.randn(2, 8, requires_grad=True)
    c_c = torch.randn(2, 8, requires_grad=True)
    y = lp(x, c_h, c_c)
    loss = y.pow(2).mean()
    loss.backward()
    for name, p in lp.named_parameters():
        assert p.grad is not None, f"{name} no grad"
        assert p.grad.abs().sum().item() > 0, f"{name} zero grad"


def test_local_processor_reproducible() -> None:
    set_all_seeds(9)
    lp1 = LocalProcessor(hidden_dim=8, n_lstm_layers=1, dropout=0.0)
    lp1.train(False)
    x = torch.randn(2, 5, 8)
    c_h = torch.randn(2, 8)
    c_c = torch.randn(2, 8)
    y1 = lp1(x, c_h, c_c)

    set_all_seeds(9)
    lp2 = LocalProcessor(hidden_dim=8, n_lstm_layers=1, dropout=0.0)
    lp2.train(False)
    _ = torch.randn(2, 5, 8)
    _ = torch.randn(2, 8)
    _ = torch.randn(2, 8)
    y2 = lp2(x, c_h, c_c)

    assert torch.allclose(y1, y2)
