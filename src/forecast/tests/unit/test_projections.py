"""Unit tests for :class:`forecast.models.projections.ModalityProjections`.

Covers:

* Shape contract per modality.
* Absent modality in forward() batch raises ``KeyError``.
* Subset of modalities can be instantiated independently.
* Gradient flow.
* Reproducibility.
"""

from __future__ import annotations

import pytest
import torch

from forecast.models.projections import ModalityProjections
from mmfp.utils.seeding import set_all_seeds


def test_projections_shape_contract_per_modality() -> None:
    dims = {"price": 13, "news": 129, "macro": 9, "social": 7, "graph": 64}
    proj = ModalityProjections(feature_dims=dims, hidden_dim=64)
    batch = {name: torch.randn(4, 20, d) for name, d in dims.items()}
    out = proj(batch)
    assert set(out.keys()) == set(dims.keys())
    for name in dims:
        assert out[name].shape == (4, 20, 64)


def test_projections_absent_modality_raises_keyerror() -> None:
    """If we pass a batch missing a configured modality, raise."""
    dims = {"price": 13, "news": 129}
    proj = ModalityProjections(feature_dims=dims, hidden_dim=32)
    batch = {"price": torch.randn(2, 10, 13)}  # missing 'news'
    with pytest.raises(KeyError, match="news"):
        proj(batch)


def test_projections_subset_of_modalities() -> None:
    """Price-only config works as expected."""
    proj = ModalityProjections(feature_dims={"price": 13}, hidden_dim=64)
    batch = {"price": torch.randn(3, 60, 13)}
    out = proj(batch)
    assert list(out.keys()) == ["price"]
    assert out["price"].shape == (3, 60, 64)


def test_projections_layer_norm_applied() -> None:
    """LayerNorm should centre + scale the output; raw Linear alone won't."""
    proj = ModalityProjections(feature_dims={"x": 32}, hidden_dim=16)
    proj.train(False)
    batch = {"x": torch.randn(8, 50, 32) * 10.0}  # large variance
    out = proj(batch)["x"]
    # After LayerNorm, per-timestep mean ~ 0 and std ~ 1. Torch's
    # ``Tensor.std(unbiased=True)`` uses (n-1), while LayerNorm's
    # normalisation uses biased variance (n); at hidden_dim=16 this
    # introduces a ~3% gap. Use a loose tolerance that still catches
    # gross bugs like "no LayerNorm at all" where std would be ~10.
    mean = out.mean(dim=-1)
    std_biased = out.std(dim=-1, unbiased=False)
    assert mean.abs().max().item() < 1e-5
    assert (std_biased - 1.0).abs().max().item() < 1e-3


def test_projections_gradient_flow() -> None:
    dims = {"a": 8, "b": 16}
    proj = ModalityProjections(feature_dims=dims, hidden_dim=16)
    batch = {name: torch.randn(2, 4, d, requires_grad=True) for name, d in dims.items()}
    out = proj(batch)
    loss = sum(v.pow(2).mean() for v in out.values())
    loss.backward()
    for name, p in proj.named_parameters():
        assert p.grad is not None, f"{name} missing grad"
        assert p.grad.abs().sum().item() > 0, f"{name} zero grad"


def test_projections_reproducible_under_fixed_seed() -> None:
    dims = {"x": 10, "y": 20}

    set_all_seeds(123)
    p1 = ModalityProjections(feature_dims=dims, hidden_dim=32)
    p1.train(False)
    batch = {
        "x": torch.randn(2, 5, 10),
        "y": torch.randn(2, 5, 20),
    }
    out1 = p1(batch)

    set_all_seeds(123)
    p2 = ModalityProjections(feature_dims=dims, hidden_dim=32)
    p2.train(False)
    _ = torch.randn(2, 5, 10)
    _ = torch.randn(2, 5, 20)
    out2 = p2(batch)

    for k in out1:
        assert torch.allclose(out1[k], out2[k])
