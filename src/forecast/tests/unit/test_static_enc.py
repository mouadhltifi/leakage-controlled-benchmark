"""Unit tests for :class:`forecast.models.tft.static_enc.StaticCovariateEncoders`.

Covers:

* Four distinct context tensors are returned with the correct shapes.
* Different ticker/sector IDs produce distinct contexts.
* Gradient flow through embeddings + GRNs.
* Input-validation error paths.
* Reproducibility under ``set_all_seeds``.
"""

from __future__ import annotations

import pytest
import torch

from forecast.models.tft.static_enc import (
    StaticCovariateEncoders,
    _embed_dim_for_cardinality,
)
from mmfp.utils.seeding import set_all_seeds


def test_embed_dim_heuristic_matches_spec_defaults() -> None:
    assert _embed_dim_for_cardinality(55) == 16
    assert _embed_dim_for_cardinality(11) == 8


def test_static_enc_returns_four_contexts_with_correct_shape() -> None:
    """All four contexts have shape (B, H)."""
    enc = StaticCovariateEncoders(
        static_cardinalities=[55, 11], hidden_dim=64, dropout=0.0
    )
    enc.train(False)
    static_cat = torch.zeros((4, 2), dtype=torch.long)
    out = enc(static_cat)
    assert set(out.keys()) == {"c_s", "c_c", "c_h", "c_e"}
    for key, tensor in out.items():
        assert tensor.shape == (4, 64), f"{key}: {tuple(tensor.shape)}"


def test_static_enc_four_contexts_are_distinct_projections() -> None:
    """Each of the four GRNs has independent weights, so contexts differ
    for the same input."""
    set_all_seeds(0)
    enc = StaticCovariateEncoders(
        static_cardinalities=[55, 11], hidden_dim=64, dropout=0.0
    )
    enc.train(False)
    static_cat = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    out = enc(static_cat)
    keys = list(out.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not torch.allclose(out[keys[i]], out[keys[j]]), (
                f"{keys[i]} and {keys[j]} produced identical tensors"
            )


def test_static_enc_different_inputs_yield_different_contexts() -> None:
    """Different ticker/sector IDs should push each context apart."""
    set_all_seeds(0)
    enc = StaticCovariateEncoders(
        static_cardinalities=[55, 11], hidden_dim=32, dropout=0.0
    )
    enc.train(False)
    a = torch.tensor([[0, 0]], dtype=torch.long)
    b = torch.tensor([[10, 5]], dtype=torch.long)
    out_a = enc(a)
    out_b = enc(b)
    for key in out_a:
        assert not torch.allclose(out_a[key], out_b[key]), (
            f"{key} did not change between distinct static inputs"
        )


def test_static_enc_gradient_flow_to_embeddings_and_grns() -> None:
    enc = StaticCovariateEncoders(
        static_cardinalities=[55, 11], hidden_dim=32, dropout=0.0
    )
    static_cat = torch.tensor(
        [[0, 0], [1, 2], [2, 3]], dtype=torch.long
    )
    out = enc(static_cat)
    loss = sum(v.pow(2).mean() for v in out.values())
    loss.backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"{name} missing grad"
        assert p.grad.abs().sum().item() > 0, f"{name} zero grad"


def test_static_enc_wrong_column_count_raises() -> None:
    enc = StaticCovariateEncoders(
        static_cardinalities=[55, 11], hidden_dim=32
    )
    with pytest.raises(ValueError, match="expected 2"):
        enc(torch.zeros((4, 3), dtype=torch.long))


def test_static_enc_reproducible_under_fixed_seed() -> None:
    set_all_seeds(7)
    enc1 = StaticCovariateEncoders(
        static_cardinalities=[55, 11], hidden_dim=16, dropout=0.0
    )
    enc1.train(False)
    static_cat = torch.tensor([[3, 4], [5, 6]], dtype=torch.long)
    out1 = enc1(static_cat)

    set_all_seeds(7)
    enc2 = StaticCovariateEncoders(
        static_cardinalities=[55, 11], hidden_dim=16, dropout=0.0
    )
    enc2.train(False)
    out2 = enc2(static_cat)

    for key in out1:
        assert torch.allclose(out1[key], out2[key])
