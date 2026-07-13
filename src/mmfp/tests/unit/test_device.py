"""Tests for :func:`mmfp.utils.device.resolve_device`.

We avoid mocking the torch backend probes and instead verify invariants
that hold regardless of which backend is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mmfp.utils.device import resolve_device


@dataclass
class _StubCfg:
    """Minimal stand-in for ``ExperimentConfig`` exposing only ``device``."""

    device: str


def test_resolve_device_returns_torch_device_none_cfg() -> None:
    dev = resolve_device(None)
    assert isinstance(dev, torch.device)
    assert dev.type in {"mps", "cuda", "cpu"}


def test_resolve_device_auto_returns_torch_device() -> None:
    dev = resolve_device(_StubCfg(device="auto"))
    assert isinstance(dev, torch.device)
    assert dev.type in {"mps", "cuda", "cpu"}


def test_resolve_device_explicit_cpu_always_works() -> None:
    dev = resolve_device(_StubCfg(device="cpu"))
    assert dev == torch.device("cpu")


def test_resolve_device_invalid_backend_falls_back_to_cpu() -> None:
    # Request MPS if available; otherwise we still expect CPU.
    if torch.backends.mps.is_available():
        dev = resolve_device(_StubCfg(device="mps"))
        assert dev == torch.device("mps")
    else:
        dev = resolve_device(_StubCfg(device="mps"))
        assert dev == torch.device("cpu")


def test_resolve_device_with_experiment_config(minimal_cfg) -> None:
    """End-to-end: a fully-built ExperimentConfig resolves cleanly."""
    dev = resolve_device(minimal_cfg)
    assert isinstance(dev, torch.device)
    assert dev.type in {"mps", "cuda", "cpu"}
