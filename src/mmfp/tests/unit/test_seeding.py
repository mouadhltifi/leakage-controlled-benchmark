"""Tests for :mod:`mmfp.utils.seeding`.

The core reproducibility contract is: ``set_all_seeds(s)`` followed by
the same torch/numpy calls must produce bit-identical results across
invocations on CPU.
"""

from __future__ import annotations

import numpy as np
import torch

from mmfp.utils.seeding import (
    dataloader_worker_init,
    make_dataloader_generator,
    make_worker_init_fn,
    set_all_seeds,
)


def test_set_all_seeds_torch_cpu_deterministic() -> None:
    """Two ``set_all_seeds(42)`` + ``torch.rand(3)`` calls agree bit-for-bit."""
    set_all_seeds(42)
    a = torch.rand(3)

    set_all_seeds(42)
    b = torch.rand(3)

    assert torch.equal(a, b), f"expected bit-equal; got {a} vs {b}"


def test_set_all_seeds_different_seeds_different_output() -> None:
    set_all_seeds(1)
    a = torch.rand(3)
    set_all_seeds(2)
    b = torch.rand(3)
    assert not torch.equal(a, b)


def test_set_all_seeds_numpy_deterministic() -> None:
    set_all_seeds(7)
    a = np.random.rand(4)
    set_all_seeds(7)
    b = np.random.rand(4)
    assert np.array_equal(a, b)


def test_set_all_seeds_python_random_deterministic() -> None:
    import random

    set_all_seeds(100)
    a = [random.random() for _ in range(5)]
    set_all_seeds(100)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_make_dataloader_generator_is_seeded() -> None:
    g1 = make_dataloader_generator(42)
    g2 = make_dataloader_generator(42)

    # Draw the same shape tensor from each generator and compare.
    t1 = torch.empty(5).uniform_(0, 1, generator=g1)
    t2 = torch.empty(5).uniform_(0, 1, generator=g2)
    assert torch.equal(t1, t2)


def test_make_dataloader_generator_different_seeds_differ() -> None:
    g1 = make_dataloader_generator(1)
    g2 = make_dataloader_generator(2)
    t1 = torch.empty(5).uniform_(0, 1, generator=g1)
    t2 = torch.empty(5).uniform_(0, 1, generator=g2)
    assert not torch.equal(t1, t2)


def test_dataloader_worker_init_seeds_reproducibly() -> None:
    """Same (seed, worker_id) pair produces same numpy.random draws."""
    dataloader_worker_init(worker_id=0, seed=42)
    a = np.random.rand(3)

    dataloader_worker_init(worker_id=0, seed=42)
    b = np.random.rand(3)

    assert np.array_equal(a, b)


def test_dataloader_worker_init_distinct_workers_differ() -> None:
    dataloader_worker_init(worker_id=0, seed=42)
    a = np.random.rand(3)
    dataloader_worker_init(worker_id=1, seed=42)
    b = np.random.rand(3)
    assert not np.array_equal(a, b)


def test_make_worker_init_fn_binds_seed() -> None:
    """The helper returns a one-arg callable taking just worker_id."""
    fn = make_worker_init_fn(42)
    fn(0)
    a = np.random.rand(3)
    fn(0)
    b = np.random.rand(3)
    assert np.array_equal(a, b)


def test_pythonhashseed_set_by_set_all_seeds() -> None:
    import os

    set_all_seeds(12345)
    assert os.environ.get("PYTHONHASHSEED") == "12345"
