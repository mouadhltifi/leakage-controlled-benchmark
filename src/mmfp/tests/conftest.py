"""Shared pytest fixtures for mmfp tests.

Kept small and dependency-free. The ``minimal_cfg_dict`` fixture is the
canonical starting point for every config test.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure mmfp is importable when pytest is launched from ``work/``.
_WORK_DIR = Path(__file__).resolve().parents[2]
if str(_WORK_DIR) not in sys.path:
    sys.path.insert(0, str(_WORK_DIR))

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.config.schema import ExperimentConfig


@pytest.fixture
def minimal_cfg_dict() -> dict[str, Any]:
    """Return a deep-copied canonical defaults dict.

    Tests can freely mutate this without affecting other tests.
    """
    return copy.deepcopy(DEFAULT_CONFIG)


@pytest.fixture
def minimal_cfg(minimal_cfg_dict: dict[str, Any]) -> ExperimentConfig:
    """Return a validated :class:`ExperimentConfig` from the canonical defaults."""
    return ExperimentConfig.model_validate(minimal_cfg_dict)


@pytest.fixture
def defaults_toml_path() -> Path:
    """Filesystem path to the canonical defaults.toml."""
    return _WORK_DIR / "mmfp" / "configs" / "defaults.toml"
