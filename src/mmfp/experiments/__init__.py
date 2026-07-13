"""Experiment runner and sweep orchestration.

Milestone 7 deliverable. ``run_one_experiment(cfg) -> ResultRecord`` is
the atomic unit of work; ``run_sweep`` is a thin wrapper that iterates
config overrides and appends to a CSV with resume support.
"""

from mmfp.experiments.result_schema import (
    ResultRecord,
    append_result,
    config_hash,
    fingerprint_from_row,
    load_results,
    record_fingerprint,
)
from mmfp.experiments.runner import evaluate_on_test, run_one_experiment
from mmfp.experiments.sweep import apply_override, run_sweep

__all__ = [
    "ResultRecord",
    "append_result",
    "apply_override",
    "config_hash",
    "evaluate_on_test",
    "fingerprint_from_row",
    "load_results",
    "record_fingerprint",
    "run_one_experiment",
    "run_sweep",
]
