"""Tests for :mod:`mmfp.experiments.sweep`.

These tests never call :func:`run_one_experiment` — that's covered by
the integration test. Here we exercise:

* ``apply_override`` dotted-key semantics
* ``parallelism > 1`` rejection
* Resume-skip logic driven by an already-written CSV
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from mmfp.config.defaults import DEFAULT_CONFIG
from mmfp.experiments.result_schema import ResultRecord, append_result
from mmfp.experiments.sweep import apply_override, run_sweep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_toml(tmp_path: Path) -> Path:
    """Write defaults.toml-equivalent to disk and return the path."""
    # Minimal valid TOML rendered from DEFAULT_CONFIG. We shortcut by
    # pointing at the real defaults.toml that ships with the package.
    from mmfp.config.load import dump_config
    from mmfp.config.schema import ExperimentConfig

    cfg = ExperimentConfig.model_validate(copy.deepcopy(DEFAULT_CONFIG))
    path = tmp_path / "base.toml"
    dump_config(cfg, path)
    return path


def _make_record(seed: int = 42, fold: int = 0, name: str = "sweep-test") -> ResultRecord:
    """Utility record with known identification tuple."""
    return ResultRecord(
        experiment_name=name,
        config_hash="0" * 64,
        seed=seed,
        fold_idx=fold,
        news_encoder="none",
        news_aggregation="none",
        fusion_strategy="concat",
        head_architecture="parallel_multitask",
        targets="direction,return",
        graph_source="none",
        lookback=1,
        price_encoder="feedforward",
        mcc=0.01, accuracy=0.51, f1=0.55,
        r2=-0.001, rmse=0.012, sharpe_ratio=0.5,
        vol_rmse=None, vol_r2=None,
        n_train=1000, n_val=200, n_test=300, n_params=12345,
        epochs_trained=42, best_val_metric=0.02, elapsed_seconds=12.5,
        platform_version="2.0.0-dev", git_sha="abcdef012345",
    )


# ---------------------------------------------------------------------------
# apply_override
# ---------------------------------------------------------------------------


class TestApplyOverride:
    def test_top_level_override(self) -> None:
        base = {"a": 1, "b": 2}
        out = apply_override(base, {"a": 99})
        assert out == {"a": 99, "b": 2}
        # Source dict untouched.
        assert base == {"a": 1, "b": 2}

    def test_dotted_path_creates_nested(self) -> None:
        base: dict[str, Any] = {"training": {"batch_size": 64}}
        out = apply_override(base, {"training.batch_size": 128})
        assert out == {"training": {"batch_size": 128}}

    def test_dotted_path_creates_missing_segment(self) -> None:
        base: dict[str, Any] = {}
        out = apply_override(base, {"a.b.c": 7})
        assert out == {"a": {"b": {"c": 7}}}

    def test_reject_descent_through_scalar(self) -> None:
        base: dict[str, Any] = {"a": 1}
        with pytest.raises(ValueError, match="cannot descend"):
            apply_override(base, {"a.b": 2})

    def test_empty_key_rejected(self) -> None:
        base: dict[str, Any] = {}
        with pytest.raises(ValueError, match="malformed"):
            apply_override(base, {"a..b": 2})

    def test_multiple_overrides_applied(self) -> None:
        base = {"training": {"batch_size": 64, "learning_rate": 1e-4}}
        out = apply_override(
            base,
            {
                "training.batch_size": 128,
                "training.learning_rate": 1e-3,
                "seed": 7,
            },
        )
        assert out["training"]["batch_size"] == 128
        assert out["training"]["learning_rate"] == 1e-3
        assert out["seed"] == 7


# ---------------------------------------------------------------------------
# run_sweep: parallelism validation
# ---------------------------------------------------------------------------


class TestSweepParallelismValidation:
    """Input validation on the ``parallelism`` / ``device_strategy`` arguments.

    Milestone 9 replaced the ``NotImplementedError`` guard with a real
    multiprocess implementation. The only remaining hard failures are
    semantically invalid inputs (``parallelism < 1``, unknown device).
    """

    def test_parallelism_greater_than_one_now_supported(
        self, tmp_path: Path, base_toml: Path,
    ) -> None:
        """``parallelism > 1`` no longer raises; empty overrides still returns fast.

        This test intentionally passes zero overrides so the spawn pool is
        never constructed — we only verify the call is accepted. The real
        bit-identical check lives in
        ``tests/reproducibility/test_parallelism_equivalence.py``.
        """
        df = run_sweep(
            base_cfg_path=base_toml,
            overrides=[],
            output_csv=tmp_path / "out.csv",
            parallelism=4,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_parallelism_zero_raises_value_error(
        self, tmp_path: Path, base_toml: Path,
    ) -> None:
        with pytest.raises(ValueError, match="parallelism"):
            run_sweep(
                base_cfg_path=base_toml,
                overrides=[{}],
                output_csv=tmp_path / "out.csv",
                parallelism=0,
            )

    def test_parallelism_negative_raises_value_error(
        self, tmp_path: Path, base_toml: Path,
    ) -> None:
        with pytest.raises(ValueError, match="parallelism"):
            run_sweep(
                base_cfg_path=base_toml,
                overrides=[{}],
                output_csv=tmp_path / "out.csv",
                parallelism=-1,
            )

    def test_unknown_device_strategy_raises(
        self, tmp_path: Path, base_toml: Path,
    ) -> None:
        with pytest.raises(ValueError, match="device_strategy"):
            run_sweep(
                base_cfg_path=base_toml,
                overrides=[{}],
                output_csv=tmp_path / "out.csv",
                device_strategy="tpu",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# run_sweep: resume-skip
# ---------------------------------------------------------------------------


class TestSweepResume:
    def test_resume_skips_existing_fingerprints(
        self, tmp_path: Path, base_toml: Path,
    ) -> None:
        """Pre-seed the output CSV with a fingerprint matching our override.

        After running the sweep we expect ``run_one_experiment`` was not
        invoked — the row was skipped. We patch the runner to raise so
        the test fails loudly if the skip logic misfires.
        """
        out_csv = tmp_path / "out.csv"

        # Build the config that the sweep will produce for override {}.
        from mmfp.config.schema import ExperimentConfig
        from mmfp.experiments.result_schema import config_hash

        cfg = ExperimentConfig.model_validate(copy.deepcopy(DEFAULT_CONFIG))
        pre_existing = ResultRecord(
            experiment_name=cfg.name,
            config_hash=config_hash(cfg),
            seed=cfg.seed,
            fold_idx=cfg.data.fold_idx,
            news_encoder="none",
            news_aggregation="none",
            fusion_strategy=cfg.fusion.strategy,
            head_architecture=cfg.head.architecture,
            targets=",".join(cfg.head.targets),
            graph_source="none",
            lookback=cfg.data.lookback,
            price_encoder=cfg.model.price_encoder,
            mcc=0.0, accuracy=0.0, f1=0.0,
            r2=0.0, rmse=0.0, sharpe_ratio=0.0,
            vol_rmse=None, vol_r2=None,
            n_train=1, n_val=1, n_test=1, n_params=1,
            epochs_trained=1, best_val_metric=0.0, elapsed_seconds=0.0,
            platform_version="2.0.0-dev", git_sha="prerun",
        )
        append_result(out_csv, pre_existing)

        # Patch run_one_experiment to fail if invoked.
        def _must_not_be_called(_cfg):  # pragma: no cover
            raise AssertionError(
                "run_one_experiment should not be called when the fingerprint "
                "already exists and resume=True."
            )

        with patch(
            "mmfp.experiments.sweep.run_one_experiment",
            side_effect=_must_not_be_called,
        ):
            df = run_sweep(
                base_cfg_path=base_toml,
                overrides=[{}],  # one override producing the same fingerprint
                output_csv=out_csv,
                resume=True,
                parallelism=1,
            )

        # CSV is unchanged: still exactly one row.
        assert len(df) == 1
        assert df.iloc[0]["git_sha"] == "prerun"

    def test_no_resume_ignores_existing_fingerprints(
        self, tmp_path: Path, base_toml: Path,
    ) -> None:
        """With ``resume=False`` a matching fingerprint does NOT skip the run."""
        out_csv = tmp_path / "out.csv"
        # Pre-seed with the same fingerprint.
        from mmfp.config.schema import ExperimentConfig
        from mmfp.experiments.result_schema import config_hash

        cfg = ExperimentConfig.model_validate(copy.deepcopy(DEFAULT_CONFIG))
        append_result(
            out_csv,
            _make_record(seed=cfg.seed).__class__(
                experiment_name=cfg.name,
                config_hash=config_hash(cfg),
                seed=cfg.seed,
                fold_idx=cfg.data.fold_idx,
                news_encoder="none",
                news_aggregation="none",
                fusion_strategy=cfg.fusion.strategy,
                head_architecture=cfg.head.architecture,
                targets=",".join(cfg.head.targets),
                graph_source="none",
                lookback=cfg.data.lookback,
                price_encoder=cfg.model.price_encoder,
                mcc=0.0, accuracy=0.0, f1=0.0,
                r2=0.0, rmse=0.0, sharpe_ratio=0.0,
                vol_rmse=None, vol_r2=None,
                n_train=1, n_val=1, n_test=1, n_params=1,
                epochs_trained=1, best_val_metric=0.0, elapsed_seconds=0.0,
                platform_version="2.0.0-dev", git_sha="prerun",
            ),
        )

        # Stub the runner to return a fresh record.
        stub_record = _make_record(seed=cfg.seed, fold=cfg.data.fold_idx)
        stub_record.experiment_name = cfg.name
        stub_record.config_hash = config_hash(cfg)
        stub_record.git_sha = "freshrun"

        call_count = {"n": 0}

        def _stub_runner(_cfg):
            call_count["n"] += 1
            return stub_record

        with patch(
            "mmfp.experiments.sweep.run_one_experiment",
            side_effect=_stub_runner,
        ):
            df = run_sweep(
                base_cfg_path=base_toml,
                overrides=[{}],
                output_csv=out_csv,
                resume=False,
                parallelism=1,
            )

        assert call_count["n"] == 1
        # CSV now has two rows: the prerun + the freshrun.
        assert len(df) == 2
        assert "freshrun" in set(df["git_sha"].tolist())

    def test_override_names_propagate_to_fingerprint(
        self, tmp_path: Path, base_toml: Path,
    ) -> None:
        """Different seeds produce different fingerprints; both run."""
        out_csv = tmp_path / "out.csv"

        call_log: list[int] = []

        def _stub_runner(cfg):
            call_log.append(cfg.seed)
            from mmfp.experiments.result_schema import config_hash as _ch
            return _make_record(seed=cfg.seed, fold=cfg.data.fold_idx).__class__(
                experiment_name=cfg.name,
                config_hash=_ch(cfg),
                seed=cfg.seed,
                fold_idx=cfg.data.fold_idx,
                news_encoder="none",
                news_aggregation="none",
                fusion_strategy=cfg.fusion.strategy,
                head_architecture=cfg.head.architecture,
                targets=",".join(cfg.head.targets),
                graph_source="none",
                lookback=cfg.data.lookback,
                price_encoder=cfg.model.price_encoder,
                mcc=0.0, accuracy=0.0, f1=0.0,
                r2=0.0, rmse=0.0, sharpe_ratio=0.0,
                vol_rmse=None, vol_r2=None,
                n_train=1, n_val=1, n_test=1, n_params=1,
                epochs_trained=1, best_val_metric=0.0, elapsed_seconds=0.0,
                platform_version="2.0.0-dev", git_sha="stub",
            )

        with patch(
            "mmfp.experiments.sweep.run_one_experiment",
            side_effect=_stub_runner,
        ):
            df = run_sweep(
                base_cfg_path=base_toml,
                overrides=[{"seed": 42}, {"seed": 43}, {"seed": 44}],
                output_csv=out_csv,
                resume=True,
                parallelism=1,
            )

        assert call_log == [42, 43, 44]
        assert len(df) == 3
        # Resume run: should skip everything.
        with patch(
            "mmfp.experiments.sweep.run_one_experiment",
            side_effect=lambda _: (_ for _ in ()).throw(
                AssertionError("should not run again")
            ),
        ):
            df2 = run_sweep(
                base_cfg_path=base_toml,
                overrides=[{"seed": 42}, {"seed": 43}, {"seed": 44}],
                output_csv=out_csv,
                resume=True,
                parallelism=1,
            )
        assert len(df2) == 3


# ---------------------------------------------------------------------------
# Smoke: empty overrides + empty CSV produces empty output
# ---------------------------------------------------------------------------


def test_empty_overrides_produces_empty_dataframe(
    tmp_path: Path, base_toml: Path,
) -> None:
    df = run_sweep(
        base_cfg_path=base_toml,
        overrides=[],
        output_csv=tmp_path / "out.csv",
        resume=True,
        parallelism=1,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
