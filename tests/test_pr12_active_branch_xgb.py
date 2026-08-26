from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chem_ts_corr.xgb_runner as xgb_runner
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import (
    DOWNSTREAM_LOCK_FILENAME,
    XGB_FORMAL_INPUT_FILES,
    run_xgb_for_active_branch,
)
from chem_ts_corr.xgb_runner import XGBRunResult, XGB_OUTPUT_FILES


FORMAL_ROOT_FILES = [
    "ranked_features.csv",
    "recommended_candidates.csv",
    "causal_review_candidates.csv",
    "risk_flags.csv",
    "conditional_granger_scores.csv",
    "causal_review_report.csv",
    "causal_review_evidence.csv",
    "final_review_summary.csv",
    "validation_summary.csv",
]


def _config(tmp_path: Path, **overrides) -> AnalysisConfig:
    kwargs = {
        "input_path": tmp_path / "input.csv",
        "time_column": "time",
        "target": "target",
        "output_dir": tmp_path,
        "max_lag": 5,
        "top_k": 15,
        "residual_control_columns": [],
        "force_include_variables": [],
        "enable_model": False,
    }
    kwargs.update(overrides)
    return AnalysisConfig(**kwargs)


def _ranked_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "variable": "x",
            "final_score": 0.9,
            "driver_rank": 1,
            "lag": 2,
            "candidate_class": "upstream_driver_candidate",
            "risk_flags": "",
            "recommended_use": "strong_screening_candidate",
        }]
    )


def _final_review_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "variable": "x",
            "final_rank": 1,
            "final_recommendation": "priority_review",
            "screening_lag": 2,
            "candidate_class": "upstream_driver_candidate",
            "risk_flags": "",
            "recommended_use": "strong_screening_candidate",
        }]
    )


def _write_formal_root(run_dir: Path) -> None:
    ranked = _ranked_frame()
    final = _final_review_frame()
    ranked.to_csv(run_dir / "ranked_features.csv", index=False, encoding="utf-8-sig")
    final.to_csv(run_dir / "final_review_summary.csv", index=False, encoding="utf-8-sig")
    ranked[["variable", "final_score", "driver_rank"]].to_csv(
        run_dir / "recommended_candidates.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {"variable": ["x"], "candidate_source": ["formal"], "reason": [""]}
    ).to_csv(
        run_dir / "causal_review_candidates.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "variable": ["x"],
            "risk_flags": ["formal_root_risk"],
            "recommended_use": [""],
            "recommended_action": [""],
        }
    ).to_csv(run_dir / "risk_flags.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"variable": ["x"]}).to_csv(
        run_dir / "conditional_granger_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"variable": ["x"]}).to_csv(
        run_dir / "causal_review_report.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"variable": ["x"]}).to_csv(
        run_dir / "causal_review_evidence.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "variable": ["x"],
            "validation_status": ["not_computed"],
            "evidence_consistency": ["not_computed"],
            "supporting_methods": [""],
            "limiting_factors": [""],
        }
    ).to_csv(run_dir / "validation_summary.csv", index=False, encoding="utf-8-sig")


def _write_context(run_dir: Path, **overrides) -> None:
    context = {
        "selected_preprocessing_mode": "raw",
        "active_screening_branch": "raw",
        "active_preprocessing_mode": "raw",
        "lowpass_tau_minutes": None,
        "requested_diff_interval_minutes": None,
        "effective_diff_points": None,
        "effective_diff_interval_minutes": None,
        "resample_rule": None,
        "branch_selection_status": "not_required",
    }
    context.update(overrides)
    (run_dir / "preprocessing_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_input_csv(
    config: AnalysisConfig,
    rows: int = 700,
    load: pd.Series | None = None,
) -> None:
    time = pd.date_range("2026-01-01", periods=rows, freq="min")
    frame = pd.DataFrame(
        {
            "target": np.arange(rows, dtype=float) + 100.0,
            "x": np.arange(rows, dtype=float) * 3.0,
        },
        index=time,
    )
    if load is not None:
        frame["load"] = load.to_numpy()
    table = frame.reset_index().rename(columns={"index": config.time_column})
    columns = [config.time_column, "target", "x", *(["load"] if load is not None else [])]
    table[columns].to_csv(
        config.input_path, index=False, encoding=config.encoding
    )


def _make_raw_run(tmp_path: Path) -> AnalysisConfig:
    config = _config(tmp_path)
    _write_formal_root(tmp_path)
    _write_context(tmp_path)
    return config


def _fake_xgb_regressor():
    class FakeXGBRegressor:
        best_iteration = 1

        def __init__(self, **params):
            self.params = params

        def fit(self, X, y, *, eval_set, verbose):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=float)

    return FakeXGBRegressor


def _install_fake_dependency(monkeypatch) -> None:
    monkeypatch.setattr(xgb_runner, "XGBRegressor", _fake_xgb_regressor())


def _spy_fold_safe(monkeypatch):
    captured: dict[str, object] = {}

    def fake_fold_safe(**kwargs):
        captured.update(kwargs)
        return XGBRunResult("success", XGB_OUTPUT_FILES, None, None, None, None)

    monkeypatch.setattr(xgb_runner, "run_xgb_validation_fold_safe", fake_fold_safe)
    return captured


def _assert_no_lock(run_dir: Path) -> None:
    assert not (run_dir / DOWNSTREAM_LOCK_FILENAME).exists()


def _formal_root_bytes(run_dir: Path) -> dict[str, bytes]:
    return {name: (run_dir / name).read_bytes() for name in FORMAL_ROOT_FILES}


def test_awaiting_confirmation_blocks_xgb(tmp_path: Path):
    config = _config(tmp_path, preprocess_mode="lowpass_diff")
    _write_formal_root(tmp_path)
    _write_context(
        tmp_path,
        selected_preprocessing_mode="lowpass_diff",
        active_screening_branch=None,
        active_preprocessing_mode=None,
        branch_selection_status="awaiting_confirmation",
    )

    with pytest.raises(ValueError, match="initial_screening_branch_not_confirmed"):
        run_xgb_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)
    assert not (tmp_path / "xgb_validation").exists()


def test_missing_final_review_blocks_xgb_without_auto_three_tier(tmp_path: Path):
    config = _make_raw_run(tmp_path)
    (tmp_path / "final_review_summary.csv").unlink()

    with pytest.raises(ValueError, match="initial_screening_formal_output_missing"):
        run_xgb_for_active_branch(tmp_path, base_config=config)

    assert not (tmp_path / "final_review_summary.csv").exists()
    assert not (tmp_path / "xgb_validation").exists()
    _assert_no_lock(tmp_path)


def test_confirmed_raw_wins_over_selected_mode(tmp_path: Path, monkeypatch):
    config = _config(tmp_path, preprocess_mode="lowpass_diff")
    _write_formal_root(tmp_path)
    _write_context(
        tmp_path,
        selected_preprocessing_mode="lowpass_diff",
        active_screening_branch="raw",
        active_preprocessing_mode="raw",
        branch_selection_status="confirmed",
    )
    _write_input_csv(config)
    captured = _spy_fold_safe(monkeypatch)

    run_xgb_for_active_branch(tmp_path, base_config=config)

    assert captured["preprocess_mode"] == "raw"
    assert captured["diff_interval_minutes"] is None


def test_confirmed_processed_uses_context_parameters(tmp_path: Path, monkeypatch):
    config = _config(
        tmp_path,
        preprocess_mode="lowpass_diff",
        lowpass_tau_minutes=7.5,
        diff_interval_minutes=5.0,
        resample_rule="2min",
    )
    _write_formal_root(tmp_path)
    _write_context(
        tmp_path,
        selected_preprocessing_mode="lowpass_diff",
        active_screening_branch="processed",
        active_preprocessing_mode="lowpass_diff",
        lowpass_tau_minutes=7.5,
        requested_diff_interval_minutes=5.0,
        effective_diff_points=5,
        effective_diff_interval_minutes=5.0,
        resample_rule="2min",
        branch_selection_status="confirmed",
    )
    _write_input_csv(config)
    captured = _spy_fold_safe(monkeypatch)

    run_xgb_for_active_branch(tmp_path, base_config=config)

    assert captured["preprocess_mode"] == "lowpass_diff"
    assert captured["lowpass_tau_minutes"] == 7.5
    assert captured["diff_interval_minutes"] == 5.0
    assert captured["resample_rule"] == "2min"


def test_context_overrides_caller_preprocessing_config(tmp_path: Path, monkeypatch):
    config = _make_raw_run(tmp_path)
    caller_config = replace(
        config,
        preprocess_mode="lowpass_diff",
        lowpass_tau_minutes=3.0,
        diff_interval_minutes=2.0,
    )
    _write_input_csv(config)
    captured = _spy_fold_safe(monkeypatch)

    run_xgb_for_active_branch(tmp_path, base_config=caller_config)

    assert captured["preprocess_mode"] == "raw"
    assert captured["diff_interval_minutes"] is None


def test_xgb_runner_uses_frozen_exclude_windows_for_fold_safe_data(tmp_path: Path, monkeypatch):
    frozen_windows = [{"start": "2026-01-01T00:10:00", "end": "2026-01-01T00:15:00"}]
    config = _make_raw_run(tmp_path)
    _write_context(tmp_path, exclude_windows=frozen_windows)
    _write_input_csv(config, rows=40)
    before = {
        name: (tmp_path / name).read_bytes()
        for name in (
            "ranked_features.csv",
            "final_review_summary.csv",
            "recommended_candidates.csv",
            "preprocessing_context.json",
        )
    }
    assert not (tmp_path / "run_config.json").exists()
    captured = _spy_fold_safe(monkeypatch)
    from chem_ts_corr import web

    original_numeric_frame = web._numeric_frame

    def capture_numeric_frame(downstream_config, protected_columns=None):
        captured["config"] = downstream_config
        return original_numeric_frame(downstream_config, protected_columns)

    monkeypatch.setattr(web, "_numeric_frame", capture_numeric_frame)

    run_xgb_for_active_branch(
        tmp_path, base_config=replace(config, exclude_windows=[])
    )

    data = captured["data"]
    assert captured["config"].exclude_windows == frozen_windows
    assert captured["config"].exclude_windows != []
    excluded_timestamps = pd.date_range("2026-01-01 00:10", "2026-01-01 00:15", freq="min")
    assert data.index.intersection(excluded_timestamps).empty
    assert pd.Timestamp("2026-01-01 00:09") in data.index
    assert pd.Timestamp("2026-01-01 00:16") in data.index
    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
    assert not (tmp_path / "run_config.json").exists()


@pytest.mark.parametrize(
    ("explicit", "residual", "capacity", "expected"),
    [
        (["explicit_control"], ["residual_control"], [], ["explicit_control"]),
        (None, ["residual_control"], ["capacity_control"], ["residual_control"]),
        (None, [], ["capacity_control"], ["capacity_control"]),
        ([], ["residual_control"], ["capacity_control"], []),
    ],
)
def test_control_columns_resolution(
    tmp_path: Path, monkeypatch, explicit, residual, capacity, expected
):
    config = _make_raw_run(tmp_path)
    config = replace(config, residual_control_columns=residual, capacity_columns=capacity)
    _write_input_csv(config)
    captured = _spy_fold_safe(monkeypatch)

    run_xgb_for_active_branch(
        tmp_path, base_config=config, control_columns=explicit
    )

    assert captured["control_columns"] == expected


def test_whitelist_resolution(tmp_path: Path, monkeypatch):
    config = _make_raw_run(tmp_path)
    config = replace(config, xgb_whitelist=["A", "B"])
    _write_input_csv(config)
    captured = _spy_fold_safe(monkeypatch)

    run_xgb_for_active_branch(tmp_path, base_config=config, whitelist=None)
    assert captured["whitelist"] == ["A", "B"]

    captured.clear()
    run_xgb_for_active_branch(tmp_path, base_config=config, whitelist=[])
    assert captured["whitelist"] == []


def test_topn_and_maxlag_defaults(tmp_path: Path, monkeypatch):
    config = _make_raw_run(tmp_path)
    config = replace(config, xgb_top_n=10, xgb_max_lag=None)
    _write_input_csv(config)
    captured = _spy_fold_safe(monkeypatch)

    run_xgb_for_active_branch(tmp_path, base_config=config)

    assert captured["top_n"] == 10
    assert captured["max_lag"] is None


def test_formal_root_is_the_only_ranked_source(tmp_path: Path, monkeypatch):
    config = _make_raw_run(tmp_path)
    processed_dir = tmp_path / "screening_branches" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"variable": ["sentinel_processed"]}).to_csv(
        processed_dir / "ranked_features.csv", index=False, encoding="utf-8-sig"
    )
    _write_input_csv(config)
    captured = _spy_fold_safe(monkeypatch)

    run_xgb_for_active_branch(tmp_path, base_config=config)

    ranked = captured["ranked_features"]
    assert "sentinel_processed" not in ranked["variable"].astype(str).tolist()
    assert ranked["variable"].astype(str).tolist() == ["x"]


def test_model_discovery_does_not_extend_xgb(tmp_path: Path, monkeypatch):
    config = _make_raw_run(tmp_path)
    pd.DataFrame({"variable": ["sentinel_model"]}).to_csv(
        tmp_path / "model_discovered_candidates.csv", index=False, encoding="utf-8-sig"
    )
    _write_input_csv(config)
    captured = _spy_fold_safe(monkeypatch)

    run_xgb_for_active_branch(tmp_path, base_config=config)

    final_review = captured["final_review_summary"]
    assert "sentinel_model" not in final_review["variable"].astype(str).tolist()


def test_xgb_keeps_formal_root_byte_identical(tmp_path: Path, monkeypatch):
    config = _make_raw_run(tmp_path)
    _write_input_csv(config)
    before = _formal_root_bytes(tmp_path)
    before_ranked = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")
    before_top_k = before_ranked.head(config.top_k)[["variable", "final_score", "driver_rank"]]
    _install_fake_dependency(monkeypatch)

    result = run_xgb_for_active_branch(tmp_path, base_config=config)

    assert result["status"] == "success"
    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
    after_ranked = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")
    pd.testing.assert_frame_equal(
        after_ranked[["variable", "final_score", "driver_rank"]],
        before_ranked[["variable", "final_score", "driver_rank"]],
    )
    pd.testing.assert_frame_equal(
        after_ranked.head(config.top_k)[["variable", "final_score", "driver_rank"]],
        before_top_k,
    )
    forbidden_decision_fields = {
        "final_score",
        "driver_rank",
        "final_rank",
        "validation_score",
        "validation_rank",
        "candidate_priority_score",
        "candidate_priority_rank",
        "candidate_pool_rank",
    }
    for name in ("xgb_fold_metrics.csv", "xgb_model_summary.csv", "xgb_candidate_uplift.csv"):
        columns = set(pd.read_csv(tmp_path / "xgb_validation" / name).columns)
        assert columns.isdisjoint(forbidden_decision_fields)
    summary = json.loads(
        (tmp_path / "xgb_validation/xgb_validation_summary.json").read_text(encoding="utf-8")
    )
    assert set(summary).isdisjoint(forbidden_decision_fields)


def test_xgb_writes_exactly_six_outputs(tmp_path: Path, monkeypatch):
    config = _make_raw_run(tmp_path)
    _write_input_csv(config)
    _install_fake_dependency(monkeypatch)

    result = run_xgb_for_active_branch(tmp_path, base_config=config)

    assert result["status"] == "success"
    output_dir = tmp_path / "xgb_validation"
    assert set(path.name for path in output_dir.iterdir()) == set(XGB_OUTPUT_FILES)


def test_xgb_creates_lock_as_first_stage(tmp_path: Path, monkeypatch):
    config = _make_raw_run(tmp_path)
    _install_fake_dependency(monkeypatch)
    _write_input_csv(config)
    _assert_no_lock(tmp_path)

    run_xgb_for_active_branch(tmp_path, base_config=config)

    assert (tmp_path / DOWNSTREAM_LOCK_FILENAME).exists()


def test_xgb_runs_after_existing_downstream_lock(tmp_path: Path, monkeypatch):
    config = _make_raw_run(tmp_path)
    (tmp_path / DOWNSTREAM_LOCK_FILENAME).write_text("downstream-locked", encoding="utf-8")
    _write_input_csv(config)
    _install_fake_dependency(monkeypatch)

    result = run_xgb_for_active_branch(tmp_path, base_config=config)

    assert result["status"] == "success"
    assert (tmp_path / DOWNSTREAM_LOCK_FILENAME).exists()


def test_missing_dependency_returns_missing_dependency(tmp_path: Path, monkeypatch):
    config = _make_raw_run(tmp_path)
    _write_input_csv(config)
    monkeypatch.setattr(xgb_runner, "XGBRegressor", None)

    result = run_xgb_for_active_branch(tmp_path, base_config=config)

    assert result["status"] == "missing_dependency"
    assert "xgboost" in result["error_message"]


def test_pr12_runner_source_preserves_formal_contract():
    source = Path("chem_ts_corr/pipeline.py").read_text(encoding="utf-8")
    start = source.index("def run_xgb_for_active_branch")
    end = source.index("\ndef _resolve_base_config", start)
    body = source[start:end]
    for forbidden in (
        "screening_branches",
        "preprocessing_comparison.csv",
        "secondary_candidate_context",
        "model_discovered_candidates",
        "abs(lag",
        "abs(best_lag",
        "_prepared_frame_for_validation",
    ):
        assert forbidden not in body
    assert "required_formal_files=XGB_FORMAL_INPUT_FILES" in body
    assert "run_xgb_validation_fold_safe" in body


def test_formal_runner_returns_invalid_input_for_effective_train_shortfall(
    tmp_path: Path, monkeypatch
):
    config = _config(
        tmp_path,
        segment_column="load",
        segment_mode="custom",
        segment_min=0,
        segment_max=0,
    )
    _write_formal_root(tmp_path)
    _write_context(tmp_path)
    load = pd.Series(np.zeros(700, dtype=float))
    load.iloc[:135] = 1.0
    _write_input_csv(config, load=load)
    _install_fake_dependency(monkeypatch)

    result = run_xgb_for_active_branch(tmp_path, base_config=config)

    assert result["status"] == "invalid_input"
    assert "effective train rows" in result["error_message"]
    assert "min_train_rows 100" in result["error_message"]


def test_formal_runner_row_count_matches_predictions(tmp_path: Path, monkeypatch):
    config = _make_raw_run(tmp_path)
    _write_input_csv(config)
    _install_fake_dependency(monkeypatch)

    result = run_xgb_for_active_branch(tmp_path, base_config=config)

    assert result["status"] == "success"
    payload = json.loads(
        (tmp_path / "xgb_validation/xgb_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    predictions = pd.read_csv(tmp_path / "xgb_validation/xgb_predictions.csv")
    assert payload["row_count"] == len(predictions)
