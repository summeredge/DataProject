from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chem_ts_corr.pipeline as pipeline
import chem_ts_corr.service as service
import chem_ts_corr.xgb_runner as runner
from chem_ts_corr.xgb_runner import XGBRunResult, XGB_OUTPUT_FILES, run_xgb_validation
from chem_ts_corr.xgb_validation import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_XGB_PARAMS,
    XGBFeatureSets,
    XGBTimeSplit,
    XGBValidationProvenance,
    XGBValidationResult,
)


CANDIDATE_COLUMNS = [
    "variable",
    "fold_count",
    "positive_rmse_fold_count",
    "positive_mae_fold_count",
    "positive_rmse_fold_ratio",
    "median_rmse_improvement_pct",
    "median_mae_improvement_pct",
    "mean_rmse_improvement_pct",
    "mean_mae_improvement_pct",
    "worst_fold_rmse_improvement_pct",
    "validation_status",
]


def _inputs(rows: int = 20):
    data = pd.DataFrame(
        {
            "target": np.arange(rows, dtype=float),
            "control": np.arange(rows, dtype=float) * 2,
            "x": np.arange(rows, dtype=float) * 3,
        },
        index=pd.date_range("2025-01-01", periods=rows, freq="min"),
    )
    final = pd.DataFrame(
        [{"final_rank": 1, "variable": "x", "final_recommendation": "priority_review", "screening_lag": 3}]
    )
    ranked = pd.DataFrame(
        [{
            "variable": "x", "lag": 3, "candidate_class": "upstream_driver_candidate",
            "risk_flags": "", "recommended_use": "strong_screening_candidate",
        }]
    )
    return data, final, ranked


def _feature_sets() -> XGBFeatureSets:
    index = pd.RangeIndex(12)
    features = pd.DataFrame(
        {"target__lag_1": np.arange(12), "x__lag_3": np.arange(12) * 2}, index=index
    )
    target = pd.Series(np.arange(12, dtype=float), index=index, name="target")
    return XGBFeatureSets(
        features=features,
        target=target,
        m0_features=("target__lag_1",),
        m1_features=("target__lag_1",),
        m2_features=("target__lag_1", "x__lag_3"),
        candidate_feature_map={"x": ("x__lag_3",)},
        max_used_lag=3,
    )


def _model_result() -> XGBValidationResult:
    return XGBValidationResult(
        fold_metrics=pd.DataFrame(
            [{
                "fold": 0, "model_name": "M1", "train_rows": 6, "validation_rows": 3,
                "test_rows": 3, "best_iteration": 4, "rmse": 1.0, "mae": 0.8, "r2": 0.5,
            }]
        ),
        summary=pd.DataFrame(
            [{
                "model_name": "M1", "mean_rmse": 1.0, "median_rmse": 1.0,
                "mean_mae": 0.8, "median_mae": 0.8, "mean_r2": 0.5, "fold_count": 1,
            }]
        ),
        predictions=pd.DataFrame(
            [{
                "fold": 0, "timestamp_index": 9, "y_true": 9.0,
                "M0_prediction": 8.0, "M1_prediction": 8.5, "M2_prediction": 9.0,
            }]
        ),
        provenance=XGBValidationProvenance(
            m1_features=("target__lag_1",),
            split_signature=(),
            parameter_signature=(),
            early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
            data_fingerprint="test-provenance-fingerprint",
        ),
    )


def _candidate_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [["x", 1, 1, 1, 1.0, 10.0, 8.0, 10.0, 8.0, 10.0, "weak_incremental_value"]],
        columns=CANDIDATE_COLUMNS,
    )


def _mock_success(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    *,
    expected_top_n: int = 8,
    expected_max_lag: int = 5,
):
    pool = pd.DataFrame([{"candidate_order": 1, "variable": "x", "screening_lag": 3}])
    feature_sets = _feature_sets()
    splits = [XGBTimeSplit(0, slice(0, 6), slice(6, 9), slice(9, 12), 3)]
    model_result = _model_result()

    monkeypatch.setattr(runner, "XGBRegressor", object)

    def build_pool(*args, **kwargs):
        calls.append("candidate_pool")
        assert kwargs["top_n"] == expected_top_n
        return pool.copy(deep=True)

    def build_features(*args, **kwargs):
        calls.append("features")
        assert kwargs["max_lag"] == expected_max_lag
        return feature_sets

    def build_splits(n_samples, **kwargs):
        calls.append("splits")
        assert n_samples == 12
        assert kwargs["gap"] == 3
        return splits

    def run_models(received_features, received_splits):
        calls.append("models")
        assert received_features is feature_sets
        assert received_splits is splits
        return model_result

    def run_candidates(received_features, received_splits, received_pool, **kwargs):
        calls.append("uplift")
        assert received_features is feature_sets
        assert received_splits is splits
        assert kwargs["baseline_result"] is model_result
        return pd.DataFrame(), _candidate_summary()

    monkeypatch.setattr(runner, "build_xgb_candidate_pool", build_pool)
    monkeypatch.setattr(runner, "build_xgb_feature_sets", build_features)
    monkeypatch.setattr(runner, "build_expanding_time_splits", build_splits)
    monkeypatch.setattr(runner, "run_xgb_time_validation", run_models)
    monkeypatch.setattr(runner, "run_candidate_uplift_validation", run_candidates)


def test_successful_run_uses_fixed_orchestration_and_creates_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls)
    data, final, ranked = _inputs()
    run_dir = tmp_path / "new-run"

    result = run_xgb_validation(
        run_dir=run_dir, target="target", data=data, final_review_summary=final,
        ranked_features=ranked, control_columns=["control"],
    )

    assert result.status == "success"
    assert calls == ["candidate_pool", "features", "splits", "models", "uplift"]
    assert (run_dir / "xgb_validation").is_dir()
    assert result.output_files == XGB_OUTPUT_FILES


def test_success_writes_all_fixed_files_and_exact_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls)
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
    )
    output_dir = tmp_path / "xgb_validation"

    assert set(path.name for path in output_dir.iterdir()) == set(XGB_OUTPUT_FILES)
    pd.testing.assert_frame_equal(
        pd.read_csv(output_dir / "xgb_fold_metrics.csv"), _model_result().fold_metrics,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(output_dir / "xgb_model_summary.csv"), _model_result().summary,
        check_dtype=False,
    )
    assert pd.read_csv(output_dir / "xgb_candidate_uplift.csv").columns.tolist() == CANDIDATE_COLUMNS
    assert pd.read_csv(output_dir / "xgb_predictions.csv").columns.tolist() == (
        _model_result().predictions.columns.tolist()
    )
    assert result.fold_metrics_path == str(output_dir / "xgb_fold_metrics.csv")
    assert result.summary_path == str(output_dir / "xgb_model_summary.csv")
    assert result.candidate_uplift_path == str(output_dir / "xgb_candidate_uplift.csv")


def test_json_summary_is_auditable_and_has_no_ranking_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls)
    data, final, ranked = _inputs()

    run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
    )
    payload = json.loads((tmp_path / "xgb_validation/xgb_validation_summary.json").read_text())

    assert payload["status"] == "success"
    assert payload["target"] == "target"
    assert payload["candidate_count"] == 1
    assert payload["candidate_pool_count"] == 1
    assert payload["fold_count"] == 1
    assert payload["row_count"] == len(_feature_sets().features)
    assert payload["m0_feature_count"] == len(_feature_sets().m0_features)
    assert payload["m1_feature_count"] == len(_feature_sets().m1_features)
    assert payload["m2_feature_count"] == len(_feature_sets().m2_features)
    assert payload["max_used_lag"] == _feature_sets().max_used_lag
    assert payload["resolved_max_lag"] == 5
    assert payload["top_n"] == 8
    assert payload["data_fingerprint"] == "test-provenance-fingerprint"
    assert payload["early_stopping_rounds"] == DEFAULT_EARLY_STOPPING_ROUNDS
    assert payload["model_parameters"] == DEFAULT_XGB_PARAMS
    assert set(payload["timings_seconds"]) == {
        "input_validation", "candidate_pool", "feature_build", "split_build",
        "model_validation", "candidate_uplift", "write_outputs", "total",
    }
    assert all(value >= 0 for value in payload["timings_seconds"].values())
    assert payload["timings_seconds"]["total"] >= max(
        value
        for key, value in payload["timings_seconds"].items()
        if key != "total"
    )
    assert payload["files"] == list(XGB_OUTPUT_FILES)
    assert payload["created_at"]
    assert "final_score" not in payload
    assert "driver_rank" not in payload
    assert "final_rank" not in payload


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_target", "target column not found"),
        ("empty_summary", "missing final_review_summary"),
        ("empty_data", "missing data"),
        ("summary_missing_variable", "final_review_summary missing column: variable"),
        (
            "summary_missing_recommendation",
            "final_review_summary missing column: final_recommendation",
        ),
    ],
)
def test_invalid_required_inputs_return_invalid_input(
    tmp_path: Path, mutation: str, match: str
):
    data, final, ranked = _inputs()
    target = "target"
    if mutation == "missing_target":
        target = "missing"
    elif mutation == "empty_summary":
        final = pd.DataFrame()
    elif mutation == "empty_data":
        data = pd.DataFrame()
    elif mutation == "summary_missing_variable":
        final = final.drop(columns="variable")
    else:
        final = final.drop(columns="final_recommendation")

    result = run_xgb_validation(
        run_dir=tmp_path, target=target, data=data,
        final_review_summary=final, ranked_features=ranked,
    )

    assert result.status == "invalid_input"
    assert match in result.error_message
    assert result.output_files == ()


def test_run_dir_that_is_a_file_returns_invalid_input(tmp_path: Path):
    run_dir = tmp_path / "not-a-directory"
    run_dir.write_text("occupied")
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=run_dir, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
    )

    assert result.status == "invalid_input"
    assert "run_dir is not writable" in result.error_message


def test_missing_xgboost_returns_install_instruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(runner, "XGBRegressor", None)
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
    )

    assert result.status == "missing_dependency"
    assert result.error_message == (
        "xgboost is not installed. Install optional dependency: pip install -e '.[xgb]'"
    )


def test_model_failure_is_isolated_and_does_not_delete_existing_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls)
    output_dir = tmp_path / "xgb_validation"
    output_dir.mkdir()
    old = output_dir / "xgb_fold_metrics.csv"
    old.write_text("existing-result")
    monkeypatch.setattr(
        runner, "run_xgb_time_validation", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("training failed"))
    )
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
    )

    assert result.status == "failed"
    assert result.error_message == "training failed"
    assert old.read_text() == "existing-result"


def _write_old_xgb_outputs(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(exist_ok=True)
    contents = {name: f"old::{name}" for name in XGB_OUTPUT_FILES}
    for name, content in contents.items():
        (output_dir / name).write_text(content)
    return contents


@pytest.mark.parametrize("failure_position", [2, 3, 4, 5])
def test_staging_write_failure_preserves_all_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_position: int
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls)
    output_dir = tmp_path / "xgb_validation"
    old_contents = _write_old_xgb_outputs(output_dir)
    original_to_csv = pd.DataFrame.to_csv
    original_write_text = Path.write_text
    write_count = 0

    def failing_to_csv(frame, path, *args, **kwargs):
        nonlocal write_count
        write_count += 1
        if write_count == failure_position:
            raise OSError("injected staging failure")
        return original_to_csv(frame, path, *args, **kwargs)

    def failing_write_text(path, *args, **kwargs):
        if failure_position == 5 and path.name == "xgb_validation_summary.json":
            raise OSError("injected staging failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", failing_to_csv)
    monkeypatch.setattr(Path, "write_text", failing_write_text)
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
    )

    assert result.status == "invalid_input"
    for name, content in old_contents.items():
        assert (output_dir / name).read_text() == content
    assert not list(tmp_path.glob(".xgb-stage-*"))


def test_first_run_staging_failure_leaves_no_partial_formal_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls)
    original_to_csv = pd.DataFrame.to_csv
    write_count = 0

    def failing_to_csv(frame, path, *args, **kwargs):
        nonlocal write_count
        write_count += 1
        if write_count == 3:
            raise OSError("injected staging failure")
        return original_to_csv(frame, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", failing_to_csv)
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
    )

    assert result.status == "invalid_input"
    output_dir = tmp_path / "xgb_validation"
    assert not any((output_dir / name).exists() for name in XGB_OUTPUT_FILES)


@pytest.mark.parametrize("failure_position", [2, 5, 6])
def test_commit_failure_rolls_back_every_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_position: int
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls)
    output_dir = tmp_path / "xgb_validation"
    old_contents = _write_old_xgb_outputs(output_dir)
    original_replace = Path.replace
    replace_count = 0

    def failing_replace(path, target):
        nonlocal replace_count
        replace_count += 1
        if replace_count == failure_position:
            raise OSError("injected commit failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
    )

    assert result.status == "invalid_input"
    for name, content in old_contents.items():
        assert (output_dir / name).read_text() == content


@pytest.mark.parametrize("failure_position", [3, 6])
def test_first_run_commit_failure_removes_new_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_position: int
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls)
    original_replace = Path.replace
    replace_count = 0

    def failing_replace(path, target):
        nonlocal replace_count
        replace_count += 1
        if replace_count == failure_position:
            raise OSError("injected commit failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
    )

    assert result.status == "invalid_input"
    output_dir = tmp_path / "xgb_validation"
    assert not any((output_dir / name).exists() for name in XGB_OUTPUT_FILES)


def test_output_timings_include_json_write_and_atomic_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls)
    clock = {"seconds": 0.0}
    original_write_text = Path.write_text
    original_replace = Path.replace

    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock["seconds"])

    def delayed_write_text(path, *args, **kwargs):
        if path.name in {"xgb_validation_summary.json", ".final-summary.json"}:
            clock["seconds"] += 0.2
        return original_write_text(path, *args, **kwargs)

    def delayed_replace(path, target):
        clock["seconds"] += 0.1
        return original_replace(path, target)

    monkeypatch.setattr(Path, "write_text", delayed_write_text)
    monkeypatch.setattr(Path, "replace", delayed_replace)
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path,
        target="target",
        data=data,
        final_review_summary=final,
        ranked_features=ranked,
    )
    payload = json.loads(
        (tmp_path / "xgb_validation/xgb_validation_summary.json").read_text()
    )

    assert result.status == "success"
    assert payload["timings_seconds"]["write_outputs"] >= 0.7
    assert payload["timings_seconds"]["total"] >= 0.7


def test_json_candidate_count_matches_uplift_when_whitelist_expands_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls)
    pool = pd.DataFrame(
        [{"candidate_order": index + 1, "variable": f"c{index}"} for index in range(9)]
    )
    summary = pd.concat(
        [_candidate_summary().assign(variable=f"c{index}") for index in range(8)],
        ignore_index=True,
    )
    monkeypatch.setattr(runner, "build_xgb_candidate_pool", lambda *args, **kwargs: pool)
    monkeypatch.setattr(
        runner,
        "run_candidate_uplift_validation",
        lambda *args, **kwargs: (pd.DataFrame(), summary),
    )
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
        whitelist=[f"c{index}" for index in range(9)],
    )
    payload = json.loads((tmp_path / "xgb_validation/xgb_validation_summary.json").read_text())
    uplift = pd.read_csv(tmp_path / "xgb_validation/xgb_candidate_uplift.csv")

    assert result.status == "success"
    assert payload["candidate_pool_count"] == 9
    assert payload["candidate_count"] == 8
    assert payload["candidate_count"] == uplift["variable"].nunique()


def test_top_n_above_eight_is_rejected_explicitly(tmp_path: Path):
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked, top_n=9,
    )

    assert result.status == "invalid_input"
    assert result.error_message == "top_n must be an integer between 1 and 8"


@pytest.mark.parametrize("top_n", [0, 9, True, 4.0])
def test_runner_rejects_invalid_top_n_types_or_range(tmp_path: Path, top_n: object):
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path,
        target="target",
        data=data,
        final_review_summary=final,
        ranked_features=ranked,
        top_n=top_n,
    )

    assert result.status == "invalid_input"
    assert result.error_message == "top_n must be an integer between 1 and 8"


@pytest.mark.parametrize("max_lag", [0, 5001, True, 1.0])
def test_runner_rejects_invalid_max_lag_types_or_range(tmp_path: Path, max_lag: object):
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path,
        target="target",
        data=data,
        final_review_summary=final,
        ranked_features=ranked,
        max_lag=max_lag,
    )

    assert result.status == "invalid_input"
    assert result.error_message == "max_lag must be an integer between 1 and 5000"


def test_runner_accepts_max_lag_hard_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls, expected_max_lag=5000)
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path,
        target="target",
        data=data,
        final_review_summary=final,
        ranked_features=ranked,
        max_lag=5000,
    )

    assert result.status == "success"


def test_runner_accepts_numpy_integral_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []
    _mock_success(monkeypatch, calls, expected_top_n=4, expected_max_lag=5)
    data, final, ranked = _inputs()

    result = run_xgb_validation(
        run_dir=tmp_path,
        target="target",
        data=data,
        final_review_summary=final,
        ranked_features=ranked,
        top_n=np.int64(4),
        max_lag=np.int64(5),
    )

    assert result.status == "success"


def test_inputs_are_not_modified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    _mock_success(monkeypatch, calls)
    data, final, ranked = _inputs()
    before_data = data.copy(deep=True)
    before_final = final.copy(deep=True)
    before_ranked = ranked.copy(deep=True)

    run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
    )

    pd.testing.assert_frame_equal(data, before_data)
    pd.testing.assert_frame_equal(final, before_final)
    pd.testing.assert_frame_equal(ranked, before_ranked)


def test_real_xgb1_setup_is_reused_while_training_is_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(runner, "XGBRegressor", object)
    data, final, ranked = _inputs(rows=600)
    captured: dict[str, object] = {}

    def model_run(feature_sets, splits):
        captured["feature_sets"] = feature_sets
        captured["splits"] = splits
        return _model_result()

    def uplift_run(feature_sets, splits, pool, **kwargs):
        assert feature_sets is captured["feature_sets"]
        assert splits is captured["splits"]
        assert kwargs["baseline_result"] is not None
        return pd.DataFrame(), _candidate_summary()

    monkeypatch.setattr(runner, "run_xgb_time_validation", model_run)
    monkeypatch.setattr(runner, "run_candidate_uplift_validation", uplift_run)

    result = run_xgb_validation(
        run_dir=tmp_path, target="target", data=data,
        final_review_summary=final, ranked_features=ranked,
        control_columns=["control"], max_lag=5,
    )

    assert result.status == "success"
    feature_sets = captured["feature_sets"]
    assert feature_sets.candidate_feature_map["x"] == ("x__lag_1", "x__lag_2", "x__lag_3", "x__lag_4", "x__lag_5")
    assert all(split.gap == feature_sets.max_used_lag for split in captured["splits"])


def test_service_exports_unified_runner_without_pipeline_side_effects():
    assert service.run_xgb_validation is runner.run_xgb_validation
    assert service.XGBRunResult is runner.XGBRunResult
    assert callable(service.run_xgb_analysis)
    assert not hasattr(pipeline, "run_xgb_validation")


def test_service_hook_forwards_all_parameters_and_returns_xgb_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data, final, ranked = _inputs()
    expected = XGBRunResult("success", (), None, None, None, None)
    captured: dict[str, object] = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(service, "run_xgb_validation", fake_runner)
    result = service.run_xgb_analysis(
        run_dir=tmp_path,
        data=data,
        target="target",
        final_review_summary=final,
        ranked_features=ranked,
        control_columns=["control"],
        whitelist=["manual"],
        top_n=4,
        max_lag=9,
    )

    assert result is expected
    assert captured["run_dir"] == tmp_path
    assert captured["data"] is data
    assert captured["target"] == "target"
    assert captured["final_review_summary"] is final
    assert captured["ranked_features"] is ranked
    assert captured["control_columns"] == ["control"]
    assert captured["whitelist"] == ["manual"]
    assert captured["top_n"] == 4
    assert captured["max_lag"] == 9


def test_service_hook_missing_final_review_returns_xgb_error(tmp_path: Path):
    data, _, ranked = _inputs()

    result = service.run_xgb_analysis(
        run_dir=tmp_path,
        data=data,
        target="target",
        final_review_summary=None,
        ranked_features=ranked,
    )

    assert isinstance(result, XGBRunResult)
    assert result.status == "invalid_input"
    assert "final_review_summary" in result.error_message


def test_service_hook_failure_does_not_modify_analysis_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data, final, ranked = _inputs()
    before_data = data.copy(deep=True)
    before_final = final.copy(deep=True)
    before_ranked = ranked.copy(deep=True)
    failed = XGBRunResult("failed", (), None, None, None, "training failed")
    monkeypatch.setattr(service, "run_xgb_validation", lambda **kwargs: failed)

    result = service.run_xgb_analysis(
        run_dir=tmp_path,
        data=data,
        target="target",
        final_review_summary=final,
        ranked_features=ranked,
    )

    assert result is failed
    pd.testing.assert_frame_equal(data, before_data)
    pd.testing.assert_frame_equal(final, before_final)
    pd.testing.assert_frame_equal(ranked, before_ranked)


def test_service_hook_is_not_called_by_existing_analysis_or_pipeline():
    service_source = inspect.getsource(service.analyze_numeric_frame)
    pipeline_source = Path("chem_ts_corr/pipeline.py").read_text(encoding="utf-8")

    assert "run_xgb_analysis" not in service_source
    assert "run_xgb_analysis" not in pipeline_source


def test_service_and_pipeline_have_no_xgb_ranking_writeback_patterns():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [Path("chem_ts_corr/service.py"), Path("chem_ts_corr/pipeline.py")]
    )
    for forbidden in ["final_score =", "driver_rank =", "update_rank", "screening修改"]:
        assert forbidden not in source


def test_xgb_run_result_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        XGBRunResult("unknown", (), None, None, None, "bad")
