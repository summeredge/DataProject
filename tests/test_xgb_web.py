from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chem_ts_corr.llm_report as llm_report
import chem_ts_corr.service as service
import chem_ts_corr.web as web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.xgb_validation import build_xgb_feature_sets


def _write_run(tmp_path: Path) -> tuple[Path, AnalysisConfig, pd.DataFrame, pd.DataFrame]:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    input_path = tmp_path / "input.csv"
    input_path.write_text("timestamp,target,x\n2025-01-01,1,2\n", encoding="utf-8")
    config = AnalysisConfig(
        input_path=input_path,
        time_column="timestamp",
        target="target",
        output_dir=run_dir,
    )
    web._write_run_config(run_dir, config, "file-id")
    final = pd.DataFrame(
        [{"variable": "x", "final_recommendation": "priority_review", "screening_lag": 2}]
    )
    ranked = pd.DataFrame([{"variable": "x", "lag": 2}])
    final.to_csv(run_dir / "final_review_summary.csv", index=False)
    ranked.to_csv(run_dir / "ranked_features.csv", index=False)
    ranked.to_csv(run_dir / "recommended_candidates.csv", index=False)
    return run_dir, config, final, ranked


def _handler_form(monkeypatch: pytest.MonkeyPatch, form: dict[str, str]) -> None:
    monkeypatch.setattr(web, "_multipart_form", lambda handler: form)


def test_xgb_config_defaults_are_disabled_and_isolated(tmp_path: Path):
    first = AnalysisConfig(tmp_path / "a.csv", "time", "target", tmp_path / "a")
    second = AnalysisConfig(tmp_path / "b.csv", "time", "target", tmp_path / "b")

    assert first.enable_xgb_validation is False
    assert first.xgb_top_n == 8
    assert first.xgb_max_lag is None
    assert first.xgb_whitelist == []
    assert first.xgb_whitelist is not second.xgb_whitelist


def test_disabled_xgb_request_is_skipped_without_calling_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir, _, _, _ = _write_run(tmp_path)
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    _handler_form(monkeypatch, {"run_id": run_dir.name, "enable_xgb_validation": "false"})
    monkeypatch.setattr(
        web,
        "run_xgb_for_active_branch",
        lambda **kwargs: pytest.fail("runner must not be called"),
    )

    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == "skipped"
    assert payload["xgbModelSummary"] == []
    assert payload["xgbCandidateUplift"] == []


def test_xgb_web_forwards_inputs_through_formal_runner_and_returns_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir, _, final, _ = _write_run(tmp_path)
    ranked = pd.DataFrame([
        {"variable": "control_high_score", "lag": 1},
        {"variable": "candidate_a", "lag": 2},
        {"variable": "candidate_b", "lag": 3},
    ])
    recommended = ranked.iloc[1:].copy(deep=True)
    ranked.to_csv(run_dir / "ranked_features.csv", index=False)
    recommended.to_csv(run_dir / "recommended_candidates.csv", index=False)
    before_final = final.copy(deep=True)
    before_ranked = ranked.copy(deep=True)
    captured: dict[str, object] = {}

    def fake_runner(output_dir, **kwargs):
        captured["run_dir"] = Path(output_dir)
        captured["base_config"] = kwargs.get("base_config")
        captured["top_n"] = kwargs.get("top_n")
        captured["max_lag"] = kwargs.get("max_lag")
        captured["whitelist"] = kwargs.get("whitelist")
        captured["control_columns"] = kwargs.get("control_columns")
        output_dir = run_dir / "xgb_validation"
        output_dir.mkdir()
        pd.DataFrame(
            [{"model_name": "M2", "mean_rmse": 1.0, "mean_mae": 0.8, "mean_r2": 0.4}]
        ).to_csv(output_dir / "xgb_model_summary.csv", index=False)
        pd.DataFrame(
            [{"variable": "x", "median_rmse_improvement_pct": 5.0, "validation_status": "validated_incremental_signal"}]
        ).to_csv(output_dir / "xgb_candidate_uplift.csv", index=False)
        pd.DataFrame(
            [{
                "variable": "x", "fold": 0,
                "train_start": "2025-01-01T00:00:00",
                "train_end": "2025-01-01T00:09:00",
                "validation_start": "2025-01-01T00:11:00",
                "validation_end": "2025-01-01T00:14:00",
                "test_start": "2025-01-01T00:16:00",
                "test_end": "2025-01-01T00:19:00",
                "train_rows": 10, "validation_rows": 4, "test_rows": 4,
                "baseline_rmse": 1.0, "candidate_rmse": 0.9,
                "rmse_improvement_pct": 10.0,
                "baseline_mae": 0.8, "candidate_mae": 0.7,
                "mae_improvement_pct": 12.5, "candidate_r2": 0.5,
                "best_iteration": 4,
            }]
        ).to_csv(output_dir / "xgb_candidate_fold_metrics.csv", index=False)
        (output_dir / "xgb_validation_summary.json").write_text(
            json.dumps({"status": "success", "candidate_count": 1}), encoding="utf-8"
        )
        return {
            "status": "success",
            "error_message": None,
            "fold_metrics_path": str(output_dir / "xgb_fold_metrics.csv"),
            "summary_path": str(output_dir / "xgb_model_summary.csv"),
            "candidate_uplift_path": str(output_dir / "xgb_candidate_uplift.csv"),
        }

    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(web, "run_xgb_for_active_branch", fake_runner)
    _handler_form(
        monkeypatch,
        {
            "run_id": run_dir.name,
            "enable_xgb_validation": "true",
            "top_n": "10",
            "max_lag": "6",
            "whitelist": "manual_a, manual_b",
            "control_columns": "control",
        },
    )

    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == "success"
    assert captured["run_dir"] == run_dir
    assert captured["top_n"] == 10
    assert captured["max_lag"] == 6
    assert captured["whitelist"] == ["manual_a", "manual_b"]
    assert captured["control_columns"] == ["control"]
    pd.testing.assert_frame_equal(pd.read_csv(run_dir / "ranked_features.csv"), before_ranked)
    assert payload["xgbModelSummary"][0]["model_name"] == "M2"
    assert payload["xgbCandidateUplift"][0]["variable"] == "x"
    assert payload["xgbCandidateFoldMetrics"][0]["fold"] == 0
    assert payload["xgbValidationSummary"]["candidate_count"] == 1
    assert "候选变量预测增量证据" in payload["message"]
    assert "人工复核参考" in payload["message"]
    assert "不改变前三层结果" in payload["message"]
    download_names = {item["name"] for item in payload["downloads"]}
    assert "xgb_validation/xgb_model_summary.csv" in download_names
    assert "xgb_validation/xgb_candidate_uplift.csv" in download_names
    assert "xgb_validation/xgb_candidate_fold_metrics.csv" in download_names
    assert "xgb_validation/xgb_validation_summary.json" in download_names


def test_missing_formal_input_error_token_is_preserved_in_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir, _, _, _ = _write_run(tmp_path)
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    _handler_form(monkeypatch, {"run_id": run_dir.name, "enable_xgb_validation": "true"})

    def rejected_runner(output_dir, **kwargs):
        raise ValueError(
            "initial_screening_formal_output_missing: missing ranked_features.csv"
        )

    monkeypatch.setattr(web, "run_xgb_for_active_branch", rejected_runner)

    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == "invalid_input"
    assert "initial_screening_formal_output_missing" in payload["error_message"]
    assert "ranked_features.csv" in payload["error_message"]


def test_xgb_web_endpoint_delegates_to_formal_fold_safe_runner():
    source = inspect.getsource(web._run_xgb_validation_response)

    assert "run_xgb_for_active_branch(" in source
    assert "_prepared_frame_for_validation" not in source
    assert "run_xgb_analysis" not in source
    assert "recommended_candidates.csv" not in source


def test_missing_final_review_error_token_is_preserved_in_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir, _, _, _ = _write_run(tmp_path)
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    _handler_form(monkeypatch, {"run_id": run_dir.name, "enable_xgb_validation": "true"})

    def rejected_runner(output_dir, **kwargs):
        raise ValueError(
            "initial_screening_formal_output_missing: missing final_review_summary.csv"
        )

    monkeypatch.setattr(web, "run_xgb_for_active_branch", rejected_runner)

    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == "invalid_input"
    assert "final_review_summary.csv" in payload["error_message"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("top_n", "0", "top_n must be an integer between 1 and 10"),
        ("top_n", "11", "top_n must be an integer between 1 and 10"),
        ("top_n", "8.5", "top_n must be an integer between 1 and 10"),
        ("top_n", "abc", "top_n must be an integer between 1 and 10"),
        ("max_lag", "0", "max_lag must be an integer between 1 and 5000"),
        ("max_lag", "abc", "max_lag must be an integer between 1 and 5000"),
        ("max_lag", "5001", "max_lag must be an integer between 1 and 5000"),
    ],
)
def test_invalid_xgb_parameters_are_rejected_before_data_or_service_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
):
    run_dir, _, _, _ = _write_run(tmp_path)
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    _handler_form(
        monkeypatch,
        {"run_id": run_dir.name, "enable_xgb_validation": "true", field: value},
    )
    monkeypatch.setattr(
        web,
        "run_xgb_for_active_branch",
        lambda **kwargs: pytest.fail("runner must not be called"),
    )

    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == "invalid_input"
    assert payload["error_message"] == message


@pytest.mark.parametrize("status", ["missing_dependency", "invalid_input", "failed"])
def test_xgb_service_errors_remain_isolated_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
):
    run_dir, _, _, _ = _write_run(tmp_path)
    (run_dir / "summary.md").write_text("existing analysis", encoding="utf-8")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(
        web,
        "run_xgb_for_active_branch",
        lambda output_dir, **kwargs: {
            "status": status,
            "error_message": "xgb error",
        },
    )
    _handler_form(monkeypatch, {"run_id": run_dir.name, "enable_xgb_validation": "true"})

    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == status
    assert payload["error_message"] == "xgb error"
    assert (run_dir / "ranked_features.csv").exists()
    assert (run_dir / "final_review_summary.csv").exists()
    assert "summary.md" in {item["name"] for item in payload["downloads"]}


def _validation_config(
    tmp_path: Path, frame: pd.DataFrame, **kwargs
) -> AnalysisConfig:
    input_path = tmp_path / "validation.csv"
    frame.to_csv(input_path, index=False)
    return AnalysisConfig(
        input_path=input_path,
        time_column="timestamp",
        target="target",
        output_dir=tmp_path / "out",
        min_valid_ratio=0.0,
        **kwargs,
    )


def test_xgb_prepared_frame_preserves_resampled_lag_units_without_standardizing(
    tmp_path: Path,
):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=60, freq="min"),
            "target": 100.0 + np.arange(60) * 2.0,
            "x": 1000.0 + np.arange(60) * 10.0,
        }
    )
    config = _validation_config(tmp_path, frame, resample_rule="5min")

    prepared = web._prepared_frame_for_validation(config)

    assert len(prepared) == 12
    assert prepared.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=5)).all()
    assert prepared.index[2] - prepared.index[0] == pd.Timedelta(minutes=10)
    assert prepared.iloc[0]["target"] == 104.0
    assert not np.isclose(prepared["target"].mean(), 0.0)
    assert "standardize_frame" not in inspect.getsource(web._prepared_frame_for_validation)


def test_xgb_prepared_frame_applies_operating_segment(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=30, freq="min"),
            "target": np.arange(30, dtype=float),
            "load": np.resize([1.0, 0.0], 30),
            "x": np.arange(30, dtype=float) * 3,
        }
    )
    config = _validation_config(
        tmp_path,
        frame,
        segment_column="load",
        segment_mode="custom",
        segment_min=0,
        segment_max=0,
    )

    prepared = web._prepared_frame_for_validation(config)
    target_mask = web._target_segment_mask(prepared)
    feature_sets = build_xgb_feature_sets(
        prepared,
        "target",
        pd.DataFrame([{"variable": "x", "screening_lag": 1, "candidate_order": 1}]),
        max_lag=1,
        baseline_lags=[1],
        candidate_lag_radius=0,
        target_mask=target_mask,
    )
    first_target_time = prepared.index[1]

    assert len(prepared) == 30
    assert target_mask is not None
    assert int(target_mask.sum()) == 15
    assert first_target_time in feature_sets.features.index
    assert prepared.loc[first_target_time, "load"] == 0
    assert prepared.loc[first_target_time - pd.Timedelta(minutes=1), "load"] == 1
    assert feature_sets.features.loc[first_target_time, "x__lag_1"] == prepared.loc[
        first_target_time - pd.Timedelta(minutes=1), "x"
    ]


def test_xgb_prepared_frame_applies_ignore_roles(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=30, freq="min"),
            "target": np.arange(30, dtype=float),
            "ignored": np.arange(30, dtype=float) * 2,
            "kept": np.arange(30, dtype=float) * 3,
        }
    )
    roles_path = tmp_path / "roles.csv"
    roles_path.write_text("variable,role\nignored,IGNORE\nkept,PV\n", encoding="utf-8")
    config = _validation_config(tmp_path, frame, roles_path=roles_path)

    prepared = web._prepared_frame_for_validation(config)

    assert "ignored" not in prepared.columns
    assert "kept" in prepared.columns


@pytest.mark.parametrize("mode", ["diff", "detrend"])
def test_xgb_prepared_frame_applies_configured_transform(tmp_path: Path, mode: str):
    values = np.arange(40, dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=40, freq="min"),
            "target": values**2 + 100,
            "x": values**3 + 1000,
        }
    )
    config = _validation_config(
        tmp_path,
        frame,
        preprocess_mode=mode,
        detrend_window=7,
    )

    prepared = web._prepared_frame_for_validation(config)

    if mode == "diff":
        assert len(prepared) == len(frame) - 1
        assert prepared.index[0] > pd.Timestamp(frame.iloc[0]["timestamp"])
        assert prepared.iloc[0]["target"] == 1.0
    else:
        assert len(prepared) == len(frame) - 1
        assert not prepared["target"].equals(
            pd.Series(frame["target"].to_numpy(), index=pd.to_datetime(frame["timestamp"]))
        )


def test_xgb_api_delegates_to_fold_safe_runner_without_raw_loader():
    source = inspect.getsource(web._run_xgb_validation_response)

    assert "run_xgb_for_active_branch(" in source
    assert "load_timeseries_csv" not in source
    assert "_prepared_frame_for_validation" not in source


def test_xgb_web_surface_and_architecture_guards():
    source = inspect.getsource(web)
    assert "/api/run_xgb_validation" in web.INDEX_HTML
    for marker in [
        'id="enableXgbValidation"', 'id="xgbTopN"', 'id="xgbMaxLag"',
        'id="xgbWhitelist"', 'id="runXgbValidation"', 'id="xgbModelSummaryTable"',
        'id="xgbCandidateUpliftTable"', 'id="xgbCandidateFoldDetails"',
        'id="xgbCandidateFoldMetricsTable"', "xgb_validation/xgb_model_summary.csv",
        "xgb_validation/xgb_candidate_uplift.csv", "xgb_validation/xgb_validation_summary.json",
        "xgb_validation/xgb_candidate_fold_metrics.csv", "逐时间折验证明细",
        'id="xgbRunSummary"', 'max="5000"', "renderXgbRunSummary",
        "row_count", "candidate_count", "fold_count", "m0_feature_count",
        "m1_feature_count", "m2_feature_count", "max_used_lag", "timings.total",
    ]:
        assert marker in web.INDEX_HTML or marker in web.DOWNLOAD_FILES
    for forbidden in [
        "XGBRegressor", "import xgboost", ".fit(", ".predict(",
        "build_xgb_feature_sets", "final_score =", "driver_rank =",
    ]:
        assert forbidden not in source
    assert "from chem_ts_corr.xgb_runner" not in source
    assert "run_xgb_for_active_branch" in source
    assert "run_xgb_analysis" not in source
    assert service.run_xgb_analysis is not None
    assert "XGBoost 时间外预测验证" in web.INDEX_HTML
    assert "候选变量预测增量证据" in web.INDEX_HTML
    assert "Baseline（M1）" in web.INDEX_HTML
    assert "Candidate：同一 M1 基线 + 单个候选变量历史信息" in web.INDEX_HTML
    assert "不参与 ranking、scoring 或 candidate selection" in web.INDEX_HTML
    xgb_section = web.INDEX_HTML.split('<div id="xgbValidationTab"', 1)[1].split(
        '<div id="llmReportTab"', 1
    )[0]
    for forbidden in ["因果证明", "根因确认", "最终驱动变量", "最终验证", "最终证明", "最佳变量", "因果验证", "驱动变量确认"]:
        assert forbidden not in xgb_section
    assert 'renderXgbDownloads(data.status === "success" ?' in web.INDEX_HTML


def test_xgb_product_copy_avoids_misleading_validation_claims():
    sources = [
        web.INDEX_HTML,
        inspect.getsource(web._xgb_response_payload),
        inspect.getsource(web._run_xgb_validation_response),
        inspect.getsource(llm_report.build_llm_prompt),
        Path("README.md").read_text(encoding="utf-8"),
        Path("docs/product.md").read_text(encoding="utf-8"),
        Path("docs/architecture.md").read_text(encoding="utf-8"),
        Path("docs/contracts.md").read_text(encoding="utf-8"),
        Path("docs/xgb_validation.md").read_text(encoding="utf-8"),
    ]
    copy = "\n".join(sources)

    for forbidden in [
        "因果证明",
        "根因确认",
        "最终驱动变量",
        "最终验证",
        "最终证明",
        "最佳变量",
        "因果验证",
        "驱动变量确认",
    ]:
        assert forbidden not in copy


def test_second_layer_model_explanation_copy_remains_legacy():
    source = web.INDEX_HTML
    model_section = source.split('<details id="modelExplanationDetails"', 1)[1].split(
        "</details>", 1
    )[0]

    for marker in [
        "随机森林重要性表示模型依赖，不等于可操作性或因果结论。",
        "随机森林模型解释变量排序",
        "该表按变量汇总随机森林/SHAP 重要性，每个变量仅显示最强 lag。结果表示预测模型依赖，不代表因果关系或可操作性。",
        "运行随机森林模型解释后显示变量排序。",
        "结果仅表示预测模型依赖，不代表因果关系或可操作性。",
    ]:
        assert marker in model_section
    for marker in ["模型重要性排名", "最大重要性", "变量总重要性", "重要性排名"]:
        assert marker in source
    assert "模型输入重要性（不代表工艺因果贡献）" not in model_section
    assert "随机森林模型解释变量级输入重要性" not in model_section


def test_xgb_candidate_count_input_keeps_default_and_exposes_both_limits():
    source = web.INDEX_HTML

    assert 'id="xgbTopN" type="number" min="1" max="10" value="8"' in source
    assert "自动候选默认 8 个、最多 10 个" in source
    assert "加入白名单后，总候选数量最多 12 个" in source
    assert 'form.append("top_n", el("xgbTopN").value || "8")' in source
    assert 'id="xgbTopN" type="number" min="1" max="8"' not in source


def _xgb_run_function_body() -> str:
    source = web.INDEX_HTML
    start = source.index("async function runXgbValidation()")
    end = source.index("\n\n\nasync function testLlmConnection()", start)
    return source[start:end]


def test_xgb_candidate_uplift_help_and_columns_remain_explicit():
    source = web.INDEX_HTML
    uplift_section = source.split("<h2>候选变量增量验证</h2>", 1)[1].split(
        'id="xgbCandidateUpliftDownload"', 1
    )[0]

    for marker in [
        "RMSE 改善中位数", "MAE 改善中位数", "RMSE 改善折占比", "M1 基线模型",
        "大于 0 表示加入该候选后预测误差下降", "0.67 表示约 67%", "不代表工艺因果成立",
    ]:
        assert marker in uplift_section
    assert "<div class=\"help\">" in uplift_section
    assert (
        'return ["variable", "median_rmse_improvement_pct", '
        '"median_mae_improvement_pct", "positive_rmse_fold_ratio", "validation_status"];'
    ) in source
    assert (
        'return ["variable", "fold", "test_time_range", "rmse_improvement_pct", '
        '"mae_improvement_pct", "test_rows"];'
    ) in source


def test_xgb_run_uses_shared_global_status_timer_for_its_full_lifecycle():
    function_body = _xgb_run_function_body()
    success_body, catch_and_finally = function_body.split("  } catch (error) {", 1)
    catch_body, finally_body = catch_and_finally.split("  } finally {", 1)

    assert "const startedAt = performance.now();" in function_body
    assert 'startStatusTimer("正在运行 XGB 时间外预测验证...", startedAt)' in function_body
    assert function_body.index("startStatusTimer(") < function_body.index("await postForm(")
    assert "appendElapsed" in success_body
    assert "appendElapsed" in catch_body
    assert "stopStatusTimer(timerId);" in finally_body
    assert "updateXgbRunAvailability();" in finally_body
    assert finally_body.index("stopStatusTimer(timerId);") < finally_body.index(
        "updateXgbRunAvailability();"
    )
    assert "setInterval(" not in function_body
    assert "elapsedSeconds(" not in function_body


def test_xgb_run_summary_uses_json_and_hides_missing_or_failed_values():
    function_body = web.INDEX_HTML.split("function renderXgbRunSummary", 1)[1].split(
        "function", 1
    )[0]

    assert 'summary.status !== "success"' in function_body
    assert 'container.innerHTML = ""' in function_body
    assert "summary.row_count" in function_body
    assert "summary.candidate_count" in function_body
    assert "summary.fold_count" in function_body
    assert "summary.max_used_lag" in function_body
    assert "timings.total" in function_body
    assert "|| 0" not in function_body
