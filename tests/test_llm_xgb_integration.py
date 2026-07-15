from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

import chem_ts_corr.llm_report as llm_report
from chem_ts_corr.llm_api import LLMCallConfig, generate_llm_report
from chem_ts_corr.llm_report import PACKAGE_KEYS, build_llm_analysis_package, build_llm_prompt


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _set_candidate_count(xgb_dir: Path, value: object, *, remove: bool = False) -> None:
    summary_path = xgb_dir / "xgb_validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if remove:
        summary.pop("candidate_count", None)
    else:
        summary["candidate_count"] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def _write_xgb_outputs(run_dir: Path, *, status: str = "success") -> Path:
    xgb_dir = run_dir / "xgb_validation"
    xgb_dir.mkdir()
    (xgb_dir / "xgb_validation_summary.json").write_text(
        json.dumps(
            {
                "status": status,
                "target": "Y.PV",
                "row_count": 120,
                "candidate_count": 3,
                "candidate_pool_count": 5,
                "fold_count": 3,
                "m0_feature_count": 1,
                "m1_feature_count": 4,
                "m2_feature_count": 7,
                "max_used_lag": 12,
                "resolved_max_lag": 12,
                "top_n": 3,
                "created_at": "2026-07-15T12:00:00Z",
                "model_parameters": {"n_estimators": 500},
                "data_fingerprint": "must-not-enter-prompt",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_csv(
        xgb_dir / "xgb_model_summary.csv",
        [
            {"model_name": "M0", "mean_rmse": 4.0, "median_rmse": 4.1, "mean_mae": 3.0, "median_mae": 3.1, "mean_r2": 0.1, "fold_count": 3},
            {"model_name": "M1", "mean_rmse": 3.0, "median_rmse": 3.1, "mean_mae": 2.0, "median_mae": 2.1, "mean_r2": 0.3, "fold_count": 3},
            {"model_name": "M2", "mean_rmse": 2.0, "median_rmse": 2.1, "mean_mae": 1.5, "median_mae": 1.6, "mean_r2": 0.5, "fold_count": 3, "M2_vs_M1_rmse_improvement_pct": 33.3, "M2_vs_M1_mae_improvement_pct": 25.0},
        ],
    )
    _write_csv(
        xgb_dir / "xgb_candidate_uplift.csv",
        [
            {"variable": "FEED.PV", "fold_count": 3, "positive_rmse_fold_count": 3, "positive_mae_fold_count": 2, "positive_rmse_fold_ratio": 1.0, "median_rmse_improvement_pct": 12.5, "median_mae_improvement_pct": 8.0, "mean_rmse_improvement_pct": 10.0, "mean_mae_improvement_pct": 7.0, "worst_fold_rmse_improvement_pct": 4.0, "validation_status": "validated_incremental_signal"},
            {"variable": "FLOW.PV", "fold_count": 3, "positive_rmse_fold_count": 2, "positive_mae_fold_count": 2, "positive_rmse_fold_ratio": 0.67, "median_rmse_improvement_pct": 3.0, "median_mae_improvement_pct": 1.0, "mean_rmse_improvement_pct": 2.0, "mean_mae_improvement_pct": 1.0, "worst_fold_rmse_improvement_pct": -1.0, "validation_status": "unstable_out_of_time"},
            {"variable": "TEMP.PV", "fold_count": 3, "positive_rmse_fold_count": 0, "positive_mae_fold_count": 0, "positive_rmse_fold_ratio": 0.0, "median_rmse_improvement_pct": -2.0, "median_mae_improvement_pct": -1.0, "mean_rmse_improvement_pct": -2.0, "mean_mae_improvement_pct": -1.0, "worst_fold_rmse_improvement_pct": -3.0, "validation_status": "redundant_with_baseline"},
        ],
    )
    return xgb_dir


def test_xgb_success_is_compactly_included_in_package_and_prompt(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_xgb_outputs(run_dir)

    package = build_llm_analysis_package(run_dir, top_n=2)
    xgb = package["xgb_out_of_time_validation"]

    assert "xgb_out_of_time_validation" in PACKAGE_KEYS
    assert xgb["status"] == "success"
    assert xgb["available"] is True
    assert xgb["summary"]["row_count"] == 120
    assert xgb["summary"]["candidate_count"] == 3
    assert xgb["summary"]["fold_count"] == 3
    assert xgb["summary"]["max_used_lag"] == 12
    assert [row["model_name"] for row in xgb["model_comparison"]] == ["M0", "M1", "M2"]
    assert [row["variable"] for row in xgb["candidate_uplift"]] == ["FEED.PV", "FLOW.PV"]
    assert xgb["candidate_uplift"][0]["positive_rmse_fold_ratio"] == 1.0
    assert xgb["candidate_uplift"][0]["validation_status"] == "validated_incremental_signal"
    assert "不是工艺因果结论" in xgb["evidence_scope"]
    assert package["overview"]["xgb_available_files"] == [
        "xgb_validation/xgb_validation_summary.json",
        "xgb_validation/xgb_model_summary.csv",
        "xgb_validation/xgb_candidate_uplift.csv",
    ]
    json.dumps(package, ensure_ascii=False)

    prompt = build_llm_prompt(package)
    prompt_json = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
    assert prompt_json["xgb_out_of_time_validation"]["candidate_uplift"][0]["variable"] == "FEED.PV"
    assert "model_parameters" not in prompt_json["xgb_out_of_time_validation"]["summary"]
    assert "data_fingerprint" not in prompt
    assert "xgb_predictions" not in prompt
    for marker in [
        "XGBoost 时间外增量验证", "不代表确定性因果", "不改变前三层排名",
        "M1 + 单个候选变量", "改善率 = (基线误差 - 候选模型误差) / 基线误差 × 100%",
        "positive_rmse_fold_ratio", "validated_incremental_signal", "weak_incremental_value",
        "redundant_with_baseline", "unstable_out_of_time", "insufficient_features",
        "未运行时不得编造结果", "跨层证据解释规则", "建议优先进行工程复核",
        "不得直接删除或否定候选", "只能作为补充预测线索", "降低对时间稳定性的判断",
        "不得断言上述任一因素已经发生",
    ]:
        assert marker in prompt


def test_xgb_package_excludes_fold_and_prediction_details(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    _write_csv(xgb_dir / "xgb_predictions.csv", [{"secret_prediction": "must-not-enter"}])
    _write_csv(xgb_dir / "xgb_fold_metrics.csv", [{"secret_fold": "must-not-enter"}])

    package = build_llm_analysis_package(run_dir, top_n=2)

    assert len(package["xgb_out_of_time_validation"]["candidate_uplift"]) == 2
    assert len(package["xgb_out_of_time_validation"]["model_comparison"]) == 3
    assert "must-not-enter" not in json.dumps(package, ensure_ascii=False)


def test_xgb_not_run_invalid_and_incomplete_outputs_fail_closed(tmp_path: Path):
    not_run = tmp_path / "not-run"
    not_run.mkdir()
    assert build_llm_analysis_package(not_run)["xgb_out_of_time_validation"] == {
        "status": "not_run",
        "available": False,
        "summary": {},
        "model_comparison": [],
        "candidate_uplift": [],
        "evidence_scope": "时间外预测增量证据，不是工艺因果结论，也不改变前三层排名",
    }

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    xgb_dir = invalid / "xgb_validation"
    xgb_dir.mkdir()
    (xgb_dir / "xgb_validation_summary.json").write_text("not-json", encoding="utf-8")
    _write_csv(xgb_dir / "xgb_model_summary.csv", [{"model_name": "M1"}])
    _write_csv(xgb_dir / "xgb_candidate_uplift.csv", [{"variable": "must-not-use"}])
    invalid_xgb = build_llm_analysis_package(invalid)["xgb_out_of_time_validation"]
    assert invalid_xgb["status"] == "invalid_summary"
    assert invalid_xgb["available"] is False
    assert invalid_xgb["model_comparison"] == []
    assert invalid_xgb["candidate_uplift"] == []

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    xgb_dir = _write_xgb_outputs(incomplete)
    (xgb_dir / "xgb_candidate_uplift.csv").unlink()
    incomplete_xgb = build_llm_analysis_package(incomplete)["xgb_out_of_time_validation"]
    assert incomplete_xgb["status"] == "incomplete_outputs"
    assert incomplete_xgb["available"] is False
    assert incomplete_xgb["model_comparison"] == []
    assert incomplete_xgb["candidate_uplift"] == []


@pytest.mark.parametrize("status", ["missing_dependency", "invalid_input", "failed"])
def test_unsuccessful_xgb_statuses_are_not_interpreted_as_results(tmp_path: Path, status: str):
    run_dir = tmp_path / status
    run_dir.mkdir()
    _write_xgb_outputs(run_dir, status=status)

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == status
    assert xgb["available"] is False
    assert xgb["model_comparison"] == []
    assert xgb["candidate_uplift"] == []


def test_generate_llm_report_receives_xgb_package_without_api_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_xgb_outputs(run_dir)
    captured: dict[str, str] = {}

    def fake_call(config: LLMCallConfig, prompt: str) -> dict[str, object]:
        captured["prompt"] = prompt
        return {"report": "# report\n", "usage": {}, "raw": {}}

    monkeypatch.setattr("chem_ts_corr.llm_api.call_openai_compatible_chat", fake_call)

    result = generate_llm_report(
        run_dir,
        LLMCallConfig(base_url="https://api.example.com", model="test", api_key="sk-test"),
    )

    assert "XGBoost 时间外增量验证" in captured["prompt"]
    assert "跨层证据解释规则" in captured["prompt"]
    assert "xgb_out_of_time_validation" in captured["prompt"]
    assert "FEED.PV" in captured["prompt"]
    assert "median_rmse_improvement_pct" in captured["prompt"]
    assert result["package"]["xgb_out_of_time_validation"]["available"] is True


@pytest.mark.parametrize(
    "invalid_model",
    [
        "empty", "missing_m2", "duplicate_m2", "blank_model_name",
        "all_metrics_empty", "missing_model_name", "missing_mean_rmse", "missing_fold_count",
    ],
)
def test_invalid_xgb_model_summary_is_incomplete(tmp_path: Path, invalid_model: str):
    run_dir = tmp_path / invalid_model
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    model_path = xgb_dir / "xgb_model_summary.csv"
    model = pd.read_csv(model_path)
    if invalid_model == "empty":
        model = model.head(0)
    elif invalid_model == "missing_m2":
        model = model[model["model_name"] != "M2"]
    elif invalid_model == "duplicate_m2":
        model = pd.concat([model, model[model["model_name"] == "M2"]], ignore_index=True)
    elif invalid_model == "blank_model_name":
        model.loc[0, "model_name"] = ""
    elif invalid_model == "all_metrics_empty":
        metric_columns = ["mean_rmse", "median_rmse", "mean_mae", "median_mae", "mean_r2", "fold_count"]
        model = model.assign(
            **{
                column: pd.Series([None] * len(model), index=model.index, dtype="object")
                for column in metric_columns
            }
        )
    else:
        model = model.drop(columns=[invalid_model.removeprefix("missing_")])
    model.to_csv(model_path, index=False, encoding="utf-8-sig")

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == "incomplete_outputs"
    assert xgb["available"] is False
    assert xgb["model_comparison"] == []
    assert xgb["candidate_uplift"] == []


def test_empty_candidate_uplift_is_invalid_when_summary_has_candidates(tmp_path: Path):
    run_dir = tmp_path / "empty-candidates"
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    candidate_path = xgb_dir / "xgb_candidate_uplift.csv"
    pd.read_csv(candidate_path).head(0).to_csv(candidate_path, index=False, encoding="utf-8-sig")

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == "incomplete_outputs"
    assert xgb["available"] is False


def test_empty_candidate_uplift_is_valid_when_summary_has_no_candidates(tmp_path: Path):
    run_dir = tmp_path / "no-candidates"
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    summary_path = xgb_dir / "xgb_validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["candidate_count"] = 0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    candidate_path = xgb_dir / "xgb_candidate_uplift.csv"
    pd.read_csv(candidate_path).head(0).to_csv(candidate_path, index=False, encoding="utf-8-sig")

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == "success"
    assert xgb["available"] is True
    assert xgb["candidate_uplift"] == []


@pytest.mark.parametrize(
    ("invalid_candidate", "value"),
    [
        ("missing_variable", None),
        ("empty_variable", None),
        ("missing_validation_status", None),
        ("missing_positive_rmse_fold_ratio", None),
        ("missing_median_rmse_improvement_pct", None),
        ("unknown_validation_status", "unknown_status"),
        ("invalid_ratio", -0.1),
        ("invalid_ratio", 1.1),
    ],
)
def test_invalid_xgb_candidate_uplift_is_incomplete(
    tmp_path: Path, invalid_candidate: str, value: object
):
    run_dir = tmp_path / f"{invalid_candidate}-{value}"
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    candidate_path = xgb_dir / "xgb_candidate_uplift.csv"
    candidates = pd.read_csv(candidate_path)
    if invalid_candidate.startswith("missing_"):
        candidates = candidates.drop(columns=[invalid_candidate.removeprefix("missing_")])
    elif invalid_candidate == "empty_variable":
        candidates.loc[:, "variable"] = ""
    elif invalid_candidate == "unknown_validation_status":
        candidates.loc[0, "validation_status"] = value
    else:
        candidates.loc[0, "positive_rmse_fold_ratio"] = value
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == "incomplete_outputs"
    assert xgb["available"] is False
    assert xgb["model_comparison"] == []
    assert xgb["candidate_uplift"] == []


def test_insufficient_features_allows_missing_improvement_values(tmp_path: Path):
    run_dir = tmp_path / "insufficient"
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    _set_candidate_count(xgb_dir, 1)
    candidate_path = xgb_dir / "xgb_candidate_uplift.csv"
    candidates = pd.read_csv(candidate_path).head(1)
    candidates.loc[0, "validation_status"] = "insufficient_features"
    candidates.loc[0, "fold_count"] = 0
    for column in [
        "positive_rmse_fold_ratio", "median_rmse_improvement_pct",
        "median_mae_improvement_pct", "mean_rmse_improvement_pct",
        "mean_mae_improvement_pct", "worst_fold_rmse_improvement_pct",
    ]:
        candidates.loc[0, column] = None
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == "success"
    assert xgb["available"] is True
    assert xgb["candidate_uplift"][0]["validation_status"] == "insufficient_features"


def test_invalid_utf8_xgb_summary_fails_closed(tmp_path: Path):
    run_dir = tmp_path / "invalid-utf8"
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    (xgb_dir / "xgb_validation_summary.json").write_bytes(b"\xff\xfe\x00\x80")

    package = build_llm_analysis_package(run_dir)
    xgb = package["xgb_out_of_time_validation"]

    assert xgb["status"] == "invalid_summary"
    assert xgb["available"] is False
    assert xgb["model_comparison"] == []
    assert xgb["candidate_uplift"] == []
    assert "invalid_summary" in build_llm_prompt(package)


@pytest.mark.parametrize(
    ("case", "value"),
    [
        ("missing", None),
        ("none", None),
        ("true", True),
        ("false", False),
        ("empty", ""),
        ("invalid_string", "abc"),
        ("negative", -1),
        ("fractional", 1.5),
        ("nan", float("nan")),
        ("infinity", float("inf")),
        ("zero_with_rows", 0),
        ("fewer_in_summary", 2),
    ],
)
def test_invalid_candidate_count_is_incomplete(
    tmp_path: Path, case: str, value: object
):
    run_dir = tmp_path / case
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    _set_candidate_count(xgb_dir, value, remove=case == "missing")

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == "incomplete_outputs"
    assert xgb["available"] is False
    assert xgb["model_comparison"] == []
    assert xgb["candidate_uplift"] == []


def test_candidate_count_rejects_fewer_csv_rows(tmp_path: Path):
    run_dir = tmp_path / "fewer-csv-rows"
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    candidate_path = xgb_dir / "xgb_candidate_uplift.csv"
    pd.read_csv(candidate_path).head(2).to_csv(
        candidate_path, index=False, encoding="utf-8-sig"
    )

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == "incomplete_outputs"
    assert xgb["available"] is False


@pytest.mark.parametrize("candidate_count", [3, "3", 3.0])
def test_candidate_count_accepts_exact_integer_representations(
    tmp_path: Path, candidate_count: object
):
    run_dir = tmp_path / f"valid-{candidate_count!r}"
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    _set_candidate_count(xgb_dir, candidate_count)

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == "success"
    assert xgb["available"] is True
    assert len(xgb["candidate_uplift"]) == 3


@pytest.mark.parametrize(
    ("model_name", "column", "value"),
    [
        ("M0", "fold_count", 0),
        ("M1", "fold_count", 0),
        ("M1", "fold_count", 2.5),
        ("M2", "fold_count", "bad"),
        ("M2", "fold_count", float("inf")),
        ("M0", "mean_rmse", None),
        ("M2", "mean_rmse", None),
        ("M1", "median_rmse", -0.1),
        ("M2", "mean_mae", "bad"),
        ("M0", "median_mae", float("inf")),
    ],
)
def test_invalid_required_model_metric_is_incomplete(
    tmp_path: Path, model_name: str, column: str, value: object
):
    run_dir = tmp_path / f"{model_name}-{column}-{value}"
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    model_path = xgb_dir / "xgb_model_summary.csv"
    model = pd.read_csv(model_path)
    model[column] = model[column].astype("object")
    model.loc[model["model_name"] == model_name, column] = value
    model.to_csv(model_path, index=False, encoding="utf-8-sig")

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == "incomplete_outputs"
    assert xgb["available"] is False
    assert xgb["model_comparison"] == []
    assert xgb["candidate_uplift"] == []


def test_valid_m0_row_does_not_mask_empty_m1_and_m2_metrics(tmp_path: Path):
    run_dir = tmp_path / "partially-valid-models"
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    model_path = xgb_dir / "xgb_model_summary.csv"
    model = pd.read_csv(model_path)
    error_columns = ["mean_rmse", "median_rmse", "mean_mae", "median_mae"]
    model[error_columns] = model[error_columns].astype("object")
    model.loc[model["model_name"].isin(["M1", "M2"]), error_columns] = None
    model.to_csv(model_path, index=False, encoding="utf-8-sig")

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == "incomplete_outputs"
    assert xgb["available"] is False


def test_negative_mean_r2_and_empty_m2_improvements_remain_valid(tmp_path: Path):
    run_dir = tmp_path / "valid-optional-metrics"
    run_dir.mkdir()
    xgb_dir = _write_xgb_outputs(run_dir)
    model_path = xgb_dir / "xgb_model_summary.csv"
    model = pd.read_csv(model_path)
    model.loc[model["model_name"] == "M1", "mean_r2"] = -0.25
    improvement_columns = [
        "M2_vs_M1_rmse_improvement_pct", "M2_vs_M1_mae_improvement_pct",
    ]
    model[improvement_columns] = model[improvement_columns].astype("object")
    model.loc[model["model_name"] == "M2", improvement_columns] = None
    model.to_csv(model_path, index=False, encoding="utf-8-sig")

    xgb = build_llm_analysis_package(run_dir)["xgb_out_of_time_validation"]

    assert xgb["status"] == "success"
    assert xgb["available"] is True
    m1 = next(row for row in xgb["model_comparison"] if row["model_name"] == "M1")
    assert m1["mean_r2"] == -0.25


def test_xgb_validators_do_not_use_the_old_lenient_patterns():
    candidate_source = inspect.getsource(llm_report._validate_xgb_candidate_uplift)
    model_source = inspect.getsource(llm_report._validate_xgb_model_summary)

    assert 'int(summary.get("candidate_count") or 0)' not in candidate_source
    assert "len(frame) != candidate_count" in candidate_source
    assert "notna().any().any()" not in model_source
