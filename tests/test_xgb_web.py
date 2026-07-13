from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chem_ts_corr.service as service
import chem_ts_corr.web as web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.xgb_runner import XGBRunResult


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
        web, "run_xgb_analysis", lambda **kwargs: pytest.fail("service must not be called")
    )

    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == "skipped"
    assert payload["xgbModelSummary"] == []
    assert payload["xgbCandidateUplift"] == []


def test_xgb_web_forwards_inputs_through_service_and_returns_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir, _, final, ranked = _write_run(tmp_path)
    data = pd.DataFrame(
        {"target": [1.0, 2.0], "control": [3.0, 4.0], "x": [5.0, 6.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )
    before_final = final.copy(deep=True)
    before_ranked = ranked.copy(deep=True)
    captured: dict[str, object] = {}

    def fake_service(**kwargs):
        captured.update(kwargs)
        output_dir = run_dir / "xgb_validation"
        output_dir.mkdir()
        pd.DataFrame(
            [{"model_name": "M2", "mean_rmse": 1.0, "mean_mae": 0.8, "mean_r2": 0.4}]
        ).to_csv(output_dir / "xgb_model_summary.csv", index=False)
        pd.DataFrame(
            [{"variable": "x", "median_rmse_improvement_pct": 5.0, "validation_status": "validated_incremental_signal"}]
        ).to_csv(output_dir / "xgb_candidate_uplift.csv", index=False)
        (output_dir / "xgb_validation_summary.json").write_text(
            json.dumps({"status": "success", "candidate_count": 1}), encoding="utf-8"
        )
        return XGBRunResult(
            "success", (), None, str(output_dir / "xgb_model_summary.csv"),
            str(output_dir / "xgb_candidate_uplift.csv"), None,
        )

    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(web, "_prepared_frame_for_validation", lambda config: data)
    monkeypatch.setattr(web, "run_xgb_analysis", fake_service)
    _handler_form(
        monkeypatch,
        {
            "run_id": run_dir.name,
            "enable_xgb_validation": "true",
            "top_n": "4",
            "max_lag": "6",
            "whitelist": "manual_a, manual_b",
            "control_columns": "control",
        },
    )

    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == "success"
    assert captured["run_dir"] == run_dir
    assert captured["data"] is data
    assert captured["target"] == "target"
    assert captured["top_n"] == 4
    assert captured["max_lag"] == 6
    assert captured["whitelist"] == ["manual_a", "manual_b"]
    assert captured["control_columns"] == ["control"]
    pd.testing.assert_frame_equal(captured["final_review_summary"], before_final)
    pd.testing.assert_frame_equal(captured["ranked_features"], before_ranked)
    assert payload["xgbModelSummary"][0]["model_name"] == "M2"
    assert payload["xgbCandidateUplift"][0]["variable"] == "x"
    assert payload["xgbValidationSummary"]["candidate_count"] == 1
    download_names = {item["name"] for item in payload["downloads"]}
    assert "xgb_validation/xgb_model_summary.csv" in download_names
    assert "xgb_validation/xgb_candidate_uplift.csv" in download_names
    assert "xgb_validation/xgb_validation_summary.json" in download_names


def test_missing_final_review_is_rejected_before_loading_or_service_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir, _, _, _ = _write_run(tmp_path)
    (run_dir / "final_review_summary.csv").unlink()
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    _handler_form(monkeypatch, {"run_id": run_dir.name, "enable_xgb_validation": "true"})
    monkeypatch.setattr(
        web,
        "_prepared_frame_for_validation",
        lambda *args, **kwargs: pytest.fail("data must not be prepared"),
    )

    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == "invalid_input"
    assert "final_review_summary" in payload["error_message"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("top_n", "0", "top_n must be an integer between 1 and 8"),
        ("top_n", "abc", "top_n must be an integer between 1 and 8"),
        ("max_lag", "0", "max_lag must be a positive integer or empty"),
        ("max_lag", "abc", "max_lag must be a positive integer or empty"),
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
        "_prepared_frame_for_validation",
        lambda *args, **kwargs: pytest.fail("data must not be prepared"),
    )
    monkeypatch.setattr(
        web, "run_xgb_analysis", lambda **kwargs: pytest.fail("service must not be called")
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
    data = pd.DataFrame({"target": [1.0], "x": [2.0]})
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(web, "_prepared_frame_for_validation", lambda config: data)
    monkeypatch.setattr(
        web,
        "run_xgb_analysis",
        lambda **kwargs: XGBRunResult(status, (), None, None, None, "xgb error"),
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
            "load": np.arange(30, dtype=float),
            "x": np.arange(30, dtype=float) * 3,
        }
    )
    config = _validation_config(
        tmp_path,
        frame,
        segment_column="load",
        segment_mode="custom",
        segment_min=10,
        segment_max=20,
    )

    prepared = web._prepared_frame_for_validation(config)

    assert prepared["load"].min() == 10
    assert prepared["load"].max() == 20
    assert len(prepared) == 11


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
        assert len(prepared) == len(frame)
        assert not prepared["target"].equals(
            pd.Series(frame["target"].to_numpy(), index=pd.to_datetime(frame["timestamp"]))
        )


def test_xgb_api_uses_shared_prepared_frame_instead_of_raw_loader():
    source = inspect.getsource(web._run_xgb_validation_response)

    assert "_prepared_frame_for_validation(config)" in source
    assert "load_timeseries_csv" not in source


def test_xgb_web_surface_and_architecture_guards():
    source = inspect.getsource(web)
    assert "/api/run_xgb_validation" in web.INDEX_HTML
    for marker in [
        'id="enableXgbValidation"', 'id="xgbTopN"', 'id="xgbMaxLag"',
        'id="xgbWhitelist"', 'id="runXgbValidation"', 'id="xgbModelSummaryTable"',
        'id="xgbCandidateUpliftTable"', "xgb_validation/xgb_model_summary.csv",
        "xgb_validation/xgb_candidate_uplift.csv", "xgb_validation/xgb_validation_summary.json",
    ]:
        assert marker in web.INDEX_HTML or marker in web.DOWNLOAD_FILES
    for forbidden in [
        "XGBRegressor", "import xgboost", ".fit(", ".predict(",
        "build_xgb_feature_sets", "final_score =", "driver_rank =",
    ]:
        assert forbidden not in source
    assert "from chem_ts_corr.xgb_runner" not in source
    assert "run_xgb_analysis" in source
    assert service.run_xgb_analysis is not None
    assert "XGB 结果表示时间外预测增量，不代表工艺因果成立，也不改变前三层排名。" in web.INDEX_HTML
    assert 'renderXgbDownloads(data.status === "success" ?' in web.INDEX_HTML
