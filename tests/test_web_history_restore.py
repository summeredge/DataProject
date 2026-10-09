from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import web, xgb_runner
from chem_ts_corr.history import query_history
from chem_ts_corr.pipeline import confirm_initial_screening_branch, run_initial_screening_workflow


def snapshot(directory):
    return {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_restore_panel_keeps_wide_results_inside_existing_table_scroll():
    assert "#overviewTab, #overviewTab > div { min-width:0; }" in web.INDEX_HTML
    assert "#restoredAnalysis { overflow-wrap:anywhere; }" in web.INDEX_HTML


@pytest.fixture
def saved_runs(tmp_path, monkeypatch):
    uploads, runs = tmp_path / "uploads", tmp_path / "runs"
    uploads.mkdir()
    runs.mkdir()
    file_id = "a" * 32
    source = uploads / f"{file_id}.csv"
    source.write_text("time,target,feature,unused\n" + "\n".join(
        f"2025-01-01 {i // 60:02d}:{i % 60:02d}:00,{math.sin(i / 5)},{math.sin((i - 1) / 5)},{i}"
        for i in range(180)
    ), encoding="utf-8")
    source.with_suffix(".json").write_text(json.dumps({"original_filename": "装置.csv"}), encoding="utf-8")
    monkeypatch.setattr(web, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(web, "RUNS_DIR", runs)
    monkeypatch.setattr(web, "EXCLUDE_WINDOW_CONTEXTS", {})
    monkeypatch.setattr(web, "TASKS", {})

    def create(mode="raw", run_id="b" * 32):
        config = web.AnalysisConfig(input_path=source, time_column="time", target="target",
            output_dir=runs / run_id, max_lag=2, top_k=1, min_valid_ratio=0.8,
            preprocess_mode=mode, lowpass_tau_minutes=2.0, diff_interval_minutes=1,
            excluded_columns=["unused"], force_include_variables=["feature"],
            exclude_windows=[{"start": "2025-01-01T00:00:00", "end": "2025-01-01T00:02:00"}],
            skip_model_lift=True, skip_rolling_corr=True)
        web._write_run_config(config.output_dir, config, file_id)
        run_initial_screening_workflow(config)
        return config
    return create, file_id, source, runs


@pytest.mark.parametrize("mode", ["raw", "lowpass"])
def test_restore_parameters_context_and_outputs_are_read_only(saved_runs, monkeypatch, mode):
    create, file_id, _, runs = saved_runs
    config = create(mode)
    before = snapshot(runs)
    monkeypatch.setattr(web, "run_initial_screening_workflow", lambda *a, **k: pytest.fail("restore must not screen"))
    payload = web._restore_run_payload(config.output_dir.name)
    info = payload["restoration"]
    parameters = asdict(config)
    for field in ["input_path", "output_dir", "roles_path"]:
        parameters.pop(field)
    assert info["parameters"] == parameters
    assert info["file_id"] == file_id
    assert info["original_filename"] == "装置.csv"
    assert info["input_available"] is True
    assert info["excludeWindows"] == config.exclude_windows
    assert web.EXCLUDE_WINDOW_CONTEXTS[(file_id, "time")]["exclude_windows"] == config.exclude_windows
    assert payload["branchSelectionStatus"] == ("not_required" if mode == "raw" else "awaiting_confirmation")
    assert bool(payload.get("rankedFeatures")) == (mode == "raw")
    assert payload["run_id"] == config.output_dir.name
    assert snapshot(runs) == before
    assert web.TASKS == {}


def test_restore_confirmed_locked_and_continue_existing_run(saved_runs, monkeypatch):
    create, _, _, runs = saved_runs
    config = create("lowpass")
    confirm_initial_screening_branch(config.output_dir, branch="raw")
    original = (config.output_dir / "ranked_features.csv").read_bytes()
    payload = web._restore_run_payload(config.output_dir.name)
    assert payload["activeScreeningBranch"] == "raw"
    assert payload["selectedPreprocessingMode"] == "lowpass"
    assert payload["activePreprocessingMode"] == "raw"
    assert payload["branchLocked"] is False
    monkeypatch.setattr(web, "_multipart_form", lambda handler: {"run_id": config.output_dir.name})
    enhanced = web._run_enhanced_screening_response(object())
    assert enhanced["branchLocked"] is True
    granger = web._run_granger_response(object())
    assert granger["branchLocked"] is True
    before = snapshot(runs)
    restored = web._restore_run_payload(config.output_dir.name)
    assert restored["branchLocked"] is True
    assert restored["branchSelectionStatus"] == "confirmed"
    assert restored["enhancedValidationSummary"] == enhanced["enhancedValidationSummary"]
    assert restored["grangerTests"] == granger["grangerTests"]
    assert len(list(runs.iterdir())) == 1
    assert (config.output_dir / "ranked_features.csv").read_bytes() == original
    assert snapshot(runs) == before


def test_missing_source_still_restores_saved_results_and_downloads(saved_runs):
    create, _, source, runs = saved_runs
    config = create()
    before = snapshot(runs)
    source.unlink()
    payload = web._restore_run_payload(config.output_dir.name)
    assert payload["rankedFeatures"]
    assert payload["restoration"]["input_available"] is False
    assert "原始数据缺失" in payload["restoration"]["input_error"]
    assert payload["restoration"]["original_filename"] == "装置.csv"
    assert payload["restoration"]["excludeWindowStats"]["exclude_window_count"] == 1
    assert payload["restoration"]["excludeWindowStats"]["excluded_rows"] == 3
    assert query_history(web.UPLOADS_DIR, runs)["analyses"][0]["original_filename"] == "装置.csv"
    assert any(item["name"] == "ranked_features.csv" for item in payload["downloads"])
    assert snapshot(runs) == before


def test_saved_review_and_xgb_historical_success_survive_failed_attempt(saved_runs, monkeypatch):
    from test_pr12_active_branch_xgb import _fake_xgb_regressor

    create, _, source, runs = saved_runs
    index = pd.date_range("2025-01-01", periods=900, freq="min")
    values = np.arange(900)
    pd.DataFrame({"time": index, "target": np.sin(values / 5),
                  "feature": np.sin((values - 1) / 5), "unused": values}).to_csv(source, index=False)
    config = create()
    directory = config.output_dir
    original = (directory / "ranked_features.csv").read_bytes()
    for name in ["conditional_granger_scores.csv", "causal_review_report.csv", "causal_review_evidence.csv", "final_review_summary.csv"]:
        pd.DataFrame([{"variable": "feature", "status": "not_computed", "final_recommendation": "priority_review"}]).to_csv(directory / name, index=False)
    matrix = {column: "not_computed" for column in web.EVIDENCE_MATRIX_COLUMNS}
    matrix.update(variable="feature", final_score=None)
    pd.DataFrame([matrix]).to_csv(directory / "evidence_matrix.csv", index=False)
    monkeypatch.setattr(xgb_runner, "XGBRegressor", _fake_xgb_regressor())
    monkeypatch.setattr(web, "_multipart_form", lambda handler: {
        "run_id": directory.name, "enable_xgb_validation": "true", "top_n": "1", "max_lag": "2",
    })
    result = web._run_xgb_validation_response(object())
    assert result["status"] == "success", result
    xgb_runner.record_xgb_execution(directory, "failed", "failed later attempt")
    before = snapshot(runs)
    restored = web._restore_run_payload(directory.name)
    assert restored["xgbResult"]["status"] == "failed"
    assert restored["xgbResult"]["historicalResult"] is True
    assert restored["xgbResult"]["xgbModelSummary"] == result["xgbModelSummary"]
    assert restored["finalReviewSummary"][0]["variable"] == "feature"
    assert restored["evidenceMatrix"][0]["final_score"] is None
    assert restored["evidenceMatrix"][0]["xgb_status"] == "not_computed"
    assert snapshot(runs) == before
    assert (directory / "ranked_features.csv").read_bytes() == original


@pytest.mark.parametrize("broken", ["run_config.json", "ranked_features.csv", "summary.md"])
def test_broken_run_does_not_block_another_record(saved_runs, broken):
    create, _, _, runs = saved_runs
    bad = create()
    healthy = create(run_id="c" * 32)
    path = bad.output_dir / broken
    if broken == "run_config.json":
        path.write_bytes(b"bad json")
    else:
        path.unlink()
    before = snapshot(runs)
    with pytest.raises(ValueError, match="运行配置无法恢复|初筛结果缺失"):
        web._restore_run_payload(bad.output_dir.name)
    assert web._restore_run_payload(healthy.output_dir.name)["run_id"] == healthy.output_dir.name
    assert snapshot(runs) == before


def test_incomplete_stage_does_not_become_zero_or_success(saved_runs):
    create, _, _, _ = saved_runs
    config = create()
    (config.output_dir / "granger_tests.csv").write_bytes(b"")
    (config.output_dir / "model_variable_importance.csv").write_text("variable,importance\nfeature,0.2\n")
    (config.output_dir / "xgb_execution_state.json").write_text(json.dumps({"status": "failed", "error_message": "failed attempt"}))
    payload = web._restore_run_payload(config.output_dir.name)
    assert "granger" in payload["restoration"]["stage_issues"]
    assert "model" in payload["restoration"]["stage_issues"]
    assert payload["grangerTests"] == []
    assert payload["xgbResult"]["status"] == "failed"
    assert payload["xgbResult"]["historicalResult"] is False


def test_legacy_missing_context_is_viewable_but_not_continuable(saved_runs):
    create, _, _, runs = saved_runs
    config = create()
    (config.output_dir / "preprocessing_context.json").unlink()
    before = snapshot(runs)
    payload = web._restore_run_payload(config.output_dir.name)
    assert payload["rankedFeatures"]
    assert payload["restoration"]["input_available"] is True
    assert payload["restoration"]["context_available"] is False
    assert payload["branchSelectionStatus"] is None
    assert snapshot(runs) == before


def test_http_restore_after_server_restart_needs_no_task_memory(saved_runs):
    create, _, _, runs = saved_runs
    config = create()
    before = snapshot(runs)
    responses = []
    for _ in range(2):
        server = ThreadingHTTPServer(("127.0.0.1", 0), web._Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/api/result?run_id={config.output_dir.name}&restore=1") as response:
                responses.append(json.load(response))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            web.TASKS.clear()
            web.EXCLUDE_WINDOW_CONTEXTS.clear()
    assert responses[0] == responses[1]
    assert snapshot(runs) == before
