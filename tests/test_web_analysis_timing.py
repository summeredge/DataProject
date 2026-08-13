from types import SimpleNamespace

import pytest
import pandas as pd

import chem_ts_corr.pipeline as pipeline
import chem_ts_corr.web as web
from chem_ts_corr.config import AnalysisConfig


@pytest.fixture(autouse=True)
def clear_tasks():
    with web.TASKS_LOCK:
        web.TASKS.clear()
    yield
    with web.TASKS_LOCK:
        web.TASKS.clear()


def test_run_analysis_returns_deterministic_stage_timings(monkeypatch, tmp_path):
    tables = SimpleNamespace(
        ranked_features=None,
        lag_scores=None,
        granger_tests=None,
        importance=None,
        metrics=None,
        diagnostics=None,
        residual_corr_scores=None,
        regime_scores=None,
        risk_flags=None,
        model_lift_scores=None,
        lag_peak_quality=None,
        rolling_corr_scores=None,
    )
    monkeypatch.setattr(
        pipeline, "load_timeseries_csv", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_numeric_frame",
        lambda *args, **kwargs: tables,
    )
    monkeypatch.setattr(pipeline, "write_outputs", lambda *args, **kwargs: None)
    clock = iter([0.0, 1.0, 3.0, 4.0, 8.0, 9.0, 12.0, 15.0])
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: next(clock))
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path / "out")

    timings = pipeline.run_analysis(config)

    assert timings == {
        "read_data_seconds": 2.0,
        "analysis_core_seconds": 4.0,
        "write_outputs_seconds": 3.0,
        "pipeline_total_seconds": 15.0,
    }
    assert all(value >= 0 for value in timings.values())
    assert timings["pipeline_total_seconds"] >= max(
        timings["read_data_seconds"],
        timings["analysis_core_seconds"],
        timings["write_outputs_seconds"],
    )


def test_analyze_task_adds_authoritative_total_and_payload_timing(monkeypatch, tmp_path):
    task_id = "task-1"
    output_dir = tmp_path / "run-1"
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", output_dir)
    pipeline_timings = {
        "read_data_seconds": 1.0,
        "analysis_core_seconds": 4.0,
        "write_outputs_seconds": 1.0,
        "pipeline_total_seconds": 6.0,
    }
    with web.TASKS_LOCK:
        web.TASKS.clear()
        web.TASKS[task_id] = {
            "status": "running",
            "start_time": 100.0,
            "created_at": 100.0,
            "updated_at": 100.0,
        }

    monkeypatch.setattr(web, "_write_run_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        web,
        "run_initial_screening_workflow",
        lambda *args, **kwargs: {"timings": pipeline_timings},
    )
    monkeypatch.setattr(
        web,
        "_build_result_payload",
        lambda *args, **kwargs: {"run_id": "run-1", "overview": {}},
    )
    perf_clock = iter([20.0, 21.25])
    monkeypatch.setattr(web.time, "perf_counter", lambda: next(perf_clock))
    monkeypatch.setattr(web.time, "time", lambda: 110.5)
    monkeypatch.setattr(web, "_cleanup_tasks", lambda: None)

    web._analyze_task(task_id, config, "file-1")

    result = web.TASKS[task_id]["result"]
    assert result["elapsed_seconds"] == 10.5
    assert result["overview"]["analysis_elapsed_seconds"] == 10.5
    assert result["analysis_timings"] == {
        **pipeline_timings,
        "result_payload_seconds": 1.25,
        "task_total_seconds": 10.5,
    }
    assert result["analysis_timings"]["task_total_seconds"] > pipeline_timings[
        "pipeline_total_seconds"
    ]
    assert web.TASKS[task_id]["status"] == "done"


def test_completed_status_uses_backend_elapsed_time_and_has_legacy_fallback():
    body = web.INDEX_HTML.split("function formatCompletedAnalysisStatus", 1)[1].split(
        "function", 1
    )[0]

    assert "result.elapsed_seconds" in body
    assert "formatAnalysisSeconds" in body
    assert "分析完成。总耗时" in body
    assert "分析完成。运行 ID" in body
    assert "performance.now()" not in body


def test_overview_and_timing_breakdown_static_contract():
    for token in [
        "初步分析总耗时",
        "analysis_elapsed_seconds",
        "analysisTimingBreakdown",
        "read_data_seconds",
        "analysis_core_seconds",
        "write_outputs_seconds",
        "result_payload_seconds",
        "task_total_seconds",
    ]:
        assert token in web.INDEX_HTML
    formatter = web.INDEX_HTML.split("function formatAnalysisSeconds", 1)[1].split(
        "function", 1
    )[0]
    assert "finiteAnalysisSeconds" in formatter
    assert "toFixed(1)" in formatter
    assert "秒" in formatter
