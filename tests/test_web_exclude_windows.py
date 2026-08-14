from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.web import INDEX_HTML


def _write_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_id: str) -> pd.DataFrame:
    uploads = tmp_path / "uploads"
    uploads.mkdir(exist_ok=True)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-08-14 08:00", periods=12, freq="15min"),
            "alt_time": pd.date_range("2026-08-15 08:00", periods=12, freq="15min"),
            "target": range(12),
        }
    )
    frame.to_csv(uploads / f"{file_id}.csv", index=False, encoding="utf-8-sig")
    monkeypatch.setattr(web, "UPLOADS_DIR", uploads)
    return frame


@pytest.fixture(autouse=True)
def _clear_exclude_window_contexts():
    with web.EXCLUDE_WINDOW_CONTEXTS_LOCK:
        web.EXCLUDE_WINDOW_CONTEXTS.clear()
    yield
    with web.EXCLUDE_WINDOW_CONTEXTS_LOCK:
        web.EXCLUDE_WINDOW_CONTEXTS.clear()


def _post(monkeypatch: pytest.MonkeyPatch, response, **form: str) -> dict[str, object]:
    monkeypatch.setattr(web, "_multipart_form", lambda _handler: form)
    return response(object())


def _trend_params(file_id: str, time_column: str = "time") -> dict[str, list[str]]:
    values = {
        "file_id": file_id,
        "encoding": "utf-8-sig",
        "time_column": time_column,
        "variables": "target",
        "trend_start": "",
        "trend_end": "",
        "trend_max_points": "100",
        "time_range_mode": "auto",
        "segment_mode": "all",
        "preprocess_mode": "raw",
        "detrend_window": "24",
        "excluded_columns": "",
    }
    return {key: [value] for key, value in values.items()}


def test_exclude_window_api_updates_trend_payload_and_preserves_uploaded_data(tmp_path, monkeypatch):
    file_id = "a" * 32
    source = _write_upload(tmp_path, monkeypatch, file_id)
    before = web.load_timeseries_csv(web._resolve_upload(file_id), "time")

    added = _post(
        monkeypatch,
        web._exclude_window_response,
        file_id=file_id,
        time_column="time",
        encoding="utf-8-sig",
        start="2026-08-14T08:00:00",
        end="2026-08-14T09:00:00",
    )
    trend = web._trend_response(_trend_params(file_id))

    assert added["excludeWindows"] == [
        {"start": "2026-08-14T08:00:00", "end": "2026-08-14T09:00:00"}
    ]
    assert added["excludeWindowStats"] == {
        "original_rows": 12,
        "excluded_rows": 5,
        "remaining_rows": 7,
        "excluded_ratio": 5 / 12,
        "exclude_window_count": 1,
    }
    assert trend["excludeWindows"] == added["excludeWindows"]
    assert trend["excludeWindowStats"] == added["excludeWindowStats"]
    assert len(trend["series"][0]["points"]) == len(source)
    pd.testing.assert_frame_equal(web.load_timeseries_csv(web._resolve_upload(file_id), "time"), before)


def test_overlapping_windows_restore_one_and_restore_all(tmp_path, monkeypatch):
    file_id = "b" * 32
    _write_upload(tmp_path, monkeypatch, file_id)
    base = {"file_id": file_id, "time_column": "time", "encoding": "utf-8-sig"}
    _post(
        monkeypatch,
        web._exclude_window_response,
        **base,
        start="2026-08-14T08:00:00",
        end="2026-08-14T09:00:00",
    )
    added = _post(
        monkeypatch,
        web._exclude_window_response,
        **base,
        start="2026-08-14T08:30:00",
        end="2026-08-14T09:30:00",
    )

    assert added["excludeWindowStats"]["exclude_window_count"] == 2
    assert added["excludeWindowStats"]["excluded_rows"] == 7

    restored = _post(
        monkeypatch,
        web._restore_exclude_window_response,
        file_id=file_id,
        time_column="time",
        index="0",
    )
    assert restored["excludeWindows"] == [
        {"start": "2026-08-14T08:30:00", "end": "2026-08-14T09:30:00"}
    ]
    assert restored["excludeWindowStats"]["excluded_rows"] == 5

    cleared = _post(
        monkeypatch,
        web._restore_all_exclude_windows_response,
        file_id=file_id,
        time_column="time",
    )
    assert cleared["excludeWindows"] == []
    assert cleared["excludeWindowStats"] == {
        "original_rows": 12,
        "excluded_rows": 0,
        "remaining_rows": 12,
        "excluded_ratio": 0.0,
        "exclude_window_count": 0,
    }


def test_new_upload_context_does_not_inherit_exclude_windows(tmp_path, monkeypatch):
    first_id = "c" * 32
    second_id = "d" * 32
    _write_upload(tmp_path, monkeypatch, first_id)
    _write_upload(tmp_path, monkeypatch, second_id)
    _post(
        monkeypatch,
        web._exclude_window_response,
        file_id=first_id,
        time_column="time",
        encoding="utf-8-sig",
        start="2026-08-14T08:00:00",
        end="2026-08-14T08:15:00",
    )

    second_trend = web._trend_response(_trend_params(second_id))

    assert second_trend["excludeWindows"] == []
    assert second_trend["excludeWindowStats"]["excluded_rows"] == 0


def test_exclude_window_context_cache_evicts_oldest_and_keeps_latest(tmp_path, monkeypatch):
    file_ids = [f"{index:032x}" for index in range(web.MAX_EXCLUDE_WINDOW_CONTEXTS + 1)]
    for file_id in file_ids:
        _write_upload(tmp_path, monkeypatch, file_id)
        web._trend_response(_trend_params(file_id))

    assert len(web.EXCLUDE_WINDOW_CONTEXTS) == web.MAX_EXCLUDE_WINDOW_CONTEXTS
    assert (file_ids[0], "time") not in web.EXCLUDE_WINDOW_CONTEXTS
    assert (file_ids[-1], "time") in web.EXCLUDE_WINDOW_CONTEXTS
    assert web._resolve_upload(file_ids[0]).exists()

    latest = _post(
        monkeypatch,
        web._exclude_window_response,
        file_id=file_ids[-1],
        time_column="time",
        encoding="utf-8-sig",
        start="2026-08-14T08:00:00",
        end="2026-08-14T08:15:00",
    )
    trend = web._trend_response(_trend_params(file_ids[-1]))

    assert latest["excludeWindowStats"]["excluded_rows"] == 2
    assert trend["excludeWindows"] == latest["excludeWindows"]


def test_exclude_window_contexts_are_isolated_by_time_column(tmp_path, monkeypatch):
    file_id = "e" * 32
    _write_upload(tmp_path, monkeypatch, file_id)
    _post(
        monkeypatch,
        web._exclude_window_response,
        file_id=file_id,
        time_column="time",
        encoding="utf-8-sig",
        start="2026-08-14T08:00:00",
        end="2026-08-14T08:15:00",
    )

    alternate = web._trend_response(_trend_params(file_id, time_column="alt_time"))
    original = web._trend_response(_trend_params(file_id))

    assert alternate["excludeWindows"] == []
    assert len(original["excludeWindows"]) == 1


def test_exclude_window_apis_do_not_trigger_or_change_formal_analysis(tmp_path, monkeypatch):
    file_id = "f" * 32
    _write_upload(tmp_path, monkeypatch, file_id)
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run"
    run_dir.mkdir(parents=True)
    ranked_path = run_dir / "ranked_features.csv"
    ranked = pd.DataFrame(
        {"variable": ["A", "B"], "final_score": [0.8, 0.5], "driver_rank": [1, 2]}
    )
    ranked.to_csv(ranked_path, index=False, encoding="utf-8-sig")
    run_config_path = run_dir / "run_config.json"
    context_path = run_dir / "preprocessing_context.json"
    run_config_path.write_text('{"exclude_windows": []}\n', encoding="utf-8")
    context_path.write_text('{"exclude_windows": []}\n', encoding="utf-8")
    before = {
        path: path.read_bytes()
        for path in (ranked_path, run_config_path, context_path)
    }
    monkeypatch.setattr(web, "RUNS_DIR", runs_dir)
    calls: list[str] = []

    def unexpected_runner(*_args, **_kwargs):
        calls.append("runner")
        raise AssertionError("排除窗口操作不得启动分析")

    for name in (
        "_analyze_response",
        "run_initial_screening_workflow",
        "run_enhanced_screening_for_active_branch",
        "run_granger_for_active_branch",
        "run_model_for_active_branch",
        "run_causal_review_for_active_branch",
        "run_xgb_for_active_branch",
    ):
        monkeypatch.setattr(web, name, unexpected_runner)

    base = {"file_id": file_id, "time_column": "time", "encoding": "utf-8-sig"}
    _post(
        monkeypatch,
        web._exclude_window_response,
        **base,
        start="2026-08-14T08:00:00",
        end="2026-08-14T08:30:00",
    )
    _post(
        monkeypatch,
        web._restore_exclude_window_response,
        file_id=file_id,
        time_column="time",
        index="0",
    )
    _post(
        monkeypatch,
        web._exclude_window_response,
        **base,
        start="2026-08-14T09:00:00",
        end="2026-08-14T09:30:00",
    )
    _post(
        monkeypatch,
        web._restore_all_exclude_windows_response,
        file_id=file_id,
        time_column="time",
    )
    web._trend_response(_trend_params(file_id))

    assert calls == []
    assert {path: path.read_bytes() for path in before} == before
    pd.testing.assert_frame_equal(pd.read_csv(ranked_path, encoding="utf-8-sig"), ranked)
    assert list(runs_dir.iterdir()) == [run_dir]


def test_analyze_snapshots_exclude_windows_into_run_config(tmp_path, monkeypatch):
    file_id = "1" * 32
    _write_upload(tmp_path, monkeypatch, file_id)
    _post(
        monkeypatch,
        web._exclude_window_response,
        file_id=file_id,
        time_column="time",
        encoding="utf-8-sig",
        start="2026-08-14T08:00:00",
        end="2026-08-14T08:15:00",
    )
    threads: list[object] = []

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            self.args = args
            threads.append(self)

        def start(self):
            pass

    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda _handler: {
            "file_id": file_id,
            "time_column": "time",
            "target": "target",
            "preprocess_mode": "raw",
            "resample_rule": "",
        },
    )
    monkeypatch.setattr(web.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        web, "_validate_analysis_excluded_columns", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(web, "_cleanup_tasks_locked", lambda **_kwargs: None)

    response = web._analyze_response(object())

    config = threads[0].args[1]
    expected = [{"start": "2026-08-14T08:00:00", "end": "2026-08-14T08:15:00"}]
    assert config.exclude_windows == expected
    web._write_run_config(tmp_path / "run", config, file_id)
    assert json.loads((tmp_path / "run" / "run_config.json").read_text(encoding="utf-8"))["exclude_windows"] == expected
    with web.TASKS_LOCK:
        web.TASKS.pop(response["task_id"], None)


def test_exclude_window_ui_reuses_trend_selection_and_renders_background_markers():
    selection_source = INDEX_HTML.split("function updateTrendSelectionInfo()", 1)[1].split(
        "function setTrendWindowFromSelection", 1
    )[0]
    add_source = INDEX_HTML.split("async function addExcludeWindow()", 1)[1].split(
        "async function restoreExcludeWindow", 1
    )[0]
    chart_source = INDEX_HTML.split("function renderTrendChart", 1)[1].split(
        "function trendChartWidth", 1
    )[0]

    assert 'id="addExcludeWindow"' in INDEX_HTML
    assert 'id="restoreAllExcludeWindows"' in INDEX_HTML
    assert "trendSelection.start" in add_source
    assert "trendSelection.end" in add_source
    assert "let excludeWindowSelection" not in INDEX_HTML
    assert 'data-exclude-window' in chart_source
    assert 'postForm("/api/exclude_window", form)' in add_source
    assert 'postForm("/api/restore_exclude_window", form)' in INDEX_HTML
    assert 'postForm("/api/restore_all_exclude_windows", form)' in INDEX_HTML
    assert 'addButton.disabled = !trendSelection;' in selection_source
