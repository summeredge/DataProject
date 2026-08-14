from __future__ import annotations

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


def _trend_params(file_id: str) -> dict[str, list[str]]:
    values = {
        "file_id": file_id,
        "encoding": "utf-8-sig",
        "time_column": "time",
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
