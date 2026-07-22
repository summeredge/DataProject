from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.web import INDEX_HTML


def _write_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    periods: int,
    frequency: str,
) -> tuple[str, pd.DatetimeIndex]:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    file_id = "0123456789abcdef0123456789abcdef"
    times = pd.date_range("2025-01-01", periods=periods, freq=frequency)
    pd.DataFrame({"time": times, "target": range(periods)}).to_csv(
        uploads / f"{file_id}.csv", index=False, encoding="utf-8-sig"
    )
    monkeypatch.setattr(web, "UPLOADS_DIR", uploads)
    return file_id, times


def _params(file_id: str, max_points: int, **overrides: str) -> dict[str, list[str]]:
    query = {
        "file_id": file_id,
        "encoding": "utf-8-sig",
        "time_column": "time",
        "variables": "target",
        "trend_start": "",
        "trend_end": "",
        "trend_max_points": str(max_points),
        "time_range_mode": "auto",
        "segment_mode": "all",
        "preprocess_mode": "raw",
        "detrend_window": "24",
        "excluded_columns": "",
    }
    query.update(overrides)
    return {key: [value] for key, value in query.items()}


def _series_times(response: dict[str, object]) -> list[pd.Timestamp]:
    series = response["series"]
    assert isinstance(series, list)
    points = series[0]["points"]
    return [pd.Timestamp(point["x"]) for point in points]


def test_auto_range_uses_max_points_times_one_minute_interval(tmp_path, monkeypatch):
    file_id, times = _write_upload(
        tmp_path, monkeypatch, periods=10 * 24 * 60, frequency="1min"
    )

    metadata = web._columns_response(file_id, "utf-8-sig")
    response = web._trend_response(_params(file_id, 5000))
    plotted_times = _series_times(response)

    assert metadata["trendStartDefault"] == "2025-01-07T23:59"
    assert metadata["trendEndDefault"] == "2025-01-10T23:59"
    assert metadata["trendSamplingIntervalMs"] == 60_000
    assert plotted_times[0] == times[-1] - pd.Timedelta(minutes=5000)
    assert plotted_times[-1] == times[-1]
    assert response["raw_rows"] == 5001
    assert response["rows"] == 5000


def test_increasing_max_points_moves_auto_start_back_and_keeps_end(tmp_path, monkeypatch):
    file_id, times = _write_upload(
        tmp_path, monkeypatch, periods=10 * 24 * 60, frequency="1min"
    )

    smaller = _series_times(web._trend_response(_params(file_id, 5000)))
    larger = _series_times(web._trend_response(_params(file_id, 20000)))

    assert larger[0] < smaller[0]
    assert smaller[-1] == larger[-1] == times[-1]


def test_manual_range_is_unchanged_when_max_points_changes(tmp_path, monkeypatch):
    file_id, times = _write_upload(
        tmp_path, monkeypatch, periods=10 * 24 * 60, frequency="1min"
    )
    end = times[-1]
    start = end - pd.Timedelta(days=3)
    overrides = {
        "time_range_mode": "manual",
        "trend_start": start.isoformat(),
        "trend_end": end.isoformat(),
    }

    smaller = web._trend_response(_params(file_id, 5000, **overrides))
    larger = web._trend_response(_params(file_id, 20000, **overrides))

    assert _series_times(smaller)[0] == _series_times(larger)[0] == start
    assert _series_times(smaller)[-1] == _series_times(larger)[-1] == end
    assert smaller["raw_rows"] == larger["raw_rows"] == 3 * 24 * 60 + 1


def test_auto_range_uses_actual_five_minute_interval(tmp_path, monkeypatch):
    file_id, times = _write_upload(
        tmp_path, monkeypatch, periods=10 * 24 * 12, frequency="5min"
    )

    response = web._trend_response(_params(file_id, 1000))
    plotted_times = _series_times(response)

    assert plotted_times[0] == times[-1] - pd.Timedelta(minutes=5000)
    assert plotted_times[-1] == times[-1]
    assert response["raw_rows"] == 1001


def test_auto_range_falls_back_to_existing_bounds_without_sampling_interval():
    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2025-01-04")

    assert web._trend_time_bounds(
        pd.RangeIndex(12), start, end, max_points=5000, mode="auto"
    ) == (start, end)


def test_frontend_tracks_auto_and_manual_trend_time_modes():
    assert 'let trendTimeRangeMode = "auto";' in INDEX_HTML
    assert 'params.set("time_range_mode", trendTimeRangeMode);' in INDEX_HTML
    assert 'el("trendStart").addEventListener("input", markTrendTimeRangeManual);' in INDEX_HTML
    assert 'el("trendEnd").addEventListener("input", markTrendTimeRangeManual);' in INDEX_HTML
    assert 'el("trendMaxPoints").addEventListener("change", updateAutoTrendTimeRange);' in INDEX_HTML
