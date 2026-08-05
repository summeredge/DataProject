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


def _write_multi_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    periods: int,
    frequency: str,
    variable_count: int,
) -> tuple[str, pd.DatetimeIndex]:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    file_id = "0123456789abcdef0123456789abcdef"
    times = pd.date_range("2025-01-01", periods=periods, freq=frequency)
    frame = {"time": times}
    for index in range(1, variable_count + 1):
        frame[f"v{index}"] = [value + index * 1000 for value in range(periods)]
    pd.DataFrame(frame).to_csv(
        uploads / f"{file_id}.csv", index=False, encoding="utf-8-sig"
    )
    monkeypatch.setattr(web, "UPLOADS_DIR", uploads)
    return file_id, times


def _multi_variables(count: int) -> str:
    return ",".join(f"v{index}" for index in range(1, count + 1))


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


def test_auto_range_keeps_start_and_uses_max_points_times_one_minute_interval(
    tmp_path, monkeypatch
):
    file_id, times = _write_upload(
        tmp_path, monkeypatch, periods=10 * 24 * 60, frequency="1min"
    )

    metadata = web._columns_response(file_id, "utf-8-sig")
    response = web._trend_response(_params(file_id, 5000))
    plotted_times = _series_times(response)

    assert metadata["trendStartDefault"] == "2025-01-01T00:00"
    assert metadata["trendEndDefault"] == "2025-01-04T00:00"
    assert metadata["trendSamplingIntervalMs"] == 60_000
    assert plotted_times[0] == times[0]
    assert plotted_times[-1] == times[0] + pd.Timedelta(minutes=4999)
    assert response["raw_rows"] == 5000
    assert response["rows"] == 5000


def test_increasing_max_points_keeps_auto_start_and_moves_end_forward(tmp_path, monkeypatch):
    file_id, times = _write_upload(
        tmp_path, monkeypatch, periods=10 * 24 * 60, frequency="1min"
    )

    smaller = _series_times(web._trend_response(_params(file_id, 5000)))
    larger = _series_times(web._trend_response(_params(file_id, 20000)))

    assert smaller[0] == larger[0] == times[0]
    assert larger[-1] > smaller[-1]
    assert larger[-1] == times[-1]
    assert web._trend_time_bounds(
        times,
        times[0],
        times[0] + pd.Timedelta(days=3),
        max_points=20000,
        mode="auto",
    ) == (times[0], times[-1])


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

    assert plotted_times[0] == times[0]
    assert plotted_times[-1] == times[0] + pd.Timedelta(minutes=4995)
    assert response["raw_rows"] == 1000


def test_auto_range_falls_back_to_existing_bounds_without_sampling_interval():
    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2025-01-04")

    assert web._trend_time_bounds(
        pd.RangeIndex(12), start, end, max_points=5000, mode="auto"
    ) == (start, end)


def test_trend_response_accepts_eight_variables(tmp_path, monkeypatch):
    file_id, _ = _write_multi_upload(
        tmp_path, monkeypatch, periods=120, frequency="1min", variable_count=8
    )

    response = web._trend_response(
        _params(file_id, 100, variables=_multi_variables(8))
    )

    assert [series["name"] for series in response["series"]] == [
        f"v{index}" for index in range(1, 9)
    ]


def test_trend_response_rejects_nine_variables(tmp_path, monkeypatch):
    file_id, _ = _write_multi_upload(
        tmp_path, monkeypatch, periods=120, frequency="1min", variable_count=9
    )

    with pytest.raises(ValueError, match="^最多选择 8 个趋势变量$"):
        web._trend_response(_params(file_id, 100, variables=_multi_variables(9)))


def test_eight_variables_30241_points_keep_all_time_points(tmp_path, monkeypatch):
    file_id, _ = _write_multi_upload(
        tmp_path, monkeypatch, periods=30241, frequency="1min", variable_count=8
    )

    response = web._trend_response(
        _params(file_id, 30241, variables=_multi_variables(8))
    )

    assert response["rows"] == 30241
    assert response["max_points"] == 30241
    for series in response["series"]:
        assert len(series["points"]) == 30241


def test_eight_variables_50000_points_are_capped_at_37500(tmp_path, monkeypatch):
    file_id, _ = _write_multi_upload(
        tmp_path, monkeypatch, periods=50000, frequency="1min", variable_count=8
    )

    response = web._trend_response(
        _params(file_id, 50000, variables=_multi_variables(8))
    )

    assert response["rows"] == 37500
    assert response["max_points"] == 37500
    for series in response["series"]:
        assert len(series["points"]) == 37500


def test_four_variables_still_allow_50000_points(tmp_path, monkeypatch):
    file_id, _ = _write_multi_upload(
        tmp_path, monkeypatch, periods=50000, frequency="1min", variable_count=4
    )

    response = web._trend_response(
        _params(file_id, 50000, variables=_multi_variables(4))
    )

    assert response["rows"] == 50000
    assert response["max_points"] == 50000
    for series in response["series"]:
        assert len(series["points"]) == 50000


def test_sampled_curves_share_identical_time_coordinates(tmp_path, monkeypatch):
    file_id, _ = _write_multi_upload(
        tmp_path, monkeypatch, periods=50000, frequency="1min", variable_count=8
    )

    response = web._trend_response(
        _params(file_id, 50000, variables=_multi_variables(8))
    )

    first_times = [point["x"] for point in response["series"][0]["points"]]
    assert len(first_times) == 37500
    for series in response["series"][1:]:
        assert [point["x"] for point in series["points"]] == first_times


def test_duplicate_trend_variables_return_single_series(tmp_path, monkeypatch):
    file_id, _ = _write_multi_upload(
        tmp_path, monkeypatch, periods=120, frequency="1min", variable_count=2
    )

    response = web._trend_response(_params(file_id, 100, variables="v1,v2,v1"))

    assert [series["name"] for series in response["series"]] == ["v1", "v2"]


def test_frontend_tracks_auto_and_manual_trend_time_modes():
    assert 'let trendTimeRangeMode = "auto";' in INDEX_HTML
    assert "let trendAutoWindowActive = false;" in INDEX_HTML
    assert 'id="trendMaxPoints" type="number" min="100" max="100000" value="10000"' in INDEX_HTML
    assert 'el("trendStart").addEventListener("input", markTrendTimeRangeManual);' in INDEX_HTML
    assert 'el("trendEnd").addEventListener("input", markTrendTimeRangeManual);' in INDEX_HTML
    assert 'el("trendMaxPoints").addEventListener("change", updateAutoTrendTimeRange);' in INDEX_HTML
    load_columns_body = INDEX_HTML.split("async function loadColumns()", 1)[1].split(
        "async function analyze()", 1
    )[0]
    assert 'el("trendMaxPoints"' not in load_columns_body
    assert "updateAutoTrendTimeRange();" not in load_columns_body
    auto_range_body = INDEX_HTML.split("function updateAutoTrendTimeRange()", 1)[1].split(
        "function appendChartQueryParams", 1
    )[0]
    assert 'el("trendEnd").value =' in auto_range_body
    assert 'el("trendStart").value =' not in auto_range_body
    assert "(maxPoints - 1) * trendSamplingIntervalMs" in auto_range_body
    assert "calculatedEnd > latest ? latest : calculatedEnd" in auto_range_body
    draw_body = INDEX_HTML.split("async function drawTrend()", 1)[1].split(
        "async function drawScatterMatrix()", 1
    )[0]
    assert 'params.set("time_range_mode", trendAutoWindowActive ? "auto" : "manual");' in draw_body
