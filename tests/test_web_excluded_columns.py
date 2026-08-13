from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.web import INDEX_HTML


def _write_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    file_id = "0123456789abcdef0123456789abcdef"
    pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=20, freq="h"),
            "target": range(20),
            "keep": range(20),
            "drop_me": range(20),
            "load": range(20),
        }
    ).to_csv(uploads / f"{file_id}.csv", index=False, encoding="utf-8-sig")
    monkeypatch.setattr(web, "UPLOADS_DIR", uploads)
    return file_id


def _params(file_id: str, **overrides: str) -> dict[str, list[str]]:
    values = {
        "file_id": file_id,
        "encoding": "utf-8-sig",
        "time_column": "time",
        "trend_max_points": "10000",
        "segment_mode": "all",
        "preprocess_mode": "raw",
        "detrend_window": "24",
        "excluded_columns": "drop_me",
    }
    values.update(overrides)
    return parse_qs(
        "&".join(f"{key}={value}" for key, value in values.items()),
        keep_blank_values=True,
    )


def test_excluded_columns_control_is_between_input_and_base_parameters():
    input_position = INDEX_HTML.index('<div class="control-group-title">数据输入</div>')
    exclusion_position = INDEX_HTML.index('<div class="control-group-title">数据剔除</div>')
    base_position = INDEX_HTML.index('<div class="control-group-title">基础分析参数</div>')

    assert input_position < exclusion_position < base_position
    for token in [
        "excludedColumnsDropdown",
        "excludedColumnsSummary",
        "excludedColumnsOptions",
        "强制剔除列（多选）",
        "原始上传文件不会被修改",
    ]:
        assert token in INDEX_HTML


def test_frontend_submits_filters_and_resets_excluded_columns():
    load_columns = INDEX_HTML.split("async function loadColumns()", 1)[1].split(
        "async function analyze()", 1
    )[0]
    analyze = INDEX_HTML.split("async function analyze()", 1)[1].split(
        "async function waitForAnalysisResult", 1
    )[0]
    chart_params = INDEX_HTML.split("function appendChartQueryParams", 1)[1].split(
        "async function drawTrend", 1
    )[0]
    reset = INDEX_HTML.split("function reset()", 1)[1].split("</script>", 1)[0]

    assert "fillExcludedColumnOptions" in load_columns
    assert 'form.append("excluded_columns", getExcludedColumnSelection().join(","))' in analyze
    assert 'params.set("excluded_columns", getExcludedColumnSelection().join(","))' in chart_params
    assert "refreshColumnSelectors" in INDEX_HTML
    assert "recognizedColumns" in INDEX_HTML
    assert "recognizedNumericColumns" in INDEX_HTML
    assert "excludedColumnsOptions" in reset
    assert "未选择剔除列" in reset


def test_frontend_disables_time_and_target_and_filters_other_selectors():
    options = INDEX_HTML.split(
        "function updateExcludedColumnDisabledState", 1
    )[1].split("async function uploadFile", 1)[0]

    assert 'el("timeColumn").value' in options
    assert 'el("targetColumn").value' in options
    assert "input.disabled" in options
    for token in [
        "segmentColumn",
        "fillCapacityOptions",
        "fillForceIncludeOptions",
        "trendVar1",
        "trendVar4",
        "scatterX1",
        "scatterY3",
    ]:
        assert token in options


def test_trend_and_scatter_reject_excluded_variables(tmp_path: Path, monkeypatch):
    file_id = _write_upload(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="已剔除列不能用于图表：drop_me"):
        web._trend_response(_params(file_id, variables="drop_me"))
    with pytest.raises(ValueError, match="已剔除列不能用于图表：drop_me"):
        web._scatter_matrix_response(
            _params(file_id, x_variables="keep", y_variables="drop_me")
        )


def test_charts_still_return_non_excluded_data(tmp_path: Path, monkeypatch):
    file_id = _write_upload(tmp_path, monkeypatch)

    trend = web._trend_response(_params(file_id, variables="keep"))
    scatter = web._scatter_matrix_response(
        _params(file_id, x_variables="keep", y_variables="target")
    )

    assert [series["name"] for series in trend["series"]] == ["keep"]
    assert scatter["columns"] == ["keep", "target"]
    assert "drop_me" not in scatter["columns"]


def test_chart_rejects_excluded_segment_and_unknown_columns(tmp_path: Path, monkeypatch):
    file_id = _write_upload(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="剔除列不能同时作为工况列：load"):
        web._trend_response(
            _params(
                file_id,
                variables="keep",
                excluded_columns="load",
                segment_column="load",
            )
        )
    with pytest.raises(ValueError, match="剔除列不存在：missing"):
        web._trend_response(
            _params(file_id, variables="keep", excluded_columns="missing")
        )


def test_source_lists_all_three_data_read_exclusion_boundaries():
    pipeline_source = Path("chem_ts_corr/pipeline.py").read_text(encoding="utf-8")
    web_source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")

    pipeline_read = pipeline_source.split("raw = load_timeseries_csv", 1)[1].split(
        "analyze_numeric_frame", 1
    )[0]
    numeric_read = web_source.split("def _numeric_frame", 1)[1].split(
        "def _protected_validation_columns", 1
    )[0]
    chart_read = web_source.split("def _chart_frame_from_params", 1)[1].split(
        "def _trend_response", 1
    )[0]

    assert "drop_excluded_columns" in pipeline_read
    assert "drop_excluded_columns" in numeric_read
    assert "drop_excluded_columns" in chart_read
