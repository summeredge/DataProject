from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.web import INDEX_HTML


def _write_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> str:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    file_id = "0123456789abcdef0123456789abcdef"
    frame.to_csv(uploads / f"{file_id}.csv", index=False, encoding="utf-8-sig")
    monkeypatch.setattr(web, "UPLOADS_DIR", uploads)
    return file_id


def _params(file_id: str, **overrides: str) -> dict[str, list[str]]:
    query = {
        "file_id": file_id,
        "encoding": "utf-8-sig",
        "time_column": "time",
        "x_variables": "A",
        "y_variables": "B",
        "trend_max_points": "10000",
        "segment_mode": "all",
        "preprocess_mode": "raw",
        "detrend_window": "24",
    }
    query.update(overrides)
    return parse_qs("&".join(f"{key}={value}" for key, value in query.items()), keep_blank_values=True)


def _frame(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "A": range(rows),
            "B": [value * 2 for value in range(rows)],
            "C": [value * 3 for value in range(rows)],
            "D": [value * 4 for value in range(rows)],
            "load": range(rows),
        }
    )


def test_scatter_matrix_requires_x_variables(tmp_path, monkeypatch):
    file_id = _write_upload(tmp_path, monkeypatch, _frame())
    with pytest.raises(ValueError, match="请选择至少一个 X 轴变量"):
        web._scatter_matrix_response(_params(file_id, x_variables=""))


def test_scatter_matrix_requires_y_variables(tmp_path, monkeypatch):
    file_id = _write_upload(tmp_path, monkeypatch, _frame())
    with pytest.raises(ValueError, match="请选择至少一个 Y 轴变量"):
        web._scatter_matrix_response(_params(file_id, y_variables=""))


def test_scatter_matrix_rejects_more_than_three_x_variables(tmp_path, monkeypatch):
    file_id = _write_upload(tmp_path, monkeypatch, _frame())
    with pytest.raises(ValueError, match="X 轴变量最多选择 3 个"):
        web._scatter_matrix_response(_params(file_id, x_variables="A,B,C,D"))


def test_scatter_matrix_rejects_more_than_three_y_variables(tmp_path, monkeypatch):
    file_id = _write_upload(tmp_path, monkeypatch, _frame())
    with pytest.raises(ValueError, match="Y 轴变量最多选择 3 个"):
        web._scatter_matrix_response(_params(file_id, y_variables="A,B,C,D"))


def test_scatter_matrix_deduplicates_variables_and_keeps_order(tmp_path, monkeypatch):
    file_id = _write_upload(tmp_path, monkeypatch, _frame())
    payload = web._scatter_matrix_response(_params(file_id, x_variables="A,B,A", y_variables="C,D,C"))
    assert payload["x_variables"] == ["A", "B"]
    assert payload["y_variables"] == ["C", "D"]
    assert payload["columns"] == ["A", "B", "C", "D"]


def test_scatter_matrix_returns_shared_matrix_payload(tmp_path, monkeypatch):
    file_id = _write_upload(tmp_path, monkeypatch, _frame(12))
    payload = web._scatter_matrix_response(_params(file_id, x_variables="A,C", y_variables="B,D", trend_max_points="5"))
    assert payload["x_variables"] == ["A", "C"]
    assert payload["y_variables"] == ["B", "D"]
    assert payload["columns"] == ["A", "C", "B", "D"]
    assert payload["rows"] <= payload["max_points"]
    assert len(payload["values"]) <= payload["max_points"]
    assert all(len(row) == len(payload["columns"]) for row in payload["values"])


def test_scatter_matrix_converts_nan_and_infinity_to_null(tmp_path, monkeypatch):
    frame = _frame(12).astype({"A": "float64", "B": "float64", "C": "float64"})
    frame.loc[0, "A"] = float("nan")
    frame.loc[1, "B"] = float("inf")
    frame.loc[2, "C"] = float("-inf")
    file_id = _write_upload(tmp_path, monkeypatch, frame)
    payload = web._scatter_matrix_response(_params(file_id, x_variables="A,C", y_variables="B"))
    assert payload["values"][0][payload["columns"].index("A")] is None
    assert payload["values"][1][payload["columns"].index("B")] is None
    assert payload["values"][2][payload["columns"].index("C")] is None


def test_scatter_matrix_preserves_real_zero_values(tmp_path, monkeypatch):
    frame = _frame(12).astype({"A": "float64"})
    frame.loc[0, "A"] = 0.0
    file_id = _write_upload(tmp_path, monkeypatch, frame)
    payload = web._scatter_matrix_response(_params(file_id, x_variables="A", y_variables="B"))
    assert payload["values"][0][payload["columns"].index("A")] == 0.0


def test_scatter_matrix_applies_time_segment_preprocess_and_max_points(tmp_path, monkeypatch):
    frame = _frame(150)
    file_id = _write_upload(tmp_path, monkeypatch, frame)
    payload = web._scatter_matrix_response(
        _params(
            file_id,
            x_variables="A",
            y_variables="B",
            trend_start="2024-01-01 03:00:00",
            trend_end="2024-01-06 18:00:00",
            trend_max_points="100",
            segment_column="load",
            segment_mode="custom",
            segment_min="5",
            segment_max="130",
            preprocess_mode="detrend",
            detrend_window="3",
        )
    )
    assert payload["raw_rows"] == 126
    assert payload["rows"] <= 100
    assert len(payload["values"]) <= 100
    assert payload["max_points"] == 100


def test_scatter_matrix_static_frontend_contract():
    for token in ["scatterX1", "scatterX2", "scatterX3", "scatterY1", "scatterY2", "scatterY3", "drawScatterMatrix", "scatterMatrixChart", "scatterMatrixMeta"]:
        assert token in INDEX_HTML
    assert "/api/scatter_matrix" in INDEX_HTML
    assert "x_variables" in INDEX_HTML
    assert "y_variables" in INDEX_HTML
    assert "scatterX4" not in INDEX_HTML
    assert "scatterY4" not in INDEX_HTML
    assert 'document.createElement("canvas")' in INDEX_HTML
    assert 'getContext("2d")' in INDEX_HTML
    body = INDEX_HTML.split("function renderScatterMatrix", 1)[1].split("function renderTrendChart", 1)[0]
    assert "<circle" not in body
    assert "createElementNS" not in body
    assert "Number(valueRow[xIndex])" not in body
    assert "Number(valueRow[yIndex])" not in body
    assert "function finiteScatterNumber(value)" in INDEX_HTML
    assert "value === null" in INDEX_HTML
    assert "value === undefined" in INDEX_HTML
    assert 'value === ""' in INDEX_HTML
    assert "Number.isFinite(numeric)" in INDEX_HTML
    assert "finiteScatterNumber (valueRow[xIndex])" in body
    assert "finiteScatterNumber (valueRow[yIndex])" in body
    assert "const panelWidth = 260;" not in body
    assert "container.clientWidth" in body
    assert "/ Math.max(columnCount, 1)" in body or "/ columnCount" in body
    assert "vs ${xName}" not in body
    assert "vs ${" not in body
    assert "yName, xName" not in body
    assert "rotate(-Math.PI / 2)" not in body
    assert "n=${validCount}" in body or "`n=" in body
    assert "yVariables.forEach" in body or "for (let row = 0;" in body
    assert body.count("fillText(xName") <= 1
    assert "Math.min(...points" not in body
    assert "Math.max(...points" not in body
    assert "Math.min(...points.map" not in body
    assert "Math.max(...points.map" not in body
    assert "Math.min(...values" not in body
    assert "Math.max(...values" not in body
    assert "const points = values.map" not in body
    assert ".map((valueRow) => ({ x:" not in body
    assert "points.forEach" not in body
    assert "let validCount = 0" in body
    assert "let xMin = Infinity" in body
    assert "let xMax = -Infinity" in body
    assert "let yMin = Infinity" in body
    assert "let yMax = -Infinity" in body
    assert body.count("for (const valueRow of values)") >= 2
    assert 'const context = canvas.getContext("2d")' in body
    assert "if (!context)" in body
    assert "context.save()" in body
    assert "context.restore()" in body
    assert "xIndex === undefined" in body
    assert "yIndex === undefined" in body
    assert "appendChartQueryParams(params)" in INDEX_HTML.split("async function drawTrend", 1)[1].split("async function drawScatterMatrix", 1)[0]
    assert "appendChartQueryParams(params)" in INDEX_HTML.split("async function drawScatterMatrix", 1)[1]
    assert "lastScatterMatrixPayload = null" in INDEX_HTML
    assert "clearScatterMatrix()" in INDEX_HTML
    assert 'el("drawScatterMatrix").disabled = true' in INDEX_HTML


def test_scatter_matrix_response_uses_itertuples_instead_of_iterrows():
    source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")
    body = source.split("def _scatter_matrix_response", 1)[1].split("\ndef ", 1)[0]
    assert ".iterrows()" not in body
    assert ".itertuples(" in body
    assert "index=False" in body
    assert "name=None" in body


def test_scatter_matrix_rejects_empty_segment_result(tmp_path, monkeypatch):
    frame = _frame(20)
    file_id = _write_upload(tmp_path, monkeypatch, frame)

    with pytest.raises(ValueError, match="没有可绘制的散点数据"):
        web._scatter_matrix_response(
            _params(
                file_id,
                segment_column="load",
                segment_mode="custom",
                segment_min="1000",
                segment_max="2000",
            )
        )


def test_scatter_matrix_resize_and_tab_visibility_guards():
    assert "function isElementVisible(node)" in INDEX_HTML
    assert "node.offsetParent !== null" in INDEX_HTML
    assert "node.getClientRects().length" in INDEX_HTML
    assert "isElementVisible(scatterContainer)" in INDEX_HTML
    assert "lastScatterMatrixPayload" in INDEX_HTML
    assert "renderScatterMatrix(lastScatterMatrixPayload)" in INDEX_HTML

    activate_body = (
        INDEX_HTML
        .split("function activateTab", 1)[1]
        .split("function handleTabKeydown", 1)[0]
    )
    assert 'tabId === "trendTab"' in activate_body
    assert "requestAnimationFrame" in activate_body
    assert "lastScatterMatrixPayload" in activate_body
    assert "renderScatterMatrix" in activate_body
    assert "isElementVisible" in activate_body


def test_scatter_matrix_frontend_empty_response_guard_before_success():
    draw_body = (
        INDEX_HTML
        .split("async function drawScatterMatrix", 1)[1]
        .split("function finiteScatterNumber", 1)[0]
    )
    assert "Array.isArray(data.values)" in draw_body
    assert "data.values.length === 0" in draw_body
    assert "throw new Error" in draw_body
    assert draw_body.index("data.values.length === 0") < draw_body.index("lastScatterMatrixPayload = data")


def test_scatter_matrix_canvas_layout_review_guards():
    render_body = INDEX_HTML.split("function renderScatterMatrix", 1)[1].split("function renderTrendChart", 1)[0]
    assert "Math.min(" in render_body
    assert "360" in render_body or "380" in render_body
    assert "const panelHeight = Math.max(220, Math.round(panelWidth * 0.72));" not in render_body
    assert "maxYLabelWidth" in render_body
    assert "measureText(yName)" in render_body
    assert "220" in render_body
    assert "96" in render_body
    assert "const leftLabelWidth = 96;" not in render_body
    assert "function fitCanvasText" in INDEX_HTML
    assert "measureText" in INDEX_HTML
    assert '"…"' in INDEX_HTML or "'…'" in INDEX_HTML
    assert "fitCanvasText(" in render_body
    scatter_loop_position = render_body.rindex("for (const valueRow of values)")
    count_position = render_body.rindex("n=${validCount}")
    assert count_position > scatter_loop_position
    assert "fillRect(" in render_body
    assert "measureText(countText)" in render_body
    assert "vs ${xName}" not in render_body
    assert "rotate(-Math.PI / 2)" not in render_body
    assert render_body.count("fillText(xName") <= 1
    assert "const points = values.map" not in render_body
    assert "points.forEach" not in render_body
    assert "Math.min(...points" not in render_body
    assert "Math.max(...points" not in render_body
    assert "Number(valueRow[xIndex])" not in render_body
    assert "Number(valueRow[yIndex])" not in render_body
