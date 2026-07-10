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
    assert "appendChartQueryParams(params)" in INDEX_HTML.split("async function drawTrend", 1)[1].split("async function drawScatterMatrix", 1)[0]
    assert "appendChartQueryParams(params)" in INDEX_HTML.split("async function drawScatterMatrix", 1)[1]
    assert "lastScatterMatrixPayload = null" in INDEX_HTML
    assert "clearScatterMatrix()" in INDEX_HTML
    assert 'el("drawScatterMatrix").disabled = true' in INDEX_HTML
