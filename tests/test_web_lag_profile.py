from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.web import INDEX_HTML


RUN_ID = "a" * 32


def _write_run(
    runs_dir: Path,
    *,
    lag_rows: list[dict[str, object]] | None = None,
    ranked_rows: list[dict[str, object]] | None = None,
) -> Path:
    run_dir = runs_dir / RUN_ID
    run_dir.mkdir(parents=True)
    if lag_rows is not None:
        pd.DataFrame(lag_rows).to_csv(run_dir / "lag_scores.csv", index=False)
    if ranked_rows is not None:
        pd.DataFrame(ranked_rows).to_csv(run_dir / "ranked_features.csv", index=False)
    (run_dir / "run_config.json").write_text(
        json.dumps({"resample_rule": "5min"}), encoding="utf-8"
    )
    return run_dir


def _params(run_id: str = RUN_ID, variable: str = "A") -> dict[str, list[str]]:
    return {"run_id": [run_id], "variable": [variable]}


def test_lag_profile_filters_one_variable_sorts_and_preserves_signed_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(web, "RUNS_DIR", runs_dir)
    _write_run(
        runs_dir,
        lag_rows=[
            {"variable": "B", "lag": 0, "pearson": 0.9, "spearman": 0.8},
            {
                "variable": "A",
                "lag": 1,
                "pearson": -0.7,
                "spearman": -0.8,
                "pearson_q": float("nan"),
                "spearman_q": 0.02,
                "n": 79,
                "lag_boundary_flag": True,
            },
            {
                "variable": "A",
                "lag": -1,
                "pearson": -0.2,
                "spearman": -0.1,
                "pearson_q": 0.4,
                "spearman_q": float("inf"),
                "n": 79,
                "lag_boundary_flag": True,
            },
            {
                "variable": "A",
                "lag": 0,
                "pearson": 0.1,
                "spearman": 0.2,
                "pearson_q": 0.3,
                "spearman_q": 0.2,
                "n": 80,
                "lag_boundary_flag": False,
            },
        ],
        ranked_rows=[{"variable": "A", "lag": 1, "method": "spearman"}],
    )

    result = web._lag_profile_response(_params())

    assert result["variable"] == "A"
    assert result["best_lag"] == 1
    assert result["method"] == "spearman"
    assert result["max_lag"] == 1
    assert result["sampling_interval_minutes"] == 5
    assert [point["lag"] for point in result["points"]] == [-1, 0, 1]
    assert {point["variable"] for point in result["points"]} == {"A"}
    assert result["points"][0]["pearson"] == pytest.approx(-0.2)
    assert result["points"][2]["spearman"] == pytest.approx(-0.8)
    assert result["points"][0]["spearman_q"] is None
    assert result["points"][2]["pearson_q"] is None
    assert result["points"][0]["lag_boundary_flag"] is True
    assert result["points"][1]["lag_boundary_flag"] is False
    assert "B" not in json.dumps(result, ensure_ascii=False)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("run_id", ["", "not-a-run-id", "../outside", "..\\outside"])
def test_lag_profile_rejects_invalid_run_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_id: str
):
    runs_dir = tmp_path / "runs"
    outside = tmp_path / "outside"
    runs_dir.mkdir()
    outside.mkdir()
    monkeypatch.setattr(web, "RUNS_DIR", runs_dir)

    with pytest.raises(ValueError, match="运行 ID"):
        web._lag_profile_response(_params(run_id=run_id))


def test_lag_profile_reports_missing_run_file_and_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(web, "RUNS_DIR", runs_dir)

    with pytest.raises(FileNotFoundError, match="运行结果不存在"):
        web._lag_profile_response(_params())

    run_dir = runs_dir / RUN_ID
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="滞后相关结果不存在"):
        web._lag_profile_response(_params())

    pd.DataFrame([{"variable": "B", "lag": 0}]).to_csv(
        run_dir / "lag_scores.csv", index=False
    )
    with pytest.raises(ValueError, match="变量.*没有滞后相关记录"):
        web._lag_profile_response(_params())


def test_lag_profile_best_lag_and_method_only_use_ranked_candidate_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(web, "RUNS_DIR", runs_dir)
    _write_run(
        runs_dir,
        lag_rows=[
            {"variable": "A", "lag": 2, "pearson": 0.95, "spearman": 0.2},
            {"variable": "A", "lag": 4, "pearson": 0.1, "spearman": -0.96},
        ],
        ranked_rows=[
            {
                "variable": "A",
                "lag": 4,
                "method": "spearman",
                "pearson": 0.1,
                "spearman": -0.96,
            }
        ],
    )

    result = web._lag_profile_response(_params())
    source = inspect.getsource(web._lag_profile_response)

    assert result["best_lag"] == 4
    assert result["method"] == "spearman"
    for forbidden in ["compute_lag_scores", "summarize_best_lags", "idxmax"]:
        assert forbidden not in source


def test_lag_profile_route_and_frontend_curve_contract_are_present():
    handler_source = inspect.getsource(web._Handler.do_GET)
    payload_source = inspect.getsource(web._build_result_payload)

    assert 'parsed.path == "/api/lag_profile"' in handler_source
    assert '"lagScores": []' in payload_source
    assert "lag_scores.csv" not in payload_source

    for marker in [
        "滞后相关曲线",
        "正在加载滞后相关曲线……",
        "Pearson 曲线",
        "Spearman 曲线",
        "lag = 0",
        "当前最佳滞后",
        "变量领先目标",
        "变量滞后目标",
        "同步变化",
        "最佳滞后触及搜索边界",
        "correlationConsistencyMessage",
        "lagProfileRequestSerial",
        "lagProfileCacheKey",
        "clearLagProfileCache",
    ]:
        assert marker in INDEX_HTML


def test_lag_profile_frontend_guards_races_cache_scope_and_scoring_boundaries():
    loader = INDEX_HTML.split("async function loadLagProfile", 1)[1].split(
        "function renderLagProfile", 1
    )[0]
    cache_key = INDEX_HTML.split("function lagProfileCacheKey", 1)[1].split("}", 1)[0]
    curve_path = INDEX_HTML.split("function lagProfilePath", 1)[1].split("}", 1)[0]
    upload = INDEX_HTML.split("async function uploadFile", 1)[1].split(
        "async function loadColumns", 1
    )[0]
    analyze = INDEX_HTML.split("async function analyze", 1)[1].split(
        "async function waitForAnalysisResult", 1
    )[0]
    resize = INDEX_HTML.split('window.addEventListener("resize"', 1)[1].split(
        'el("testLlmConnection")', 1
    )[0]

    assert "runId" in cache_key and "variable" in cache_key
    assert "requestId !== lagProfileRequestSerial" in loader
    assert "currentRunId !== runId" in loader
    assert "panel.isConnected" in loader
    assert "panel.dataset.lagProfileKey !== key" in loader
    assert "lastLagProfile = null" in loader
    assert "lagPanel.dataset.lagProfileKey === lastLagProfile.key" in resize
    assert "clearLagProfileCache()" in upload
    assert "clearLagProfileCache()" in analyze
    assert "Math.abs" not in curve_path
    for score in ["driver_rank", "final_score", "driver_priority_score"]:
        assert f".{score} =" not in loader
