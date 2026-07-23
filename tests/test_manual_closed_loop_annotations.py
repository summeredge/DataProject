from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import run_analysis


def _frame(rows: int = 120) -> pd.DataFrame:
    values = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "target": np.sin(values / 8),
            "FCV101.PV": np.sin((values - 2) / 8),
            "反应器蒸汽阀位": np.cos(values / 9),
            "load": values,
            "note": ["text"] * rows,
        }
    )


def _config(tmp_path: Path, output_name: str, **kwargs: object) -> AnalysisConfig:
    input_path = tmp_path / "input.csv"
    _frame().to_csv(input_path, index=False, encoding="utf-8-sig")
    settings: dict[str, object] = {
        "input_path": input_path,
        "time_column": "time",
        "target": "target",
        "output_dir": tmp_path / output_name,
        "max_lag": 3,
        "top_k": 10,
        "skip_model_lift": True,
        "skip_rolling_corr": True,
    }
    settings.update(kwargs)
    return AnalysisConfig(**settings)


def test_manual_closed_loop_annotations_default_to_empty_lists(tmp_path: Path):
    first = _config(tmp_path, "first")
    second = _config(tmp_path, "second")

    assert first.manual_closed_loop_variables == []
    assert first.manual_non_closed_loop_variables == []
    assert first.manual_closed_loop_variables is not second.manual_closed_loop_variables
    assert first.manual_non_closed_loop_variables is not second.manual_non_closed_loop_variables


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("A,B", ["A", "B"]),
        ("A，B", ["A", "B"]),
        (" A , B ", ["A", "B"]),
        ("A,A,B", ["A", "B"]),
        ("", []),
    ],
)
def test_manual_annotation_list_parser_is_stable(value: str, expected: list[str]):
    assert (
        web._list_field({"manual_closed_loop_variables": value}, "manual_closed_loop_variables")
        == expected
    )
    assert web._list_field({}, "manual_closed_loop_variables") == []


@pytest.mark.parametrize(
    ("closed", "non_closed", "message"),
    [
        (["FCV101.PV"], ["FCV101.PV"], "不能同时标记"),
        (["target"], [], "目标变量"),
        (["time"], [], "时间列"),
        (["FCV101.PV"], [], "已剔除列"),
        (["missing"], [], "不存在的变量"),
        (["note"], [], "只能包含数值列"),
    ],
)
def test_manual_annotation_backend_rejects_invalid_candidates(
    tmp_path: Path, closed: list[str], non_closed: list[str], message: str
):
    config = _config(tmp_path, "run")
    excluded = ["FCV101.PV"] if message == "已剔除列" else []

    with pytest.raises(ValueError, match=message):
        web._validate_manual_closed_loop_annotations(
            config.input_path,
            config.encoding,
            time_column=config.time_column,
            target=config.target,
            excluded_columns=excluded,
            manual_closed_loop_variables=closed,
            manual_non_closed_loop_variables=non_closed,
        )


def test_empty_manual_annotations_do_not_read_data_file(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def fail_if_called(*args: object, **kwargs: object) -> tuple[pd.DataFrame, str]:
        nonlocal calls
        calls += 1
        raise AssertionError("empty annotations must not read the data file")

    monkeypatch.setattr(web, "read_timeseries_table", fail_if_called)

    web._validate_manual_closed_loop_annotations(
        Path("unused.csv"),
        "utf-8-sig",
        time_column="time",
        target="target",
        excluded_columns=[],
        manual_closed_loop_variables=[],
        manual_non_closed_loop_variables=[],
    )

    assert calls == 0


def test_run_config_persists_manual_annotations_in_utf8_and_round_trips(tmp_path: Path):
    config = _config(
        tmp_path,
        "run",
        manual_closed_loop_variables=["FCV101.PV", "反应器蒸汽阀位"],
        manual_non_closed_loop_variables=[],
    )

    web._write_run_config(config.output_dir, config, "file-id")

    raw = (config.output_dir / "run_config.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    restored = web._read_run_config(config.output_dir)
    assert payload["manual_closed_loop_variables"] == ["FCV101.PV", "反应器蒸汽阀位"]
    assert payload["manual_non_closed_loop_variables"] == []
    assert restored.manual_closed_loop_variables == payload["manual_closed_loop_variables"]
    assert restored.manual_non_closed_loop_variables == []


def test_old_run_config_without_manual_annotations_remains_compatible(tmp_path: Path):
    config = _config(tmp_path, "old")
    web._write_run_config(config.output_dir, config, "file-id")
    path = config.output_dir / "run_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("manual_closed_loop_variables")
    payload.pop("manual_non_closed_loop_variables")
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    restored = web._read_run_config(config.output_dir)

    assert restored.manual_closed_loop_variables == []
    assert restored.manual_non_closed_loop_variables == []


def test_analyze_request_propagates_manual_annotations_before_background_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _config(tmp_path, "unused")
    captured: list[tuple[object, tuple[object, ...]]] = []

    class RecordingThread:
        def __init__(self, *, target: object, args: tuple[object, ...], daemon: bool):
            captured.append((target, args))

        def start(self) -> None:
            return None

    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(web, "_resolve_upload", lambda file_id: config.input_path)
    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda handler: {
            "file_id": "file-id",
            "encoding": "utf-8-sig",
            "time_column": "time",
            "target": "target",
            "manual_closed_loop_variables": "FCV101.PV，反应器蒸汽阀位,FCV101.PV",
            "manual_non_closed_loop_variables": "",
        },
    )
    monkeypatch.setattr(web.threading, "Thread", RecordingThread)

    response = web._analyze_response(object())

    assert response["status"] == "running"
    task_config = captured[0][1][1]
    assert task_config.manual_closed_loop_variables == ["FCV101.PV", "反应器蒸汽阀位"]
    assert task_config.manual_non_closed_loop_variables == []


def test_invalid_manual_request_creates_no_background_task_or_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _config(tmp_path, "unused")
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(web, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(web, "_resolve_upload", lambda file_id: config.input_path)
    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda handler: {
            "file_id": "file-id",
            "encoding": "utf-8-sig",
            "time_column": "time",
            "target": "target",
            "manual_closed_loop_variables": "FCV101.PV",
            "manual_non_closed_loop_variables": "FCV101.PV",
        },
    )

    with pytest.raises(ValueError, match="不能同时标记"):
        web._analyze_response(object())

    assert not runs_dir.exists()


def test_manual_annotations_only_change_allowed_ranking_outputs(tmp_path: Path):
    settings = {
        "segment_column": "load",
        "capacity_columns": ["load"],
        "residual_control_columns": ["load"],
        "skip_model_lift": False,
        "skip_rolling_corr": False,
    }
    baseline = _config(tmp_path, "baseline", **settings)
    annotated = _config(
        tmp_path,
        "annotated",
        manual_closed_loop_variables=["FCV101.PV"],
        manual_non_closed_loop_variables=["反应器蒸汽阀位"],
        **settings,
    )

    web._write_run_config(annotated.output_dir, annotated, "file-id")
    run_analysis(baseline)
    run_analysis(annotated)

    forbidden_fields = {
        "manual_closed_loop_variables",
        "manual_non_closed_loop_variables",
        "manual_closed_loop_status",
        "closed_loop_evidence_level",
        "closed_loop_evidence_source",
        "closed_loop_conflict",
        "auto_closed_loop_score",
        "original_driver_rank",
    }
    for filename in [
        "risk_flags.csv",
        "residual_corr_scores.csv",
        "regime_scores.csv",
        "model_lift_scores.csv",
        "rolling_corr_scores.csv",
    ]:
        before = pd.read_csv(baseline.output_dir / filename, encoding="utf-8-sig")
        after = pd.read_csv(annotated.output_dir / filename, encoding="utf-8-sig")
        pd.testing.assert_frame_equal(before, after)
        assert not forbidden_fields.intersection(after.columns)

    ranked = pd.read_csv(annotated.output_dir / "ranked_features.csv", encoding="utf-8-sig")
    assert {
        "final_score",
        "candidate_class",
        "driver_priority_factor",
        "driver_priority_score",
        "driver_rank",
        "recommended_use",
        "recommended_action",
    }.issubset(ranked.columns)
    baseline_ranked = pd.read_csv(baseline.output_dir / "ranked_features.csv", encoding="utf-8-sig")
    allowed = {
        "candidate_class",
        "driver_priority_factor",
        "driver_priority_score",
        "driver_rank",
        "recommended_use",
        "recommended_action",
    }
    unaffected = [column for column in baseline_ranked.columns if column not in allowed and column != "variable"]
    pd.testing.assert_frame_equal(
        baseline_ranked.set_index("variable").sort_index()[unaffected],
        ranked.set_index("variable").sort_index()[unaffected],
    )


def test_web_ui_contains_manual_annotation_controls_and_lifecycle_contract():
    source = web.INDEX_HTML
    for marker in [
        "人工闭环风险确认（可选）",
        "已确认闭环/控制相关变量",
        "已确认非闭环变量",
        "该确认针对当前目标变量",
        "已确认闭环会降低工程推荐优先级",
        "fillManualAnnotationOptions",
        "getManualClosedLoopSelection",
        "setManualClosedLoopSelection",
        "getManualNonClosedLoopSelection",
        "setManualNonClosedLoopSelection",
        "updateManualAnnotationSummary",
    ]:
        assert marker in source
    assert 'kind === "manualClosedLoop" ? "manualNonClosedLoop" : "manualClosedLoop"' in source
    assert "setManualClosedLoopSelection([]);" in source
    assert "setManualNonClosedLoopSelection([]);" in source
    assert "manualCandidates.includes(name)" in source


def test_screening_does_not_read_manual_annotations_or_future_fields():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")
    for token in [
        "manual_closed_loop_variables",
        "manual_non_closed_loop_variables",
        "confirmed_closed_loop",
        "confirmed_not_closed_loop",
        "manual_closed_loop_status",
        "auto_closed_loop_score",
        "original_driver_rank",
    ]:
        assert token not in source
