from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr import web
from chem_ts_corr.closed_loop import CLOSED_LOOP_EVIDENCE_COLUMNS, build_closed_loop_evidence
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import run_analysis


def _risk_flags() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": "manual_closed", "closed_loop_suspect_flag": False, "target_leads_variable_flag": False},
            {"variable": "manual_closed_and_auto", "closed_loop_suspect_flag": True, "target_leads_variable_flag": True},
            {"variable": "manual_non_closed", "closed_loop_suspect_flag": False, "target_leads_variable_flag": False},
            {"variable": "conflict", "closed_loop_suspect_flag": True, "target_leads_variable_flag": True},
            {"variable": "automatic", "closed_loop_suspect_flag": True, "target_leads_variable_flag": False},
            {"variable": "unknown", "closed_loop_suspect_flag": False, "target_leads_variable_flag": True},
        ]
    )


def test_closed_loop_evidence_fuses_manual_and_existing_risk_flags():
    evidence = build_closed_loop_evidence(
        _risk_flags(),
        manual_closed_loop_variables=["manual_closed", "manual_closed_and_auto"],
        manual_non_closed_loop_variables=["manual_non_closed", "conflict"],
    ).set_index("variable")

    assert evidence.loc["manual_closed", "closed_loop_evidence_level"] == "confirmed"
    assert evidence.loc["manual_closed", "closed_loop_evidence_source"] == "manual"
    assert evidence.loc["manual_closed_and_auto", "closed_loop_evidence_source"] == "manual_and_automatic"
    assert evidence.loc["manual_non_closed", "closed_loop_evidence_level"] == "rejected"
    assert evidence.loc["conflict", "closed_loop_evidence_level"] == "conflict"
    assert evidence.loc["conflict", "closed_loop_evidence_source"] == "manual_and_automatic"
    assert bool(evidence.loc["conflict", "closed_loop_conflict"])
    assert evidence.loc["automatic", "closed_loop_evidence_level"] == "suspected"
    assert evidence.loc["unknown", "closed_loop_evidence_level"] == "unknown"
    assert json.loads(evidence.loc["conflict", "closed_loop_reason"]) == ["人工确认非闭环", "自动检测存在闭环嫌疑"]


def test_auto_status_only_uses_existing_closed_loop_suspect_flag():
    evidence = build_closed_loop_evidence(
        pd.DataFrame([
            {"variable": "leads_only", "closed_loop_suspect_flag": False, "target_leads_variable_flag": True},
            {"variable": "suspect", "closed_loop_suspect_flag": True, "target_leads_variable_flag": False},
        ])
    ).set_index("variable")

    assert evidence.loc["leads_only", "auto_closed_loop_status"] == "unknown"
    assert evidence.loc["suspect", "auto_closed_loop_status"] == "suspected_closed_loop"


def test_closed_loop_evidence_csv_schema_is_frozen():
    evidence = build_closed_loop_evidence(_risk_flags())

    assert evidence.columns.tolist() == CLOSED_LOOP_EVIDENCE_COLUMNS


def test_manual_only_variable_is_preserved_without_automatic_risk_result():
    evidence = build_closed_loop_evidence(
        _risk_flags(),
        manual_closed_loop_variables=["manual_only"],
        manual_non_closed_loop_variables=["manual_only_non_closed"],
    )

    assert evidence["variable"].tolist()[-2:] == ["manual_only", "manual_only_non_closed"]
    manual_only = evidence.set_index("variable").loc["manual_only"]
    assert manual_only["manual_closed_loop_status"] == "confirmed_closed_loop"
    assert manual_only["auto_closed_loop_status"] == "unknown"
    assert manual_only["closed_loop_evidence_level"] == "confirmed"
    assert json.loads(manual_only["closed_loop_reason"])[-1] == "未获得自动闭环判断结果"


def _analysis_config(tmp_path: Path, output_name: str, **kwargs: object) -> AnalysisConfig:
    values = np.arange(120, dtype=float)
    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=len(values), freq="h"),
            "target": np.sin(values / 8),
            "FCV101.PV": np.sin((values - 2) / 8),
            "load": values,
        }
    ).to_csv(input_path, index=False, encoding="utf-8-sig")
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


def test_evidence_output_does_not_change_ranked_or_risk_flags(tmp_path: Path):
    baseline = _analysis_config(tmp_path, "baseline")
    annotated = _analysis_config(
        tmp_path,
        "annotated",
        manual_closed_loop_variables=["FCV101.PV"],
    )

    run_analysis(baseline)
    run_analysis(annotated)

    for filename in ["ranked_features.csv", "risk_flags.csv"]:
        pd.testing.assert_frame_equal(
            pd.read_csv(baseline.output_dir / filename),
            pd.read_csv(annotated.output_dir / filename),
        )
    evidence = pd.read_csv(annotated.output_dir / "closed_loop_evidence.csv")
    assert evidence.columns.tolist() == CLOSED_LOOP_EVIDENCE_COLUMNS
    assert not evidence.empty


def test_old_run_without_evidence_file_returns_empty_evidence_payload(tmp_path: Path):
    config = _analysis_config(tmp_path, "old")
    config.output_dir.mkdir()
    (config.output_dir / "summary.md").write_text("# summary", encoding="utf-8")

    payload = web._build_result_payload("old", config.output_dir, config)

    assert payload["closedLoopEvidence"] == []
    for marker in [
        "闭环风险证据",
        "暂无闭环证据",
        "人工确认",
        "自动判断",
        "综合闭环状态",
        "证据来源",
        "证据冲突",
        "判断依据",
        "人工与自动判断冲突",
        "reasons.join(\"；\")",
    ]:
        assert marker in web.INDEX_HTML


def test_screening_does_not_read_closed_loop_evidence_fields():
    source = (Path(__file__).parents[1] / "chem_ts_corr" / "screening.py").read_text(encoding="utf-8")

    for field in CLOSED_LOOP_EVIDENCE_COLUMNS[1:]:
        assert field not in source
