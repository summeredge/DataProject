from __future__ import annotations

from pathlib import Path

import pandas as pd

from chem_ts_corr.llm_report import build_llm_analysis_package, build_llm_prompt


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_fic_pv_with_related_sv_represents_loop_mv_candidate(tmp_path: Path):
    """分析使用 .PV；有 .SV/.MV 的控制回路可作为 MV 候选复核。"""
    run_dir = tmp_path / "run_loop"
    run_dir.mkdir()
    (run_dir / "summary.md").write_text("- target: Y.PV\n", encoding="utf-8")
    _write_csv(
        run_dir / "ranked_features.csv",
        [
            {"variable": "FIC421002.PV", "final_score": 0.88, "candidate_grade": "A", "lag": 5, "direction": "variable_leads_target", "risk_flags": "", "risk_level": "none"},
            {"variable": "FIC421002.SV", "final_score": 0.61, "candidate_grade": "B", "lag": 5, "direction": "variable_leads_target", "risk_flags": "", "risk_level": "none"},
        ],
    )
    _write_csv(
        run_dir / "causal_review_evidence.csv",
        [
            {"variable": "FIC421002.PV", "evidence_level": "strong_predictive_evidence", "integrated_review_decision": "priority_review"},
            {"variable": "FIC421002.SV", "evidence_level": "moderate_predictive_evidence", "integrated_review_decision": "secondary_review"},
        ],
    )

    package = build_llm_analysis_package(run_dir, top_n=10)
    roles = {row["variable"]: row for row in package["control_candidate_variables"]}

    pv_row = roles["FIC421002.PV"]
    assert pv_row["suggested_control_role"] == "loop_mv_candidate"
    assert pv_row.get("related_sv") == "FIC421002.SV"
    assert pv_row.get("loop_tag") == "FIC421002"
    assert "历史数据代表" in pv_row["control_comment"]
    assert "FIC421002 控制回路" in pv_row["control_comment"]
    assert "实际操纵点" in pv_row["control_comment"]
    assert "FIC421002.SV" in pv_row["control_comment"]

    sv_row = roles["FIC421002.SV"]
    assert sv_row["suggested_control_role"] == "mv_candidate"


def test_fic_pv_without_related_sv_or_mv_still_can_be_loop_review_candidate_but_needs_dcs_check(tmp_path: Path):
    """单独出现 FIC.PV 时，不应机械排除为只能监控，而应提示核对回路写入点。"""
    run_dir = tmp_path / "run_no_loop"
    run_dir.mkdir()
    (run_dir / "summary.md").write_text("- target: Y.PV\n", encoding="utf-8")
    _write_csv(
        run_dir / "ranked_features.csv",
        [
            {"variable": "FIC421002.PV", "final_score": 0.88, "candidate_grade": "A", "lag": 5, "direction": "variable_leads_target", "risk_flags": "", "risk_level": "none"},
        ],
    )

    package = build_llm_analysis_package(run_dir, top_n=10)
    row = package["control_candidate_variables"][0]

    assert row["variable"] == "FIC421002.PV"
    assert row["suggested_control_role"] in {"loop_mv_candidate", "dv_feedforward_candidate", "monitor_candidate", "manual_review_only"}
    assert row["suggested_control_role"] != "not_recommended_for_control"
    assert "历史数据代表" in row["control_comment"]
    assert "SV/MV" in row["control_comment"] or "APC" in row["control_comment"]
    assert "核对" in row["control_comment"] or "确认" in row["control_comment"]


def test_prompt_requires_pv_as_analysis_representative_and_loop_as_mv_candidate(tmp_path: Path):
    run_dir = tmp_path / "run_prompt"
    run_dir.mkdir()
    (run_dir / "summary.md").write_text("- target: Y.PV\n", encoding="utf-8")
    _write_csv(
        run_dir / "ranked_features.csv",
        [
            {"variable": "FIC421002.PV", "final_score": 0.88, "candidate_grade": "A", "lag": 5, "direction": "variable_leads_target"},
            {"variable": "FIC421002.SV", "final_score": 0.61, "candidate_grade": "B", "lag": 5, "direction": "variable_leads_target"},
        ],
    )

    prompt = build_llm_prompt(build_llm_analysis_package(run_dir, top_n=10), report_type="apc_advice")

    assert "用 PV 做分析" in prompt
    assert "用回路做 MV 候选" in prompt
    assert "用 SV/MV" in prompt
    assert "实际操纵点" in prompt
    assert "loop_mv_candidate" in prompt
    assert "不得机械排除" in prompt
