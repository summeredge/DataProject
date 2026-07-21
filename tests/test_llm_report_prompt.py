from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chem_ts_corr.llm_report import build_llm_analysis_package, build_llm_prompt


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_build_llm_analysis_package_compacts_and_classifies_results(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "summary.md").write_text(
        "- target: Y.PV\n- rows_after_preprocess: 1000\n- max_lag: 24\n",
        encoding="utf-8",
    )
    _write_csv(
        run_dir / "ranked_features.csv",
        [
            {"variable": "FIC001.SV", "driver_rank": 1, "driver_priority_score": 0.91, "final_score": 0.91, "candidate_class": "upstream_driver_candidate", "driver_priority_factor": 1.0, "evidence_coverage_status": "部分完整", "evidence_missing_items": "稳定性验证", "evidence_completeness": 0.875, "data_quality_score": 0.999, "evidence_confidence": 0.935, "candidate_grade": "A", "lag": 6, "direction": "variable_leads_target", "recommended_use": "prediction_candidate", "risk_flags": "", "risk_level": "none", "score_method": "industrial_robust_v3"},
            {"variable": "PI002.PV", "final_score": 0.73, "candidate_grade": "B", "lag": -2, "direction": "target_leads_variable", "recommended_use": "manual_review", "risk_flags": "target_leads_variable", "risk_level": "medium"},
        ],
    )
    _write_csv(
        run_dir / "risk_flags.csv",
        [
            {"variable": "PI002.PV", "risk_flags": "target_leads_variable;high_collinearity_risk", "risk_count": 2, "risk_level": "medium", "recommended_use": "manual_review"}
        ],
    )
    _write_csv(
        run_dir / "conditional_granger_scores.csv",
        [
            {"variable": "FIC001.SV", "status": "ok", "best_lag": 6, "tested_lags": "4,5,6,7", "fdr_q_value": 0.03, "predictive_contribution": 0.12, "interpretation": "predictive validation only; not a causal conclusion"},
            {"variable": "PI002.PV", "status": "skipped: non-positive screening lag", "best_lag": "", "tested_lags": "", "fdr_q_value": "", "predictive_contribution": "", "interpretation": "predictive validation only; not a causal conclusion"},
        ],
    )
    _write_csv(
        run_dir / "causal_review_evidence.csv",
        [
            {"variable": "FIC001.SV", "evidence_score": 4.2, "evidence_level": "strong_predictive_evidence", "data_priority": "high", "risk_constraint_level": "none", "statistical_limit_level": "none", "integrated_review_decision": "priority_review", "evidence_reason": "candidate_grade_A;conditional_granger_supported", "statistical_limit_reason": "", "integrated_review_reason": "strong_data_evidence"},
            {"variable": "PI002.PV", "evidence_score": 1.1, "evidence_level": "weak_or_incomplete_evidence", "data_priority": "medium", "risk_constraint_level": "medium", "statistical_limit_level": "medium", "integrated_review_decision": "manual_review_only", "evidence_reason": "target_lead_risk", "statistical_limit_reason": "high_collinearity_limited_signal", "integrated_review_reason": "manual_review_only"},
        ],
    )
    _write_csv(
        run_dir / "final_review_summary.csv",
        [
            {"final_rank": 1, "variable": "FIC001.SV", "integrated_review_decision": "priority_review", "priority_label": "优先复核", "key_reason": "预测验证支持较强", "lag_boundary_hint": "", "conflict_type": "", "conflict_reason": ""},
            {"final_rank": 2, "variable": "PI002.PV", "integrated_review_decision": "manual_review_only", "priority_label": "仅人工查看", "key_reason": "目标领先变量，不建议直接作为前馈候选", "lag_boundary_hint": "", "conflict_type": "target_leads_variable", "conflict_reason": "目标领先变量"},
        ],
    )

    package = build_llm_analysis_package(run_dir, top_n=10)

    assert package["meta"]["target"] == "Y.PV"
    assert package["overview"]["score_method"] == "industrial_robust_v3"
    assert package["highly_correlated_variables"][0]["variable"] == "FIC001.SV"
    assert package["highly_correlated_variables"][0]["evidence_coverage_status"] == "部分完整"
    assert package["highly_correlated_variables"][0]["evidence_missing_items"] == "稳定性验证"
    assert package["highly_correlated_variables"][0]["evidence_confidence"] == 0.935
    assert package["attention_variables"][0]["variable"] == "FIC001.SV"
    assert package["predictive_causal_evidence"][0]["variable"] == "FIC001.SV"
    control_roles = {row["variable"]: row["suggested_control_role"] for row in package["control_candidate_variables"]}
    assert control_roles["FIC001.SV"] in {"mv_candidate", "dv_feedforward_candidate"}
    assert control_roles["PI002.PV"] == "not_recommended_for_control"
    assert package["risk_and_limitations"][0]["variable"] == "PI002.PV"


def test_build_llm_prompt_contains_strict_control_and_causality_constraints(tmp_path: Path):
    run_dir = tmp_path / "run_002"
    run_dir.mkdir()
    (run_dir / "summary.md").write_text("- target: Y.PV\n", encoding="utf-8")
    _write_csv(run_dir / "ranked_features.csv", [{"variable": "FIC001.SV", "final_score": 0.9, "candidate_grade": "A", "lag": 5, "direction": "variable_leads_target", "score_method": "industrial_robust_v3"}])

    package = build_llm_analysis_package(run_dir, top_n=5)
    prompt = build_llm_prompt(package, report_type="apc_advice")

    assert "不得声称发现确定性因果关系" in prompt
    assert "与目标变量高度相关的过程变量" in prompt
    assert "最需要关注的变量" in prompt
    assert "相关性与因果复核证据靠前" in prompt
    assert "可能适合作为控制变量" in prompt
    assert "可能 MV 候选" in prompt
    assert "可能 DV / 前馈候选" in prompt
    assert "不建议直接用于控制" in prompt
    assert "common_capacity_driver" in prompt
    assert "closed_loop_suspect" in prompt
    assert "high_collinearity_risk" in prompt
    assert "fallback_missing_ranked_lag" in prompt
    assert "evidence_confidence 的中文含义是“证据修正系数”" in prompt
    assert "不是概率、统计置信度或因果置信度" in prompt
    assert "industrial_robust_v3" in prompt
    assert "评分版本仅表示评分语义，不代表新的因果算法" in prompt
    assert "```json" in prompt
    json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
