from __future__ import annotations

import pandas as pd
from pathlib import Path

from chem_ts_corr.evidence_explanations import add_evidence_explanations
from chem_ts_corr.report import build_markdown_summary, build_recommended_candidates
from chem_ts_corr.web import AnalysisConfig, INDEX_HTML, _build_result_payload


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": "x_driver",
                "driver_rank": 1,
                "driver_priority_score": 0.8,
                "final_score": 0.8,
                "candidate_grade": "A",
                "lag": 3,
                "risk_flags": "",
                "layer1_association_status": "supported",
                "layer2_temporal_status": "supported",
                "layer3_independence_status": "supported",
                "layer4_model_status": "supported",
                "stability_status": "supported",
                "data_quality_status": "supported",
                "evidence_missing_items": "",
                "evidence_conflict_items": "",
                "recommended_use": "strong_screening_candidate",
            },
            {
                "variable": "x_response",
                "driver_rank": 2,
                "driver_priority_score": 0.4,
                "final_score": 0.8,
                "candidate_grade": "D",
                "lag": -2,
                "risk_flags": "target_leads_variable",
                "layer1_association_status": "supported",
                "layer2_temporal_status": "conflicting",
                "layer3_independence_status": "not_available",
                "layer4_model_status": "insufficient_data",
                "stability_status": "supported",
                "data_quality_status": "supported",
                "evidence_missing_items": "模型提升",
                "evidence_conflict_items": "target_leads_variable",
                "recommended_use": "unstable_candidate",
            },
        ]
    )


def test_explanation_fields_are_deterministic_and_do_not_change_ranking_values():
    source = _rows()
    explained = add_evidence_explanations(source)

    pd.testing.assert_frame_equal(
        source[["driver_rank", "driver_priority_score", "final_score", "candidate_grade"]],
        explained[["driver_rank", "driver_priority_score", "final_score", "candidate_grade"]],
    )
    driver = explained.set_index("variable").loc["x_driver"]
    response = explained.set_index("variable").loc["x_response"]
    assert "Layer 1 关联支持" in driver["evidence_support_items"]
    assert "潜在驱动因素候选" in driver["candidate_summary"]
    assert "未获得" in response["evidence_missing_items"]
    assert "数据不足" in response["evidence_missing_items"]
    assert "下游响应可能" in response["candidate_summary"]
    assert "无效" not in response["candidate_summary"]


def test_csv_and_markdown_share_the_same_explanation_fields():
    explained = add_evidence_explanations(_rows())
    recommended = build_recommended_candidates(explained)
    summary = build_markdown_summary("target", explained, pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())

    for column in [
        "layer1_association_status",
        "layer2_temporal_status",
        "layer3_independence_status",
        "layer4_model_status",
        "stability_status",
        "data_quality_status",
        "evidence_support_items",
        "evidence_against_items",
        "evidence_missing_items",
        "evidence_conflict_items",
        "candidate_summary",
    ]:
        assert column in recommended.columns
        assert column in summary


def test_api_payload_preserves_the_csv_explanation_contract(tmp_path):
    explained = add_evidence_explanations(_rows())
    explained.to_csv(tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(tmp_path / "risk_flags.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(tmp_path / "residual_corr_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(tmp_path / "regime_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(tmp_path / "model_lift_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(tmp_path / "rolling_corr_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(tmp_path / "enhanced_validation_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(tmp_path / "granger_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(tmp_path / "shap_or_importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(tmp_path / "model_variable_importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(tmp_path / "model_discovered_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(tmp_path / "near_miss_candidates.csv", index=False, encoding="utf-8-sig")
    (tmp_path / "summary.md").write_text("- rows_after_preprocess: 2\n", encoding="utf-8")
    config = AnalysisConfig(input_path=tmp_path / "input.csv", time_column="time", target="target", output_dir=tmp_path)

    row = _build_result_payload("run", tmp_path, config)["rankedFeatures"][0]
    for field in ["layer1_association_status", "evidence_support_items", "evidence_against_items", "evidence_missing_items", "evidence_conflict_items", "candidate_summary"]:
        assert (row[field] or "") == explained.iloc[0][field]


def test_web_uses_the_result_explanation_contract_without_historical_state_controls():
    for field in [
        "layer1_association_status",
        "layer2_temporal_status",
        "layer3_independence_status",
        "layer4_model_status",
        "stability_status",
        "data_quality_status",
        "evidence_support_items",
        "evidence_against_items",
        "evidence_missing_items",
        "evidence_conflict_items",
        "candidate_summary",
    ]:
        assert field in INDEX_HTML
    assert "closed" + "_loop" not in INDEX_HTML
    assert "闭" + "环" not in INDEX_HTML


def test_historical_state_terms_are_absent_from_production_and_test_sources():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for root in [Path("chem_ts_corr"), Path("tests")]
        for path in root.rglob("*.py")
    )
    for term in ["closed" + "_loop", "manual" + "_closed", "auto" + "_closed", "闭" + "环", "人工推荐" + "决策", "保存并" + "重排"]:
        assert term not in source
