from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from chem_ts_corr.evidence_explanations import _tokens, add_evidence_explanations
from chem_ts_corr.final_review_summary import build_final_review_summary
from chem_ts_corr.report import build_markdown_summary, build_recommended_candidates, write_outputs
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
    assert response["evidence_missing_items"] == "模型提升"
    assert "未获得" in response["four_layer_missing_items"]
    assert "数据不足" in response["four_layer_missing_items"]
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
        "four_layer_missing_items",
        "four_layer_coverage_status",
        "evidence_conflict_items",
        "candidate_summary",
    ]:
        assert column in recommended.columns
        assert column in summary


def _explained_single(**changes) -> pd.Series:
    row = _rows().iloc[[0]].copy()
    for field, value in changes.items():
        row[field] = value
    return add_evidence_explanations(row).iloc[0]


def test_candidate_summary_requires_every_core_layer_to_be_supported():
    assert "均支持" in _explained_single()["candidate_summary"]
    for field, status in [
        ("layer3_independence_status", "not_supported"),
        ("layer4_model_status", "not_available"),
        ("stability_status", "conflicting"),
        ("layer2_temporal_status", "partially_supported"),
    ]:
        assert "均支持" not in _explained_single(**{field: status})["candidate_summary"]


@pytest.mark.parametrize(
    "value",
    [
        "target_leads_variable;unstable_over_time",
        "target_leads_variable；unstable_over_time",
        "target_leads_variable,unstable_over_time",
        "target_leads_variable，unstable_over_time",
        "target_leads_variable|unstable_over_time",
    ],
)
def test_explanation_token_parser_supports_all_production_delimiters(value):
    assert _tokens(value) == ["target_leads_variable", "unstable_over_time"]


def test_downstream_summary_is_detected_when_multiple_risk_flags_are_present():
    row = _explained_single(
        lag=-2,
        risk_flags="target_leads_variable;unstable_over_time",
        layer2_temporal_status="conflicting",
    )
    assert "下游响应可能" in row["candidate_summary"]


def test_common_capacity_conflict_takes_priority_over_synchronous_summary():
    row = _explained_single(
        lag=0,
        layer2_temporal_status="partially_supported",
        layer3_independence_status="not_supported",
        risk_flags="common_capacity_driver",
    )
    assert "需要工程复核" in row["candidate_summary"]
    assert "存在部分统计证据支持" in row["candidate_summary"]
    assert "同步关联候选" not in row["candidate_summary"]


def test_redundant_proxy_conflict_takes_priority_over_missing_evidence_summary():
    row = _explained_single(
        layer4_model_status="not_available",
        evidence_missing_items="模型提升",
        risk_flags="redundant_proxy",
    )
    assert "需要工程复核" in row["candidate_summary"]
    assert "未获得或数据不足" not in row["candidate_summary"]


def test_stability_conflict_takes_priority_over_supported_layers():
    row = _explained_single(stability_status="conflicting")
    assert "需要工程复核" in row["candidate_summary"]
    assert "均支持" not in row["candidate_summary"]


def test_plain_zero_lag_without_conflicts_uses_synchronous_summary():
    row = _explained_single(lag=0, layer2_temporal_status="partially_supported")
    assert "同步关联候选" in row["candidate_summary"]


def test_four_layer_coverage_is_independent_of_scoring_component_coverage():
    row = _explained_single(
        evidence_coverage_status="完整",
        evidence_missing_items="",
        layer1_association_status="not_available",
    )
    assert row["evidence_coverage_status"] == "完整"
    assert row["evidence_missing_items"] == ""
    assert row["four_layer_coverage_status"] == "部分完整"
    assert "Layer 1" in row["four_layer_missing_items"]


def test_four_layer_coverage_treats_negative_and_conflicting_statuses_as_obtained():
    row = _explained_single(
        layer1_association_status="not_supported",
        layer3_independence_status="conflicting",
    )
    assert row["four_layer_coverage_status"] == "完整"
    assert row["four_layer_missing_items"] == ""


def test_conflict_summary_without_support_does_not_claim_statistical_support():
    row = _explained_single(
        layer1_association_status="not_supported",
        layer2_temporal_status="not_supported",
        layer3_independence_status="not_supported",
        layer4_model_status="not_supported",
        stability_status="conflicting",
        data_quality_status="not_supported",
    )
    assert "当前缺少明确支持证据" in row["candidate_summary"]
    assert "存在统计证据支持" not in row["candidate_summary"]
    assert "存在部分统计证据支持" not in row["candidate_summary"]


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
    for field in ["layer1_association_status", "evidence_support_items", "evidence_against_items", "evidence_missing_items", "four_layer_missing_items", "four_layer_coverage_status", "evidence_conflict_items", "candidate_summary"]:
        assert (row[field] or "") == explained.iloc[0][field]


def test_explanation_fields_flow_to_csv_markdown_and_final_review(tmp_path):
    explained = add_evidence_explanations(_rows())
    write_outputs(
        tmp_path, "target", explained, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    )
    for filename in ["ranked_features.csv", "recommended_candidates.csv"]:
        saved = pd.read_csv(tmp_path / filename, encoding="utf-8-sig")
        assert set([
            "evidence_support_items", "evidence_against_items", "evidence_missing_items", "four_layer_missing_items", "four_layer_coverage_status",
            "evidence_conflict_items", "candidate_summary",
        ]) <= set(saved)
    assert "candidate_summary" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    review = build_final_review_summary(
        pd.DataFrame([{"variable": "x_driver", "integrated_review_decision": "priority_review"}]),
        ranked_features=explained,
    )
    assert set([
        "evidence_support_items", "evidence_against_items", "evidence_missing_items", "four_layer_missing_items", "four_layer_coverage_status",
        "evidence_conflict_items", "candidate_summary",
    ]) <= set(review)


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
        "four_layer_missing_items",
        "four_layer_coverage_status",
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


def test_obsolete_score_explanations_are_absent_from_all_user_outputs():
    old = [
        "证据修正系数由证据覆盖度和数据质量共同" + "计算",
        "可用相关证据按等权几何" + "均值合并",
    ]
    required = {
        "report.py": "证据修正系数当前仅由数据质量得分决定",
        "web.py": "证据修正系数当前仅反映数据质量",
        "llm_report.py": "当前仅由 data_quality_score 决定",
    }
    for filename, expected in required.items():
        source = (Path("chem_ts_corr") / filename).read_text(encoding="utf-8")
        assert expected in source
        assert not any(text in source for text in old)


def test_explanations_are_added_only_after_final_ranking_is_completed():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")
    finalize = source.index("final = _finalize_driver_ranking(")
    sort = source.index("final = final.sort_values(PRIMARY_RANK_COLUMN", finalize)
    explain = source.index("final = add_evidence_explanations(final)", sort)
    assert finalize < sort < explain
