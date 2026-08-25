import pandas as pd

from chem_ts_corr.final_review_summary import INTERPRETATION, build_final_review_summary


def _evidence(rows):
    base = {
        "candidate_grade": "C",
        "final_score": 0.1,
        "lag": 1,
        "data_priority": "medium",
        "evidence_score": 1.0,
        "evidence_level": "moderate_predictive_evidence",
        "statistical_limit_level": "weak",
        "risk_constraint_level": "none",
        "risk_flags": "",
        "integrated_review_reason": "reason",
        "evidence_reason": "evidence",
        "statistical_limit_reason": "",
        "conditional_granger_status": "ok",
        "conditional_fdr_q_value": 0.2,
        "model_importance_rank": 99,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_priority_review_ranks_before_secondary_review():
    out = build_final_review_summary(_evidence([
        {"variable": "x2", "integrated_review_decision": "secondary_review"},
        {"variable": "x1", "integrated_review_decision": "priority_review"},
    ]))
    assert out["variable"].tolist() == ["x1", "x2"]


def test_priority_review_with_statistical_limit_ranks_before_secondary_review():
    out = build_final_review_summary(_evidence([
        {"variable": "x2", "integrated_review_decision": "secondary_review"},
        {"variable": "x1", "integrated_review_decision": "priority_review_with_statistical_limit"},
    ]))
    assert out["variable"].tolist() == ["x1", "x2"]


def test_high_data_priority_ranks_before_medium_within_same_decision():
    out = build_final_review_summary(_evidence([
        {"variable": "x2", "integrated_review_decision": "secondary_review", "data_priority": "medium"},
        {"variable": "x1", "integrated_review_decision": "secondary_review", "data_priority": "high"},
    ]))
    assert out["variable"].tolist() == ["x1", "x2"]


def test_higher_evidence_score_ranks_first_within_same_class():
    out = build_final_review_summary(_evidence([
        {"variable": "x1", "integrated_review_decision": "secondary_review", "evidence_score": 1},
        {"variable": "x2", "integrated_review_decision": "secondary_review", "evidence_score": 3},
    ]))
    assert out["variable"].tolist() == ["x2", "x1"]


def test_high_priority_and_medium_statistical_limit_generates_conflict():
    out = build_final_review_summary(_evidence([
        {"variable": "x1", "integrated_review_decision": "priority_review", "data_priority": "high", "statistical_limit_level": "medium"},
    ]))
    assert "strong_screening_but_statistical_limited" in out.loc[0, "evidence_conflict_type"]


def test_lag_boundary_generates_hint():
    out = build_final_review_summary(_evidence([
        {"variable": "x1", "integrated_review_decision": "secondary_review", "risk_flags": "lag_boundary"},
    ]))
    assert "滞后边界" in out.loc[0, "lag_boundary_hint"]


def test_large_tested_lags_without_lag_boundary_does_not_generate_hint():
    conditional = pd.DataFrame([
        {"variable": "x1", "tested_lags": "80,81,82", "baseline_maxlag": 24, "fallback_maxlag": 24}
    ])
    evidence = _evidence([
        {"variable": "x1", "integrated_review_decision": "priority_review", "risk_flags": ""}
    ])
    out = build_final_review_summary(evidence, conditional_granger_scores=conditional)
    assert out.loc[0, "lag_boundary_hint"] == ""


def test_does_not_modify_input_dataframe():
    evidence = _evidence([{"variable": "x1", "integrated_review_decision": "priority_review"}])
    before = evidence.copy(deep=True)
    build_final_review_summary(evidence)
    pd.testing.assert_frame_equal(evidence, before)


def test_missing_optional_tables_do_not_raise():
    out = build_final_review_summary(_evidence([{"variable": "x1", "integrated_review_decision": "priority_review"}]))
    assert out.loc[0, "variable"] == "x1"


def test_interpretation_is_fixed():
    out = build_final_review_summary(_evidence([{"variable": "x1", "integrated_review_decision": "priority_review"}]))
    assert set(out["interpretation"]) == {INTERPRETATION}


def test_confidence_review_fields_are_forwarded_as_explanation_only():
    evidence = _evidence([
        {
            "variable": "x1",
            "integrated_review_decision": "priority_review",
            "independent_predictive_support": "supported",
            "confounder_assessment": "no_flagged_confounder",
            "control_relation_assessment": "no_control_relation_flagged",
            "statistical_limitation": "no_flagged_statistical_limitation",
            "direction_assessment": "variable_leads_target",
        }
    ])
    ranked = pd.DataFrame([{"variable": "x1", "final_score": 0.1, "driver_rank": 9, "lag": 1}])
    ranked_before = ranked.copy(deep=True)

    out = build_final_review_summary(evidence, ranked_features=ranked)

    assert out.loc[0, "independent_predictive_support"] == "supported"
    assert out.loc[0, "confounder_assessment"] == "no_flagged_confounder"
    assert out.loc[0, "control_relation_assessment"] == "no_control_relation_flagged"
    assert out.loc[0, "statistical_limitation"] == "no_flagged_statistical_limitation"
    assert out.loc[0, "direction_assessment"] == "variable_leads_target"
    pd.testing.assert_frame_equal(ranked, ranked_before)


def test_ranked_lag_outside_maxlag_status_generates_hint():
    out = build_final_review_summary(_evidence([
        {
            "variable": "x1",
            "integrated_review_decision": "insufficient_evidence",
            "conditional_granger_status": "skipped: ranked lag outside maxlag",
        }
    ]))

    assert out.loc[0, "lag_boundary_hint"]
    assert "扩大 maxlag" in out.loc[0, "lag_boundary_hint"]
    assert "工艺停留时间" in out.loc[0, "lag_boundary_hint"]


def test_conditional_granger_conflict_uses_independent_predictive_evidence_language():
    out = build_final_review_summary(
        _evidence(
            [
                {
                    "variable": "x1",
                    "candidate_grade": "D",
                    "integrated_review_decision": "secondary_review",
                    "conditional_granger_status": "ok",
                    "conditional_fdr_q_value": 0.01,
                }
            ]
        )
    )

    reason = out.loc[0, "evidence_conflict_reason"]
    assert "条件 Granger 显示存在独立预测贡献证据" in reason
    assert "条件 Granger 支持" not in reason
