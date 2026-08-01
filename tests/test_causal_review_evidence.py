import pandas as pd
import pytest

from chem_ts_corr.causal_review_evidence import EVIDENCE_COLUMNS, build_causal_review_evidence
from chem_ts_corr.causal_review import build_causal_review_candidates
from chem_ts_corr.final_review_summary import build_final_review_summary


def _ranked(**extra):
    row = {
        "variable": "x1",
        "candidate_grade": "A",
        "final_score": 0.9,
        "lag": 1,
        "direction": "positive",
        "risk_level": "none",
        "risk_flags": "",
        "risk_count": 0,
        "recommended_use": "prediction_candidate",
        "recommended_action": "review",
    }
    row.update(extra)
    return pd.DataFrame([row])


def test_only_ranked_features_outputs_fixed_columns():
    out = build_causal_review_evidence(
        ranked_features=_ranked(candidate_grade="C"),
        conditional_granger_scores=pd.DataFrame(),
    )

    assert list(out.columns) == EVIDENCE_COLUMNS
    assert out.iloc[0]["variable"] == "x1"
    assert out.iloc[0]["interpretation"] == "integrated review evidence only; not a causal conclusion"


def test_significant_conditional_granger_low_risk_gets_priority_review():
    conditional = pd.DataFrame([
        {
            "variable": "x1",
            "status": "ok",
            "best_lag": 1,
            "min_p_value": 0.001,
            "fdr_q_value": 0.01,
            "predictive_contribution": 0.08,
        }
    ])

    out = build_causal_review_evidence(_ranked(), conditional)
    row = out.iloc[0]

    assert row["evidence_level"] == "strong_predictive_evidence"
    assert row["integrated_review_decision"] == "priority_review"
    assert "conditional_granger_supported" in row["evidence_reason"]


def test_engineering_context_does_not_change_review_priority_or_decision():
    conditional = pd.DataFrame([
        {"variable": "x1", "status": "ok", "best_lag": 1, "min_p_value": 0.001, "fdr_q_value": 0.01, "predictive_contribution": 0.08}
    ])
    baseline = _ranked()
    contextual = _ranked(
        engineering_context='{"source": "engineering_note"}',
    )

    pd.testing.assert_frame_equal(
        build_causal_review_candidates(baseline),
        build_causal_review_candidates(contextual),
    )
    pd.testing.assert_frame_equal(
        build_causal_review_evidence(baseline, conditional),
        build_causal_review_evidence(contextual, conditional),
    )


def test_high_collinearity_adds_limited_signal_without_p_value_support():
    conditional = pd.DataFrame([
        {
            "variable": "x1",
            "status": "high_collinearity_risk",
            "best_lag": 1,
            "min_p_value": 0.001,
            "fdr_q_value": 0.001,
            "predictive_contribution": 0.04,
        }
    ])

    out = build_causal_review_evidence(_ranked(candidate_grade="D"), conditional)
    row = out.iloc[0]

    assert row["risk_constraint_level"] == "medium"
    assert row["evidence_score"] == 0.8
    assert "high_collinearity_limited_signal" in row["evidence_reason"]
    assert "conditional_granger_supported" not in row["evidence_reason"]


def test_strong_formula_and_severe_data_quality_limit_to_manual_review_only():
    risks = pd.DataFrame([
        {"variable": "x1", "risk_flags": "strong_formula_leakage;severe_data_quality", "risk_level": "strong"}
    ])
    conditional = pd.DataFrame([
        {"variable": "x1", "status": "ok", "fdr_q_value": 0.01, "predictive_contribution": 0.1}
    ])

    out = build_causal_review_evidence(_ranked(risk_flags=""), conditional, risk_flags=risks)
    row = out.iloc[0]

    assert row["risk_constraint_level"] == "strong"
    assert row["evidence_level"] == "risk_limited_evidence"
    assert row["integrated_review_decision"] == "manual_review_only"
    assert "severe_data_quality_risk" in row["integrated_review_reason"]


def test_enhanced_validation_summary_increases_evidence_score():
    enhanced = pd.DataFrame([
        {
            "variable": "x1",
            "model_lift": 0.06,
            "rolling_stability": 0.75,
            "rolling_sign_consistency": 0.85,
            "status": "ok",
        }
    ])

    out = build_causal_review_evidence(
        _ranked(candidate_grade="C"),
        pd.DataFrame(),
        enhanced_validation_summary=enhanced,
    )
    row = out.iloc[0]

    assert row["evidence_score"] == 2.8
    assert "model_lift_supported" in row["evidence_reason"]
    assert "rolling_stability_supported" in row["evidence_reason"]


def test_model_variable_importance_top_rank_adds_model_explanation_support():
    model_importance = pd.DataFrame([
        {"variable": "x1", "importance_rank": 3, "max_importance": 0.2, "best_model_lag": 2}
    ])

    out = build_causal_review_evidence(
        _ranked(candidate_grade="D"),
        pd.DataFrame(),
        model_variable_importance=model_importance,
    )
    row = out.iloc[0]

    assert row["model_explanation_support"] == "model_explanation_support"
    assert row["model_importance_rank"] == 3
    assert "model_explanation_support" in row["evidence_reason"]


def test_missing_optional_evidence_tables_do_not_error():
    out = build_causal_review_evidence(
        ranked_features=_ranked(candidate_grade="B"),
        conditional_granger_scores=pd.DataFrame(columns=["variable"]),
        risk_flags=None,
        enhanced_validation_summary=None,
        granger_tests=None,
        model_variable_importance=None,
    )

    assert len(out) == 1
    assert out.iloc[0]["candidate_grade"] == "B"


def test_inputs_are_not_mutated():
    ranked = _ranked()
    conditional = pd.DataFrame([{"variable": "x1", "status": "ok", "fdr_q_value": 0.02}])
    risks = pd.DataFrame([{"variable": "x1", "risk_flags": "lag_boundary"}])
    ranked_before = ranked.copy(deep=True)
    conditional_before = conditional.copy(deep=True)
    risks_before = risks.copy(deep=True)

    build_causal_review_evidence(ranked, conditional, risk_flags=risks)

    pd.testing.assert_frame_equal(ranked, ranked_before)
    pd.testing.assert_frame_equal(conditional, conditional_before)
    pd.testing.assert_frame_equal(risks, risks_before)


def test_strong_data_high_collinearity_preserves_priority_with_statistical_limit():
    conditional = pd.DataFrame([
        {
            "variable": "x1",
            "status": "high_collinearity_risk",
            "best_lag": 1,
            "min_p_value": 0.001,
            "fdr_q_value": 0.001,
            "predictive_contribution": 0.04,
        }
    ])

    out = build_causal_review_evidence(_ranked(candidate_grade="A"), conditional)
    row = out.iloc[0]

    assert row["data_priority"] == "high"
    assert row["statistical_limit_level"] in {"medium", "strong"}
    assert row["integrated_review_decision"] == "priority_review_with_statistical_limit"
    assert "statistical_test_limited" in row["integrated_review_reason"]
    assert "high_collinearity" in row["integrated_review_reason"]


def test_strong_data_common_capacity_driver_is_not_rejected():
    risks = pd.DataFrame([
        {"variable": "x1", "risk_flags": "common_capacity_driver", "risk_level": "medium"}
    ])
    model_importance = pd.DataFrame([
        {"variable": "x1", "importance_rank": 4, "max_importance": 0.2, "best_model_lag": 2}
    ])

    out = build_causal_review_evidence(
        _ranked(candidate_grade="A", risk_flags=""),
        pd.DataFrame(),
        risk_flags=risks,
        model_variable_importance=model_importance,
    )
    row = out.iloc[0]

    assert row["data_priority"] == "high"
    assert row["statistical_limit_reason"] == "common_capacity_driver"
    assert row["integrated_review_decision"] in {
        "priority_review_with_statistical_limit",
        "secondary_review_with_statistical_limit",
    }
    assert row["integrated_review_decision"] != "not_recommended"


@pytest.mark.parametrize("other_risk", ["strong_formula_leakage", "target_leads_variable"])
def test_severe_data_quality_hard_downgrade_takes_precedence_over_strong_data(other_risk: str):
    risks = pd.DataFrame([
        {"variable": "x1", "risk_flags": f"severe_data_quality;{other_risk}", "risk_level": "strong"}
    ])
    conditional = pd.DataFrame([
        {"variable": "x1", "status": "ok", "fdr_q_value": 0.01, "predictive_contribution": 0.1}
    ])

    out = build_causal_review_evidence(_ranked(candidate_grade="A", risk_flags=""), conditional, risk_flags=risks)
    row = out.iloc[0]

    assert row["data_priority"] == "high"
    assert row["integrated_review_decision"] == "manual_review_only"
    assert row["risk_constraint_level"] == "strong"
    assert "hard_downgrade_risk" in row["integrated_review_reason"]


def test_poor_data_quality_is_warning_without_hard_downgrade():
    conditional = pd.DataFrame([
        {"variable": "x1", "status": "ok", "fdr_q_value": 0.01, "predictive_contribution": 0.1}
    ])
    risks = pd.DataFrame([
        {"variable": "x1", "risk_flags": "poor_data_quality;lag_boundary", "risk_level": "weak"}
    ])

    row = build_causal_review_evidence(_ranked(risk_flags=""), conditional, risk_flags=risks).iloc[0]

    assert row["integrated_review_decision"] != "manual_review_only"
    assert row["risk_constraint_level"] == "weak"
    assert "hard_downgrade_risk" not in row["integrated_review_reason"]
    assert "poor_data_quality_warning" in row["integrated_review_reason"]


def test_low_risk_strong_evidence_keeps_priority_review():
    conditional = pd.DataFrame([
        {"variable": "x1", "status": "ok", "fdr_q_value": 0.01, "predictive_contribution": 0.1}
    ])

    out = build_causal_review_evidence(_ranked(candidate_grade="A", risk_flags="", risk_level="none"), conditional)
    row = out.iloc[0]

    assert float(row["evidence_score"]) >= 4
    assert row["statistical_limit_level"] == "none"
    assert row["integrated_review_decision"] == "priority_review"


def test_moderate_evidence_with_statistical_limit_records_limit_reason():
    risks = pd.DataFrame([
        {"variable": "x1", "risk_flags": "lag_boundary;residual_collinearity", "risk_level": "weak"}
    ])
    enhanced = pd.DataFrame([
        {"variable": "x1", "model_lift": 0.06, "rolling_stability": 0.75, "rolling_sign_consistency": 0.72}
    ])

    out = build_causal_review_evidence(
        _ranked(candidate_grade="C", risk_flags=""),
        pd.DataFrame(),
        risk_flags=risks,
        enhanced_validation_summary=enhanced,
    )
    row = out.iloc[0]

    assert row["evidence_level"] == "moderate_predictive_evidence"
    assert row["integrated_review_decision"] in {"secondary_review", "secondary_review_with_statistical_limit"}
    assert row["statistical_limit_reason"]
    assert "lag_boundary" in row["statistical_limit_reason"]


def test_fallback_missing_ranked_lag_is_limited_not_supported():
    conditional = pd.DataFrame([
        {
            "variable": "x1",
            "status": "ok: fallback_missing_ranked_lag",
            "best_lag": 1,
            "min_p_value": 0.001,
            "fdr_q_value": 0.01,
            "predictive_contribution": 0.08,
        }
    ])

    out = build_causal_review_evidence(_ranked(candidate_grade="D"), conditional)
    row = out.iloc[0]

    assert "conditional_granger_supported" not in row["evidence_reason"]
    assert "fallback_predictive_signal" in row["evidence_reason"]
    assert row["data_priority"] != "high"
    assert row["integrated_review_decision"] != "priority_review"


def test_high_collinearity_without_q_does_not_priority_from_grade_only():
    conditional = pd.DataFrame([
        {
            "variable": "x1",
            "status": "high_collinearity_risk",
            "best_lag": 1,
            "min_p_value": pd.NA,
            "fdr_q_value": pd.NA,
            "predictive_contribution": 0.04,
        }
    ])

    out = build_causal_review_evidence(_ranked(candidate_grade="A"), conditional)
    row = out.iloc[0]

    assert row["data_priority"] == "medium"
    assert row["integrated_review_decision"] != "priority_review_with_statistical_limit"
    assert row["integrated_review_decision"] in {"manual_review_only", "secondary_review_with_statistical_limit"}
    assert "high_collinearity_limited_signal" in row["integrated_review_reason"]


def test_high_collinearity_with_independent_model_support_can_keep_limited_priority():
    conditional = pd.DataFrame([
        {
            "variable": "x1",
            "status": "high_collinearity_risk",
            "best_lag": 1,
            "min_p_value": pd.NA,
            "fdr_q_value": pd.NA,
            "predictive_contribution": 0.04,
        }
    ])
    model_importance = pd.DataFrame([
        {"variable": "x1", "importance_rank": 2, "max_importance": 0.2, "best_model_lag": 1}
    ])

    out = build_causal_review_evidence(
        _ranked(candidate_grade="A"),
        conditional,
        model_variable_importance=model_importance,
    )
    row = out.iloc[0]

    assert row["data_priority"] == "high"
    assert row["integrated_review_decision"] == "priority_review_with_statistical_limit"
