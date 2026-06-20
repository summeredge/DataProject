import pandas as pd

from chem_ts_corr.causal_review_service import REPORT_COLUMNS, build_causal_review_report


def test_causal_review_report_priority_review():
    candidates = pd.DataFrame(
        [
            {
                "variable": "x1",
                "review_priority": 1,
                "review_tier": "tier_1",
                "review_reason": "优先复核",
            }
        ]
    )
    ranked = pd.DataFrame(
        [
            {
                "variable": "x1",
                "candidate_grade": "A",
                "final_score": 0.91,
                "recommended_use": "prediction_candidate",
                "recommended_action": "manual_review",
            }
        ]
    )
    conditional = pd.DataFrame(
        [
            {
                "variable": "x1",
                "status": "ok",
                "best_lag": 2,
                "min_p_value": 0.01,
                "fdr_q_value": 0.03,
                "predictive_contribution": 0.08,
            }
        ]
    )

    out = build_causal_review_report(ranked, candidates, conditional)

    row = out.iloc[0]
    assert row["final_review_decision"] == "priority_review"
    assert row["conditional_granger_status"] == "ok"
    assert row["conditional_best_lag"] == 2
    assert row["candidate_grade"] == "A"


def test_causal_review_report_insufficient_evidence():
    candidates = pd.DataFrame([{"variable": "x1"}])
    conditional = pd.DataFrame(
        [
            {
                "variable": "x1",
                "status": "skipped: insufficient rows",
                "predictive_contribution": 0.0,
            }
        ]
    )

    out = build_causal_review_report(pd.DataFrame(), candidates, conditional)

    assert out.iloc[0]["final_review_decision"] == "insufficient_evidence"


def test_causal_review_report_handles_missing_inputs():
    candidates = pd.DataFrame([{"variable": "x1"}])

    out = build_causal_review_report(
        ranked_features=pd.DataFrame([{"variable": "x1", "candidate_grade": "B"}]),
        causal_review_candidates=candidates,
        conditional_granger_scores=pd.DataFrame(),
        risk_flags=None,
    )

    assert list(out.columns) == REPORT_COLUMNS
    assert out.iloc[0]["variable"] == "x1"
    assert out.iloc[0]["candidate_grade"] == "B"
    assert out.iloc[0]["final_review_decision"] == "insufficient_evidence"


def test_causal_review_report_does_not_claim_causality():
    candidates = pd.DataFrame([{"variable": "x1"}])
    conditional = pd.DataFrame(
        [
            {
                "variable": "x1",
                "status": "ok",
                "fdr_q_value": 0.2,
                "predictive_contribution": 0.01,
            }
        ]
    )

    out = build_causal_review_report(pd.DataFrame(), candidates, conditional)
    row = out.iloc[0]

    assert "not a causal conclusion" in row["interpretation"]
    assert "不是因果结论" in row["final_review_reason"]


def test_causal_review_report_does_not_mutate_inputs_and_fills_missing_values():
    ranked = pd.DataFrame([{"variable": "x1", "final_score": 0.7}])
    candidates = pd.DataFrame([{"variable": "x1", "final_score": pd.NA}])
    conditional = pd.DataFrame(
        [{"variable": "x1", "status": "ok", "fdr_q_value": 0.2, "predictive_contribution": 0.0}]
    )
    original_candidates = candidates.copy(deep=True)

    out = build_causal_review_report(ranked, candidates, conditional)

    assert float(out.iloc[0]["final_score"]) == 0.7
    pd.testing.assert_frame_equal(candidates, original_candidates)


def test_causal_review_report_risk_limited_takes_precedence():
    ranked = pd.DataFrame([{"variable": "x1", "recommended_use": "capacity_driven"}])
    candidates = pd.DataFrame([{"variable": "x1"}])
    conditional = pd.DataFrame(
        [
            {
                "variable": "x1",
                "status": "ok",
                "fdr_q_value": 0.01,
                "predictive_contribution": 0.08,
            }
        ]
    )
    risks = pd.DataFrame([{"variable": "x1", "risk_flags": "common_capacity_driver"}])

    out = build_causal_review_report(ranked, candidates, conditional, risk_flags=risks)
    row = out.iloc[0]

    assert row["final_review_decision"] == "risk_limited_review"
    assert "not a causal conclusion" in row["interpretation"]
    assert "不是因果结论" in row["final_review_reason"]


def test_causal_review_report_high_collinearity_with_contribution_is_stat_limited():
    candidates = pd.DataFrame([{"variable": "x1"}])
    conditional = pd.DataFrame(
        [
            {
                "variable": "x1",
                "status": "high_collinearity_risk",
                "fdr_q_value": pd.NA,
                "predictive_contribution": 0.04,
            }
        ]
    )

    out = build_causal_review_report(pd.DataFrame(), candidates, conditional)

    assert out.iloc[0]["final_review_decision"] == "manual_review_only"
    assert "高共线性" in out.iloc[0]["final_review_reason"]
