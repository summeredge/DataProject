from __future__ import annotations

import pandas as pd
import pytest

from chem_ts_corr.causal_review_evidence import build_causal_review_evidence
from chem_ts_corr.final_review_summary import build_final_review_summary
from chem_ts_corr.report import build_recommended_candidates, write_outputs


@pytest.mark.parametrize("recommended_use", ["closed_loop_confirmed", "closed_loop_conflict"])
def test_closed_loop_recommendations_are_excluded_from_fallback_and_final_priority(
    tmp_path, recommended_use: str
):
    variables = [f"loop_{index}" for index in range(10)]
    ranked = pd.DataFrame(
        [
            {
                "variable": variable,
                "candidate_grade": "A",
                "final_score": 1.0 - index * 0.01,
                "lag": 1,
                "direction": "",
                "risk_level": "none",
                "risk_flags": "",
                "risk_count": 0,
                "recommended_use": recommended_use,
                "recommended_action": "人工复核",
            }
            for index, variable in enumerate(variables)
        ]
    )
    conditional = pd.DataFrame(
        [
            {"variable": variable, "status": "ok", "fdr_q_value": 0.01, "predictive_contribution": 0.1}
            for variable in variables
        ]
    )

    evidence = build_causal_review_evidence(ranked, conditional)
    summary = build_final_review_summary(evidence)
    write_outputs(tmp_path, "target", ranked, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), metrics={})
    recommended = pd.read_csv(tmp_path / "recommended_candidates.csv", encoding="utf-8-sig")

    assert recommended.empty
    assert set(evidence["integrated_review_decision"]) == {"manual_review_only"}
    assert set(summary["final_recommendation"]) == {"manual_review_only"}


def test_confirmed_closed_loop_is_excluded_but_normal_ab_candidate_is_retained():
    ranked = pd.DataFrame(
        [
            {"variable": "A", "candidate_grade": "A", "final_score": 0.99, "recommended_use": "closed_loop_confirmed"},
            {"variable": "B", "candidate_grade": "B", "final_score": 0.80, "recommended_use": "upstream_driver_candidate"},
        ]
    )

    recommended = build_recommended_candidates(ranked)

    assert recommended["variable"].tolist() == ["B"]
    assert recommended.loc[recommended.index[0], "final_score"] == 0.80


def test_conflict_closed_loop_is_excluded_from_normal_recommendations():
    ranked = pd.DataFrame(
        [{"variable": "conflict", "candidate_grade": "A", "final_score": 0.99, "recommended_use": "closed_loop_conflict"}]
    )

    assert build_recommended_candidates(ranked).empty


def test_ordinary_candidates_keep_existing_fallback_order_and_values():
    ranked = pd.DataFrame(
        [
            {"variable": "first", "candidate_grade": "C", "final_score": 0.7, "recommended_use": "manual_review_required"},
            {"variable": "second", "candidate_grade": "D", "final_score": 0.6, "recommended_use": "manual_review_required"},
        ]
    )

    recommended = build_recommended_candidates(ranked)

    assert recommended["variable"].tolist() == ["first", "second"]
    assert recommended["final_score"].tolist() == [0.7, 0.6]
    assert recommended["fallback_reason"].tolist() == ["no_A_or_B_candidates"] * 2


def test_force_included_closed_loop_candidate_is_not_a_normal_recommendation():
    ranked = pd.DataFrame(
        [{"variable": "loop", "candidate_grade": "A", "final_score": 0.9, "recommended_use": "closed_loop_confirmed", "force_included": True}]
    )

    assert build_recommended_candidates(ranked).empty
