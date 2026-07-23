from __future__ import annotations

import inspect

import pandas as pd

from chem_ts_corr.report import build_recommended_candidates
from chem_ts_corr.screening import final_ranked_features, reorder_ranked_features
from chem_ts_corr import web


def _ranked_result() -> pd.DataFrame:
    ranked = pd.DataFrame(
        [
            {"variable": "A", "score": 0.82, "innovation_score": 0.82, "lag": 1, "direction": "领先"},
            {"variable": "B", "score": 0.60, "innovation_score": 0.60, "lag": 1, "direction": "领先"},
            {"variable": "C", "score": 0.74, "innovation_score": 0.74, "lag": 1, "direction": "领先"},
            {"variable": "ctrl", "score": 0.90, "innovation_score": 0.90, "lag": 1, "direction": "领先"},
        ]
    )
    values = ranked[["variable", "score"]]
    empty = pd.DataFrame(columns=["variable"])
    return final_ranked_features(
        ranked,
        empty,
        empty,
        values.rename(columns={"score": "model_lift_score"}).assign(status="ok"),
        pd.DataFrame([{"variable": "C", "risk_flags": "target_leads_variable"}]),
        values.rename(columns={"score": "lag_quality"}),
        values.rename(columns={"score": "rolling_stability"}),
    )


def _records(*items: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{"variable": variable, "new_status": status} for variable, status in items])


def test_confirmed_recommendation_promotes_normal_candidate_without_changing_evidence():
    original = _ranked_result()
    before = original.set_index("variable")

    reordered = reorder_ranked_features(
        original,
        candidate_decision_records=_records(("C", "confirmed_recommendation")),
    ).set_index("variable")

    assert reordered.loc["C", "driver_priority_factor"] == 1.0
    assert reordered.loc["C", "driver_priority_score"] == reordered.loc["C", "final_score"]
    assert reordered.loc["C", "driver_rank"] < before.loc["C", "driver_rank"]
    assert reordered.loc["C", "final_score"] == before.loc["C", "final_score"]
    assert reordered.loc["C", "evidence_score"] == before.loc["C", "evidence_score"]
    assert reordered.loc["C", "association_rank"] == before.loc["C", "association_rank"]


def test_confirmed_recommendation_keeps_confirmed_closed_loop_cap():
    original = _ranked_result()
    evidence = pd.DataFrame([{"variable": "A", "closed_loop_evidence_level": "confirmed"}])

    reordered = reorder_ranked_features(
        original,
        closed_loop_evidence=evidence,
        candidate_decision_records=_records(("A", "confirmed_recommendation")),
    ).set_index("variable")

    assert reordered.loc["A", "driver_priority_factor"] <= 0.55
    assert reordered.loc["A", "recommended_use"] == "closed_loop_confirmed"


def test_excluded_candidate_stays_out_of_topk_and_recommended_candidates_with_controls():
    reordered = reorder_ranked_features(
        _ranked_result(),
        candidate_decision_records=_records(("A", "excluded_recommendation")),
        top_k=3,
        control_columns=["ctrl"],
    )

    assert "A" not in reordered["variable"].values
    recommended = build_recommended_candidates(reordered)
    assert "A" not in recommended["variable"].values
    assert "ctrl" not in reordered["variable"].values

    forced = reorder_ranked_features(
        _ranked_result(),
        candidate_decision_records=_records(("A", "excluded_recommendation")),
        force_include_variables=["A"],
    )
    assert "A" not in forced["variable"].values


def test_needs_review_only_changes_recommendation_status():
    original = _ranked_result()
    baseline = reorder_ranked_features(original).set_index("variable")
    reviewed = reorder_ranked_features(
        original,
        candidate_decision_records=_records(("B", "needs_review")),
    ).set_index("variable")

    for column in ["driver_priority_factor", "driver_priority_score", "driver_rank"]:
        assert reviewed.loc["B", column] == baseline.loc["B", column]
    assert reviewed.loc["B", "recommended_use"] == "manual_review_required"
    assert reviewed.loc["B", "recommended_action"] == "人工标记需复核"


def test_latest_candidate_decision_wins_and_input_is_not_mutated():
    original = _ranked_result()
    snapshot = original.copy(deep=True)

    reordered = reorder_ranked_features(
        original,
        candidate_decision_records=_records(
            ("A", "unknown"),
            ("A", "confirmed_recommendation"),
            ("A", "excluded_recommendation"),
        ),
        top_k=3,
    )

    pd.testing.assert_frame_equal(original, snapshot)
    assert "A" not in reordered["variable"].values


def test_result_page_reorder_uses_persisted_results_without_rebuilding_analysis_inputs():
    source = inspect.getsource(web._update_candidate_decision_response)

    assert "candidateDecisionControls" in web.INDEX_HTML
    assert "applyCandidateDecision" in web.INDEX_HTML
    assert "confirmed_recommendation" in web.INDEX_HTML
    assert 'postForm("/api/update_candidate_decision", form)' in web.INDEX_HTML
    assert "reorder_ranked_features(" in source
    for forbidden_input in [
        "residual_corr_scores.csv",
        "model_lift_scores.csv",
        "risk_flags.csv",
        "lag_peak_quality.csv",
        "rolling_corr_scores.csv",
    ]:
        assert forbidden_input not in source
    assert 'to_csv(output_dir / "reordered_recommendations.csv"' in source
    assert 'to_csv(output_dir / "ranked_features.csv"' not in source
    assert "candidate_decision_records.json" in web.DOWNLOAD_FILES
    assert "reordered_recommendations.csv" in web.DOWNLOAD_FILES
    assert "recommended_candidates_reordered.csv" in web.DOWNLOAD_FILES
