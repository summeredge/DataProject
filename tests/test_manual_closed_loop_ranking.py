from __future__ import annotations

import pandas as pd

from chem_ts_corr.screening import final_ranked_features


def _run(evidence: pd.DataFrame | None = None) -> pd.DataFrame:
    ranked = pd.DataFrame([{"variable": "up", "score": 0.9, "innovation_score": 0.9, "lag": 1, "direction": ""}])
    values = ranked[["variable", "score"]]
    empty = pd.DataFrame(columns=["variable"])
    return final_ranked_features(
        ranked, empty, empty, values.rename(columns={"score": "model_lift_score"}), empty,
        values.rename(columns={"score": "lag_quality"}), values.rename(columns={"score": "rolling_stability"}),
        closed_loop_evidence=evidence,
    )


def test_manual_context_keeps_score_class_and_recommendation():
    baseline = _run()
    context = pd.DataFrame([{
        "variable": "up", "closed_loop_context": "manual_engineering_input",
        "closed_loop_status": "manual_context_requires_review", "closed_loop_reason": '["engineering input"]',
    }])
    output = _run(context)

    for field in ["candidate_class", "driver_priority_factor", "driver_priority_score", "driver_rank", "recommended_use", "recommended_action", "evidence_score", "final_score"]:
        assert output.loc[0, field] == baseline.loc[0, field]
    assert output.loc[0, "closed_loop_context"] == "manual_engineering_input"
