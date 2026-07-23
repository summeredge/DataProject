from __future__ import annotations

import pandas as pd
import pytest

from chem_ts_corr.causal_review_evidence import build_causal_review_evidence
from chem_ts_corr.final_review_summary import build_final_review_summary
from chem_ts_corr.report import write_outputs


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
