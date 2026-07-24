from __future__ import annotations

import pandas as pd

from chem_ts_corr.closed_loop import build_closed_loop_evidence
from chem_ts_corr.screening import reorder_ranked_features


def _ranked() -> pd.DataFrame:
    return pd.DataFrame([
        {"variable": "A", "final_score": 0.9, "evidence_score": 0.9, "candidate_class": "upstream_driver_candidate", "driver_priority_factor": 1.0, "driver_priority_score": 0.9, "driver_rank": 1, "lag": 1, "raw_corr": 0.9, "risk_flags": ""},
        {"variable": "B", "final_score": 0.8, "evidence_score": 0.8, "candidate_class": "upstream_driver_candidate", "driver_priority_factor": 1.0, "driver_priority_score": 0.8, "driver_rank": 2, "lag": 1, "raw_corr": 0.8, "risk_flags": ""},
    ])


def _ranking_fields(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("variable")[["final_score", "evidence_score", "driver_priority_factor", "driver_priority_score", "driver_rank"]]


def test_manual_closed_loop_context_does_not_change_ranking():
    baseline = reorder_ranked_features(_ranked())
    context = build_closed_loop_evidence(pd.DataFrame(), manual_closed_loop_variables=["A"])
    output = reorder_ranked_features(_ranked(), closed_loop_evidence=context)

    pd.testing.assert_frame_equal(_ranking_fields(output), _ranking_fields(baseline))


def test_automatic_closed_loop_indicator_does_not_change_ranking():
    baseline = reorder_ranked_features(_ranked())
    context = build_closed_loop_evidence(pd.DataFrame([{"variable": "A", "closed_loop_suspect_flag": True}]))
    output = reorder_ranked_features(_ranked(), closed_loop_evidence=context)

    pd.testing.assert_frame_equal(_ranking_fields(output), _ranking_fields(baseline))


def test_closed_loop_context_fields_are_output_for_explanation():
    context = build_closed_loop_evidence(pd.DataFrame([{"variable": "A", "closed_loop_suspect_flag": True}]), manual_closed_loop_variables=["A"])
    output = reorder_ranked_features(_ranked(), closed_loop_evidence=context).set_index("variable")

    assert output.loc["A", "closed_loop_context"] == "manual_engineering_input_and_automatic_indicator"
    assert output.loc["A", "closed_loop_status"] == "possible_closed_loop_influence"
    assert "人工工程经验输入" in output.loc["A", "closed_loop_reason"]


def test_no_closed_loop_input_preserves_historical_ranking():
    baseline = reorder_ranked_features(_ranked())
    output = reorder_ranked_features(_ranked(), closed_loop_evidence=build_closed_loop_evidence(pd.DataFrame()))

    pd.testing.assert_frame_equal(_ranking_fields(output), _ranking_fields(baseline))
