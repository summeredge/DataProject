from __future__ import annotations

import pandas as pd

from chem_ts_corr.closed_loop import build_closed_loop_evidence
from chem_ts_corr.screening import final_ranked_features


def _ranked(evidence: pd.DataFrame | None = None) -> pd.DataFrame:
    ranked = pd.DataFrame([
        {"variable": "A", "score": 0.9, "innovation_score": 0.9, "lag": 1, "direction": ""},
        {"variable": "B", "score": 0.8, "innovation_score": 0.8, "lag": 1, "direction": ""},
    ])
    values = ranked[["variable", "score"]]
    empty = pd.DataFrame(columns=["variable"])
    return final_ranked_features(
        ranked, empty, empty, values.rename(columns={"score": "model_lift_score"}), empty,
        values.rename(columns={"score": "lag_quality"}), values.rename(columns={"score": "rolling_stability"}),
        closed_loop_evidence=evidence,
    )


def _ranking_fields(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("variable")[["final_score", "evidence_score", "driver_priority_factor", "driver_priority_score", "driver_rank"]]


def test_manual_closed_loop_context_does_not_change_final_ranking():
    baseline = _ranked()
    context = build_closed_loop_evidence(pd.DataFrame(), manual_closed_loop_variables=["A"])
    output = _ranked(context)

    pd.testing.assert_frame_equal(_ranking_fields(output), _ranking_fields(baseline))


def test_manual_closed_and_non_closed_inputs_do_not_change_final_ranking():
    baseline = _ranked()
    context = build_closed_loop_evidence(
        pd.DataFrame(),
        manual_closed_loop_variables=["A"],
        manual_non_closed_loop_variables=["B"],
    )
    output = _ranked(context)

    pd.testing.assert_frame_equal(_ranking_fields(output), _ranking_fields(baseline))


def test_automatic_closed_loop_indicator_does_not_change_final_ranking():
    baseline = _ranked()
    context = build_closed_loop_evidence(pd.DataFrame([{"variable": "A", "closed_loop_suspect_flag": True}]))
    output = _ranked(context)

    pd.testing.assert_frame_equal(_ranking_fields(output), _ranking_fields(baseline))


def test_closed_loop_context_fields_are_output_for_explanation():
    context = build_closed_loop_evidence(pd.DataFrame([{"variable": "A", "closed_loop_suspect_flag": True}]), manual_closed_loop_variables=["A"])
    output = _ranked(context).set_index("variable")

    assert output.loc["A", "closed_loop_context"] == "manual_engineering_input_and_automatic_indicator"
    assert output.loc["A", "closed_loop_status"] == "possible_closed_loop_influence"
    assert "人工工程经验输入" in output.loc["A", "closed_loop_reason"]
