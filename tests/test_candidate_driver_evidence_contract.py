from __future__ import annotations

import pandas as pd
import pytest

from chem_ts_corr.screening import final_ranked_features


def _ranked(
    rows: list[tuple[str, float, float, float, float]],
    risks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ranked = pd.DataFrame(
        [
            {"variable": variable, "score": association, "innovation_score": association, "lag": 1, "direction": ""}
            for variable, association, _, _, _ in rows
        ]
    )
    values = pd.DataFrame(
        [
            {"variable": variable, "rolling_stability": stability, "lag_quality": lag_quality, "model_lift_score": predictive}
            for variable, _, stability, lag_quality, predictive in rows
        ]
    )
    empty = pd.DataFrame(columns=["variable"])
    return final_ranked_features(
        ranked,
        empty,
        values[["variable", "rolling_stability"]],
        values[["variable", "model_lift_score"]],
        pd.DataFrame(columns=["variable"]) if risks is None else risks,
        values[["variable", "lag_quality"]],
        values[["variable", "rolling_stability"]],
    )


def test_four_layer_scores_are_traceable_without_changing_final_score_formula():
    result = _ranked([("A", 0.8, 0.7, 0.6, 0.5)]).iloc[0]

    for field in ["association_score", "temporal_score", "independent_score", "predictive_score", "data_quality_score", "driver_evidence_summary"]:
        assert field in result.index
    expected = min(
        result["evidence_strength"] * result["evidence_confidence"] * (1 - result["risk_penalty_rate"]),
        result["risk_score_cap"],
    )
    assert result["final_score"] == pytest.approx(expected)
    assert result["candidate_driver_score"] == pytest.approx(result["final_score"])
    assert "Layer1关联" in result["driver_evidence_summary"]
    assert "Layer4预测" in result["driver_evidence_summary"]


def test_temporal_evidence_can_outweigh_slightly_higher_association():
    result = _ranked(
        [
            ("association_only", 0.95, 0.2, 0.2, 0.5),
            ("temporally_consistent", 0.80, 0.95, 0.95, 0.5),
        ]
    ).set_index("variable")

    assert result.loc["temporally_consistent", "temporal_score"] > result.loc["association_only", "temporal_score"]
    assert result.loc["temporally_consistent", "candidate_driver_score"] > result.loc["association_only", "candidate_driver_score"]


def test_common_driver_risk_is_retained_as_a_warning_not_a_candidate_deletion():
    risks = pd.DataFrame([{"variable": "shared_load", "risk_flags": "common_capacity_driver", "risk_count": 1}])
    result = _ranked([("shared_load", 0.9, 0.9, 0.9, 0.9)], risks).iloc[0]

    assert result["variable"] == "shared_load"
    assert "common_capacity_driver" in result["risk_flags"]


def test_predictive_evidence_cannot_override_stronger_association_and_temporal_evidence():
    result = _ranked(
        [
            ("statistically_supported", 0.9, 0.9, 0.9, 0.1),
            ("prediction_only", 0.3, 0.2, 0.2, 1.0),
        ]
    ).set_index("variable")

    assert result.loc["prediction_only", "predictive_score"] > result.loc["statistically_supported", "predictive_score"]
    assert result.loc["statistically_supported", "candidate_driver_score"] > result.loc["prediction_only", "candidate_driver_score"]
