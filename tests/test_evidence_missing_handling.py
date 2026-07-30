from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr.screening import final_ranked_features


def _evaluate(prediction: float | None) -> pd.Series:
    variable_only = pd.DataFrame(columns=["variable"])
    model_lift = (
        pd.DataFrame([{"variable": "x", "model_lift_score": prediction, "status": "ok"}])
        if prediction is not None else variable_only
    )
    return final_ranked_features(
        pd.DataFrame([{"variable": "x", "score": 0.8, "innovation_score": 0.8, "lag": 1}]),
        variable_only,
        pd.DataFrame([{"variable": "x", "regime_stability_final": 0.7}]),
        model_lift,
        pd.DataFrame([{"variable": "x", "risk_flags": ""}]),
        pd.DataFrame([{
            "variable": "x", "lag_quality": 0.5,
            "temporal_direction_status": "variable_leads_supported",
        }]),
        variable_only,
    ).iloc[0]


def test_core_evidence_is_complete_when_association_quality_and_direction_exist():
    row = _evaluate(prediction=0.6)

    assert row["evidence_available_count"] == 3
    assert row["evidence_completeness"] == 1.0
    assert row["evidence_missing_items"] == ""
    assert row["evidence_strength"] == pytest.approx(0.8)


def test_missing_prediction_is_not_a_missing_initial_score_item():
    missing = _evaluate(prediction=None)
    zero = _evaluate(prediction=0.0)

    for field in ["evidence_available_count", "evidence_completeness", "evidence_strength"]:
        assert missing[field] == zero[field]
    assert "模型提升" not in missing["evidence_missing_items"]


def test_initial_score_source_has_no_optional_evidence_component_table():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")

    assert "INDUSTRIAL_SCORE_WEIGHT_PROFILES" not in source
    assert "_available_weight_profile_scores" not in source
    assert 'final["evidence_strength"] = final["association_score"]' in source
