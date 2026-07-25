from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.screening import (
    INDUSTRIAL_SCORE_COMPONENTS,
    INDUSTRIAL_SCORE_WEIGHT_PROFILES,
    _available_weight_profile_scores,
    final_ranked_features,
)


def _evaluate(prediction: float | None) -> pd.Series:
    variable_only = pd.DataFrame(columns=["variable"])
    model_lift = (
        pd.DataFrame(
            [{"variable": "x", "model_lift_score": prediction, "status": "ok"}]
        )
        if prediction is not None
        else variable_only
    )
    return final_ranked_features(
        pd.DataFrame(
            [{"variable": "x", "score": 0.8, "innovation_score": 0.8, "lag": 1}]
        ),
        variable_only,
        pd.DataFrame([{"variable": "x", "regime_stability_final": 0.7}]),
        model_lift,
        pd.DataFrame([{"variable": "x", "risk_flags": ""}]),
        pd.DataFrame([{"variable": "x", "lag_quality": 0.5}]),
        variable_only,
    ).iloc[0]


def test_all_four_evidence_components_use_full_profile():
    row = _evaluate(prediction=0.6)

    assert row["evidence_available_count"] == 4
    assert row["evidence_missing_items"] == ""
    assert "Layer 1 关联未获得" in row["four_layer_missing_items"]
    assert "Layer 3 独立性未获得" in row["four_layer_missing_items"]
    assert row["evidence_completeness"] == 1.0
    assert row["evidence_confidence"] == 1.0
    assert row["evidence_strength"] == pytest.approx((0.8 + 0.6 + 0.7 + 0.5) / 4)


def test_missing_prediction_is_not_treated_as_zero():
    missing = _evaluate(prediction=None)
    zero = _evaluate(prediction=0.0)

    assert missing["evidence_available_count"] == 3
    assert "模型提升" in missing["evidence_missing_items"]
    assert missing["evidence_strength"] == pytest.approx((0.8 + 0.7 + 0.5) / 3)
    assert missing["evidence_completeness"] == pytest.approx(0.75)
    assert missing["evidence_confidence"] == pytest.approx(1.0)

    assert zero["evidence_available_count"] == 4
    assert "模型提升" not in zero["evidence_missing_items"]
    assert zero["evidence_strength"] == pytest.approx((0.8 + 0.0 + 0.7 + 0.5) / 4)
    assert zero["evidence_completeness"] == 1.0
    assert zero["evidence_confidence"] == 1.0
    assert missing["evidence_strength"] != pytest.approx(zero["evidence_strength"])


def test_profile_score_renormalizes_only_available_component_weights():
    components = pd.DataFrame(
        [{"association": 0.8, "prediction": np.nan, "stability": 0.7, "lag_quality": 0.5}]
    )
    weights = {
        "association": 0.10,
        "prediction": 0.40,
        "stability": 0.30,
        "lag_quality": 0.20,
    }

    score = _available_weight_profile_scores(components, [weights]).iloc[0, 0]

    assert score == pytest.approx((0.10 * 0.8 + 0.30 * 0.7 + 0.20 * 0.5) / 0.60)


def test_profile_sampling_is_fair_across_all_four_components():
    marginal_weights = {
        component: sorted(profile[component] for profile in INDUSTRIAL_SCORE_WEIGHT_PROFILES)
        for component in INDUSTRIAL_SCORE_COMPONENTS
    }

    reference = marginal_weights[INDUSTRIAL_SCORE_COMPONENTS[0]]
    assert all(
        weights == reference
        for component, weights in marginal_weights.items()
        if component != INDUSTRIAL_SCORE_COMPONENTS[0]
    )


def test_missing_components_are_not_filled_with_zero_for_evidence_strength():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")
    profile_block = source.split("components = pd.DataFrame", 1)[1].split(
        'final["score_method"]', 1
    )[0]

    for forbidden in [
        'prediction_score"].fillna(0',
        'stability_score"].fillna(0',
        'lag_quality"].fillna(0',
    ]:
        assert forbidden not in profile_block
    assert "prediction < association" not in source
    assert "stability < lag_quality" not in source
