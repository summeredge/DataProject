from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.screening import (
    INDUSTRIAL_SCORE_WEIGHT_PROFILES,
    final_ranked_features,
)


def _frame(values: dict[str, object] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=["variable"]) if values is None else pd.DataFrame([values])


def _score(
    *,
    association: float,
    innovation: object = None,
    prediction: object = None,
    rolling: object = None,
    regime: object = None,
    lag_quality: object = None,
) -> pd.Series:
    ranked = pd.DataFrame(
        [{"variable": "x", "score": association, "innovation_score": innovation, "lag": 1}]
    )
    model = (
        _frame()
        if prediction is None
        else _frame({"variable": "x", "model_lift_score": prediction, "status": "ok"})
    )
    stability = (
        _frame()
        if regime is None
        else _frame({"variable": "x", "regime_stability_final": regime})
    )
    rolling_frame = (
        _frame() if rolling is None else _frame({"variable": "x", "rolling_stability": rolling})
    )
    lag = (
        _frame() if lag_quality is None else _frame({"variable": "x", "lag_quality": lag_quality})
    )
    risks = _frame({"variable": "x", "risk_flags": "", "data_quality_score": 1.0})
    return final_ranked_features(
        ranked, _frame(), stability, model, risks, lag, rolling_frame
    ).iloc[0]


def test_weight_profiles_cover_reasonable_engineering_choices_without_one_magic_formula():
    profiles = INDUSTRIAL_SCORE_WEIGHT_PROFILES

    assert len(profiles) > 10
    assert all(sum(profile.values()) == pytest.approx(1.0) for profile in profiles)
    assert all(0.10 <= value <= 0.40 for profile in profiles for value in profile.values())
    assert all(profile["prediction"] >= profile["association"] for profile in profiles)
    assert all(profile["stability"] >= profile["lag_quality"] for profile in profiles)


def test_innovation_disagreement_reduces_slow_trend_association():
    row = _score(
        association=0.90,
        innovation=0.01,
        prediction=0.0,
        rolling=0.20,
        lag_quality=0.10,
    )

    assert row["association_score"] == pytest.approx(0.90)
    assert row["innovation_score"] == pytest.approx(0.01)
    assert row["correlation_evidence_score"] == pytest.approx(np.sqrt(0.90 * 0.01))
    assert row["evidence_score"] < 0.20


def test_prediction_and_stability_can_outrank_higher_raw_correlation():
    empty = _frame()
    ranked = pd.DataFrame(
        [
            {"variable": "trend", "score": 0.90, "innovation_score": 0.02, "lag": 1},
            {"variable": "driver", "score": 0.55, "innovation_score": 0.60, "lag": 2},
        ]
    )
    model = pd.DataFrame(
        [
            {"variable": "trend", "model_lift_score": 0.0, "status": "ok"},
            {"variable": "driver", "model_lift_score": 0.70, "status": "ok"},
        ]
    )
    rolling = pd.DataFrame(
        [
            {"variable": "trend", "rolling_stability": 0.15},
            {"variable": "driver", "rolling_stability": 0.80},
        ]
    )
    lag = pd.DataFrame(
        [
            {"variable": "trend", "lag_quality": 0.10},
            {"variable": "driver", "lag_quality": 0.75},
        ]
    )
    risks = pd.DataFrame(
        [
            {"variable": "trend", "risk_flags": "", "data_quality_score": 1.0},
            {"variable": "driver", "risk_flags": "", "data_quality_score": 1.0},
        ]
    )

    result = final_ranked_features(ranked, empty, empty, model, risks, lag, rolling)

    assert result["variable"].tolist() == ["driver", "trend"]
    assert result.set_index("variable").loc["driver", "evidence_score"] > result.set_index(
        "variable"
    ).loc["trend", "evidence_score"]


def test_missing_core_evidence_lowers_completeness_instead_of_renormalizing_raw_correlation():
    incomplete = _score(association=0.80)
    complete = _score(
        association=0.80,
        innovation=0.80,
        prediction=0.80,
        rolling=0.80,
        lag_quality=0.80,
    )

    assert incomplete["evidence_completeness"] < complete["evidence_completeness"]
    assert incomplete["evidence_score"] < complete["evidence_score"]
    assert complete["evidence_score"] == pytest.approx(0.80)


def test_regime_and_time_stability_are_combined_conservatively():
    row = _score(
        association=0.70,
        innovation=0.70,
        prediction=0.70,
        rolling=0.90,
        regime=0.40,
        lag_quality=0.70,
    )

    assert row["stability_score"] == pytest.approx(np.sqrt(0.90 * 0.40))
