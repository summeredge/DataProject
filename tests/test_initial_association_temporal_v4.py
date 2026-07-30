from __future__ import annotations

import pandas as pd
import pytest

from chem_ts_corr.screening import final_ranked_features


def _frame(row: dict[str, object] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=["variable"]) if row is None else pd.DataFrame([row])


def _score(
    association: float,
    *,
    data_quality: float = 1.0,
    temporal_status: str = "variable_leads_supported",
    innovation_score: float = 0.0,
    innovation_status: str = "innovation_verified",
    residual_corr: float = 0.0,
    model_lift_score: float = 0.0,
    rolling_stability: float = 0.0,
    regime_stability: float = 0.0,
    lag_quality: float = 0.0,
) -> pd.Series:
    ranked = pd.DataFrame([{
        "variable": "x", "score": association, "lag": 5,
        "innovation_score": innovation_score, "innovation_status": innovation_status,
    }])
    lag_peak = _frame({
        "variable": "x", "lag_quality": lag_quality, "lag_boundary_flag": False,
        "near_peak_lag_min": 5, "near_peak_lag_max": 12, "near_peak_lag_count": 2,
        "temporal_direction_status": temporal_status,
    })
    return final_ranked_features(
        ranked=ranked,
        residual=_frame({"variable": "x", "residual_corr": residual_corr}),
        stability=_frame({"variable": "x", "regime_stability_final": regime_stability}),
        model_lift=_frame({"variable": "x", "model_lift_score": model_lift_score, "status": "ok"}),
        risks=_frame({"variable": "x", "risk_flags": "", "data_quality_score": data_quality}),
        lag_peak_quality=lag_peak,
        rolling_corr_scores=_frame({"variable": "x", "rolling_stability": rolling_stability}),
    ).iloc[0]


def test_fic421002_regression_uses_association_and_data_quality_only():
    row = _score(
        0.438664,
        data_quality=0.997312,
        innovation_score=0.006014,
        innovation_status="innovation_verified",
    )

    assert row["evidence_strength"] == pytest.approx(0.438664)
    assert row["evidence_score"] == pytest.approx(0.438664 * 0.997312)
    assert row["final_score"] == pytest.approx(0.437484871168)


def test_ficq400001_conflict_does_not_enter_numeric_score():
    row = _score(
        0.391954,
        data_quality=0.999912,
        innovation_status="innovation_sign_conflict",
    )

    assert row["final_score"] == pytest.approx(0.391919508048)


def test_later_evidence_does_not_change_initial_score_or_grade():
    low = _score(0.6)
    high = _score(
        0.6,
        innovation_score=1.0,
        residual_corr=1.0,
        model_lift_score=1.0,
        rolling_stability=1.0,
        regime_stability=1.0,
        lag_quality=1.0,
    )

    for field in ["evidence_strength", "evidence_score", "final_score", "driver_rank", "candidate_grade"]:
        assert low[field] == high[field]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("variable_leads_supported", 0.8),
        ("direction_unresolved", 0.8),
        ("target_leads_supported", 0.25),
    ],
)
def test_only_supported_target_lead_is_temporally_penalized(status: str, expected: float):
    row = _score(0.8, temporal_status=status)

    assert row["final_score"] == pytest.approx(expected)


def test_target_lead_below_cap_keeps_multiplicative_penalty():
    row = _score(
        0.438664,
        data_quality=0.997312,
        temporal_status="target_leads_supported",
    )

    assert row["final_score"] == pytest.approx(0.218742435584)
    assert row["candidate_class"] == "downstream_response"
    assert row["recommended_use"] == "state_indicator"


def test_missing_core_data_quality_is_not_filled_or_scored():
    row = _score(0.8, data_quality=float("nan"))

    assert row["evidence_available_count"] == 2
    assert row["evidence_completeness"] == pytest.approx(2 / 3)
    assert row["evidence_missing_items"] == "数据质量"
    assert pd.isna(row["final_score"])
