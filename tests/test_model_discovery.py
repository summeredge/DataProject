import pandas as pd
import pytest

from chem_ts_corr.model_discovery import (
    DISCOVERY_INTERPRETATION,
    OUT_COLS,
    VARIABLE_IMPORTANCE_COLS,
    build_model_discovered_candidates,
    build_model_variable_importance,
)


def test_model_discovery_empty_importance_returns_fixed_columns():
    out = build_model_discovered_candidates(pd.DataFrame(), pd.DataFrame())

    assert out.empty
    assert list(out.columns) == OUT_COLS


def test_model_discovery_flags_model_only_signal_outside_screening_top_n():
    importance = pd.DataFrame(
        [
            {"feature": "x3__lag_1", "importance": 0.9, "variable": "x3", "lag": 1},
            {"feature": "x1__lag_1", "importance": 0.8, "variable": "x1", "lag": 1},
        ]
    )
    ranked = pd.DataFrame(
        [
            {"variable": "x1", "final_score": 0.8},
            {"variable": "x2", "final_score": 0.7},
            {"variable": "x3", "final_score": 0.2},
        ]
    )

    out = build_model_discovered_candidates(importance, ranked, screening_top_n=2)
    row = out[out["variable"] == "x3"].iloc[0]

    assert bool(row["missing_from_screening_top_n"])
    assert "model_only_signal" in row["discovery_reason"]
    assert row["interpretation"] == DISCOVERY_INTERPRETATION


def test_model_discovery_counts_multiple_lag_features():
    importance = pd.DataFrame(
        [
            {"feature": "x1__lag_1", "importance": 0.9, "variable": "x1", "lag": 1},
            {"feature": "x1__lag_2", "importance": 0.8, "variable": "x1", "lag": 2},
            {"feature": "x1__lag_3", "importance": 0.7, "variable": "x1", "lag": 3},
        ]
    )

    out = build_model_discovered_candidates(importance, pd.DataFrame())
    row = out.iloc[0]

    assert int(row["model_feature_count"]) == 3
    assert int(row["nearby_lag_count"]) == 3
    assert "multi_lag_model_signal" in row["discovery_reason"]


def test_model_discovery_flags_model_lag_boundary_risk():
    importance = pd.DataFrame(
        [{"feature": "x1__lag_10", "importance": 0.9, "variable": "x1", "lag": 10}]
    )

    out = build_model_discovered_candidates(importance, pd.DataFrame(), max_lag=12)

    assert "model_lag_boundary_risk" in out.iloc[0]["discovery_reason"]


def test_model_discovery_does_not_mutate_inputs():
    importance = pd.DataFrame(
        [{"feature": "x1__lag_1", "importance": 0.9, "variable": "x1", "lag": 1}]
    )
    ranked = pd.DataFrame([{"variable": "x1", "final_score": 0.8, "risk_flags": "lag_boundary"}])
    risk = pd.DataFrame([{"variable": "x1", "risk_flags": "target_leads_variable"}])
    importance_before = importance.copy(deep=True)
    ranked_before = ranked.copy(deep=True)
    risk_before = risk.copy(deep=True)

    build_model_discovered_candidates(importance, ranked, risk_flags=risk)

    pd.testing.assert_frame_equal(importance, importance_before)
    pd.testing.assert_frame_equal(ranked, ranked_before)
    pd.testing.assert_frame_equal(risk, risk_before)


def test_model_variable_importance_summarizes_lags_and_ranks_variables():
    importance = pd.DataFrame(
        [
            {"feature": "x1__lag_1", "importance": 0.2, "method": "rf", "variable": "x1", "lag": 1},
            {"feature": "x1__lag_2", "importance": 0.8, "method": "rf", "variable": "x1", "lag": 2},
            {"feature": "x2__lag_0", "importance": 0.7, "method": "rf", "variable": "x2", "lag": 0},
            {"feature": "x2__lag_3", "importance": 0.6, "method": "rf", "variable": "x2", "lag": 3},
        ]
    )
    ranked = pd.DataFrame(
        [
            {"variable": "x2", "final_score": 0.91, "recommended_use": "prediction_candidate"},
            {"variable": "x1", "final_score": 0.72, "risk_flags": "lag_boundary"},
        ]
    )
    risk = pd.DataFrame(
        [{"variable": "x2", "risk_flags": "unstable_over_time", "recommended_action": "manual_review_required"}]
    )

    out = build_model_variable_importance(importance, ranked, risk_flags=risk)

    assert list(out.columns) == VARIABLE_IMPORTANCE_COLS
    assert list(out["variable"]) == ["x2", "x1"]
    x1 = out[out["variable"] == "x1"].iloc[0]
    x2 = out[out["variable"] == "x2"].iloc[0]
    assert x1["best_model_feature"] == "x1__lag_2"
    assert x1["best_model_lag"] == 2
    assert x1["feature_count"] == 2
    assert x1["total_importance"] == 1.0
    assert x2["total_importance"] == pytest.approx(1.3)
    assert x2["importance_rank"] == 1
    assert x2["ranked_feature_rank"] == 1
    assert x2["ranked_final_score"] == 0.91
    assert x2["risk_flags"] == "unstable_over_time"
    assert x2["recommended_use"] == "prediction_candidate"
    assert x2["recommended_action"] == "manual_review_required"
    assert x2["interpretation"] == "model explanation only; not a causal conclusion"


def test_model_variable_importance_empty_and_does_not_mutate_inputs():
    assert list(build_model_variable_importance(pd.DataFrame()).columns) == VARIABLE_IMPORTANCE_COLS

    importance = pd.DataFrame(
        [{"feature": "x1__lag_1", "importance": 0.9, "method": "rf", "variable": "x1", "lag": 1}]
    )
    ranked = pd.DataFrame([{"variable": "x1", "final_score": 0.8}])
    risk = pd.DataFrame([{"variable": "x1", "risk_flags": "target_leads_variable"}])
    importance_before = importance.copy(deep=True)
    ranked_before = ranked.copy(deep=True)
    risk_before = risk.copy(deep=True)

    build_model_variable_importance(importance, ranked, risk_flags=risk)

    pd.testing.assert_frame_equal(importance, importance_before)
    pd.testing.assert_frame_equal(ranked, ranked_before)
    pd.testing.assert_frame_equal(risk, risk_before)
