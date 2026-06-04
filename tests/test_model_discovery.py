import pandas as pd

from chem_ts_corr.model_discovery import OUT_COLS, build_model_discovered_candidates


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
