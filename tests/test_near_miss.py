import pandas as pd

from chem_ts_corr.near_miss import OUT_COLS, build_near_miss_candidates


def test_near_miss_outputs_strong_lag_signal_missing_from_top_n():
    lag_scores = pd.DataFrame(
        [
            {"variable": "x1", "lag": 1, "abs_pearson": 0.8, "abs_spearman": 0.7, "direction": "变量领先目标"},
            {"variable": "x2", "lag": 2, "abs_pearson": 0.75, "abs_spearman": 0.6, "direction": "变量领先目标"},
        ]
    )
    ranked = pd.DataFrame([{"variable": "x1", "final_score": 0.9}, {"variable": "x2", "final_score": 0.2}])

    out = build_near_miss_candidates(lag_scores, ranked, screening_top_n=1)

    assert list(out.columns) == OUT_COLS
    assert out["variable"].tolist() == ["x2"]
    assert bool(out.iloc[0]["missing_from_screening_top_n"])
    assert "raw_lag_signal" in out.iloc[0]["near_miss_reason"]


def test_near_miss_excludes_screening_top_n_variables():
    lag_scores = pd.DataFrame(
        [{"variable": "x1", "lag": 1, "abs_pearson": 0.9, "abs_spearman": 0.8, "direction": "变量领先目标"}]
    )
    ranked = pd.DataFrame([{"variable": "x1", "final_score": 0.9}])

    out = build_near_miss_candidates(lag_scores, ranked, screening_top_n=1)

    assert out.empty


def test_near_miss_merges_risk_flags_and_residuals():
    lag_scores = pd.DataFrame(
        [{"variable": "x2", "lag": 1, "abs_pearson": 0.5, "abs_spearman": 0.4, "direction": "变量领先目标"}]
    )
    residual = pd.DataFrame([{"variable": "x2", "residual_corr": 0.35}])
    risks = pd.DataFrame([{"variable": "x2", "risk_flags": "lag_boundary;target_leads_variable"}])

    out = build_near_miss_candidates(lag_scores, pd.DataFrame(), residual_corr_scores=residual, risk_flags=risks)
    reason = out.iloc[0]["near_miss_reason"]

    assert out.iloc[0]["risk_flags"] == "lag_boundary;target_leads_variable"
    assert "residual_signal" in reason
    assert "lag_boundary_risk" in reason
    assert "target_lead_risk" in reason


def test_near_miss_quality_and_formula_risk_affect_score_and_reason():
    lag_scores = pd.DataFrame(
        [
            {"variable": "clean", "lag": 1, "abs_pearson": 0.5, "abs_spearman": 0.4, "direction": "变量领先目标"},
            {"variable": "risky", "lag": 1, "abs_pearson": 0.5, "abs_spearman": 0.4, "direction": "变量领先目标"},
        ]
    )
    lag_quality = pd.DataFrame(
        [{"variable": "clean", "lag_quality": 0.8}, {"variable": "risky", "lag_quality": 0.8}]
    )
    risks = pd.DataFrame([{"variable": "risky", "risk_flags": "strong_formula_leakage;poor_data_quality"}])

    out = build_near_miss_candidates(lag_scores, pd.DataFrame(), lag_peak_quality=lag_quality, risk_flags=risks)
    clean = out[out["variable"] == "clean"].iloc[0]
    risky = out[out["variable"] == "risky"].iloc[0]

    assert "clear_lag_peak" in clean["near_miss_reason"]
    assert "data_or_formula_risk" in risky["near_miss_reason"]
    assert float(risky["near_miss_score"]) < float(clean["near_miss_score"])


def test_near_miss_does_not_mutate_inputs():
    lag_scores = pd.DataFrame(
        [{"variable": "x2", "lag": 1, "abs_pearson": 0.5, "abs_spearman": 0.4, "direction": "变量领先目标"}]
    )
    ranked = pd.DataFrame([{"variable": "x2", "final_score": 0.2}])
    before_lag = lag_scores.copy(deep=True)
    before_ranked = ranked.copy(deep=True)

    build_near_miss_candidates(lag_scores, ranked, screening_top_n=0)

    pd.testing.assert_frame_equal(lag_scores, before_lag)
    pd.testing.assert_frame_equal(ranked, before_ranked)
