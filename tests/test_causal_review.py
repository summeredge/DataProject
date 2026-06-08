import pandas as pd

from chem_ts_corr.causal_review import build_causal_review_candidates


def test_causal_review_keeps_fixed_columns_when_ranked_features_is_minimal():
    ranked = pd.DataFrame([{"variable": "x1"}])
    out = build_causal_review_candidates(ranked)

    cols = [
        "variable", "final_score", "candidate_grade", "lag", "direction", "raw_corr", "residual_corr",
        "rolling_stability", "regime_stability_final", "lag_boundary_flag", "model_lift_score",
        "risk_level", "risk_flags", "recommended_use", "recommended_action", "force_included",
        "review_priority", "review_reason", "review_tier",
    ]

    assert not out.empty
    for c in cols:
        assert c in out.columns

    row = out.iloc[0]
    assert pd.notna(row["review_priority"])
    assert str(row["review_reason"]).strip() != ""
    assert str(row["review_tier"]).strip() != ""
