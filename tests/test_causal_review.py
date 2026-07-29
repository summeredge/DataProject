import pandas as pd

from chem_ts_corr.causal_review import build_causal_review_candidates


def test_causal_review_keeps_fixed_columns_when_ranked_features_is_minimal():
    ranked = pd.DataFrame([{"variable": "x1"}])
    out = build_causal_review_candidates(ranked)

    cols = [
        "variable", "final_score", "candidate_grade", "lag", "direction", "raw_corr", "residual_corr",
        "rolling_stability", "regime_stability_final", "lag_boundary_flag", "model_lift_score",
        "risk_level", "risk_flags", "recommended_use", "recommended_action", "force_included",
        "candidate_source", "selected_by_raw", "selected_by_residual", "raw_candidate_rank",
        "residual_candidate_rank", "candidate_pool_rank", "common_capacity_candidate_flag",
        "review_priority", "review_reason", "review_tier",
    ]

    assert not out.empty
    for c in cols:
        assert c in out.columns

    row = out.iloc[0]
    assert pd.notna(row["review_priority"])
    assert str(row["review_reason"]).strip() != ""
    assert str(row["review_tier"]).strip() != ""


def test_causal_review_preserves_candidate_source_fields():
    source_fields = {
        "candidate_source": "residual_only",
        "selected_by_raw": False,
        "selected_by_residual": True,
        "raw_candidate_rank": pd.NA,
        "residual_candidate_rank": 2,
        "candidate_pool_rank": 3,
        "common_capacity_candidate_flag": False,
    }
    candidate = pd.DataFrame([{"variable": "x1", "final_score": .2, **source_fields}])

    row = build_causal_review_candidates(candidate).iloc[0]

    for field, expected in source_fields.items():
        if pd.isna(expected):
            assert pd.isna(row[field])
        else:
            assert row[field] == expected
