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
        "residual_signal_score", "residual_evidence_status", "load_adjusted_relation_status",
        "candidate_priority_tier", "candidate_priority_score", "candidate_priority_rank",
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
        "residual_signal_score": .7,
        "residual_evidence_status": "strong",
        "load_adjusted_relation_status": "residual_only_supported",
        "candidate_priority_tier": 1,
        "candidate_priority_score": .7,
        "candidate_priority_rank": 3,
    }
    candidate = pd.DataFrame([{"variable": "x1", "final_score": .2, **source_fields}])

    row = build_causal_review_candidates(candidate).iloc[0]

    for field, expected in source_fields.items():
        if pd.isna(expected):
            assert pd.isna(row[field])
        else:
            assert row[field] == expected


def test_causal_review_output_follows_candidate_priority_rank_without_changing_review_fields():
    candidates = pd.DataFrame([
        {"variable": "review_first", "final_score": .9, "candidate_grade": "A", "recommended_use": "strong_screening_candidate", "risk_level": "none", "candidate_priority_rank": 2},
        {"variable": "priority_first", "final_score": .2, "candidate_grade": "C", "recommended_use": "manual_review_required", "risk_level": "medium", "candidate_priority_rank": 1},
    ])

    out = build_causal_review_candidates(candidates)

    assert out["variable"].tolist() == ["priority_first", "review_first"]
    rows = out.set_index("variable")
    assert rows.loc["review_first", "review_priority"] == 1
    assert rows.loc["priority_first", "review_priority"] == 5
