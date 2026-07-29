from pathlib import Path

import pandas as pd

from chem_ts_corr.report import build_recommended_candidates, write_outputs


def test_write_outputs_writes_expected_files(tmp_path: Path):
    ranked = pd.DataFrame(
        [
            {
                "variable": "v1",
                "final_score": 0.82,
                "candidate_grade": "A",
                "recommended_use": "strong_screening_candidate",
                "lag": 1,
                "direction": "变量领先目标",
                "raw_corr": 0.8,
                "residual_corr": 0.7,
                "risk_flags": "",
                "recommended_action": "优先进入机理复核",
                "lag_boundary_flag": False,
            }
        ]
    )
    lag_scores = pd.DataFrame([{"variable": "v1", "lag": 1, "score": 0.8, "corr_q_value": 0.01}])
    granger = pd.DataFrame([{"variable": "v1", "min_p_value": 0.04, "status": "ok"}])
    importance = pd.DataFrame([{"variable": "v1", "importance": 0.5}])
    risk = pd.DataFrame([{"variable": "v1", "risk_count": 0, "common_capacity_driver_flag": False}])

    write_outputs(
        output_dir=tmp_path,
        target="target",
        ranked_features=ranked,
        lag_scores=lag_scores,
        granger_tests=granger,
        importance=importance,
        metrics={"rows_after_preprocess": 100, "variables": 5},
        diagnostics=pd.DataFrame([{"variable": "v1", "missing_rate": 0.0}]),
        residual_corr_scores=pd.DataFrame([{"variable": "v1", "residual_corr": 0.7}]),
        regime_scores=pd.DataFrame([{"variable": "v1", "regime_stability_final": 0.9}]),
        risk_flags=risk,
        model_lift_scores=pd.DataFrame([{"variable": "v1", "model_lift": 0.1}]),
        lag_peak_quality=pd.DataFrame([{"variable": "v1", "lag_quality": 0.8}]),
        rolling_corr_scores=pd.DataFrame([{"variable": "v1", "rolling_stability": 0.7}]),
    )

    for name in [
        "ranked_features.csv",
        "recommended_candidates.csv",
        "lag_scores.csv",
        "risk_flags.csv",
        "causal_review_candidates.csv",
        "summary.md",
    ]:
        assert (tmp_path / name).exists(), f"missing output: {name}"

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "# 初步筛选摘要：target" in summary


    review = pd.read_csv(tmp_path / "causal_review_candidates.csv", encoding="utf-8-sig")
    for c in [
        "variable",
        "final_score",
        "candidate_grade",
        "recommended_use",
        "review_priority",
        "review_reason",
        "review_tier",
    ]:
        assert c in review.columns


def test_write_outputs_uses_metrics_top_k_for_near_miss_candidates(tmp_path: Path):
    ranked = pd.DataFrame(
        [
            {"variable": "v1", "final_score": 0.9, "candidate_grade": "A", "recommended_use": "strong_screening_candidate", "recommended_action": "review"},
            {"variable": "v2", "final_score": 0.8, "candidate_grade": "B", "recommended_use": "prediction_candidate", "recommended_action": "review"},
        ]
    )
    lag_scores = pd.DataFrame(
        [
            {"variable": "v1", "lag": 1, "score": 0.9},
            {"variable": "v2", "lag": 2, "score": 0.8},
        ]
    )

    write_outputs(
        output_dir=tmp_path,
        target="target",
        ranked_features=ranked,
        lag_scores=lag_scores,
        granger_tests=pd.DataFrame(),
        importance=pd.DataFrame(),
        metrics={"top_k": 1},
    )

    near_miss = pd.read_csv(tmp_path / "near_miss_candidates.csv", encoding="utf-8-sig")
    assert near_miss["variable"].tolist() == ["v2"]


def test_recommended_candidates_uses_candidate_grade_without_engineering_context():
    ranked = pd.DataFrame(
        [
            {"variable": "candidate_a", "candidate_grade": "A", "recommended_use": "strong_screening_candidate", "final_score": 0.9, "engineering_context": '{"source": "engineering_note"}'},
            {"variable": "candidate_b", "candidate_grade": "A", "recommended_use": "strong_screening_candidate", "final_score": 0.8},
        ]
    )

    assert build_recommended_candidates(ranked)["variable"].tolist() == ["candidate_a", "candidate_b"]


def test_causal_review_candidates_use_the_provided_recommended_pool(tmp_path: Path):
    ranked = pd.DataFrame([
        {"variable": "control", "final_score": 0.9, "candidate_grade": "A", "recommended_use": "control_variable_reference"},
        {"variable": "candidate", "final_score": 0.8, "candidate_grade": "A", "recommended_use": "strong_screening_candidate"},
    ])
    recommended = ranked[ranked["variable"].eq("candidate")].copy()

    write_outputs(
        tmp_path, "target", ranked, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {},
        recommended_candidates=recommended,
    )

    causal = pd.read_csv(tmp_path / "causal_review_candidates.csv", encoding="utf-8-sig")
    assert causal["variable"].tolist() == ["candidate"]


def test_write_outputs_preserves_candidate_source_fields_in_recommended_and_causal_csv(tmp_path: Path):
    source_fields = {
        "candidate_source": "residual_only",
        "selected_by_raw": False,
        "selected_by_residual": True,
        "raw_candidate_rank": pd.NA,
        "residual_candidate_rank": 1,
        "candidate_pool_rank": 1,
        "common_capacity_candidate_flag": False,
        "force_included": False,
    }
    ranked = pd.DataFrame([{
        "variable": "candidate",
        "final_score": .2,
        "candidate_grade": "C",
        "recommended_use": "manual_review_required",
    }])
    recommended = pd.DataFrame([{"variable": "candidate", "final_score": .2, **source_fields}])

    write_outputs(
        tmp_path, "target", ranked, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {},
        recommended_candidates=recommended,
        residual_corr_scores=pd.DataFrame([{
            "variable": "candidate", "residual_corr": .8, "residual_lag_quality": .6,
            "residual_n": 100, "residual_lag": 1, "residual_status": "ok",
        }]),
    )

    written_recommended = pd.read_csv(tmp_path / "recommended_candidates.csv", encoding="utf-8-sig")
    causal = pd.read_csv(tmp_path / "causal_review_candidates.csv", encoding="utf-8-sig")
    for field, expected in source_fields.items():
        recommended_value = written_recommended.loc[0, field]
        causal_value = causal.loc[0, field]
        if pd.isna(expected):
            assert pd.isna(recommended_value)
            assert pd.isna(causal_value)
        else:
            assert recommended_value == expected
            assert causal_value == expected
    for field in [
        "residual_signal_score", "residual_evidence_status", "load_adjusted_relation_status",
        "candidate_priority_tier", "candidate_priority_score", "candidate_priority_rank",
    ]:
        assert field in written_recommended.columns
        assert field in causal.columns
        if pd.isna(written_recommended.loc[0, field]):
            assert pd.isna(causal.loc[0, field])
        else:
            assert written_recommended.loc[0, field] == causal.loc[0, field]
    assert written_recommended.loc[0, "residual_signal_score"] == .7
    assert written_recommended.loc[0, "residual_evidence_status"] == "strong"
    assert written_recommended.loc[0, "load_adjusted_relation_status"] == "residual_only_supported"
    assert written_recommended.loc[0, "candidate_priority_rank"] == 1

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    for label in [
        "双通道支持候选数", "仅原始通道候选数", "仅残差通道候选数",
        "共同负荷风险候选数", "仅人工强制包含数", "控制参考候选数",
    ]:
        assert label in summary
