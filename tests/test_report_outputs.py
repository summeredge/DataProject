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
        "rolling_corr_scores.csv",
        "causal_review_candidates.csv",
        "summary.md",
    ]:
        assert (tmp_path / name).exists(), f"missing output: {name}"

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "自动诊断建议" in summary


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


def test_recommended_candidates_fallback_excludes_manual_closed_loop_recommendations():
    ranked = pd.DataFrame(
        [
            {"variable": "confirmed", "candidate_grade": "A", "recommended_use": "closed_loop_confirmed", "final_score": 0.9},
            {"variable": "conflict", "candidate_grade": "A", "recommended_use": "closed_loop_conflict", "final_score": 0.8},
        ]
    )

    assert build_recommended_candidates(ranked).empty
