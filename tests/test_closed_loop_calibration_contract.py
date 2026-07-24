from __future__ import annotations

import inspect
import json

import pandas as pd

from chem_ts_corr import web
from chem_ts_corr.closed_loop_calibration import (
    CALIBRATION_RESULT_COLUMNS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    build_training_labels,
    run_closed_loop_calibration,
)


def _write_inputs(tmp_path):
    ranked = pd.DataFrame([
        {"variable": "positive", "driver_priority_factor": 0.8, "driver_priority_score": 0.64, "driver_rank": 1, "final_score": 0.8, "prediction_score": 0.8},
        {"variable": "negative", "driver_priority_factor": 0.8, "driver_priority_score": 0.48, "driver_rank": 2, "final_score": 0.6, "prediction_score": 0.5},
        {"variable": "unknown", "driver_priority_factor": 0.8, "driver_priority_score": 0.4, "driver_rank": 3, "final_score": 0.5, "prediction_score": 0.4},
        {"variable": "diagnosis_only", "driver_priority_factor": 0.8, "driver_priority_score": 0.32, "driver_rank": 4, "final_score": 0.4, "prediction_score": 0.3},
    ])
    files = {
        "ranked_features.csv": ranked,
        "risk_flags.csv": pd.DataFrame([
            {"variable": "positive", "closed_loop_suspect_flag": True, "risk_count": 2},
            {"variable": "negative", "closed_loop_suspect_flag": False, "risk_count": 0},
        ]),
        "lag_peak_quality.csv": pd.DataFrame([
            {"variable": "positive", "best_lag": 0, "lag_quality": 0.9, "lag_boundary_flag": False},
            {"variable": "negative", "best_lag": 2, "lag_quality": 0.2, "lag_boundary_flag": False},
        ]),
        "rolling_corr_scores.csv": pd.DataFrame([
            {"variable": "positive", "rolling_stability": 0.9},
            {"variable": "negative", "rolling_stability": 0.2},
        ]),
        "regime_scores.csv": pd.DataFrame([
            {"variable": "positive", "regime_consistency_score": 0.9, "regime_sign_consistency": 1.0, "regime_lag_consistency": 1.0},
            {"variable": "negative", "regime_consistency_score": 0.2, "regime_sign_consistency": 0.0, "regime_lag_consistency": 0.0},
        ]),
        "model_lift_scores.csv": pd.DataFrame([
            {"variable": "positive", "model_lift_score": 0.2},
            {"variable": "negative", "model_lift_score": 0.0},
        ]),
        "auto_closed_loop_diagnosis.csv": pd.DataFrame([
            {"mv_variable": "positive", "diagnosis_status": "confirmed", "confidence_level": "high"},
            {"mv_variable": "diagnosis_only", "diagnosis_status": "confirmed", "confidence_level": "high"},
        ]),
    }
    for name, frame in files.items():
        frame.to_csv(tmp_path / name, index=False, encoding="utf-8-sig")
    return ranked


def test_human_labels_are_explicit_and_auto_diagnosis_is_not_a_label():
    labels = build_training_labels(
        ["positive"],
        ["negative"],
        [{"variable": "record_positive", "new_status": POSITIVE_LABEL}, {"variable": "ignored", "new_status": "confirmed_recommendation"}],
    ).set_index("variable")

    assert labels.loc["positive", "training_label"] == POSITIVE_LABEL
    assert labels.loc["negative", "training_label"] == NEGATIVE_LABEL
    assert labels.loc["record_positive", "training_label"] == POSITIVE_LABEL
    assert "ignored" not in labels.index


def test_calibration_outputs_probability_files_and_preserves_original_results(tmp_path):
    ranked = _write_inputs(tmp_path)
    original_files = {name: (tmp_path / name).read_bytes() for name in ["ranked_features.csv", "risk_flags.csv", "auto_closed_loop_diagnosis.csv"]}
    records = [{"variable": "positive", "new_status": "confirmed_recommendation"}]

    results, model, metrics = run_closed_loop_calibration(tmp_path, ["positive"], ["negative"], records)

    assert results.columns.tolist() == CALIBRATION_RESULT_COLUMNS
    assert results.loc[results["variable"] == "diagnosis_only", "training_label"].iloc[0] == "unknown"
    assert results["auto_closed_loop_probability"].dropna().between(0, 1).all()
    assert model["status"] == "trained"
    assert metrics["sample_count"] == 2
    for name in ["closed_loop_calibration_model.json", "closed_loop_calibration_results.csv", "closed_loop_calibration_metrics.json"]:
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "closed_loop_calibration_metrics.json").read_text(encoding="utf-8"))["status"] == "trained"
    for name, before in original_files.items():
        assert (tmp_path / name).read_bytes() == before
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")[["driver_priority_factor", "driver_priority_score", "driver_rank", "final_score"]],
        ranked[["driver_priority_factor", "driver_priority_score", "driver_rank", "final_score"]],
    )
    assert records == [{"variable": "positive", "new_status": "confirmed_recommendation"}]


def test_single_class_labels_produce_clear_non_model_status(tmp_path):
    _write_inputs(tmp_path)

    results, model, metrics = run_closed_loop_calibration(tmp_path, ["positive"], [], [])

    assert model["status"] == "insufficient_training_labels"
    assert metrics["status"] == "insufficient_training_labels"
    assert results["calibration_status"].eq("insufficient_training_labels").all()
    assert results["auto_closed_loop_probability"].isna().all()


def test_web_calibration_api_is_shadow_only():
    source = inspect.getsource(web._run_closed_loop_calibration_response)

    for name in ["closed_loop_calibration_model.json", "closed_loop_calibration_results.csv", "closed_loop_calibration_metrics.json"]:
        assert name in web.DOWNLOAD_FILES
    assert "closedLoopCalibrationTable" in web.INDEX_HTML
    assert 'postForm("/api/run_closed_loop_calibration", form)' in web.INDEX_HTML
    assert "run_closed_loop_calibration(" in source
    for forbidden in ["run_analysis", "reorder_ranked_features", "final_ranked_features", "candidate_decision_records.json", "reordered_recommendations.csv"]:
        assert forbidden not in source
