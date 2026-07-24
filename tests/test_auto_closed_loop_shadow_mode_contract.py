from __future__ import annotations

import inspect

import pandas as pd

from chem_ts_corr import web
from chem_ts_corr.auto_closed_loop import (
    AUTO_CLOSED_LOOP_DIAGNOSIS_COLUMNS,
    build_auto_closed_loop_diagnosis,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked = pd.DataFrame([
        {"variable": "mv_confirmed", "lag": 0, "final_score": 0.8, "driver_priority_factor": 0.8, "driver_priority_score": 0.64, "driver_rank": 1},
        {"variable": "mv_possible", "lag": 0, "final_score": 0.7, "driver_priority_factor": 0.8, "driver_priority_score": 0.56, "driver_rank": 2},
        {"variable": "mv_none", "lag": 2, "final_score": 0.6, "driver_priority_factor": 0.8, "driver_priority_score": 0.48, "driver_rank": 3},
    ])
    risks = pd.DataFrame([
        {"variable": "mv_confirmed", "closed_loop_suspect_flag": True},
        {"variable": "mv_possible", "closed_loop_suspect_flag": True},
    ])
    lag = pd.DataFrame([
        {"variable": "mv_confirmed", "best_lag": 0, "lag_quality": 0.8},
        {"variable": "mv_possible", "best_lag": 0, "lag_quality": 0.4},
    ])
    stability = pd.DataFrame([
        {"variable": "mv_confirmed", "rolling_stability": 0.9},
        {"variable": "mv_possible", "rolling_stability": 0.3},
    ])
    prediction = pd.DataFrame([
        {"variable": "mv_confirmed", "model_lift_score": 0.2},
        {"variable": "mv_possible", "model_lift_score": 0.01},
    ])
    return ranked, risks, lag, stability, prediction


def test_shadow_diagnosis_writes_frozen_schema_without_mutating_ranking_inputs(tmp_path):
    inputs = _inputs()
    before = [frame.copy(deep=True) for frame in inputs]

    diagnosis = build_auto_closed_loop_diagnosis(*inputs, cv_variable="cv", created_time="2026-07-24T00:00:00Z")
    diagnosis.to_csv(tmp_path / "auto_closed_loop_diagnosis.csv", index=False, encoding="utf-8-sig")

    assert diagnosis.columns.tolist() == AUTO_CLOSED_LOOP_DIAGNOSIS_COLUMNS
    assert (tmp_path / "auto_closed_loop_diagnosis.csv").exists()
    assert diagnosis.set_index("mv_variable").loc["mv_confirmed", "diagnosis_status"] == "confirmed"
    assert diagnosis.set_index("mv_variable").loc["mv_possible", "diagnosis_status"] == "possible"
    assert diagnosis.set_index("mv_variable").loc["mv_none", "diagnosis_status"] == "not_supported"
    for actual, expected in zip(inputs, before):
        pd.testing.assert_frame_equal(actual, expected)


def test_shadow_diagnosis_handles_no_relationship_or_missing_evidence():
    empty = build_auto_closed_loop_diagnosis(pd.DataFrame(), None, None, None, None, "cv")
    assert empty.columns.tolist() == AUTO_CLOSED_LOOP_DIAGNOSIS_COLUMNS
    assert empty.empty

    ranked = pd.DataFrame([{"variable": "mv", "lag": 1}])
    diagnosis = build_auto_closed_loop_diagnosis(ranked, None, None, None, None, "cv")
    assert diagnosis.loc[0, "diagnosis_status"] == "not_supported"
    assert diagnosis.loc[0, "confidence_level"] == "low"


def test_shadow_diagnosis_isolated_from_scores():
    ranked, risks, lag, stability, prediction = _inputs()
    ranking_before = ranked[["driver_priority_factor", "driver_priority_score", "driver_rank", "final_score"]].copy()

    build_auto_closed_loop_diagnosis(ranked, risks, lag, stability, prediction, "cv")

    pd.testing.assert_frame_equal(
        ranked[["driver_priority_factor", "driver_priority_score", "driver_rank", "final_score"]],
        ranking_before,
    )


def test_web_shadow_api_reads_existing_files_and_never_reorders_or_reanalyzes():
    source = inspect.getsource(web._run_auto_closed_loop_diagnosis_response)

    assert "auto_closed_loop_diagnosis.csv" in web.DOWNLOAD_FILES
    assert "build_auto_closed_loop_diagnosis(" in source
    assert "ranked_features.csv" in source
    for forbidden in [
        "run_analysis",
        'to_csv(output_dir / "ranked_features.csv"',
        'to_csv(output_dir / "risk_flags.csv"',
        'to_csv(output_dir / "closed_loop_evidence.csv"',
    ]:
        assert forbidden not in source
