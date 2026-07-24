from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import run_analysis
from chem_ts_corr.screening import final_ranked_features
from chem_ts_corr.service import analyze_numeric_frame


REGISTRY_PATH = Path("docs/four_layer_evidence_registry.json")
DIRECT_SCORE_FIELDS = {
    "association_score",
    "correlation_evidence_score",
    "lag_quality",
    "stability_score",
    "prediction_score",
    "data_quality_score",
    "evidence_completeness",
    "evidence_confidence",
    "evidence_strength",
    "evidence_score",
    "risk_penalty_rate",
    "risk_score_cap",
    "final_score",
    "driver_priority_factor",
    "driver_priority_score",
    "driver_rank",
}
RANKING_INPUT_FIELDS = {
    "raw_corr", "innovation_score", "residual_corr", "lag", "lag_quality",
    "rolling_stability", "regime_stability_final", "model_lift_score", "risk_flags",
    "data_quality_score", "candidate_class", "final_score", "driver_priority_factor",
    "driver_priority_score", "driver_rank",
}


def _registry() -> list[dict[str, object]]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _ranking_inputs() -> tuple[pd.DataFrame, ...]:
    ranked = pd.DataFrame([
        {"variable": "upstream", "score": 0.81, "innovation_score": 0.64, "lag": 2},
        {"variable": "downstream", "score": 0.86, "innovation_score": 0.72, "lag": -1},
        {"variable": "capacity", "score": 0.78, "innovation_score": 0.70, "lag": 3},
    ])
    residual = pd.DataFrame([
        {"variable": "upstream", "residual_corr": 0.75},
        {"variable": "downstream", "residual_corr": 0.80},
        {"variable": "capacity", "residual_corr": 0.20},
    ])
    stability = pd.DataFrame([
        {"variable": "upstream", "regime_stability_final": 0.80},
        {"variable": "downstream", "regime_stability_final": 0.85},
        {"variable": "capacity", "regime_stability_final": 0.72},
    ])
    lift = pd.DataFrame([
        {"variable": "upstream", "model_lift_score": 0.66, "status": "ok"},
        {"variable": "downstream", "model_lift_score": 0.73, "status": "ok"},
        {"variable": "capacity", "model_lift_score": 0.55, "status": "ok"},
    ])
    risks = pd.DataFrame([
        {"variable": "upstream", "risk_flags": "", "data_quality_score": 0.96},
        {"variable": "downstream", "risk_flags": "target_leads_variable", "data_quality_score": 0.93},
        {"variable": "capacity", "risk_flags": "common_capacity_driver", "data_quality_score": 0.90},
    ])
    lag_quality = pd.DataFrame([
        {"variable": "upstream", "lag_quality": 0.82, "lag_boundary_flag": False},
        {"variable": "downstream", "lag_quality": 0.76, "lag_boundary_flag": False},
        {"variable": "capacity", "lag_quality": 0.68, "lag_boundary_flag": False},
    ])
    rolling = pd.DataFrame([
        {"variable": "upstream", "rolling_stability": 0.84},
        {"variable": "downstream", "rolling_stability": 0.81},
        {"variable": "capacity", "rolling_stability": 0.71},
    ])
    return ranked, residual, stability, lift, risks, lag_quality, rolling


def test_final_ranking_baseline_is_frozen():
    result = final_ranked_features(*_ranking_inputs())
    actual = result[[
        "variable", "final_score", "driver_priority_factor", "driver_priority_score",
        "driver_rank", "candidate_grade", "recommended_use",
    ]].to_dict("records")
    expected = pd.DataFrame([
        {"variable": "upstream", "final_score": 0.7421023851203975, "driver_priority_factor": 1.0,
         "driver_priority_score": 0.7421023851203975, "driver_rank": 1, "candidate_grade": "B",
         "recommended_use": "prediction_candidate"},
        {"variable": "capacity", "final_score": 0.5746554492601353, "driver_priority_factor": 0.75,
         "driver_priority_score": 0.43099158694510153, "driver_rank": 2, "candidate_grade": "C",
         "recommended_use": "capacity_driven"},
        {"variable": "downstream", "final_score": 0.7500341454106302, "driver_priority_factor": 0.45,
         "driver_priority_score": 0.3375153654347836, "driver_rank": 3, "candidate_grade": "A",
         "recommended_use": "state_indicator"},
    ])
    pd.testing.assert_frame_equal(pd.DataFrame(actual), expected, check_exact=False, rtol=1e-12)


def _config(tmp_path: Path, name: str) -> AnalysisConfig:
    return AnalysisConfig(
        input_path=tmp_path / "input.csv", time_column="time", target="target",
        output_dir=tmp_path / name, max_lag=3, top_k=3, skip_model_lift=True,
        skip_rolling_corr=True,
    )


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    size = 96
    upstream = rng.normal(size=size)
    target = np.roll(upstream, 2) + rng.normal(scale=0.25, size=size)
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=size, freq="h"),
        "target": target, "upstream": upstream, "other": rng.normal(size=size),
    })


def test_service_and_pipeline_entries_write_the_same_frozen_ranked_features(tmp_path: Path):
    frame = _frame()
    config = _config(tmp_path, "service")
    service_ranked = analyze_numeric_frame(frame.set_index("time"), config).ranked_features

    frame.to_csv(config.input_path, index=False, encoding="utf-8-sig")
    pipeline_config = _config(tmp_path, "pipeline")
    run_analysis(pipeline_config)
    exported = pd.read_csv(pipeline_config.output_dir / "ranked_features.csv", encoding="utf-8-sig")

    fields = [
        "variable", "final_score", "driver_priority_factor", "driver_priority_score",
        "driver_rank", "candidate_grade", "recommended_use",
    ]
    pd.testing.assert_frame_equal(
        service_ranked[fields].reset_index(drop=True), exported[fields].reset_index(drop=True),
        check_dtype=False,
    )
    assert exported["variable"].tolist() == ["upstream", "other"]
    assert exported["driver_rank"].tolist() == [1, 2]


def test_registry_direct_score_fields_match_scoring_source():
    registry = _registry()
    registered = {entry["field"] for entry in registry}
    assert DIRECT_SCORE_FIELDS <= registered
    source = inspect.getsource(final_ranked_features) + inspect.getsource(
        __import__("chem_ts_corr.screening", fromlist=["_finalize_driver_ranking"])._finalize_driver_ranking
    )
    for field in DIRECT_SCORE_FIELDS:
        assert field in source


def test_all_final_ranking_inputs_are_registered_and_context_is_not_scored():
    registry = _registry()
    registered = {entry["field"] for entry in registry}
    assert RANKING_INPUT_FIELDS <= registered
    source = inspect.getsource(final_ranked_features)
    score_expression = source.split('final["association_score"]', 1)[1].split(
        'final["candidate_class"]', 1
    )[0]
    for entry in registry:
        if entry["score_role"] == "not_in_scoring":
            assert str(entry["field"]) not in score_expression
