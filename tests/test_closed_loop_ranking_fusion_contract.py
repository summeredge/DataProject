from __future__ import annotations

import inspect

import pandas as pd

from chem_ts_corr import web
from chem_ts_corr.closed_loop_ranking import build_closed_loop_risk_context
from chem_ts_corr.screening import reorder_ranked_features


def _ranked() -> pd.DataFrame:
    return pd.DataFrame([
        {"variable": "A", "final_score": 0.9, "evidence_score": 0.9, "candidate_class": "upstream_driver_candidate", "driver_priority_factor": 1.0, "driver_priority_score": 0.9, "driver_rank": 1, "lag": 1, "raw_corr": 0.9, "risk_flags": ""},
        {"variable": "B", "final_score": 0.8, "evidence_score": 0.8, "candidate_class": "upstream_driver_candidate", "driver_priority_factor": 1.0, "driver_priority_score": 0.8, "driver_rank": 2, "lag": 1, "raw_corr": 0.8, "risk_flags": ""},
    ])


def _context(manual_status: str, probability: float) -> pd.DataFrame:
    return build_closed_loop_risk_context(
        pd.DataFrame([{"variable": "A", "manual_closed_loop_status": manual_status}]),
        pd.DataFrame([{"variable": "A", "auto_closed_loop_probability": probability}]),
        timestamp="2026-07-24T00:00:00Z",
    )


def test_manual_closed_loop_limit_overrides_low_automatic_probability():
    output = reorder_ranked_features(_ranked(), closed_loop_risk_context=_context("confirmed_closed_loop", 0.1)).set_index("variable")

    assert output.loc["A", "driver_priority_factor"] <= 0.55
    assert output.loc["A", "closed_loop_ranking_reason"] == "manual_confirmed_closed_loop_limit"


def test_manual_non_closed_loop_overrides_automatic_high_risk():
    baseline = reorder_ranked_features(_ranked()).set_index("variable")
    output = reorder_ranked_features(_ranked(), closed_loop_risk_context=_context("confirmed_not_closed_loop", 0.9)).set_index("variable")

    assert output.loc["A", "driver_priority_factor"] == baseline.loc["A", "driver_priority_factor"]
    assert output.loc["A", "driver_rank"] == baseline.loc["A", "driver_rank"]
    assert output.loc["A", "closed_loop_ranking_reason"] == "manual_non_closed_loop_override"


def test_automatic_high_risk_changes_rank_but_low_risk_does_not():
    high = reorder_ranked_features(_ranked(), closed_loop_risk_context=_context("unknown", 0.9)).set_index("variable")
    low = reorder_ranked_features(_ranked(), closed_loop_risk_context=_context("unknown", 0.1)).set_index("variable")

    assert high.loc["A", "driver_priority_factor"] == 0.55
    assert high.loc["A", "driver_rank"] == 2
    assert high.loc["A", "closed_loop_ranking_reason"] == "automatic_closed_loop_high_risk_penalty"
    assert low.loc["A", "driver_priority_factor"] == 1.0
    assert low.loc["A", "driver_rank"] == 1


def test_automatic_thresholds_and_factors_are_configurable():
    context = build_closed_loop_risk_context(
        pd.DataFrame([{"variable": "A", "manual_closed_loop_status": "unknown"}]),
        pd.DataFrame([{"variable": "A", "auto_closed_loop_probability": 0.75}]),
        medium_threshold=0.4,
        high_threshold=0.8,
        medium_factor=0.7,
        high_factor=0.5,
    ).set_index("variable")

    assert context.loc["A", "automatic_closed_loop_risk"] == 0.75
    assert context.loc["A", "factor_adjustment"] == 0.7
    assert context.loc["A", "closed_loop_ranking_reason"] == "automatic_closed_loop_medium_risk_penalty"


def test_fusion_preserves_evidence_scores_and_ignores_recommendation_decisions():
    original = _ranked()
    output = reorder_ranked_features(
        original,
        candidate_decision_records=pd.DataFrame([{"variable": "A", "new_status": "confirmed_recommendation"}]),
        closed_loop_risk_context=_context("unknown", 0.9),
    ).set_index("variable")

    before = original.set_index("variable")
    assert output.loc["A", "final_score"] == before.loc["A", "final_score"]
    assert output.loc["A", "evidence_score"] == before.loc["A", "evidence_score"]
    assert output.loc["A", "closed_loop_ranking_reason"] == "automatic_closed_loop_high_risk_penalty"


def test_web_ranking_endpoint_uses_single_screening_entry_and_preserves_history_inputs():
    source = inspect.getsource(web._apply_closed_loop_ranking_response)

    assert "closed_loop_ranking_fusion.csv" in web.DOWNLOAD_FILES
    assert "closedLoopRankingFusionTable" in web.INDEX_HTML
    assert 'postForm("/api/apply_closed_loop_ranking", form)' in web.INDEX_HTML
    assert "reorder_ranked_features(" in source
    for forbidden in ["final_ranked_features(", "closed_loop_evidence.csv\", index=False", "candidate_decision_records.json\", index=False", "closed_loop_calibration_results.csv\", index=False"]:
        assert forbidden not in source
