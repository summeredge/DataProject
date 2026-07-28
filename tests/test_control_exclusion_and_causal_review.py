from pathlib import Path

import pandas as pd

from chem_ts_corr.report import write_outputs
from chem_ts_corr.screening import final_ranked_features, _recommended_action


def _base_frames():
    ranked = pd.DataFrame([
        {"variable": "ctrl", "lag": 0, "direction": "同步变化", "score": 0.99},
        {"variable": "sig", "lag": 1, "direction": "变量领先目标", "score": 0.80},
    ])
    empty = pd.DataFrame(columns=["variable"])
    return ranked, empty, empty, empty, pd.DataFrame(columns=["variable", "risk_flags", "risk_count", "strong_risk_count", "weak_risk_count", "risk_level", "human_reason"]), empty, empty


def test_control_column_excluded_from_top_by_default():
    ranked, residual, stability, lift, risks, lag_peak, rolling = _base_frames()
    out = final_ranked_features(
        ranked, residual, stability, lift, risks, lag_peak, rolling,
        force_include_variables=[], top_k=1, control_columns=["ctrl"],
    )
    assert out.iloc[0]["variable"] == "ctrl"
    assert out["variable"].tolist() == ["ctrl", "sig"]
    assert out.iloc[0]["variable_role"] == "residual_control"


def test_control_column_can_be_force_included_and_marked_reference():
    ranked, residual, stability, lift, risks, lag_peak, rolling = _base_frames()
    out = final_ranked_features(
        ranked, residual, stability, lift, risks, lag_peak, rolling,
        force_include_variables=["ctrl"], top_k=1, control_columns=["ctrl"],
    )
    ctrl = out[out["variable"] == "ctrl"].iloc[0]
    assert bool(ctrl["force_included"]) is True
    assert ctrl["variable_role"] == "residual_control"


def test_control_variable_reference_has_recommended_action_text():
    row = pd.Series({"recommended_use": "control_variable_reference"})
    assert "控制变量" in _recommended_action(row)


def test_write_outputs_includes_causal_review_candidates(tmp_path: Path):
    ranked = pd.DataFrame([
        {
            "variable": "v1", "final_score": 0.8, "candidate_grade": "A", "lag": 1,
            "direction": "变量领先目标", "raw_corr": 0.8, "residual_corr": 0.7,
            "rolling_stability": 0.6, "regime_stability_final": 0.7, "lag_boundary_flag": False,
            "model_lift_score": 0.2, "risk_level": "weak", "risk_flags": "", "recommended_use": "strong_screening_candidate",
            "recommended_action": "优先进入机理复核", "force_included": False,
        }
    ])
    write_outputs(tmp_path, "target", ranked, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), metrics={})
    assert (tmp_path / "causal_review_candidates.csv").exists()
