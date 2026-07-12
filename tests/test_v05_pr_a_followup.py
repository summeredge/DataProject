import inspect

import pandas as pd
import pytest

from chem_ts_corr import screening
from chem_ts_corr.preprocess import preprocess_frame
from chem_ts_corr.screening import final_ranked_features


def test_screening_negative_best_lag_is_not_abs_converted_for_model_lift():
    assert screening._nearby_lags(-3, 12) == [0]
    source = inspect.getsource(screening._nearby_lags)
    assert "abs(best_lag)" not in source


def test_preprocess_keeps_minimum_row_guard_after_recording_drop_metadata():
    index = pd.date_range("2026-01-01", periods=4, freq="min")
    frame = pd.DataFrame(
        {
            "target": [1, 2, 3, 4],
            "x": [None, 2, 3, None],
        },
        index=index,
    )
    with pytest.raises(ValueError, match="Not enough usable rows after preprocessing"):
        preprocess_frame(frame, target="target", resample_rule=None, min_valid_ratio=0.0)


def test_final_ranked_features_removes_redundant_residual_score_assignment():
    source = inspect.getsource(final_ranked_features)
    assert "residual_corr_score" not in source
    assert "display_residual" not in source


def test_final_ranked_features_missing_residual_uses_association_only():
    ranked = pd.DataFrame({"variable": ["x"], "lag": [1], "score": [0.5], "direction": ["变量领先目标"]})
    empty = pd.DataFrame(columns=["variable"])
    result = final_ranked_features(
        ranked=ranked,
        residual=empty,
        stability=empty,
        model_lift=empty,
        risks=pd.DataFrame(columns=["variable", "risk_flags", "risk_count", "strong_risk_count", "weak_risk_count", "risk_level", "human_reason"]),
        lag_peak_quality=empty,
        rolling_corr_scores=empty,
        top_k=10,
    )
    assert pd.isna(result.loc[0, "independent_signal_score"])
    assert result.loc[0, "correlation_evidence_score"] == pytest.approx(0.5)
    assert result.loc[0, "correlation_evidence_status"] == "association_only"
    assert result.loc[0, "residual_status"] == "not_computed"
