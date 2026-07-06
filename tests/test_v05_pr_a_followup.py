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
    index = pd.date_range("2026-01-01", periods=6, freq="min")
    frame = pd.DataFrame(
        {
            "target": [1, 2, 3, 4, 5, 6],
            "x": [None, 2, 3, 4, 5, None],
        },
        index=index,
    )
    with pytest.raises(ValueError, match="Not enough usable rows after preprocessing"):
        preprocess_frame(frame, target="target", resample_rule=None, min_valid_ratio=0.0)


def test_final_ranked_features_removes_redundant_residual_score_assignment():
    source = inspect.getsource(final_ranked_features)
    assert source.count('final["residual_corr_score"]') == 1
    assert "display_residual" in source


def test_final_ranked_features_residual_display_semantics_unchanged():
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
    assert result.loc[0, "residual_corr_score"] == pytest.approx(0.5)
    assert result.loc[0, "residual_status"] == "not_computed"
