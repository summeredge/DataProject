import pandas as pd
import pytest

from chem_ts_corr.modeling import _best_only_lag, _nearby_lags
from chem_ts_corr.preprocess import preprocess_frame
from chem_ts_corr.web import _float_field, _int_field, _optional_float_field, _optional_int_field
from chem_ts_corr.screening import final_ranked_features


def test_negative_best_lag_is_not_abs_converted_for_model_features():
    # A negative screening lag means the candidate lags the target.
    # It must not be converted to a positive lag feature that reverses direction.
    assert _best_only_lag(-3, 12) == 0
    assert _nearby_lags(-3, 12) == [0]


def test_invalid_numeric_form_fields_fall_back_to_defaults():
    form = {
        "max_lag": "abc",
        "min_valid_ratio": "bad",
        "segment_min": "not-a-number",
        "top_n": "bad-int",
    }
    assert _int_field(form, "max_lag", 12) == 12
    assert _float_field(form, "min_valid_ratio", 0.7) == 0.7
    assert _optional_float_field(form, "segment_min") is None
    assert _optional_int_field(form, "top_n") is None


def test_preprocess_records_row_drop_metadata_after_interpolation():
    index = pd.date_range("2026-01-01", periods=12, freq="min")
    frame = pd.DataFrame(
        {
            "target": list(range(1, 13)),
            "x": [None] + list(range(2, 12)) + [None],
        },
        index=index,
    )
    cleaned = preprocess_frame(frame, target="target", resample_rule=None, min_valid_ratio=0.0)
    assert len(cleaned) == 10
    assert cleaned.attrs.get("rows_before_dropna") == 12
    assert cleaned.attrs.get("rows_after_dropna") == 10
    assert cleaned.attrs.get("rows_dropped_by_dropna") == 2
    assert "rows_dropped_by_dropna" in cleaned.attrs.get("preprocess_warnings", "")


def test_final_ranked_features_does_not_compute_residual_score_twice():
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
    assert result.loc[0, "correlation_evidence_score"] == pytest.approx(0.5)
    assert result.loc[0, "correlation_evidence_status"] == "association_only"
    assert "independent_signal_score" not in result.columns
    assert "residual_status" not in result.columns
