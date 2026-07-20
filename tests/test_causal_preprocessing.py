from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import preprocess, web


def _frame(rows: int = 80) -> pd.DataFrame:
    values = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {"target": values**2 + 10.0, "x": values**3 + 100.0},
        index=pd.date_range("2026-01-01", periods=rows, freq="5min"),
    )


def _causal(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    cleaned = preprocess.preprocess_frame_causal(frame, "target", None, 2)
    return preprocess.transform_frame_causal(cleaned, mode, detrend_window=8)


@pytest.mark.parametrize("mode", ["raw", "diff", "detrend", "detrend_diff"])
def test_causal_preprocessing_is_prefix_invariant(mode: str):
    frame = _frame()
    prefix = frame.iloc[:50]

    full_result = _causal(frame, mode)
    prefix_result = _causal(prefix, mode)

    pd.testing.assert_frame_equal(full_result.loc[prefix_result.index], prefix_result)


@pytest.mark.parametrize("mode", ["raw", "diff", "detrend", "detrend_diff"])
def test_extreme_future_value_does_not_change_causal_prefix(mode: str):
    frame = _frame()
    baseline = _causal(frame.iloc[:50], mode)
    extended = frame.iloc[:51].copy()
    extended.iloc[-1] = 1e30

    result = _causal(extended, mode)

    pd.testing.assert_frame_equal(result.loc[baseline.index], baseline)


def test_validation_value_does_not_fill_missing_training_tail():
    frame = _frame(30)
    frame.iloc[19, frame.columns.get_loc("x")] = np.nan
    frame.iloc[20, frame.columns.get_loc("x")] = 999999.0

    result = preprocess.preprocess_frame_causal(frame, "target", None, 2)

    assert result.iloc[19]["x"] == frame.iloc[18]["x"]
    assert result.iloc[19]["x"] != frame.iloc[20]["x"]


@pytest.mark.parametrize("mode", ["raw", "diff", "detrend", "detrend_diff"])
def test_missing_target_is_removed_before_causal_transformation(mode: str):
    frame = _frame(40)
    missing_time = frame.index[20]
    expected_target = frame["target"].drop(index=missing_time)
    frame.loc[missing_time, "target"] = np.nan

    cleaned = preprocess.preprocess_frame_causal(frame, "target", None, 2)
    transformed = preprocess.transform_frame_causal(cleaned, mode, detrend_window=8)

    assert missing_time not in cleaned.index
    assert missing_time not in transformed.index
    pd.testing.assert_series_equal(cleaned["target"], expected_target, check_names=True)


def test_causal_forward_fill_does_not_cross_physical_gap():
    frame = _frame(30).drop(index=_frame(30).index[15])
    after_gap = frame.index[15]
    frame.loc[after_gap, "x"] = np.nan

    cleaned = preprocess.preprocess_frame_causal(frame, "target", None, 5)

    assert pd.isna(cleaned.loc[after_gap, "x"])


def test_causal_diff_and_trailing_detrend_restart_after_physical_gap():
    complete = _frame(40)
    frame = complete.drop(index=complete.index[20])
    after_gap = complete.index[21]

    differenced = preprocess.transform_frame_causal(frame, "diff", detrend_window=8)
    detrended = preprocess.transform_frame_causal(frame, "detrend", detrend_window=8)

    assert after_gap not in differenced.index
    assert after_gap not in detrended.index


def test_trailing_detrend_does_not_use_future_values():
    frame = _frame(30)
    baseline = preprocess.detrend_trailing_average(frame.iloc[:20], 8)
    changed = frame.copy()
    changed.iloc[20:] = 1e30

    result = preprocess.detrend_trailing_average(changed, 8)

    pd.testing.assert_frame_equal(result.loc[baseline.index], baseline)


def test_retrospective_detrend_remains_centered_but_xgb_path_is_causal():
    retrospective_source = inspect.getsource(preprocess.detrend_moving_average)
    causal_source = inspect.getsource(preprocess.detrend_trailing_average)
    xgb_source = inspect.getsource(web._prepared_frame_for_validation)

    assert "center=True" in retrospective_source
    assert "center=False" in causal_source
    assert "preprocess_frame_causal" in xgb_source
    assert "transform_frame_causal" in xgb_source
    assert "preprocess_frame(" not in xgb_source
    assert "transform_frame(" not in xgb_source
    assert "interpolate(" not in causal_source
    assert "bfill" not in causal_source
