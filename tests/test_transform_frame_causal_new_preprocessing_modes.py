from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.config import (
    NOT_IMPLEMENTED_PREPROCESS_MODES,
    NOT_WIRED_ANALYSIS_PREPROCESS_MODES,
    AnalysisConfig,
)
from chem_ts_corr.preprocess import (
    detrend_trailing_average,
    difference_by_physical_interval,
    lowpass_filter_frame,
    transform_frame_causal,
)
from chem_ts_corr.service import analyze_numeric_frame
from chem_ts_corr.time_axis import sample_period_ns


MINUTE_NS = 60 * 1_000_000_000


def _regular_frame(
    rows: int = 10,
    freq: str = "1min",
    columns: tuple[str, ...] = ("x",),
) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq=freq)
    return pd.DataFrame(
        {column: np.arange(rows, dtype=float) for column in columns},
        index=index,
    )


def _timestamps_from_intervals(intervals_minutes: list[int]) -> pd.DatetimeIndex:
    offsets = np.cumsum([0, *intervals_minutes])
    return pd.to_datetime("2026-01-01 00:00") + pd.to_timedelta(offsets, unit="min")


def _raw_frame() -> pd.DataFrame:
    rows = 120
    time = np.arange(rows, dtype=float)
    controls = [f"control_{index}" for index in range(8)]
    candidates = [f"candidate_{index}" for index in range(5)]
    return pd.DataFrame(
        {
            "target": np.sin(time / 7),
            **{name: np.sin((time + index + 1) / 7) for index, name in enumerate(candidates)},
            **{name: np.cos((time + index + 1) / 7) for index, name in enumerate(controls)},
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="min"),
    )


def _raw_config(tmp_path: Path, **overrides) -> AnalysisConfig:
    controls = [f"control_{index}" for index in range(8)]
    kwargs = {
        "input_path": tmp_path / "input.csv",
        "time_column": "time",
        "target": "target",
        "output_dir": tmp_path,
        "max_lag": 3,
        "top_k": 15,
        "residual_control_columns": controls,
        "force_include_variables": [],
        "enable_model": False,
        "skip_model_lift": True,
        "skip_rolling_corr": True,
    }
    kwargs.update(overrides)
    return AnalysisConfig(**kwargs)


@pytest.mark.parametrize("tau", [2.0, 10.0])
def test_causal_lowpass_is_equivalent_to_lowpass_filter_frame(tau: float):
    frame = _regular_frame(20)

    result = transform_frame_causal(
        frame,
        "lowpass",
        24,
        lowpass_tau_minutes=tau,
    )
    expected = lowpass_filter_frame(frame, tau_minutes=tau)

    pd.testing.assert_frame_equal(result, expected)


def test_causal_lowpass_tau_parameter_actually_changes_the_output():
    frame = _regular_frame(20)

    result_fast = transform_frame_causal(
        frame, "lowpass", 24, lowpass_tau_minutes=2.0
    )
    result_slow = transform_frame_causal(
        frame, "lowpass", 24, lowpass_tau_minutes=10.0
    )

    assert not np.allclose(
        result_fast["x"].to_numpy(),
        result_slow["x"].to_numpy(),
        equal_nan=True,
    )
    pd.testing.assert_frame_equal(
        result_fast,
        lowpass_filter_frame(frame, tau_minutes=2.0),
    )
    pd.testing.assert_frame_equal(
        result_slow,
        lowpass_filter_frame(frame, tau_minutes=10.0),
    )


def test_causal_lowpass_detrend_locks_lowpass_then_trailing_detrend_order():
    rows = 40
    values = np.concatenate([np.zeros(10), np.ones(30)]) + np.arange(rows) * 0.1
    frame = pd.DataFrame(
        {"x": values.astype(float)},
        index=pd.date_range("2026-01-01", periods=rows, freq="1min"),
    )
    tau = 5.0

    result = transform_frame_causal(
        frame,
        "lowpass_detrend",
        6,
        lowpass_tau_minutes=tau,
    )
    expected = detrend_trailing_average(
        lowpass_filter_frame(frame, tau_minutes=tau),
        6,
    )
    reversed_order = lowpass_filter_frame(
        detrend_trailing_average(frame, 6),
        tau_minutes=tau,
    )

    pd.testing.assert_frame_equal(result, expected)
    # The two orders differ on this data, so equivalence with the specified
    # composition really locks the execution order.
    assert not np.allclose(
        result["x"].to_numpy(),
        reversed_order["x"].reindex(result.index).to_numpy(),
        atol=1e-9,
        equal_nan=True,
    )


def test_causal_lowpass_detrend_does_not_use_future_values():
    rows = 40
    frame = pd.DataFrame(
        {"x": np.arange(rows, dtype=float) * 0.1},
        index=pd.date_range("2026-01-01", periods=rows, freq="1min"),
    )
    prefix = frame.iloc[:20]
    future_modified = frame.copy()
    future_modified.iloc[20:] = 1e30

    prefix_result = transform_frame_causal(
        prefix, "lowpass_detrend", 6, lowpass_tau_minutes=5.0
    )
    full_result = transform_frame_causal(
        future_modified, "lowpass_detrend", 6, lowpass_tau_minutes=5.0
    )

    pd.testing.assert_frame_equal(full_result.loc[prefix_result.index], prefix_result)
    # Sanity check: the future spike really changes the suffix output, so the
    # prefix invariance above is not vacuous.
    unmodified = transform_frame_causal(
        frame, "lowpass_detrend", 6, lowpass_tau_minutes=5.0
    )
    assert not np.allclose(
        full_result.loc[future_modified.index[20:], "x"].to_numpy(),
        unmodified.loc[future_modified.index[20:], "x"].to_numpy(),
        atol=1e-6,
        equal_nan=True,
    )


def test_causal_lowpass_diff_is_equivalent_to_lowpass_then_interval_diff():
    frame = _regular_frame(12)
    tau = 2.0
    interval = 5.0

    result = transform_frame_causal(
        frame,
        "lowpass_diff",
        24,
        lowpass_tau_minutes=tau,
        diff_interval_minutes=interval,
    )
    expected = difference_by_physical_interval(
        lowpass_filter_frame(frame, tau_minutes=tau),
        diff_interval_minutes=interval,
    ).dropna(how="all")

    assert result.index.equals(expected.index)
    assert result.columns.tolist() == expected.columns.tolist()
    pd.testing.assert_frame_equal(result, expected)
    assert sample_period_ns(result) == sample_period_ns(expected) == MINUTE_NS
    assert result.attrs == expected.attrs


@pytest.mark.parametrize("mode", ["lowpass", "lowpass_detrend", "lowpass_diff"])
def test_causal_modes_are_prefix_invariant(mode: str):
    frame = _regular_frame(30, columns=("x", "y"))
    prefix = frame.iloc[:20]

    full_result = transform_frame_causal(
        frame,
        mode,
        8,
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=5.0,
    )
    prefix_result = transform_frame_causal(
        prefix,
        mode,
        8,
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=5.0,
    )

    pd.testing.assert_frame_equal(full_result.loc[prefix_result.index], prefix_result)


@pytest.mark.parametrize("mode", ["lowpass", "lowpass_detrend", "lowpass_diff"])
def test_extreme_future_value_does_not_change_causal_prefix(mode: str):
    frame = _regular_frame(30, columns=("x", "y"))
    prefix = frame.iloc[:20]
    extended = frame.iloc[:21].copy()
    extended.iloc[-1] = 1e30

    prefix_result = transform_frame_causal(
        prefix,
        mode,
        8,
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=5.0,
    )
    full_result = transform_frame_causal(
        extended,
        mode,
        8,
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=5.0,
    )

    pd.testing.assert_frame_equal(full_result.loc[prefix_result.index], prefix_result)


def test_causal_lowpass_state_does_not_cross_physical_gap():
    complete = pd.date_range("2026-01-01", periods=9, freq="1min")
    index = complete.drop([complete[2], complete[5]])
    frame = pd.DataFrame(
        {"x": [0.0, 0.0, 10.0, 11.0, 100.0, 101.0, 102.0]},
        index=index,
    )

    result = transform_frame_causal(frame, "lowpass", 24, lowpass_tau_minutes=5.0)
    expected = lowpass_filter_frame(frame, tau_minutes=5.0)

    pd.testing.assert_frame_equal(result, expected)
    # New segments start with the raw value directly, never borrowing state.
    assert result["x"].iloc[2] == 10.0
    assert result["x"].iloc[4] == 100.0


def test_causal_lowpass_diff_does_not_cross_physical_gap():
    complete = pd.date_range("2026-01-01", periods=9, freq="1min")
    index = complete.drop([complete[2]])
    frame = pd.DataFrame(
        {"x": np.arange(8, dtype=float)},
        index=index,
    )
    smoothed = lowpass_filter_frame(frame, tau_minutes=5.0)

    result = transform_frame_causal(
        frame,
        "lowpass_diff",
        24,
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=2.0,
    )
    expected = difference_by_physical_interval(
        smoothed,
        diff_interval_minutes=2.0,
    ).dropna(how="all")

    pd.testing.assert_frame_equal(result, expected)
    # The gap resets the filter, and the first positions of the new segment
    # stay missing so the multi-point difference never borrows pre-gap history.
    assert complete[3] not in result.index
    assert complete[4] not in result.index
    assert complete[5] in result.index
    assert result.loc[complete[5], "x"] == pytest.approx(
        smoothed.loc[complete[5], "x"] - smoothed.loc[complete[3], "x"],
        abs=1e-12,
    )


def test_causal_lowpass_irregular_contiguous_axis_keeps_one_segment():
    frame = pd.DataFrame(
        {"x": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]},
        index=_timestamps_from_intervals([4, 2, 4, 2, 4]),
    )
    assert sample_period_ns(frame) == 4 * MINUTE_NS

    result = transform_frame_causal(frame, "lowpass", 24, lowpass_tau_minutes=5.0)
    expected = lowpass_filter_frame(frame, tau_minutes=5.0)

    pd.testing.assert_frame_equal(result, expected)
    # Actual historical deltas stay in one filter state: 4/2/4/2/4 minutes.
    alpha4 = 1.0 - np.exp(-4.0 / 5.0)
    alpha2 = 1.0 - np.exp(-2.0 / 5.0)
    y0 = 0.0
    y1 = y0 + alpha4 * (10.0 - y0)
    y2 = y1 + alpha2 * (20.0 - y1)
    y3 = y2 + alpha4 * (30.0 - y2)
    y4 = y3 + alpha2 * (40.0 - y3)
    y5 = y4 + alpha4 * (50.0 - y4)
    np.testing.assert_allclose(
        result["x"].to_numpy(),
        [y0, y1, y2, y3, y4, y5],
        atol=1e-12,
    )


def test_causal_lowpass_diff_irregular_contiguous_axis_does_not_split_segments():
    frame = pd.DataFrame(
        {"x": np.arange(7, dtype=float)},
        index=_timestamps_from_intervals([4, 2, 4, 5, 6, 4]),
    )
    assert sample_period_ns(frame) == 4 * MINUTE_NS
    smoothed = lowpass_filter_frame(frame, tau_minutes=5.0)

    result = transform_frame_causal(
        frame,
        "lowpass_diff",
        24,
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=8.0,
    )
    expected = difference_by_physical_interval(
        smoothed,
        diff_interval_minutes=8.0,
    ).dropna(how="all")

    pd.testing.assert_frame_equal(result, expected)
    # A 2-point difference on the 4-minute nominal period: only the first two
    # positions are dropped, so every 2/4/5/6-minute interval stays contiguous.
    assert len(result) == len(smoothed) - 2
    assert result["x"].iloc[0] == pytest.approx(
        smoothed["x"].iloc[2] - smoothed["x"].iloc[0],
        abs=1e-12,
    )


def test_causal_lowpass_diff_none_interval_uses_one_analysis_sampling_period():
    frame = _regular_frame(10)

    result = transform_frame_causal(
        frame,
        "lowpass_diff",
        24,
        diff_interval_minutes=None,
    )
    smoothed = lowpass_filter_frame(frame, tau_minutes=5.0)
    expected = difference_by_physical_interval(
        smoothed,
        diff_interval_minutes=None,
    ).dropna(how="all")

    pd.testing.assert_frame_equal(result, expected)
    # One analysis sampling period on a 1-minute axis: the first surviving row
    # is the one-point difference, never interpreted as 0.0.
    assert result.index[0] == frame.index[1]
    assert result["x"].iloc[0] == pytest.approx(
        smoothed["x"].iloc[1] - smoothed["x"].iloc[0],
        abs=1e-12,
    )


def test_causal_lowpass_diff_five_minute_interval_uses_five_point_difference():
    frame = _regular_frame(12, "1min")
    smoothed = lowpass_filter_frame(frame, tau_minutes=5.0)

    result = transform_frame_causal(
        frame,
        "lowpass_diff",
        24,
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=5.0,
    )
    expected = difference_by_physical_interval(
        smoothed,
        diff_interval_minutes=5.0,
    ).dropna(how="all")

    pd.testing.assert_frame_equal(result, expected)
    # 5 points, not a fixed 1-point difference: the first row compares t5 - t0.
    assert result.index[0] == frame.index[5]
    assert result["x"].iloc[0] == pytest.approx(
        smoothed["x"].iloc[5] - smoothed["x"].iloc[0],
        abs=1e-12,
    )


def test_causal_lowpass_diff_multi_column_nan_semantics():
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        },
        index=pd.date_range("2026-01-01", periods=6, freq="1min"),
    )
    smoothed = lowpass_filter_frame(frame, tau_minutes=5.0)

    result = transform_frame_causal(
        frame,
        "lowpass_diff",
        24,
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=2.0,
    )
    expected = difference_by_physical_interval(
        smoothed,
        diff_interval_minutes=2.0,
    ).dropna(how="all")

    pd.testing.assert_frame_equal(result, expected)
    # Only rows where every column is uncomputable are dropped: rows t0 and t1
    # have no 2-point history, while t2 survives because column a still has a
    # valid difference even though column b is missing there.
    assert result.index[0] == frame.index[2]
    assert len(result) == len(frame) - 2
    # Original missing stays missing, never backfilled or written as 0.0.
    assert pd.isna(result.loc[frame.index[2], "b"])
    assert pd.notna(result.loc[frame.index[2], "a"])
    # Fixed history positions are never skipped: b at t3 uses t1 (valid),
    # while b at t4 uses t2 (missing) and therefore stays missing.
    assert pd.notna(result.loc[frame.index[3], "b"])
    assert pd.isna(result.loc[frame.index[4], "b"])
    assert pd.notna(result.loc[frame.index[5], "b"])


@pytest.mark.parametrize(
    "mode",
    ["lowpass", "lowpass_detrend", "lowpass_diff"],
)
@pytest.mark.parametrize(
    "bad_tau",
    [0, -1.0, -0.001, float("nan"), float("inf"), -float("inf"), "abc"],
)
def test_causal_invalid_lowpass_tau_minutes_fails_in_all_lowpass_modes(
    mode: str,
    bad_tau: object,
):
    frame = _regular_frame(10)

    with pytest.raises(ValueError, match="tau_minutes"):
        transform_frame_causal(frame, mode, 24, lowpass_tau_minutes=bad_tau)


@pytest.mark.parametrize(
    "bad_interval",
    [0, -1.0, -0.001, float("nan"), float("inf"), -float("inf"), "abc"],
)
def test_causal_invalid_diff_interval_minutes_fails_in_lowpass_diff(
    bad_interval: object,
):
    frame = _regular_frame(10)

    with pytest.raises(ValueError, match="diff_interval_minutes"):
        transform_frame_causal(
            frame,
            "lowpass_diff",
            24,
            diff_interval_minutes=bad_interval,
        )


def test_stage_constant_semantics():
    assert NOT_WIRED_ANALYSIS_PREPROCESS_MODES == {
        "lowpass",
        "lowpass_detrend",
        "lowpass_diff",
    }
    assert NOT_IMPLEMENTED_PREPROCESS_MODES == NOT_WIRED_ANALYSIS_PREPROCESS_MODES


@pytest.mark.parametrize("mode", sorted(NOT_WIRED_ANALYSIS_PREPROCESS_MODES))
def test_analysis_flow_still_rejects_lowpass_modes(tmp_path: Path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode)

    with pytest.raises(ValueError, match="analysis/screening flow"):
        analyze_numeric_frame(_raw_frame(), config)


@pytest.mark.parametrize("mode", ["raw", "detrend", "diff", "detrend_diff"])
def test_causal_legacy_modes_are_unaffected_by_new_parameters(mode: str):
    frame = _regular_frame(20, columns=("x", "y"))

    baseline = transform_frame_causal(frame, mode, 8)
    varied = transform_frame_causal(
        frame,
        mode,
        8,
        lowpass_tau_minutes=30.0,
        diff_interval_minutes=3.0,
    )

    pd.testing.assert_frame_equal(baseline, varied)
