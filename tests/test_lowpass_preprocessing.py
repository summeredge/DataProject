from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.preprocess import (
    lowpass_filter_frame,
    transform_frame,
    transform_frame_causal,
)
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


def test_constant_signal_stays_constant():
    frame = pd.DataFrame(
        {"x": [5.0] * 20},
        index=pd.date_range("2026-01-01", periods=20, freq="1min"),
    )

    result = lowpass_filter_frame(frame, tau_minutes=5.0)

    assert np.allclose(result["x"].to_numpy(), 5.0)


def test_step_response_matches_known_alpha_per_point():
    tau = 5.0
    rows = 21
    frame = pd.DataFrame(
        {"x": np.concatenate([[0.0], np.ones(rows - 1)])},
        index=pd.date_range("2026-01-01", periods=rows, freq="1min"),
    )

    result = lowpass_filter_frame(frame, tau_minutes=tau)

    alpha = 1.0 - np.exp(-1.0 / tau)
    expected = np.zeros(rows)
    for i in range(1, rows):
        expected[i] = expected[i - 1] + alpha * (1.0 - expected[i - 1])
    # Closed form for a step input with initial value 0: y(t) = 1 - exp(-t/tau).
    assert np.allclose(expected, 1.0 - np.exp(-np.arange(rows) / tau), atol=1e-12)
    assert np.allclose(result["x"].to_numpy(), expected, atol=1e-12)


def test_physical_time_consistency_between_sampling_rates():
    tau = 5.0
    fine = pd.DataFrame(
        {"x": np.concatenate([[0.0], np.ones(30)])},
        index=pd.date_range("2026-01-01", periods=31, freq="1min"),
    )
    coarse = pd.DataFrame(
        {"x": np.concatenate([[0.0], np.ones(6)])},
        index=pd.date_range("2026-01-01", periods=7, freq="5min"),
    )

    fine_result = lowpass_filter_frame(fine, tau_minutes=tau)
    coarse_result = lowpass_filter_frame(coarse, tau_minutes=tau)

    for i in range(7):
        minutes = 5 * i
        expected = 1.0 - np.exp(-minutes / tau)
        assert fine_result["x"].iloc[5 * i] == pytest.approx(expected, abs=1e-12)
        assert coarse_result["x"].iloc[i] == pytest.approx(expected, abs=1e-12)


def test_irregular_sampling_uses_actual_delta_t():
    # Adjacent intervals are 4/2/4/2/4 minutes. The nominal period is the
    # 4-minute mode and the gap threshold is 1.5 * 4 = 6 minutes, so every
    # interval stays in the same physical segment. Under the old rule the
    # 2-minute diffs reset the state and this exact recurrence would fail.
    frame = pd.DataFrame(
        {"x": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]},
        index=_timestamps_from_intervals([4, 2, 4, 2, 4]),
    )
    assert sample_period_ns(frame) == 4 * MINUTE_NS
    tau = 5.0

    result = lowpass_filter_frame(frame, tau_minutes=tau)

    alpha4 = 1.0 - np.exp(-4.0 / tau)
    alpha2 = 1.0 - np.exp(-2.0 / tau)
    y0 = 0.0
    y1 = y0 + alpha4 * (10.0 - y0)
    y2 = y1 + alpha2 * (20.0 - y1)
    y3 = y2 + alpha4 * (30.0 - y2)
    y4 = y3 + alpha2 * (40.0 - y3)
    y5 = y4 + alpha4 * (50.0 - y4)
    expected = np.array([y0, y1, y2, y3, y4, y5])
    np.testing.assert_allclose(result["x"].to_numpy(), expected, atol=1e-12)


def test_interval_shorter_than_nominal_period_stays_contiguous():
    frame = pd.DataFrame(
        {"x": [0.0, 10.0, 20.0, 30.0]},
        index=_timestamps_from_intervals([4, 2, 4]),
    )
    assert sample_period_ns(frame) == 4 * MINUTE_NS
    tau = 5.0

    result = lowpass_filter_frame(frame, tau_minutes=tau)

    alpha4 = 1.0 - np.exp(-4.0 / tau)
    alpha2 = 1.0 - np.exp(-2.0 / tau)
    y0 = 0.0
    y1 = y0 + alpha4 * (10.0 - y0)
    y2 = y1 + alpha2 * (20.0 - y1)
    y3 = y2 + alpha4 * (30.0 - y2)
    np.testing.assert_allclose(result["x"].to_numpy(), [y0, y1, y2, y3], atol=1e-12)


def test_interval_within_gap_threshold_stays_contiguous():
    frame = pd.DataFrame(
        {"x": [0.0, 10.0, 20.0, 30.0]},
        index=_timestamps_from_intervals([4, 5, 4]),
    )
    assert sample_period_ns(frame) == 4 * MINUTE_NS
    tau = 5.0

    result = lowpass_filter_frame(frame, tau_minutes=tau)

    alpha4 = 1.0 - np.exp(-4.0 / tau)
    alpha5 = 1.0 - np.exp(-5.0 / tau)
    y0 = 0.0
    y1 = y0 + alpha4 * (10.0 - y0)
    y2 = y1 + alpha5 * (20.0 - y1)
    y3 = y2 + alpha4 * (30.0 - y2)
    np.testing.assert_allclose(result["x"].to_numpy(), [y0, y1, y2, y3], atol=1e-12)


def test_interval_beyond_gap_threshold_starts_new_segment():
    frame = pd.DataFrame(
        {"x": [0.0, 10.0, 20.0, 30.0]},
        index=_timestamps_from_intervals([4, 7, 4]),
    )
    assert sample_period_ns(frame) == 4 * MINUTE_NS
    tau = 5.0

    result = lowpass_filter_frame(frame, tau_minutes=tau)

    alpha4 = 1.0 - np.exp(-4.0 / tau)
    y0 = 0.0
    y1 = y0 + alpha4 * (10.0 - y0)
    # 7 minutes > 1.5 * 4 minutes: the third point starts a new segment and
    # its raw value becomes the new state directly.
    y2 = 20.0
    y3 = y2 + alpha4 * (30.0 - y2)
    np.testing.assert_allclose(result["x"].to_numpy(), [y0, y1, y2, y3], atol=1e-12)
    assert result["x"].iloc[2] == 20.0


def test_interval_equal_to_gap_threshold_stays_contiguous():
    frame = pd.DataFrame(
        {"x": [0.0, 10.0, 20.0, 30.0]},
        index=_timestamps_from_intervals([4, 6, 4]),
    )
    assert sample_period_ns(frame) == 4 * MINUTE_NS
    tau = 5.0

    result = lowpass_filter_frame(frame, tau_minutes=tau)

    alpha4 = 1.0 - np.exp(-4.0 / tau)
    alpha6 = 1.0 - np.exp(-6.0 / tau)
    y0 = 0.0
    y1 = y0 + alpha4 * (10.0 - y0)
    y2 = y1 + alpha6 * (20.0 - y1)
    y3 = y2 + alpha4 * (30.0 - y2)
    np.testing.assert_allclose(result["x"].to_numpy(), [y0, y1, y2, y3], atol=1e-12)


def test_physical_gap_resets_filter_state():
    # Drop timestamps 2 and 5 minutes from a regular 1-minute grid, creating
    # three contiguous segments: [0,1], [3,4], [6,7,8].
    complete = pd.date_range("2026-01-01", periods=9, freq="1min")
    index = complete.drop([complete[2], complete[5]])
    frame = pd.DataFrame(
        {"x": [0.0, 0.0, 10.0, 11.0, 100.0, 101.0, 102.0]},
        index=index,
    )
    tau = 5.0

    result = lowpass_filter_frame(frame, tau_minutes=tau)

    alpha1 = 1.0 - np.exp(-1.0 / tau)
    y7 = 100.0 + alpha1
    expected = [
        0.0,
        0.0,
        10.0,
        10.0 + alpha1,
        100.0,
        y7,
        y7 + alpha1 * (102.0 - y7),
    ]
    np.testing.assert_allclose(result["x"].to_numpy(), expected, atol=1e-12)
    # New segment's first valid value is used directly, with no state carried
    # over from the previous segment.
    assert result["x"].iloc[2] == 10.0
    assert result["x"].iloc[4] == 100.0


def test_second_segment_changes_do_not_affect_first_segment():
    complete = pd.date_range("2026-01-01", periods=9, freq="1min")
    index = complete.drop([complete[2], complete[5]])
    baseline = pd.DataFrame(
        {"x": [0.0, 0.0, 10.0, 11.0, 100.0, 101.0, 102.0]},
        index=index,
    )
    changed = pd.DataFrame(
        {"x": [0.0, 0.0, 1000.0, 1001.0, 100.0, 101.0, 102.0]},
        index=index,
    )

    baseline_result = lowpass_filter_frame(baseline, tau_minutes=5.0)
    changed_result = lowpass_filter_frame(changed, tau_minutes=5.0)

    first_segment = [0, 1]
    last_segment = [4, 5, 6]
    assert np.array_equal(
        baseline_result["x"].to_numpy()[first_segment],
        changed_result["x"].to_numpy()[first_segment],
    )
    assert not np.array_equal(
        baseline_result["x"].to_numpy()[2:4],
        changed_result["x"].to_numpy()[2:4],
    )
    assert np.array_equal(
        baseline_result["x"].to_numpy()[last_segment],
        changed_result["x"].to_numpy()[last_segment],
    )


def test_columns_are_filtered_independently():
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
        },
        index=pd.date_range("2026-01-01", periods=6, freq="1min"),
    )
    tau = 5.0

    result = lowpass_filter_frame(frame, tau_minutes=tau)
    single_a = lowpass_filter_frame(frame[["a"]], tau_minutes=tau)

    pd.testing.assert_series_equal(result["a"], single_a["a"], check_names=False)
    assert result.index.equals(frame.index)
    assert pd.notna(result["a"].iloc[2])
    assert pd.isna(result["b"].iloc[2])

    alpha1 = 1.0 - np.exp(-1.0 / tau)
    alpha2 = 1.0 - np.exp(-2.0 / tau)
    y1 = alpha1
    y3 = y1 + alpha2 * (3.0 - y1)
    y4 = y3 + alpha1 * (4.0 - y3)
    y5 = y4 + alpha1 * (5.0 - y4)
    expected_b = [0.0, y1, np.nan, y3, y4, y5]
    for actual, expected in zip(result["b"].to_numpy(), expected_b):
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected, abs=1e-12)


def test_missing_values_keep_missing_and_never_backfill():
    frame = pd.DataFrame(
        {"x": [np.nan, 0.0, 1.0, np.nan, 2.0, 3.0]},
        index=pd.date_range("2026-01-01", periods=6, freq="1min"),
    )
    tau = 5.0

    result = lowpass_filter_frame(frame, tau_minutes=tau)

    alpha1 = 1.0 - np.exp(-1.0 / tau)
    alpha2 = 1.0 - np.exp(-2.0 / tau)
    y1 = 0.0
    y2 = y1 + alpha1 * (1.0 - y1)
    y4 = y2 + alpha2 * (2.0 - y2)
    y5 = y4 + alpha1 * (3.0 - y4)
    expected = [np.nan, y1, y2, np.nan, y4, y5]
    for actual, expected_value in zip(result["x"].to_numpy(), expected):
        if np.isnan(expected_value):
            assert np.isnan(actual)
            assert actual != 0.0
        else:
            assert actual == pytest.approx(expected_value, abs=1e-12)

    # A prefix of the frame gives the same values: later data never feeds back
    # into earlier outputs and missing outputs are never filled.
    prefix = lowpass_filter_frame(frame.iloc[:3], tau_minutes=tau)
    assert prefix["x"].iloc[2] == pytest.approx(y2, abs=1e-12)


def test_input_frame_is_not_modified_and_structure_is_preserved():
    frame = pd.DataFrame(
        {
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "a": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        },
        index=pd.date_range("2026-01-01", periods=6, freq="5min"),
    )
    frame.attrs["lag_sample_period_ns"] = 5 * MINUTE_NS
    frame.attrs["custom_attr"] = "keep"
    snapshot = frame.copy(deep=True)
    snapshot.attrs = dict(frame.attrs)

    result = lowpass_filter_frame(frame, tau_minutes=5.0)

    pd.testing.assert_frame_equal(frame, snapshot)
    assert frame.attrs == snapshot.attrs
    assert result.index.equals(frame.index)
    assert result.columns.tolist() == ["b", "a"]
    assert sample_period_ns(result) == 5 * MINUTE_NS
    assert result.attrs["custom_attr"] == "keep"


@pytest.mark.parametrize(
    "bad_tau",
    [0, -1.0, -0.001, float("nan"), float("inf"), -float("inf"), "abc"],
)
def test_rejects_invalid_tau_minutes(bad_tau):
    with pytest.raises(ValueError, match="tau_minutes"):
        lowpass_filter_frame(_regular_frame(5), tau_minutes=bad_tau)


def test_rejects_non_datetime_index():
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.RangeIndex(2))

    with pytest.raises(ValueError, match="DatetimeIndex"):
        lowpass_filter_frame(frame, tau_minutes=5.0)


def test_rejects_non_monotonic_index():
    frame = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2026-01-01 00:01", "2026-01-01 00:00"]),
    )

    with pytest.raises(ValueError, match="monotonic"):
        lowpass_filter_frame(frame, tau_minutes=5.0)


def test_rejects_duplicate_timestamps():
    frame = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 00:00"]),
    )

    with pytest.raises(ValueError, match="unique"):
        lowpass_filter_frame(frame, tau_minutes=5.0)


def test_empty_frame_returns_consistent_empty_frame():
    index = pd.DatetimeIndex([])
    frame = pd.DataFrame(index=index, columns=["x"], dtype=float)

    result = lowpass_filter_frame(frame, tau_minutes=5.0)

    assert result.empty
    assert result.index.equals(index)
    assert result.columns.tolist() == ["x"]


def test_zero_column_frame_keeps_index():
    index = pd.date_range("2026-01-01", periods=5, freq="1min")
    frame = pd.DataFrame(index=index)

    result = lowpass_filter_frame(frame, tau_minutes=5.0)

    assert result.empty
    assert len(result.columns) == 0
    assert result.index.equals(index)


def test_single_point_passes_through():
    frame = pd.DataFrame(
        {"x": [3.5]},
        index=pd.DatetimeIndex(["2026-01-01 00:00"]),
    )

    result = lowpass_filter_frame(frame, tau_minutes=5.0)

    assert result["x"].iloc[0] == 3.5
    assert result.index.equals(frame.index)


def test_all_missing_column_stays_missing_without_dropping_other_columns():
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [np.nan, np.nan, np.nan, np.nan],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="1min"),
    )

    result = lowpass_filter_frame(frame, tau_minutes=5.0)
    single_a = lowpass_filter_frame(frame[["a"]], tau_minutes=5.0)

    assert result["b"].isna().all()
    assert len(result) == len(frame)
    pd.testing.assert_series_equal(result["a"], single_a["a"], check_names=False)


@pytest.mark.parametrize("mode", ["lowpass", "lowpass_detrend", "lowpass_diff"])
def test_lowpass_modes_run_in_transform_frame_and_causal_paths(mode: str):
    frame = _regular_frame(20)

    transformed = transform_frame(frame, mode, 24)
    causal = transform_frame_causal(frame, mode, 24)

    assert transformed.columns.tolist() == frame.columns.tolist()
    assert not transformed.empty
    assert causal.columns.tolist() == frame.columns.tolist()
    assert not causal.empty
