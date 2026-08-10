from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.preprocess import (
    difference_by_physical_interval,
    resolve_diff_interval,
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


def test_default_interval_is_one_sampling_period():
    frame = _regular_frame(10, "1min")

    points, interval = resolve_diff_interval(frame, None)

    assert points == 1
    assert interval == 1.0

    points_coarse, interval_coarse = resolve_diff_interval(
        _regular_frame(10, "5min"), None
    )
    assert points_coarse == 1
    assert interval_coarse == 5.0


def test_five_minute_interval_uses_fixed_five_point_difference():
    frame = _regular_frame(12, "1min")

    points, interval = resolve_diff_interval(frame, 5.0)

    assert points == 5
    assert interval == 5.0
    result = difference_by_physical_interval(frame, 5.0)
    pd.testing.assert_series_equal(
        result["x"], frame["x"].diff(periods=5), check_names=False
    )
    assert result["x"].iloc[5] == 5.0


def test_interval_rounding_locks_python_round_rule():
    frame = _regular_frame(10, "2min")
    assert sample_period_ns(frame) == 2 * MINUTE_NS

    points, interval = resolve_diff_interval(frame, 5.0)

    assert round(5 / 2) == 2
    assert points == 2
    assert interval == 4.0


def test_interval_shorter_than_sampling_period_clamps_to_one_point():
    frame = _regular_frame(10, "5min")

    points, interval = resolve_diff_interval(frame, 1.0)

    assert points == 1
    assert interval == 5.0


def test_irregular_but_contiguous_timestamps_do_not_split_segments():
    frame = pd.DataFrame(
        {"x": np.arange(7, dtype=float)},
        index=_timestamps_from_intervals([4, 2, 4, 5, 6, 4]),
    )
    assert sample_period_ns(frame) == 4 * MINUTE_NS

    points, interval = resolve_diff_interval(frame, 8.0)
    assert points == 2
    assert interval == 8.0
    result = difference_by_physical_interval(frame, 8.0)
    pd.testing.assert_series_equal(
        result["x"], frame["x"].diff(periods=2), check_names=False
    )
    # The 6-minute delta equals the threshold and stays in one segment.
    assert result["x"].iloc[5] == frame["x"].iloc[5] - frame["x"].iloc[3]

    single = difference_by_physical_interval(frame, None)
    assert pd.isna(single["x"].iloc[0])
    assert single["x"].iloc[1:].notna().all()


def test_real_physical_gap_prevents_cross_segment_difference():
    assert 7 > 1.5 * 4
    frame = pd.DataFrame(
        {"x": [1.0, 10.0, 100.0, 110.0]},
        index=_timestamps_from_intervals([4, 7, 4]),
    )
    assert sample_period_ns(frame) == 4 * MINUTE_NS

    points, interval = resolve_diff_interval(frame, 4.0)
    assert points == 1
    assert interval == 4.0
    result = difference_by_physical_interval(frame, 4.0)

    assert np.isnan(result["x"].iloc[0])
    assert result["x"].iloc[1] == 9.0
    # First position of the new segment stays missing even though x1 exists.
    assert np.isnan(result["x"].iloc[2])
    assert result["x"].iloc[3] == 10.0


def test_new_segment_first_diff_points_positions_stay_missing():
    frame = pd.DataFrame(
        {"x": np.arange(5, dtype=float)},
        index=_timestamps_from_intervals([4, 7, 4, 4]),
    )

    result = difference_by_physical_interval(frame, 8.0)

    assert result["x"].iloc[:2].isna().all()
    assert result["x"].iloc[2:4].isna().all()
    # Segment B may not borrow x1 from segment A for its second position.
    assert result["x"].iloc[4] == frame["x"].iloc[4] - frame["x"].iloc[2]


def test_gap_threshold_boundary_stays_contiguous():
    frame = pd.DataFrame(
        {"x": [1.0, 10.0, 20.0, 30.0]},
        index=_timestamps_from_intervals([4, 6, 4]),
    )
    assert sample_period_ns(frame) == 4 * MINUTE_NS

    result = difference_by_physical_interval(frame, 4.0)

    assert np.isnan(result["x"].iloc[0])
    assert result["x"].iloc[1] == 9.0
    assert result["x"].iloc[2] == 10.0
    assert result["x"].iloc[3] == 10.0


def test_difference_direction_is_current_minus_history():
    frame = pd.DataFrame(
        {"x": [10.0, 12.0, 15.0]},
        index=pd.date_range("2026-01-01", periods=3, freq="1min"),
    )

    result = difference_by_physical_interval(frame, None)

    assert np.isnan(result["x"].iloc[0])
    assert result["x"].iloc[1] == 2.0
    assert result["x"].iloc[2] == 3.0
    assert (result["x"].dropna() > 0).all()


def test_missing_values_use_fixed_position_without_skipping():
    frame = pd.DataFrame(
        {"x": [1.0, 2.0, np.nan, 5.0]},
        index=pd.date_range("2026-01-01", periods=4, freq="1min"),
    )

    result = difference_by_physical_interval(frame, 1.0)

    assert np.isnan(result["x"].iloc[0])
    assert result["x"].iloc[1] == 1.0
    assert np.isnan(result["x"].iloc[2])
    # t3 = 5 - NaN stays missing; the NaN is never skipped to reach x1 = 2.
    assert np.isnan(result["x"].iloc[3])

    frame_two = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0, 5.0]},
        index=pd.date_range("2026-01-01", periods=4, freq="1min"),
    )
    result_two = difference_by_physical_interval(frame_two, 2.0)

    assert np.isnan(result_two["x"].iloc[:2]).all()
    assert result_two["x"].iloc[2] == 2.0
    assert np.isnan(result_two["x"].iloc[3])


def test_columns_are_differenced_independently():
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        },
        index=pd.date_range("2026-01-01", periods=6, freq="1min"),
    )

    result = difference_by_physical_interval(frame, 2.0)
    single_a = difference_by_physical_interval(frame[["a"]], 2.0)

    pd.testing.assert_series_equal(result["a"], single_a["a"], check_names=False)
    assert result["a"].iloc[2] == 2.0
    assert np.isnan(result["b"].iloc[2])
    assert result["b"].iloc[3] == 2.0
    assert np.isnan(result["b"].iloc[4])
    assert result["b"].iloc[5] == 2.0


def test_future_values_do_not_change_earlier_outputs():
    frame = _regular_frame(12)

    full = difference_by_physical_interval(frame, 5.0)
    prefix = difference_by_physical_interval(frame.iloc[:7], 5.0)

    pd.testing.assert_frame_equal(full.loc[prefix.index], prefix)


def test_input_is_not_modified_and_structure_is_preserved():
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

    resolve_diff_interval(frame, 10.0)
    result = difference_by_physical_interval(frame, 10.0)

    pd.testing.assert_frame_equal(frame, snapshot)
    assert frame.attrs == snapshot.attrs
    assert result.index.equals(frame.index)
    assert result.columns.tolist() == ["b", "a"]
    assert sample_period_ns(result) == 5 * MINUTE_NS
    assert result.attrs["custom_attr"] == "keep"
    pd.testing.assert_frame_equal(result, frame.diff(periods=2))


@pytest.mark.parametrize(
    "bad_interval",
    [0, -1.0, -0.001, float("nan"), float("inf"), -float("inf"), "abc"],
)
def test_rejects_invalid_diff_interval_minutes(bad_interval):
    frame = _regular_frame(5)

    with pytest.raises(ValueError, match="diff_interval_minutes"):
        resolve_diff_interval(frame, bad_interval)
    with pytest.raises(ValueError, match="diff_interval_minutes"):
        difference_by_physical_interval(frame, bad_interval)


@pytest.mark.parametrize(
    "func",
    [resolve_diff_interval, difference_by_physical_interval],
)
def test_rejects_non_datetime_index(func):
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.RangeIndex(2))

    with pytest.raises(ValueError, match="DatetimeIndex"):
        func(frame, None)


@pytest.mark.parametrize(
    "func",
    [resolve_diff_interval, difference_by_physical_interval],
)
def test_rejects_non_monotonic_index(func):
    frame = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2026-01-01 00:01", "2026-01-01 00:00"]),
    )

    with pytest.raises(ValueError, match="monotonic"):
        func(frame, None)


@pytest.mark.parametrize(
    "func",
    [resolve_diff_interval, difference_by_physical_interval],
)
def test_rejects_duplicate_timestamps(func):
    frame = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 00:00"]),
    )

    with pytest.raises(ValueError, match="unique"):
        func(frame, None)


def test_empty_frame_returns_consistent_empty_frame():
    index = pd.DatetimeIndex([])
    frame = pd.DataFrame(index=index, columns=["x"], dtype=float)

    points, interval = resolve_diff_interval(frame, None)

    assert points == 1
    assert np.isnan(interval)
    result = difference_by_physical_interval(frame, None)
    assert result.empty
    assert result.index.equals(index)
    assert result.columns.tolist() == ["x"]


def test_empty_frame_with_explicit_interval_returns_empty_frame():
    index = pd.DatetimeIndex([])
    frame = pd.DataFrame(index=index, columns=["x"], dtype=float)
    frame.attrs["custom_attr"] = "keep"
    snapshot = frame.copy(deep=True)
    snapshot.attrs = dict(frame.attrs)

    result = difference_by_physical_interval(frame, 5.0)

    assert result.empty
    assert result.index.equals(index)
    assert result.columns.tolist() == ["x"]
    assert result.attrs["custom_attr"] == "keep"
    pd.testing.assert_frame_equal(frame, snapshot)


@pytest.mark.parametrize(
    "bad_interval",
    [0, -1.0, float("nan"), float("inf"), -float("inf"), "abc"],
)
def test_empty_frame_rejects_invalid_diff_interval_minutes(bad_interval):
    frame = pd.DataFrame(index=pd.DatetimeIndex([]), columns=["x"], dtype=float)

    with pytest.raises(ValueError, match="diff_interval_minutes"):
        difference_by_physical_interval(frame, bad_interval)


def test_resolve_diff_interval_still_requires_period_for_explicit_interval_on_empty_frame():
    frame = pd.DataFrame(index=pd.DatetimeIndex([]), columns=["x"], dtype=float)

    with pytest.raises(ValueError, match="sampling period"):
        resolve_diff_interval(frame, 5.0)


def test_zero_column_frame_keeps_index():
    index = pd.date_range("2026-01-01", periods=5, freq="1min")
    frame = pd.DataFrame(index=index)

    result = difference_by_physical_interval(frame, None)

    assert result.empty
    assert len(result.columns) == 0
    assert result.index.equals(index)


def test_single_point_returns_nan_without_sampling_period():
    frame = pd.DataFrame(
        {"x": [3.5]},
        index=pd.DatetimeIndex(["2026-01-01 00:00"]),
    )

    points, interval = resolve_diff_interval(frame, None)
    assert points == 1
    assert np.isnan(interval)

    result = difference_by_physical_interval(frame, None)

    assert np.isnan(result["x"].iloc[0])
    assert result.index.equals(frame.index)


def test_single_point_with_stored_period_still_returns_nan():
    frame = pd.DataFrame(
        {"x": [3.5]},
        index=pd.DatetimeIndex(["2026-01-01 00:00"]),
    )
    frame.attrs["lag_sample_period_ns"] = 5 * MINUTE_NS

    points, interval = resolve_diff_interval(frame, None)
    assert points == 1
    assert interval == 5.0

    result = difference_by_physical_interval(frame, None)

    assert np.isnan(result["x"].iloc[0])
    assert sample_period_ns(result) == 5 * MINUTE_NS


def test_all_missing_column_stays_missing_without_affecting_other_columns():
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [np.nan, np.nan, np.nan, np.nan],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="1min"),
    )

    result = difference_by_physical_interval(frame, 1.0)
    single_a = difference_by_physical_interval(frame[["a"]], 1.0)

    assert result["b"].isna().all()
    assert len(result) == len(frame)
    pd.testing.assert_series_equal(result["a"], single_a["a"], check_names=False)
