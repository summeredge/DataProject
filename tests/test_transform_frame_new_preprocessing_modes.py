from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.preprocess import (
    detrend_moving_average,
    difference_by_physical_interval,
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


@pytest.mark.parametrize("tau", [2.0, 10.0])
def test_lowpass_mode_is_equivalent_to_lowpass_filter_frame(tau: float):
    frame = _regular_frame(20)

    result = transform_frame(
        frame,
        "lowpass",
        24,
        lowpass_tau_minutes=tau,
    )
    expected = lowpass_filter_frame(frame, tau_minutes=tau)

    pd.testing.assert_frame_equal(result, expected)


def test_lowpass_tau_parameter_actually_changes_the_output():
    frame = _regular_frame(20)

    result_fast = transform_frame(frame, "lowpass", 24, lowpass_tau_minutes=2.0)
    result_slow = transform_frame(frame, "lowpass", 24, lowpass_tau_minutes=10.0)

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


def test_lowpass_detrend_mode_locks_lowpass_then_detrend_order():
    rows = 40
    values = np.concatenate([np.zeros(10), np.ones(30)]) + np.arange(rows) * 0.1
    frame = pd.DataFrame(
        {"x": values.astype(float)},
        index=pd.date_range("2026-01-01", periods=rows, freq="1min"),
    )
    tau = 5.0

    result = transform_frame(
        frame,
        "lowpass_detrend",
        6,
        lowpass_tau_minutes=tau,
    )
    expected = detrend_moving_average(
        lowpass_filter_frame(frame, tau_minutes=tau),
        6,
        5,
        "inside",
    )
    reversed_order = lowpass_filter_frame(
        detrend_moving_average(frame, 6, 5, "inside"),
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


def test_lowpass_diff_mode_is_equivalent_to_lowpass_then_interval_diff():
    frame = _regular_frame(12)
    tau = 2.0
    interval = 5.0

    result = transform_frame(
        frame,
        "lowpass_diff",
        24,
        lowpass_tau_minutes=tau,
        diff_interval_minutes=interval,
    )
    expected = difference_by_physical_interval(
        lowpass_filter_frame(frame, tau_minutes=tau),
        diff_interval_minutes=interval,
    ).dropna()

    assert result.index.equals(expected.index)
    assert result.columns.tolist() == expected.columns.tolist()
    pd.testing.assert_frame_equal(result, expected)
    assert sample_period_ns(result) == sample_period_ns(expected) == MINUTE_NS
    assert result.attrs == expected.attrs


def test_lowpass_diff_none_interval_uses_one_analysis_sampling_period():
    frame = _regular_frame(10)

    result = transform_frame(
        frame,
        "lowpass_diff",
        24,
        diff_interval_minutes=None,
    )
    expected = difference_by_physical_interval(
        lowpass_filter_frame(frame, tau_minutes=5.0),
        diff_interval_minutes=None,
    ).dropna()
    smoothed = lowpass_filter_frame(frame, tau_minutes=5.0)

    pd.testing.assert_frame_equal(result, expected)
    # One analysis sampling period on a 1-minute axis: the first surviving row
    # is the one-point difference, never interpreted as 0.0.
    assert result.index[0] == frame.index[1]
    assert result["x"].iloc[0] == pytest.approx(
        smoothed["x"].iloc[1] - smoothed["x"].iloc[0],
        abs=1e-12,
    )


def test_lowpass_diff_five_minute_interval_uses_five_point_difference():
    frame = _regular_frame(12, "1min")
    smoothed = lowpass_filter_frame(frame, tau_minutes=5.0)

    result = transform_frame(
        frame,
        "lowpass_diff",
        24,
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=5.0,
    )

    expected = difference_by_physical_interval(
        smoothed,
        diff_interval_minutes=5.0,
    ).dropna()
    pd.testing.assert_frame_equal(result, expected)
    # 5 points, not a fixed 1-point difference: the first row compares t5 - t0.
    assert result.index[0] == frame.index[5]
    assert result["x"].iloc[0] == pytest.approx(
        smoothed["x"].iloc[5] - smoothed["x"].iloc[0],
        abs=1e-12,
    )


def test_lowpass_diff_multi_point_does_not_cross_physical_gap():
    complete = pd.date_range("2026-01-01", periods=9, freq="1min")
    index = complete.drop([complete[2]])
    frame = pd.DataFrame(
        {"x": np.arange(8, dtype=float)},
        index=index,
    )

    result = transform_frame(
        frame,
        "lowpass_diff",
        24,
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=2.0,
    )
    smoothed = lowpass_filter_frame(frame, tau_minutes=5.0)
    expected = difference_by_physical_interval(
        smoothed,
        diff_interval_minutes=2.0,
    ).dropna()

    pd.testing.assert_frame_equal(result, expected)
    # The gap resets the filter, and the first two positions of the new segment
    # stay missing so the multi-point difference never borrows pre-gap history.
    assert complete[3] not in result.index
    assert complete[4] not in result.index
    assert complete[5] in result.index
    assert result.loc[complete[5], "x"] == pytest.approx(
        smoothed.loc[complete[5], "x"] - smoothed.loc[complete[3], "x"],
        abs=1e-12,
    )


def test_lowpass_diff_irregular_contiguous_axis_does_not_split_segments():
    frame = pd.DataFrame(
        {"x": np.arange(7, dtype=float)},
        index=_timestamps_from_intervals([4, 2, 4, 5, 6, 4]),
    )
    assert sample_period_ns(frame) == 4 * MINUTE_NS
    smoothed = lowpass_filter_frame(frame, tau_minutes=5.0)

    result = transform_frame(
        frame,
        "lowpass_diff",
        24,
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=8.0,
    )
    expected = difference_by_physical_interval(
        smoothed,
        diff_interval_minutes=8.0,
    ).dropna()

    pd.testing.assert_frame_equal(result, expected)
    # A 2-point difference on the 4-minute nominal period: only the first two
    # positions are dropped, so every 2/4/5/6-minute interval stays contiguous.
    assert len(result) == len(smoothed) - 2
    assert result["x"].iloc[0] == pytest.approx(
        smoothed["x"].iloc[2] - smoothed["x"].iloc[0],
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "mode",
    ["lowpass", "lowpass_detrend", "lowpass_diff"],
)
@pytest.mark.parametrize(
    "bad_tau",
    [0, -1.0, -0.001, float("nan"), float("inf"), -float("inf"), "abc"],
)
def test_invalid_lowpass_tau_minutes_fails_in_all_lowpass_modes(
    mode: str,
    bad_tau: object,
):
    frame = _regular_frame(10)

    with pytest.raises(ValueError, match="tau_minutes"):
        transform_frame(frame, mode, 24, lowpass_tau_minutes=bad_tau)


@pytest.mark.parametrize(
    "bad_interval",
    [0, -1.0, -0.001, float("nan"), float("inf"), -float("inf"), "abc"],
)
def test_invalid_diff_interval_minutes_fails_in_lowpass_diff(bad_interval: object):
    frame = _regular_frame(10)

    with pytest.raises(ValueError, match="diff_interval_minutes"):
        transform_frame(
            frame,
            "lowpass_diff",
            24,
            diff_interval_minutes=bad_interval,
        )


@pytest.mark.parametrize(
    "mode",
    ["lowpass", "lowpass_detrend", "lowpass_diff"],
)
def test_causal_entry_still_rejects_lowpass_modes(mode: str):
    frame = _regular_frame(10)

    with pytest.raises(ValueError, match="not implemented"):
        transform_frame_causal(frame, mode, 24)
