from __future__ import annotations

import pandas as pd
import pytest

from chem_ts_corr.data import apply_exclude_windows, exclude_window_stats


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"target": range(6), "predictor": range(10, 16)},
        index=pd.date_range("2026-08-14 08:00", periods=6, freq="15min", name="time"),
    )


def _window(start: str, end: str) -> dict[str, str]:
    return {"start": start, "end": end}


def test_empty_exclude_windows_leave_data_and_input_unchanged():
    frame = _frame()
    original = frame.copy(deep=True)

    result = apply_exclude_windows(frame, [])
    stats = exclude_window_stats(frame, [])

    pd.testing.assert_frame_equal(result, original)
    pd.testing.assert_frame_equal(frame, original)
    assert result is not frame
    assert stats == {
        "original_rows": 6,
        "excluded_rows": 0,
        "remaining_rows": 6,
        "excluded_ratio": 0.0,
        "exclude_window_count": 0,
    }


def test_single_window_excludes_inclusive_boundaries():
    frame = _frame()

    result = apply_exclude_windows(frame, [_window("2026-08-14T08:15:00", "2026-08-14T08:45:00")])

    assert result.index.tolist() == [
        pd.Timestamp("2026-08-14 08:00"),
        pd.Timestamp("2026-08-14 09:00"),
        pd.Timestamp("2026-08-14 09:15"),
    ]
    assert result.columns.tolist() == frame.columns.tolist()


def test_multiple_and_overlapping_windows_exclude_each_row_once():
    frame = _frame()
    windows = [
        _window("2026-08-14T08:15:00", "2026-08-14T08:30:00"),
        _window("2026-08-14T08:30:00", "2026-08-14T08:45:00"),
        _window("2026-08-14T09:15:00", "2026-08-14T09:15:00"),
    ]

    result = apply_exclude_windows(frame, windows)
    stats = exclude_window_stats(frame, windows)

    assert result.index.tolist() == [
        pd.Timestamp("2026-08-14 08:00"),
        pd.Timestamp("2026-08-14 09:00"),
    ]
    assert stats == {
        "original_rows": 6,
        "excluded_rows": 4,
        "remaining_rows": 2,
        "excluded_ratio": 4 / 6,
        "exclude_window_count": 3,
    }


@pytest.mark.parametrize(
    "window, expected_rows",
    [
        (_window("2026-08-14T07:00:00", "2026-08-14T08:00:00"), 5),
        (_window("2026-08-14T09:15:00", "2026-08-14T10:00:00"), 5),
        (_window("2026-08-14T07:00:00", "2026-08-14T08:15:00"), 4),
        (_window("2026-08-14T09:00:00", "2026-08-14T10:00:00"), 4),
        (_window("2026-08-14T06:00:00", "2026-08-14T07:00:00"), 6),
        (_window("2026-08-14T10:00:00", "2026-08-14T11:00:00"), 6),
    ],
)
def test_windows_at_or_outside_data_boundaries_are_valid(window, expected_rows):
    result = apply_exclude_windows(_frame(), [window])

    assert len(result) == expected_rows


@pytest.mark.parametrize(
    "windows",
    [
        [_window("2026-08-14T09:00:00", "2026-08-14T08:00:00")],
        [_window("not-a-time", "2026-08-14T08:00:00")],
        [{"end": "2026-08-14T08:00:00"}],
        [{"start": "2026-08-14T08:00:00"}],
        [{"start": "2026-08-14T08:00:00", "end": "2026-08-14T08:15:00", "reason": "x"}],
    ],
)
def test_invalid_exclude_windows_fail_clearly(windows):
    with pytest.raises(ValueError, match="exclude_windows"):
        apply_exclude_windows(_frame(), windows)


def test_clearing_windows_restores_the_complete_source_data():
    frame = _frame()
    original = frame.copy(deep=True)

    excluded = apply_exclude_windows(
        frame, [_window("2026-08-14T08:00:00", "2026-08-14T08:45:00")]
    )
    restored = apply_exclude_windows(frame, [])

    assert len(excluded) == 2
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(restored, frame)


def test_empty_frame_stats_do_not_divide_by_zero():
    frame = _frame().iloc[0:0]

    assert exclude_window_stats(frame, []) == {
        "original_rows": 0,
        "excluded_rows": 0,
        "remaining_rows": 0,
        "excluded_ratio": 0.0,
        "exclude_window_count": 0,
    }
