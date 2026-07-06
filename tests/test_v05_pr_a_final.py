import inspect

import pandas as pd
import pytest

from chem_ts_corr import screening
from chem_ts_corr.preprocess import preprocess_frame


def test_screening_zero_and_negative_best_lag_use_zero_only():
    assert screening._nearby_lags(-3, 12) == [0]
    assert screening._nearby_lags(0, 12) == [0]
    source = inspect.getsource(screening._nearby_lags)
    assert "abs(best_lag)" not in source


def test_preprocess_requires_at_least_ten_rows_after_dropna():
    index = pd.date_range("2026-01-01", periods=9, freq="min")
    frame = pd.DataFrame(
        {
            "target": list(range(9)),
            "x": list(range(10, 19)),
        },
        index=index,
    )
    with pytest.raises(ValueError, match="at least 10"):
        preprocess_frame(frame, target="target", resample_rule=None, min_valid_ratio=0.0)


def test_preprocess_allows_exactly_ten_rows_after_dropna():
    index = pd.date_range("2026-01-01", periods=10, freq="min")
    frame = pd.DataFrame(
        {
            "target": list(range(10)),
            "x": list(range(10, 20)),
        },
        index=index,
    )
    cleaned = preprocess_frame(frame, target="target", resample_rule=None, min_valid_ratio=0.0)
    assert len(cleaned) == 10
    assert cleaned.attrs.get("rows_before_dropna") == 10
    assert cleaned.attrs.get("rows_after_dropna") == 10
    assert cleaned.attrs.get("rows_dropped_by_dropna") == 0


def test_preprocess_source_does_not_lower_row_guard_to_three():
    source = inspect.getsource(preprocess_frame)
    assert "at least 10" in source
    assert "< 3" not in source
    assert "at least 3" not in source
