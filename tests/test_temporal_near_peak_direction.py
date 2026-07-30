from __future__ import annotations

import pandas as pd
import pytest

from chem_ts_corr.lag import build_lag_peak_quality


def _quality(scores: dict[int, float], max_lag: int = 12) -> pd.Series:
    rows = [
        {"variable": "x", "lag": lag, "abs_pearson": score, "abs_spearman": score}
        for lag, score in scores.items()
    ]
    return build_lag_peak_quality(pd.DataFrame(rows), max_lag=max_lag).iloc[0]


@pytest.mark.parametrize(
    ("scores", "expected_min", "expected_max", "expected_status"),
    [
        ({5: 0.90, 12: 0.86, 0: 0.20}, 5, 12, "variable_leads_supported"),
        ({-12: 0.90, -3: 0.86, 0: 0.20}, -12, -3, "target_leads_supported"),
        ({0: 0.90, 3: 0.20}, 0, 0, "synchronous"),
        ({-1: 0.90, 5: 0.86, 8: 0.20}, -1, 5, "direction_unresolved"),
        ({-8: 0.90, 2: 0.86, 5: 0.20}, -8, 2, "direction_unresolved"),
        ({-1: 0.90, -5: 0.20}, -1, -1, "direction_unresolved"),
        ({1: 0.90, 5: 0.20}, 1, 1, "direction_unresolved"),
    ],
)
def test_near_peak_interval_owns_temporal_direction(
    scores: dict[int, float], expected_min: int, expected_max: int, expected_status: str,
):
    row = _quality(scores)

    assert row["near_peak_lag_min"] == expected_min
    assert row["near_peak_lag_max"] == expected_max
    assert row["temporal_direction_status"] == expected_status


def test_boundary_reduces_quality_proportionally_without_erasing_direction():
    row = _quality({-12: 0.90, -5: 0.86, 0: 0.20})

    assert bool(row["lag_boundary_flag"])
    assert row["lag_quality"] == pytest.approx(row["shape_quality"] * 0.75)
    assert row["temporal_direction_status"] == "target_leads_supported"

