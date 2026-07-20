from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from chem_ts_corr import lag
from scripts.benchmark_lag_scores import _legacy_compute_lag_scores


def _comparison_frame() -> pd.DataFrame:
    target = np.array([0, 1, 1, 2, 4, 4, 3, 2, 5, 7, 7, 6, 8, 9, 9], dtype=float)
    x = np.roll(target, -2)
    x[-2:] = np.nan
    return pd.DataFrame(
        {"target": target, "x": x, "constant": 1.0},
        index=pd.date_range("2026-01-01", periods=len(target), freq="5min"),
    )


def test_optimized_alignment_matches_legacy_statistics_and_schema():
    frame = _comparison_frame()

    legacy = _legacy_compute_lag_scores(frame, "target", 3)
    optimized = lag.compute_lag_scores(frame, "target", 3)

    assert optimized.columns.tolist() == legacy.columns.tolist()
    assert optimized[["variable", "lag", "p_value_status"]].equals(
        legacy[["variable", "lag", "p_value_status"]]
    )
    numeric = [column for column in optimized.columns if column not in {
        "variable", "p_value_status", "lag_boundary_flag"
    }]
    np.testing.assert_allclose(
        optimized[numeric].to_numpy(dtype=float),
        legacy[numeric].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    assert optimized["lag_boundary_flag"].equals(legacy["lag_boundary_flag"])


def test_each_variable_lag_pair_aligns_once(monkeypatch):
    frame = _comparison_frame()
    original = lag._aligned_corr_stats
    calls = []

    def counted(x, y):
        calls.append((x.name, y.name))
        return original(x, y)

    monkeypatch.setattr(lag, "_aligned_corr_stats", counted)

    result = lag.compute_lag_scores(frame, "target", max_lag=3, lag_values=[-3, 0, 3])

    assert len(calls) == 2 * 3
    assert len(result) == 2 * 3


def test_compute_lag_scores_has_one_shared_alignment_call_site():
    source = inspect.getsource(lag.compute_lag_scores)

    assert source.count("_aligned_corr_stats(") == 1
    assert "_safe_corr_stats" not in source
