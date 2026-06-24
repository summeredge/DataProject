import numpy as np
import pandas as pd

from chem_ts_corr.conditional_granger import OUT_COLS, run_conditional_granger_tests


def test_conditional_granger_records_explicit_baseline_maxlag():
    n = 160
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(111)
    x = rng.normal(size=n)
    target = np.zeros(n)
    for t in range(8, n):
        target[t] = 0.4 * target[t - 1] + 0.5 * x[t - 8] + 0.01 * rng.normal()
    frame = pd.DataFrame({"target": target, "x": x}, index=idx)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=10,
        min_rows=40,
        candidate_lags={"x": [8]},
        baseline_maxlag=3,
        lag_mode="ranked_window",
        lag_window=5,
        fallback_maxlag=24,
    )

    row = out.iloc[0]
    assert row["baseline_maxlag"] == 3
    assert row["tested_lags"] == "8"
    assert int(row["best_lag"]) == 8
    assert row["lag_mode"] == "ranked_window"
    assert row["lag_window"] == 5
    assert row["fallback_maxlag"] == 24


def test_conditional_granger_defaults_baseline_maxlag_to_maxlag():
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    frame = pd.DataFrame({"target": np.arange(n, dtype=float), "x": np.arange(n, dtype=float)}, index=idx)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=7,
        min_rows=40,
        candidate_lags={"x": [2]},
    )

    assert out.iloc[0]["baseline_maxlag"] == 7
    assert out.iloc[0]["tested_lags"] == "2"


def test_conditional_granger_records_full_scan_tested_lags():
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(222)
    x = rng.normal(size=n)
    target = np.zeros(n)
    for t in range(2, n):
        target[t] = 0.3 * target[t - 1] + 0.2 * x[t - 1] + 0.01 * rng.normal()
    frame = pd.DataFrame({"target": target, "x": x}, index=idx)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=4,
        min_rows=40,
        lag_mode="full_scan",
    )

    row = out.iloc[0]
    assert row["tested_lags"] == "1,2,3,4"
    assert row["lag_mode"] == "full_scan"
    assert row["baseline_maxlag"] == 4


def test_conditional_granger_handles_empty_candidate_lags():
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    frame = pd.DataFrame({"target": np.arange(n, dtype=float), "x": np.arange(n, dtype=float)}, index=idx)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=7,
        min_rows=40,
        candidate_lags={"x": []},
    )

    row = out.iloc[0]
    assert row["status"] == "skipped: no candidate lags"
    assert row["tested_lags"] == ""
    assert pd.isna(row["best_lag"])
    assert list(out.columns) == OUT_COLS
