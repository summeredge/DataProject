import numpy as np
import pandas as pd

from chem_ts_corr.conditional_granger import run_conditional_granger_tests


EXPECTED_OUT_COLS = [
    "variable",
    "status",
    "best_lag",
    "min_p_value",
    "fdr_q_value",
    "baseline_rmse",
    "full_rmse",
    "predictive_contribution",
    "control_columns",
    "n_rows",
    "interpretation",
]


def test_conditional_granger_detects_predictive_candidate():
    n = 220
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] + 0.35 * x[t - 1] + 0.05 * rng.normal()
    frame = pd.DataFrame({"target": y, "x": x}, index=idx)

    out = run_conditional_granger_tests(frame, target="target", variables=["x"], maxlag=4, min_rows=80)
    row = out.iloc[0]
    assert str(row["status"]).startswith("ok")
    assert float(row["predictive_contribution"]) >= 0
    assert 1 <= int(row["best_lag"]) <= 4
    assert "not a causal conclusion" in str(row["interpretation"])


def test_conditional_granger_handles_missing_variable():
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    frame = pd.DataFrame({"target": np.arange(n, dtype=float)}, index=idx)
    out = run_conditional_granger_tests(frame, target="target", variables=["missing"], maxlag=3, min_rows=40)
    assert out.iloc[0]["status"] == "skipped: variable not found"


def test_conditional_granger_keeps_output_columns_when_insufficient_rows():
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    frame = pd.DataFrame({"target": np.arange(n, dtype=float), "x": np.arange(n, dtype=float)}, index=idx)
    out = run_conditional_granger_tests(frame, target="target", variables=["x"], maxlag=8, min_rows=60)
    assert out.iloc[0]["status"] == "skipped: insufficient rows"
    assert list(out.columns) == EXPECTED_OUT_COLS


def test_conditional_granger_with_control_columns():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(7)
    c = rng.normal(size=n)
    x = 0.8 * c + 0.2 * rng.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] + 0.3 * x[t - 1] + 0.25 * c[t - 1] + 0.05 * rng.normal()
    frame = pd.DataFrame({"target": y, "x": x, "c": c}, index=idx)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        control_columns=["c", "missing_control"],
        maxlag=4,
        min_rows=80,
    )
    row = out.iloc[0]
    assert "c" in str(row["control_columns"])
    assert str(row["status"]).startswith("ok") or str(row["status"]).startswith("skipped")


def test_conditional_granger_controls_common_driver_effect():
    n = 320
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(123)
    c = rng.normal(size=n)
    x = c + 0.05 * rng.normal(size=n)
    target = np.zeros(n)
    for t in range(2, n):
        target[t] = 0.45 * target[t - 1] + 0.75 * c[t - 1] + 0.05 * rng.normal()
    frame = pd.DataFrame({"target": target, "x": x, "c": c}, index=idx)

    uncontrolled = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=3,
        min_rows=100,
    ).iloc[0]
    controlled = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        control_columns=["c"],
        maxlag=3,
        min_rows=100,
    ).iloc[0]

    uncontrolled_contribution = float(uncontrolled["predictive_contribution"])
    controlled_contribution = float(controlled["predictive_contribution"])

    assert str(uncontrolled["status"]).startswith("ok")
    assert str(controlled["status"]).startswith("ok")
    assert uncontrolled_contribution > 0.5
    assert controlled_contribution < uncontrolled_contribution
    assert controlled_contribution < uncontrolled_contribution * 0.05
    assert controlled_contribution < 0.01
    assert controlled["interpretation"] == "predictive validation only; not a causal conclusion"
    assert list(controlled.index) == EXPECTED_OUT_COLS
