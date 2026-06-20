import numpy as np
import pandas as pd

from chem_ts_corr.conditional_granger import build_candidate_lag_windows, run_conditional_granger_tests


EXPECTED_OUT_COLS = [
    "variable",
    "status",
    "best_lag",
    "min_p_value",
    "fdr_q_value",
    "baseline_rmse",
    "full_rmse",
    "predictive_contribution",
    "base_condition_number",
    "full_condition_number",
    "condition_number",
    "control_columns",
    "n_rows",
    "tested_lags",
    "lag_mode",
    "lag_window",
    "fallback_maxlag",
    "baseline_maxlag",
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
    assert np.isfinite(float(row["condition_number"]))
    assert "base_condition_number" in out.columns
    assert "full_condition_number" in out.columns
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


def test_conditional_granger_flags_high_collinearity_with_controls():
    n = 180
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(321)
    c = rng.normal(size=n)
    x = c.copy()
    target = np.zeros(n)
    for t in range(2, n):
        target[t] = 0.55 * target[t - 1] + 0.5 * c[t - 1] + 0.02 * rng.normal()
    frame = pd.DataFrame({"target": target, "x": x, "c": c}, index=idx)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        control_columns=["c"],
        maxlag=3,
        min_rows=80,
    )
    row = out.iloc[0]

    assert row["status"] == "high_collinearity_risk"
    assert pd.isna(row["min_p_value"])
    assert pd.isna(row["fdr_q_value"])
    assert np.isfinite(float(row["predictive_contribution"]))
    assert float(row["predictive_contribution"]) >= 0
    assert (not np.isfinite(float(row["condition_number"]))) or float(row["condition_number"]) > 1e8
    assert row["interpretation"] == "predictive validation only; not a causal conclusion"


def test_conditional_granger_non_collinear_sample_keeps_ok_status_and_condition_numbers():
    n = 220
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(654)
    x = rng.normal(size=n)
    c = rng.normal(size=n)
    target = np.zeros(n)
    for t in range(2, n):
        target[t] = 0.45 * target[t - 1] + 0.3 * x[t - 1] + 0.2 * c[t - 1] + 0.05 * rng.normal()
    frame = pd.DataFrame({"target": target, "x": x, "c": c}, index=idx)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        control_columns=["c"],
        maxlag=3,
        min_rows=80,
    )
    row = out.iloc[0]

    assert str(row["status"]).startswith("ok")
    assert "condition_number" in out.columns
    assert np.isfinite(float(row["base_condition_number"]))
    assert np.isfinite(float(row["full_condition_number"]))
    assert float(row["condition_number"]) <= 1e8


def test_build_candidate_lag_windows_centers_long_ranked_lag():
    ranked = pd.DataFrame([{"variable": "x", "lag": 80}])

    out = build_candidate_lag_windows(ranked, ["x"], maxlag=100, window=5, fallback_maxlag=24)

    assert out["x"] == list(range(75, 86))


def test_build_candidate_lag_windows_clips_at_maxlag():
    ranked = pd.DataFrame([{"variable": "x", "lag": 58}])

    out = build_candidate_lag_windows(ranked, ["x"], maxlag=60, window=5, fallback_maxlag=24)

    assert out["x"] == list(range(53, 61))


def test_build_candidate_lag_windows_clips_at_one():
    ranked = pd.DataFrame([{"variable": "x", "lag": 3}])

    out = build_candidate_lag_windows(ranked, ["x"], maxlag=100, window=5, fallback_maxlag=24)

    assert out["x"] == list(range(1, 9))


def test_build_candidate_lag_windows_uses_fallback_for_missing_lag():
    ranked = pd.DataFrame([{"variable": "x", "lag": np.nan}])

    out = build_candidate_lag_windows(ranked, ["x", "missing"], maxlag=10, window=5, fallback_maxlag=6)

    assert out["x"] == list(range(1, 7))
    assert out["missing"] == list(range(1, 7))


def test_conditional_granger_respects_candidate_lags_and_keeps_columns():
    n = 180
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(987)
    x = rng.normal(size=n)
    target = np.zeros(n)
    for t in range(10, n):
        target[t] = 0.4 * target[t - 1] + 0.5 * x[t - 10] + 0.01 * rng.normal()
    frame = pd.DataFrame({"target": target, "x": x}, index=idx)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=12,
        min_rows=80,
        candidate_lags={"x": [3, 3, 0, 99]},
    )

    assert int(out.iloc[0]["best_lag"]) == 3
    assert list(out.columns) == EXPECTED_OUT_COLS


def test_conditional_granger_records_explicit_baseline_maxlag():
    n = 180
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(111)
    x = rng.normal(size=n)
    target = np.zeros(n)
    for t in range(8, n):
        target[t] = 0.35 * target[t - 1] + 0.4 * x[t - 8] + 0.02 * rng.normal()
    frame = pd.DataFrame({"target": target, "x": x}, index=idx)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=10,
        min_rows=80,
        candidate_lags={"x": [8]},
        baseline_maxlag=3,
    )

    row = out.iloc[0]
    assert int(row["baseline_maxlag"]) == 3
    assert row["tested_lags"] == "8"
    assert int(row["best_lag"]) == 8


def test_conditional_granger_defaults_baseline_maxlag_to_maxlag():
    n = 160
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(222)
    x = rng.normal(size=n)
    target = np.zeros(n)
    for t in range(2, n):
        target[t] = 0.4 * target[t - 1] + 0.3 * x[t - 1] + 0.02 * rng.normal()
    frame = pd.DataFrame({"target": target, "x": x}, index=idx)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=7,
        min_rows=80,
        baseline_maxlag=None,
    )

    assert int(out.iloc[0]["baseline_maxlag"]) == 7
    assert out.iloc[0]["tested_lags"] == "1,2,3,4,5,6,7"


def test_conditional_granger_skips_empty_candidate_lags_and_records_tested_lags():
    n = 160
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    frame = pd.DataFrame({"target": np.arange(n, dtype=float), "x": np.arange(n, dtype=float)}, index=idx)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=10,
        min_rows=80,
        candidate_lags={"x": []},
        baseline_maxlag=3,
    )

    row = out.iloc[0]
    assert row["status"] == "skipped: no candidate lags"
    assert row["tested_lags"] == ""
    assert int(row["baseline_maxlag"]) == 3


def test_build_candidate_lag_windows_skips_negative_lag_without_abs_center():
    ranked = pd.DataFrame([{"variable": "x", "lag": -12}])

    out = build_candidate_lag_windows(ranked, ["x"], maxlag=20, window=2, fallback_maxlag=5)

    assert out["x"] == []


def test_build_candidate_lag_windows_skips_zero_lag():
    ranked = pd.DataFrame([{"variable": "x", "lag": 0}])

    out = build_candidate_lag_windows(ranked, ["x"], maxlag=20, window=2, fallback_maxlag=4)

    assert out["x"] == []


def test_build_candidate_lag_windows_keeps_positive_lag_center():
    ranked = pd.DataFrame([{"variable": "x", "lag": 8}])

    out = build_candidate_lag_windows(ranked, ["x"], maxlag=20, window=2, fallback_maxlag=4)

    assert out["x"] == [6, 7, 8, 9, 10]


def test_conditional_granger_excludes_current_candidate_from_controls():
    n = 180
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(456)
    x1 = rng.normal(size=n)
    load = rng.normal(size=n)
    target = np.zeros(n)
    for t in range(2, n):
        target[t] = 0.45 * target[t - 1] + 0.35 * x1[t - 1] + 0.2 * load[t - 1] + 0.02 * rng.normal()
    frame = pd.DataFrame({"target": target, "x1": x1, "load": load}, index=idx)

    overlap = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x1"],
        control_columns=["x1"],
        maxlag=3,
        min_rows=80,
    ).iloc[0]
    with_load = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x1"],
        control_columns=["load"],
        maxlag=3,
        min_rows=80,
    ).iloc[0]

    assert overlap["control_columns"] == ""
    assert str(overlap["status"]).startswith("ok")
    assert with_load["control_columns"] == "load"


def test_conditional_granger_ranked_window_marks_negative_lag_skipped():
    frame = pd.DataFrame({"target": np.arange(120, dtype=float), "x": np.arange(120, dtype=float)})

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=10,
        min_rows=60,
        candidate_lags={"x": []},
        lag_mode="ranked_window",
    )

    row = out.iloc[0]
    assert row["status"] == "skipped: non-positive screening lag"
    assert row["tested_lags"] == ""


def test_conditional_granger_full_scan_still_scans_all_lags():
    frame = pd.DataFrame({"target": np.arange(120, dtype=float), "x": np.arange(120, dtype=float)})

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=4,
        min_rows=60,
        candidate_lags=None,
        lag_mode="full_scan",
    )

    assert out.iloc[0]["tested_lags"] == "1,2,3,4"


def test_conditional_granger_normalizes_control_columns_per_candidate():
    n = 180
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(789)
    frame = pd.DataFrame(
        {
            "target": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "load": rng.normal(size=n),
        },
        index=idx,
    )

    duplicate_controls = run_conditional_granger_tests(
        frame, target="target", variables=["x1"], control_columns=["load", "load", "x1"], maxlag=3, min_rows=80
    ).iloc[0]
    only_self = run_conditional_granger_tests(
        frame, target="target", variables=["x1"], control_columns=["x1"], maxlag=3, min_rows=80
    ).iloc[0]
    missing_control = run_conditional_granger_tests(
        frame, target="target", variables=["x1"], control_columns=["missing", "load"], maxlag=3, min_rows=80
    ).iloc[0]

    assert duplicate_controls["control_columns"] == "load"
    assert only_self["control_columns"] == ""
    assert missing_control["control_columns"] == "load"


def test_conditional_granger_ranked_window_marks_positive_lag_outside_maxlag():
    frame = pd.DataFrame({"target": np.arange(140, dtype=float), "x": np.arange(140, dtype=float)})
    ranked = pd.DataFrame([{"variable": "x", "lag": 120}])
    candidate_lags = build_candidate_lag_windows(ranked, ["x"], maxlag=24, window=2, fallback_maxlag=6)

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=24,
        min_rows=60,
        candidate_lags=candidate_lags,
        candidate_lag_status={"x": "ranked_lag_outside_maxlag"},
        lag_mode="ranked_window",
    )

    assert out.iloc[0]["status"] == "skipped: ranked lag outside maxlag"


def test_conditional_granger_ranked_window_marks_zero_lag_skipped():
    frame = pd.DataFrame({"target": np.arange(120, dtype=float), "x": np.arange(120, dtype=float)})

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=10,
        min_rows=60,
        candidate_lags={"x": []},
        candidate_lag_status={"x": "non_positive_screening_lag"},
        lag_mode="ranked_window",
    )

    assert out.iloc[0]["status"] == "skipped: non-positive screening lag"


def test_conditional_granger_internal_columns_do_not_collide_with_x_y_controls():
    n = 160
    rng = np.random.default_rng(987)
    frame = pd.DataFrame(
        {
            "target": rng.normal(size=n),
            "candidate": rng.normal(size=n),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )

    out = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["candidate"],
        control_columns=["x", "y"],
        maxlag=3,
        min_rows=80,
    )

    assert out.iloc[0]["tested_lags"] == "1,2,3"
    assert out.iloc[0]["control_columns"] == "x,y"
