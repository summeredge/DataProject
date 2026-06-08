import pandas as pd

from chem_ts_corr.modeling import build_lag_features


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": range(20),
            "x1": range(20, 40),
            "x2": range(40, 60),
        }
    )


def test_build_lag_features_best_only_uses_one_lag_per_variable():
    features, _ = build_lag_features(
        _frame(),
        target="target",
        max_lag=6,
        candidate_variables=["x1", "x2"],
        max_features=10,
        best_lags={"x1": 2, "x2": -3},
        lag_mode="best_only",
    )

    assert list(features.columns) == ["x1__lag_2", "x2__lag_3"]


def test_build_lag_features_nearby_keeps_existing_lag_window_behavior():
    features, _ = build_lag_features(
        _frame(),
        target="target",
        max_lag=5,
        candidate_variables=["x1"],
        max_features=10,
        best_lags={"x1": 3},
        lag_mode="nearby",
    )

    assert list(features.columns) == [
        "x1__lag_1",
        "x1__lag_2",
        "x1__lag_3",
        "x1__lag_4",
        "x1__lag_5",
    ]
