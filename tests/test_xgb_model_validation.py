from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chem_ts_corr.xgb_validation as xgb_module
from chem_ts_corr.xgb_validation import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_XGB_PARAMS,
    XGBFeatureSets,
    XGBFoldMetric,
    XGBTimeSplit,
    XGBValidationResult,
    run_xgb_time_validation,
    train_xgb_fold,
)


class RecordingRegressor:
    instances: list["RecordingRegressor"] = []
    offsets = {1: 3.0, 2: 2.0, 3: 1.0}

    def __init__(self, **params: object):
        self.params = params
        self.fit_calls: list[dict[str, object]] = []
        self.predict_calls: list[pd.DataFrame] = []
        self.best_iteration: int | None = None
        type(self).instances.append(self)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: list[tuple[pd.DataFrame, pd.Series]],
        verbose: bool,
    ) -> "RecordingRegressor":
        self.fit_calls.append(
            {
                "X": X.copy(deep=True),
                "y": y.copy(deep=True),
                "eval_set": [(a.copy(deep=True), b.copy(deep=True)) for a, b in eval_set],
                "verbose": verbose,
            }
        )
        self.best_iteration = int(float(eval_set[0][1].sum())) % 11
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_calls.append(X.copy(deep=True))
        offset = self.offsets[len(X.columns)]
        return X.iloc[:, 0].to_numpy(dtype=float) + offset


@pytest.fixture(autouse=True)
def _reset_recording_regressor():
    RecordingRegressor.instances = []
    RecordingRegressor.offsets = {1: 3.0, 2: 2.0, 3: 1.0}


def _feature_sets(n_rows: int = 18) -> XGBFeatureSets:
    index = pd.Index([f"t{row:02d}" for row in range(n_rows)], name="timestamp")
    target = pd.Series(np.arange(n_rows, dtype=float) + 10.0, index=index, name="y")
    features = pd.DataFrame(
        {
            "y__lag_1": target.to_numpy(),
            "control__lag_1": target.to_numpy() * 0.5,
            "candidate__lag_2": target.to_numpy() * 2.0,
        },
        index=index,
    )
    return XGBFeatureSets(
        features=features,
        target=target,
        m0_features=("y__lag_1",),
        m1_features=("y__lag_1", "control__lag_1"),
        m2_features=("y__lag_1", "control__lag_1", "candidate__lag_2"),
        candidate_feature_map={"candidate": ("candidate__lag_2",)},
        max_used_lag=2,
    )


def _splits() -> list[XGBTimeSplit]:
    return [
        XGBTimeSplit(0, slice(0, 6), slice(6, 9), slice(9, 12), 0),
        XGBTimeSplit(1, slice(0, 9), slice(9, 12), slice(12, 18), 0),
    ]


def test_missing_xgboost_dependency_returns_install_instruction(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(xgb_module, "XGBRegressor", None)

    with pytest.raises(RuntimeError, match=r'xgboost is not installed[\s\S]*pip install -e "\.\[xgb\]"'):
        run_xgb_time_validation(_feature_sets(), _splits())


def test_train_fold_creates_xgb_with_fixed_defaults_and_early_stopping(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)
    data = _feature_sets()

    metric, _ = train_xgb_fold(
        data.features.iloc[:6, :1], data.target.iloc[:6],
        data.features.iloc[6:9, :1], data.target.iloc[6:9],
        data.features.iloc[9:12, :1], data.target.iloc[9:12],
        fold=2, model_name="M0",
    )

    model = RecordingRegressor.instances[0]
    assert model.params["tree_method"] == "hist"
    assert model.params["random_state"] == 42
    assert model.params["objective"] == "reg:squarederror"
    assert model.params["early_stopping_rounds"] == DEFAULT_EARLY_STOPPING_ROUNDS
    assert metric.best_iteration == int(data.target.iloc[6:9].sum()) % 11


def test_train_fold_uses_validation_only_as_eval_set_and_test_only_for_predict(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)
    data = _feature_sets()
    X_train, y_train = data.features.iloc[:6, :1], data.target.iloc[:6]
    X_valid, y_valid = data.features.iloc[6:9, :1], data.target.iloc[6:9]
    X_test, y_test = data.features.iloc[9:12, :1], data.target.iloc[9:12]

    train_xgb_fold(
        X_train, y_train, X_valid, y_valid, X_test, y_test, fold=0, model_name="M0"
    )

    model = RecordingRegressor.instances[0]
    call = model.fit_calls[0]
    pd.testing.assert_frame_equal(call["X"], X_train)
    pd.testing.assert_series_equal(call["y"], y_train)
    assert len(call["eval_set"]) == 1
    pd.testing.assert_frame_equal(call["eval_set"][0][0], X_valid)
    pd.testing.assert_series_equal(call["eval_set"][0][1], y_valid)
    assert call["verbose"] is False
    assert len(model.predict_calls) == 1
    pd.testing.assert_frame_equal(model.predict_calls[0], X_test)
    assert not model.fit_calls[0]["X"].index.isin(X_test.index).any()


def test_train_fold_metrics_use_test_truth_and_prediction(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)
    data = _feature_sets()

    metric, prediction = train_xgb_fold(
        data.features.iloc[:6, :1], data.target.iloc[:6],
        data.features.iloc[6:9, :1], data.target.iloc[6:9],
        data.features.iloc[9:12, :1], data.target.iloc[9:12],
        fold=0, model_name="M0",
    )

    assert isinstance(metric, XGBFoldMetric)
    assert prediction.tolist() == pytest.approx((data.target.iloc[9:12] + 3.0).tolist())
    assert metric.rmse == pytest.approx(3.0)
    assert metric.mae == pytest.approx(3.0)
    assert metric.r2 == pytest.approx(-12.5)
    assert (metric.train_rows, metric.validation_rows, metric.test_rows) == (6, 3, 3)


def test_fold_metric_rejects_unknown_model_name():
    with pytest.raises(ValueError, match="model_name"):
        XGBFoldMetric(0, "bad", 1, 1, 1, None, 1.0, 1.0, 0.0)


@pytest.mark.parametrize("model_name", ["bad", "m0", ""])
def test_train_fold_rejects_unknown_model_name(monkeypatch: pytest.MonkeyPatch, model_name: str):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)
    data = _feature_sets()

    with pytest.raises(ValueError, match="model_name"):
        train_xgb_fold(
            data.features.iloc[:6, :1], data.target.iloc[:6],
            data.features.iloc[6:9, :1], data.target.iloc[6:9],
            data.features.iloc[9:12, :1], data.target.iloc[9:12],
            fold=0, model_name=model_name,
        )


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_train_fold_rejects_empty_partitions(
    monkeypatch: pytest.MonkeyPatch, partition: str
):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)
    data = _feature_sets()
    parts = {
        "train": [data.features.iloc[:6, :1], data.target.iloc[:6]],
        "validation": [data.features.iloc[6:9, :1], data.target.iloc[6:9]],
        "test": [data.features.iloc[9:12, :1], data.target.iloc[9:12]],
    }
    parts[partition] = [data.features.iloc[0:0, :1], data.target.iloc[0:0]]

    with pytest.raises(ValueError, match=f"no {partition} rows"):
        train_xgb_fold(
            *parts["train"], *parts["validation"], *parts["test"], fold=0, model_name="M0"
        )


def test_run_trains_m0_m1_m2_for_every_fold_with_identical_test_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)

    result = run_xgb_time_validation(_feature_sets(), _splits())

    assert isinstance(result, XGBValidationResult)
    assert result.fold_metrics[["fold", "model_name"]].to_records(index=False).tolist() == [
        (0, "M0"), (0, "M1"), (0, "M2"), (1, "M0"), (1, "M1"), (1, "M2")
    ]
    assert result.fold_metrics.groupby("fold")["test_rows"].nunique().eq(1).all()
    assert [len(model.fit_calls[0]["X"].columns) for model in RecordingRegressor.instances] == [
        1, 2, 3, 1, 2, 3
    ]


def test_fold_metric_and_prediction_output_columns_are_fixed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)

    result = run_xgb_time_validation(_feature_sets(), _splits())

    assert result.fold_metrics.columns.tolist() == [
        "fold", "model_name", "train_rows", "validation_rows", "test_rows",
        "best_iteration", "rmse", "mae", "r2",
    ]
    assert result.predictions.columns.tolist() == [
        "fold", "timestamp_index", "y_true", "M0_prediction", "M1_prediction", "M2_prediction"
    ]
    assert result.summary.columns.tolist() == [
        "model_name", "mean_rmse", "median_rmse", "mean_mae", "median_mae", "mean_r2",
        "fold_count", "M2_vs_M1_rmse_improvement_pct", "M2_vs_M1_mae_improvement_pct",
    ]


def test_predictions_preserve_each_fold_test_index_for_all_models(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)
    data = _feature_sets()

    result = run_xgb_time_validation(data, _splits())

    assert result.predictions.loc[result.predictions["fold"].eq(0), "timestamp_index"].tolist() == (
        data.features.index[9:12].tolist()
    )
    assert result.predictions.loc[result.predictions["fold"].eq(1), "timestamp_index"].tolist() == (
        data.features.index[12:18].tolist()
    )
    assert result.predictions.filter(like="_prediction").notna().all().all()


def test_m2_better_than_m1_has_positive_improvement(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)

    summary = run_xgb_time_validation(_feature_sets(), _splits()).summary.set_index("model_name")

    assert summary.loc["M2", "M2_vs_M1_rmse_improvement_pct"] == pytest.approx(50.0)
    assert summary.loc["M2", "M2_vs_M1_mae_improvement_pct"] == pytest.approx(50.0)


def test_m2_worse_than_m1_has_negative_improvement(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)
    RecordingRegressor.offsets = {1: 3.0, 2: 2.0, 3: 4.0}

    summary = run_xgb_time_validation(_feature_sets(), _splits()).summary.set_index("model_name")

    assert summary.loc["M2", "M2_vs_M1_rmse_improvement_pct"] == pytest.approx(-100.0)
    assert summary.loc["M2", "M2_vs_M1_mae_improvement_pct"] == pytest.approx(-100.0)


def test_test_label_changes_do_not_change_fit_or_prediction_inputs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)
    data = _feature_sets()
    args = [
        data.features.iloc[:6, :1], data.target.iloc[:6],
        data.features.iloc[6:9, :1], data.target.iloc[6:9],
        data.features.iloc[9:12, :1], data.target.iloc[9:12],
    ]
    first_metric, first_prediction = train_xgb_fold(*args, fold=0, model_name="M0")
    changed_truth = data.target.iloc[9:12] + 1000
    second_metric, second_prediction = train_xgb_fold(
        *args[:5], changed_truth, fold=0, model_name="M0"
    )

    first, second = RecordingRegressor.instances
    pd.testing.assert_frame_equal(first.fit_calls[0]["X"], second.fit_calls[0]["X"])
    pd.testing.assert_series_equal(first.fit_calls[0]["y"], second.fit_calls[0]["y"])
    pd.testing.assert_series_equal(
        first.fit_calls[0]["eval_set"][0][1], second.fit_calls[0]["eval_set"][0][1]
    )
    pd.testing.assert_frame_equal(first.predict_calls[0], second.predict_calls[0])
    assert first_prediction.tolist() == second_prediction.tolist()
    assert first_metric.rmse != second_metric.rmse


def test_validation_label_changes_only_fit_eval_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)
    data = _feature_sets()
    common = [
        data.features.iloc[:6, :1], data.target.iloc[:6], data.features.iloc[6:9, :1]
    ]
    tail = [data.features.iloc[9:12, :1], data.target.iloc[9:12]]
    train_xgb_fold(*common, data.target.iloc[6:9], *tail, fold=0, model_name="M0")
    train_xgb_fold(*common, data.target.iloc[6:9] + 1, *tail, fold=0, model_name="M0")

    first, second = RecordingRegressor.instances
    pd.testing.assert_series_equal(first.fit_calls[0]["y"], second.fit_calls[0]["y"])
    assert not first.fit_calls[0]["eval_set"][0][1].equals(second.fit_calls[0]["eval_set"][0][1])
    pd.testing.assert_frame_equal(first.predict_calls[0], second.predict_calls[0])
    assert first.best_iteration != second.best_iteration


def test_repeated_run_is_deterministic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)

    first = run_xgb_time_validation(_feature_sets(), _splits())
    RecordingRegressor.instances = []
    second = run_xgb_time_validation(_feature_sets(), _splits())

    pd.testing.assert_frame_equal(first.fold_metrics, second.fold_metrics)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)


@pytest.mark.parametrize(
    ("feature_sets", "splits", "match"),
    [
        (_feature_sets(), [], "No time splits provided"),
        (
            XGBFeatureSets(pd.DataFrame(), pd.Series(dtype=float), (), (), (), {}, 0),
            _splits(),
            "No valid XGB features available",
        ),
    ],
)
def test_run_rejects_missing_splits_or_features(
    monkeypatch: pytest.MonkeyPatch,
    feature_sets: XGBFeatureSets,
    splits: list[XGBTimeSplit],
    match: str,
):
    monkeypatch.setattr(xgb_module, "XGBRegressor", RecordingRegressor)
    with pytest.raises(ValueError, match=match):
        run_xgb_time_validation(feature_sets, splits)


def test_run_does_not_rebuild_features_split_or_drop_rows():
    source = inspect.getsource(run_xgb_time_validation)

    assert "build_xgb_feature_sets" not in source
    assert "build_expanding_time_splits" not in source
    assert "dropna" not in source
    assert "shift(" not in source


@pytest.mark.parametrize(
    "forbidden",
    [
        "train_test_split", "KFold", "ShuffleSplit", "shuffle=True", "GridSearchCV",
        "RandomizedSearchCV", "Optuna", "fit_transform", "StandardScaler", "MinMaxScaler",
        "shift(-", "abs(best_lag)", "final_score", "driver_rank",
    ],
)
def test_xgb_model_source_keeps_architecture_and_leakage_guards(forbidden: str):
    source = Path("chem_ts_corr/xgb_validation.py").read_text(encoding="utf-8")
    assert forbidden not in source


def test_default_xgb_parameters_are_centralized_and_auditable():
    assert DEFAULT_XGB_PARAMS == {
        "objective": "reg:squarederror", "tree_method": "hist", "n_estimators": 1500,
        "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 20,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
        "reg_lambda": 10.0, "n_jobs": -1, "random_state": 42,
    }
    assert DEFAULT_EARLY_STOPPING_ROUNDS == 50


@pytest.mark.skipif(xgb_module.XGBRegressor is None, reason="xgboost optional dependency is absent")
def test_real_xgboost_small_fold_smoke():
    rows = 60
    values = np.arange(rows, dtype=float)
    X = pd.DataFrame({"signal": values, "secondary": values % 7})
    y = pd.Series(0.5 * values + np.sin(values / 3))

    metric, prediction = train_xgb_fold(
        X.iloc[:40], y.iloc[:40], X.iloc[40:50], y.iloc[40:50], X.iloc[50:], y.iloc[50:],
        fold=0, model_name="M2",
        params={"n_estimators": 40, "max_depth": 2, "min_child_weight": 1, "n_jobs": 1},
        early_stopping_rounds=5,
    )

    assert len(prediction) == 10
    assert np.isfinite([metric.rmse, metric.mae, metric.r2]).all()
    assert metric.best_iteration is not None


@pytest.mark.skipif(xgb_module.XGBRegressor is None, reason="xgboost optional dependency is absent")
def test_real_xgboost_end_to_end_small_validation():
    split = XGBTimeSplit(0, slice(0, 8), slice(8, 12), slice(12, 18), 0)

    result = run_xgb_time_validation(
        _feature_sets(),
        [split],
        params={"n_estimators": 25, "max_depth": 2, "min_child_weight": 1, "n_jobs": 1},
        early_stopping_rounds=5,
    )

    assert result.fold_metrics["model_name"].tolist() == ["M0", "M1", "M2"]
    assert len(result.predictions) == 6
    assert np.isfinite(result.fold_metrics[["rmse", "mae", "r2"]]).all().all()
