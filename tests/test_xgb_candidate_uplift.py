from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chem_ts_corr.xgb_validation as xgb_module
from chem_ts_corr.xgb_validation import (
    CandidateUpliftMetric,
    CandidateUpliftSummary,
    XGBFeatureSets,
    XGBFoldMetric,
    XGBTimeSplit,
    XGBValidationResult,
    XGB_VALIDATION_STATUS,
    run_candidate_uplift_validation,
    run_xgb_time_validation,
    summarize_candidate_uplift,
)


def _feature_sets(candidate_count: int = 2) -> XGBFeatureSets:
    rows = 18
    index = pd.Index([f"t{row:02d}" for row in range(rows)], name="timestamp")
    target = pd.Series(np.arange(rows, dtype=float) + 10.0, index=index, name="y")
    data: dict[str, np.ndarray] = {
        "y__lag_1": target.to_numpy(),
        "control__lag_1": target.to_numpy() * 0.5,
    }
    candidate_map: dict[str, tuple[str, ...]] = {}
    for number in range(candidate_count):
        variable = f"c{number}"
        feature = f"{variable}__lag_2"
        data[feature] = target.to_numpy() * (number + 2)
        candidate_map[variable] = (feature,)
    features = pd.DataFrame(data, index=index)
    return XGBFeatureSets(
        features=features,
        target=target,
        m0_features=("y__lag_1",),
        m1_features=("y__lag_1", "control__lag_1"),
        m2_features=tuple(features.columns),
        candidate_feature_map=candidate_map,
        max_used_lag=2,
    )


def _pool(variables: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"candidate_order": order, "variable": variable} for order, variable in enumerate(variables, 1)]
    )


def _splits() -> list[XGBTimeSplit]:
    return [
        XGBTimeSplit(0, slice(0, 6), slice(6, 9), slice(9, 12), 0),
        XGBTimeSplit(1, slice(0, 9), slice(9, 12), slice(12, 18), 0),
    ]


def _baseline_result(
    feature_sets: XGBFeatureSets | None = None,
    splits: list[XGBTimeSplit] | None = None,
    *,
    params: dict[str, object] | None = None,
    early_stopping_rounds: int = 50,
) -> XGBValidationResult:
    feature_sets = feature_sets or _feature_sets()
    splits = splits or _splits()
    fold_metrics = []
    prediction_rows = []
    baseline_values = {0: (12.0, 5), 1: (11.0, 6)}
    for split in splits:
        error, best_iteration = baseline_values[split.fold]
        test_target = feature_sets.target.iloc[split.test_slice]
        prediction = test_target.to_numpy() + error
        fold_metrics.append(
            {
                "fold": split.fold,
                "model_name": "M1",
                "train_rows": len(feature_sets.target.iloc[split.train_slice]),
                "validation_rows": len(feature_sets.target.iloc[split.validation_slice]),
                "test_rows": len(feature_sets.target.iloc[split.test_slice]),
                "best_iteration": best_iteration,
                "rmse": error,
                "mae": error,
                "r2": xgb_module._r2_score(test_target.to_numpy(), prediction),
            }
        )
        prediction_rows.extend(
            {
                "fold": split.fold,
                "timestamp_index": timestamp,
                "y_true": value,
                "M1_prediction": predicted,
            }
            for (timestamp, value), predicted in zip(test_target.items(), prediction)
        )
    return XGBValidationResult(
        fold_metrics=pd.DataFrame(fold_metrics),
        summary=pd.DataFrame(),
        predictions=pd.DataFrame(prediction_rows),
        provenance=xgb_module._xgb_validation_provenance(
            feature_sets, splits, params, early_stopping_rounds
        ),
    )


def _fake_trainer(
    calls: list[dict[str, object]],
    candidate_errors: dict[tuple[str, int], tuple[float, float]] | None = None,
):
    errors = candidate_errors or {}

    def train(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        *,
        fold: int,
        model_name: str,
        params: dict[str, object] | None = None,
        early_stopping_rounds: int = 50,
    ) -> tuple[XGBFoldMetric, np.ndarray]:
        calls.append(
            {
                "fold": fold,
                "model_name": model_name,
                "columns": tuple(X_train.columns),
                "train_index": X_train.index.copy(),
                "validation_index": X_valid.index.copy(),
                "test_index": X_test.index.copy(),
                "y_train_index": y_train.index.copy(),
                "y_valid_index": y_valid.index.copy(),
                "y_test_index": y_test.index.copy(),
                "params": params,
                "early_stopping_rounds": early_stopping_rounds,
            }
        )
        if model_name == "M1":
            rmse, mae = 10.0, 10.0
            prediction = y_test.to_numpy(dtype=float) + 10.0
            r2 = xgb_module._r2_score(y_test.to_numpy(dtype=float), prediction)
        else:
            variable = X_train.columns[-1].split("__lag_", 1)[0]
            rmse, mae = errors.get((variable, fold), (8.0, 6.0))
            prediction = np.zeros(len(X_test))
            r2 = 0.5
        metric = XGBFoldMetric(
            fold=fold,
            model_name=model_name,
            train_rows=len(X_train),
            validation_rows=len(X_valid),
            test_rows=len(X_test),
            best_iteration=4,
            rmse=rmse,
            mae=mae,
            r2=r2,
        )
        return metric, prediction

    return train


@pytest.fixture
def fake_dependency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(xgb_module, "XGBRegressor", object)


def test_each_candidate_gets_an_independent_m1_plus_candidate_model(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))

    metrics, _ = run_candidate_uplift_validation(_feature_sets(), _splits(), _pool(["c0", "c1"]))

    assert metrics[["fold", "variable"]].to_records(index=False).tolist() == [
        (0, "c0"), (0, "c1"), (1, "c0"), (1, "c1")
    ]
    candidate_calls = [call for call in calls if call["model_name"] == "CANDIDATE"]
    assert [call["columns"] for call in candidate_calls] == [
        ("y__lag_1", "control__lag_1", "c0__lag_2"),
        ("y__lag_1", "control__lag_1", "c1__lag_2"),
        ("y__lag_1", "control__lag_1", "c0__lag_2"),
        ("y__lag_1", "control__lag_1", "c1__lag_2"),
    ]


def test_m1_baseline_is_trained_once_per_fold_not_once_per_candidate(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))

    run_candidate_uplift_validation(_feature_sets(), _splits(), _pool(["c0", "c1"]))

    baseline_calls = [call for call in calls if call["model_name"] == "M1"]
    assert [(call["fold"], call["columns"]) for call in baseline_calls] == [
        (0, ("y__lag_1", "control__lag_1")),
        (1, ("y__lag_1", "control__lag_1")),
    ]
    assert len(calls) == len(_splits()) * 3


def test_xgb2_baseline_result_is_reused_without_retraining_m1(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))

    metrics, _ = run_candidate_uplift_validation(
        _feature_sets(),
        _splits(),
        _pool(["c0", "c1"]),
        baseline_result=_baseline_result(),
    )

    assert all(call["model_name"] == "CANDIDATE" for call in calls)
    assert len(calls) == len(_splits()) * 2
    assert metrics["baseline_rmse"].tolist() == [12.0, 12.0, 11.0, 11.0]
    assert metrics["baseline_mae"].tolist() == [12.0, 12.0, 11.0, 11.0]


def test_formal_xgb2_result_provenance_is_reused_without_m1_training(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    feature_sets = _feature_sets()
    splits = _splits()
    baseline = run_xgb_time_validation(feature_sets, splits)
    assert baseline.provenance is not None
    calls.clear()

    run_candidate_uplift_validation(
        feature_sets, splits, _pool(["c0", "c1"]), baseline_result=baseline
    )

    assert calls
    assert all(call["model_name"] == "CANDIDATE" for call in calls)


def test_external_baseline_without_provenance_is_rejected(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    valid = _baseline_result()
    baseline = XGBValidationResult(
        valid.fold_metrics, valid.summary, valid.predictions
    )

    with pytest.raises(ValueError, match="provenance"):
        run_candidate_uplift_validation(
            _feature_sets(), _splits(), _pool(["c0"]), baseline_result=baseline
        )
    assert calls == []


@pytest.mark.parametrize(("defect", "match"), [("index", "test index"), ("truth", "y_true")])
def test_external_baseline_predictions_must_match_current_test_data(
    fake_dependency, monkeypatch: pytest.MonkeyPatch, defect: str, match: str
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    baseline = _baseline_result()
    if defect == "index":
        baseline.predictions.loc[0, "timestamp_index"] = "different_timestamp"
    else:
        baseline.predictions.loc[0, "y_true"] += 1.0

    with pytest.raises(ValueError, match=match):
        run_candidate_uplift_validation(
            _feature_sets(), _splits(), _pool(["c0"]), baseline_result=baseline
        )
    assert calls == []


@pytest.mark.parametrize("field", ["rmse", "mae"])
def test_external_baseline_metrics_must_match_m1_predictions(
    fake_dependency, monkeypatch: pytest.MonkeyPatch, field: str
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    baseline = _baseline_result()
    baseline.fold_metrics.loc[0, field] += 1.0

    with pytest.raises(ValueError, match=f"{field} does not match M1_prediction"):
        run_candidate_uplift_validation(
            _feature_sets(), _splits(), _pool(["c0"]), baseline_result=baseline
        )
    assert calls == []


def test_external_baseline_requires_m1_prediction(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    baseline = _baseline_result()
    baseline.predictions.drop(columns="M1_prediction", inplace=True)

    with pytest.raises(ValueError, match="M1_prediction"):
        run_candidate_uplift_validation(
            _feature_sets(), _splits(), _pool(["c0"]), baseline_result=baseline
        )
    assert calls == []


def test_external_baseline_m1_prediction_must_match_metrics(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    baseline = _baseline_result()
    baseline.predictions.loc[0, "M1_prediction"] += 1.0

    with pytest.raises(ValueError, match="does not match M1_prediction"):
        run_candidate_uplift_validation(
            _feature_sets(), _splits(), _pool(["c0"]), baseline_result=baseline
        )
    assert calls == []


@pytest.mark.parametrize("defect", ["params", "early_stopping", "m1_features", "m1_values"])
def test_external_baseline_training_provenance_must_match(
    fake_dependency, monkeypatch: pytest.MonkeyPatch, defect: str
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    current = _feature_sets()
    kwargs: dict[str, object] = {}
    if defect == "params":
        kwargs["params"] = {"max_depth": 4}
        baseline = _baseline_result(current)
    elif defect == "early_stopping":
        kwargs["early_stopping_rounds"] = 49
        baseline = _baseline_result(current)
    elif defect == "m1_features":
        baseline_features = _feature_sets()
        baseline_features = replace(
            baseline_features,
            m1_features=tuple(reversed(baseline_features.m1_features)),
        )
        baseline = _baseline_result(baseline_features)
    else:
        baseline_features = _feature_sets()
        baseline_features.features.loc[:, "control__lag_1"] += 1.0
        baseline = _baseline_result(baseline_features)

    with pytest.raises(ValueError, match="provenance"):
        run_candidate_uplift_validation(
            current, _splits(), _pool(["c0"]), baseline_result=baseline, **kwargs
        )
    assert calls == []


@pytest.mark.parametrize("defect", ["missing_fold", "duplicate_fold", "row_count_mismatch"])
def test_invalid_xgb2_baseline_result_is_rejected(
    fake_dependency, monkeypatch: pytest.MonkeyPatch, defect: str
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    baseline = _baseline_result()
    if defect == "missing_fold":
        baseline.fold_metrics.drop(index=1, inplace=True)
    elif defect == "duplicate_fold":
        baseline = XGBValidationResult(
            fold_metrics=pd.concat(
                [baseline.fold_metrics, baseline.fold_metrics.iloc[[0]]], ignore_index=True
            ),
            summary=pd.DataFrame(),
            predictions=baseline.predictions,
            provenance=baseline.provenance,
        )
    else:
        baseline.fold_metrics.loc[0, "test_rows"] = 4

    with pytest.raises(ValueError, match="baseline_result"):
        run_candidate_uplift_validation(
            _feature_sets(), _splits(), _pool(["c0"]), baseline_result=baseline
        )
    assert calls == []


def test_candidate_and_baseline_use_identical_fold_indexes(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))

    run_candidate_uplift_validation(_feature_sets(), _splits(), _pool(["c0", "c1"]))

    for fold in [0, 1]:
        fold_calls = [call for call in calls if call["fold"] == fold]
        baseline = fold_calls[0]
        for candidate in fold_calls[1:]:
            assert candidate["train_index"].equals(baseline["train_index"])
            assert candidate["validation_index"].equals(baseline["validation_index"])
            assert candidate["test_index"].equals(baseline["test_index"])
            assert candidate["test_index"].equals(candidate["y_test_index"])


@pytest.mark.parametrize(
    ("mapping", "expected_status"),
    [
        ({"c0": ()}, "insufficient_features"),
        ({"c0": ("missing__lag_2",)}, "insufficient_features"),
        ({}, "insufficient_features"),
    ],
)
def test_invalid_candidate_features_are_preserved_without_training(
    fake_dependency,
    monkeypatch: pytest.MonkeyPatch,
    mapping: dict[str, tuple[str, ...]],
    expected_status: str,
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    feature_sets = _feature_sets(1)
    feature_sets.candidate_feature_map.clear()
    feature_sets.candidate_feature_map.update(mapping)

    metrics, summary = run_candidate_uplift_validation(feature_sets, _splits(), _pool(["c0"]))

    assert metrics.empty
    assert calls == []
    assert summary.loc[0, "variable"] == "c0"
    assert summary.loc[0, "validation_status"] == expected_status
    assert summary.loc[0, "fold_count"] == 0


def test_all_invalid_candidates_ignore_unused_stale_baseline(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    feature_sets = _feature_sets(1)
    feature_sets.candidate_feature_map.clear()
    stale_baseline = XGBValidationResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    metrics, summary = run_candidate_uplift_validation(
        feature_sets,
        _splits(),
        _pool(["c0"]),
        baseline_result=stale_baseline,
    )

    assert metrics.empty
    assert calls == []
    assert summary.loc[0, "validation_status"] == "insufficient_features"


def test_improvement_formulas_use_baseline_minus_candidate_direction(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    errors = {("c0", 0): (8.0, 7.5), ("c0", 1): (12.0, 12.5)}
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls, errors))

    metrics, _ = run_candidate_uplift_validation(_feature_sets(1), _splits(), _pool(["c0"]))

    assert metrics["rmse_improvement_pct"].tolist() == pytest.approx([20.0, -20.0])
    assert metrics["mae_improvement_pct"].tolist() == pytest.approx([25.0, -25.0])
    assert metrics["baseline_rmse"].tolist() == [10.0, 10.0]
    assert metrics["baseline_mae"].tolist() == [10.0, 10.0]


def _summary_metrics(variable: str, rmse: list[float], mae: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable": variable,
            "fold": range(len(rmse)),
            "rmse_improvement_pct": rmse,
            "mae_improvement_pct": mae,
        }
    )


@pytest.mark.parametrize(
    ("rmse", "mae", "expected"),
    [
        ([5.0, 4.0, 3.0], [2.0, 0.0, 1.0], "validated_incremental_signal"),
        ([5.0, -2.0, 1.0], [2.0, -1.0, 1.0], "unstable_out_of_time"),
        ([-1.0, 0.0, -2.0], [0.0, -1.0, -2.0], "redundant_with_baseline"),
        ([1.0], [1.0], "weak_incremental_value"),
    ],
)
def test_candidate_validation_status_rules(
    rmse: list[float], mae: list[float], expected: str
):
    summary = summarize_candidate_uplift(_summary_metrics("x", rmse, mae))

    assert summary.loc[0, "validation_status"] == expected


def test_unstable_status_takes_priority_when_validated_thresholds_also_pass():
    summary = summarize_candidate_uplift(
        _summary_metrics("x", [5.0, 4.0, 3.0, -1.0], [2.0, 2.0, 2.0, 0.0])
    )

    assert summary.loc[0, "positive_rmse_fold_ratio"] == pytest.approx(0.75)
    assert summary.loc[0, "median_rmse_improvement_pct"] == pytest.approx(3.5)
    assert summary.loc[0, "median_mae_improvement_pct"] == pytest.approx(2.0)
    assert summary.loc[0, "validation_status"] == "unstable_out_of_time"


def test_candidate_summary_counts_ratios_and_aggregates():
    summary = summarize_candidate_uplift(
        _summary_metrics("x", [6.0, 3.0, -3.0], [4.0, 0.0, -2.0])
    ).iloc[0]

    assert summary["fold_count"] == 3
    assert summary["positive_rmse_fold_count"] == 2
    assert summary["positive_mae_fold_count"] == 1
    assert summary["positive_rmse_fold_ratio"] == pytest.approx(2 / 3)
    assert summary["median_rmse_improvement_pct"] == pytest.approx(3.0)
    assert summary["mean_rmse_improvement_pct"] == pytest.approx(2.0)
    assert summary["worst_fold_rmse_improvement_pct"] == pytest.approx(-3.0)


def test_summary_sorts_by_median_rmse_descending_stably():
    metrics = pd.concat(
        [
            _summary_metrics("b", [2.0, 2.0], [1.0, 1.0]),
            _summary_metrics("a", [4.0, 4.0], [1.0, 1.0]),
            _summary_metrics("c", [2.0, 2.0], [1.0, 1.0]),
        ],
        ignore_index=True,
    )

    summary = summarize_candidate_uplift(metrics)

    assert summary["variable"].tolist() == ["a", "b", "c"]


def test_candidate_limit_uses_first_ten_in_candidate_order(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    variables = [f"c{number}" for number in range(10)]
    pool = _pool(list(reversed(variables)))
    pool["candidate_order"] = list(reversed(range(1, 11)))

    metrics, summary = run_candidate_uplift_validation(
        _feature_sets(10), [_splits()[0]], pool
    )

    assert metrics["variable"].tolist() == variables
    assert set(summary["variable"]) == set(variables)
    assert len([call for call in calls if call["model_name"] == "M1"]) == 1
    assert len([call for call in calls if call["model_name"] == "CANDIDATE"]) == 10


def test_direct_uplift_allows_twelve_normalized_unique_candidates(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    pool = _pool([*[f"c{number}" for number in range(10)], " c10 ", "c10", "c11"])
    pool["force_included"] = [False] * 10 + [True, True, True]

    metrics, summary = run_candidate_uplift_validation(
        _feature_sets(12), [_splits()[0]], pool
    )

    assert metrics["variable"].tolist() == [f"c{number}" for number in range(12)]
    assert set(summary["variable"]) == {f"c{number}" for number in range(12)}
    assert len([call for call in calls if call["model_name"] == "M1"]) == 1
    assert len([call for call in calls if call["model_name"] == "CANDIDATE"]) == 12


def test_direct_uplift_rejects_thirteen_candidates_before_any_fit(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    pool = _pool([f"c{number}" for number in range(13)])
    pool["force_included"] = [False] * 10 + [True] * 3

    with pytest.raises(
        ValueError, match="XGB total candidate count including whitelist must not exceed 12"
    ):
        run_candidate_uplift_validation(_feature_sets(13), [_splits()[0]], pool)

    assert calls == []


def test_repeated_inputs_produce_identical_results(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))

    first = run_candidate_uplift_validation(_feature_sets(), _splits(), _pool(["c0", "c1"]))
    calls.clear()
    second = run_candidate_uplift_validation(_feature_sets(), _splits(), _pool(["c0", "c1"]))

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_inputs_are_not_modified(fake_dependency, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))
    feature_sets = _feature_sets()
    pool = _pool(["c0", "c1"])
    before_features = feature_sets.features.copy(deep=True)
    before_target = feature_sets.target.copy(deep=True)
    before_map = feature_sets.candidate_feature_map.copy()
    before_pool = pool.copy(deep=True)
    baseline = _baseline_result()
    before_baseline = baseline.fold_metrics.copy(deep=True)

    run_candidate_uplift_validation(
        feature_sets, _splits(), pool, baseline_result=baseline
    )

    pd.testing.assert_frame_equal(feature_sets.features, before_features)
    pd.testing.assert_series_equal(feature_sets.target, before_target)
    assert feature_sets.candidate_feature_map == before_map
    pd.testing.assert_frame_equal(pool, before_pool)
    pd.testing.assert_frame_equal(baseline.fold_metrics, before_baseline)


def test_output_columns_match_metric_and_summary_contracts(
    fake_dependency, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(xgb_module, "train_xgb_fold", _fake_trainer(calls))

    metrics, summary = run_candidate_uplift_validation(
        _feature_sets(1), _splits(), _pool(["c0"])
    )

    assert metrics.columns.tolist() == list(CandidateUpliftMetric.__dataclass_fields__)
    assert summary.columns.tolist() == list(CandidateUpliftSummary.__dataclass_fields__)


def test_validation_status_constant_and_dataclass_guard():
    assert XGB_VALIDATION_STATUS == {
        "validated_incremental_signal", "weak_incremental_value", "redundant_with_baseline",
        "unstable_out_of_time", "insufficient_features",
    }
    with pytest.raises(ValueError, match="validation_status"):
        CandidateUpliftSummary("x", 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "bad")


def test_uplift_runner_does_not_rebuild_drop_or_split():
    source = inspect.getsource(run_candidate_uplift_validation)

    assert "build_xgb_candidate_pool" not in source
    assert "build_xgb_feature_sets" not in source
    assert "build_expanding_time_splits" not in source
    assert "dropna" not in source
    assert "shift(" not in source


@pytest.mark.parametrize(
    "forbidden",
    [
        "GridSearchCV", "RandomizedSearchCV", "Optuna", "train_test_split", "KFold",
        "shuffle=True", "shift(-", "abs(best_lag)", "final_score", "driver_rank",
    ],
)
def test_candidate_uplift_source_keeps_architecture_guards(forbidden: str):
    source = Path("chem_ts_corr/xgb_validation.py").read_text(encoding="utf-8")
    assert forbidden not in source


@pytest.mark.skipif(xgb_module.XGBRegressor is None, reason="xgboost optional dependency is absent")
def test_real_candidate_uplift_small_fold_smoke():
    feature_sets = _feature_sets(1)
    split = XGBTimeSplit(0, slice(0, 8), slice(8, 12), slice(12, 18), 0)

    metrics, summary = run_candidate_uplift_validation(
        feature_sets,
        [split],
        _pool(["c0"]),
        params={"n_estimators": 25, "max_depth": 2, "min_child_weight": 1, "n_jobs": 1},
        early_stopping_rounds=5,
    )

    assert metrics["variable"].tolist() == ["c0"]
    assert np.isfinite(metrics[["rmse", "mae", "r2"]]).all().all()
    assert summary.loc[0, "validation_status"] in XGB_VALIDATION_STATUS
