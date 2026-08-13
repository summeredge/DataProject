from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import chem_ts_corr.xgb_runner as runner
from chem_ts_corr.preprocess import (
    preprocess_frame_causal,
    transform_frame_causal,
)
from chem_ts_corr.xgb_validation import (
    XGBFoldMetric,
    build_expanding_time_splits,
    resolve_xgb_max_used_lag,
)


def _frames(rows: int = 700) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="min")
    return pd.DataFrame(
        {
            "target": np.arange(rows, dtype=float) + 100.0,
            "control": np.arange(rows, dtype=float) * 2.0,
            "x": np.arange(rows, dtype=float) * 3.0,
        },
        index=index,
    )


def _final_review() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "variable": "x",
            "final_rank": 1,
            "final_recommendation": "priority_review",
            "screening_lag": 2,
        }]
    )


def _ranked() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "variable": "x",
            "lag": 2,
            "candidate_class": "upstream_driver_candidate",
            "risk_flags": "",
            "recommended_use": "strong_screening_candidate",
        }]
    )


def _fake_trainer(captured: list[dict[str, object]]):
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
    ):
        captured.append(
            {
                "fold": fold,
                "model_name": model_name,
                "X_valid": X_valid.copy(deep=True),
                "X_test": X_test.copy(deep=True),
            }
        )
        metric = XGBFoldMetric(
            fold=fold,
            model_name=model_name,
            train_rows=len(X_train),
            validation_rows=len(X_valid),
            test_rows=len(X_test),
            best_iteration=None,
            rmse=0.0,
            mae=0.0,
            r2=0.0,
        )
        return metric, np.zeros(len(X_test))

    return train


def _run_fold_safe(tmp_path, data: pd.DataFrame, monkeypatch, **kwargs):
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "XGBRegressor", object)
    monkeypatch.setattr(runner, "train_xgb_fold", _fake_trainer(captured))
    result = runner.run_xgb_validation_fold_safe(
        run_dir=tmp_path,
        target="target",
        data=data,
        final_review_summary=_final_review(),
        ranked_features=_ranked(),
        control_columns=["control"],
        max_lag=5,
        **kwargs,
    )
    return result, captured


def test_resolve_max_used_lag_matches_baseline_and_candidate_windows():
    pool = pd.DataFrame([{"variable": "x", "screening_lag": 2}])

    assert resolve_xgb_max_used_lag(pool, max_lag=5) == 5
    assert resolve_xgb_max_used_lag(
        pool, max_lag=5, available_columns=["target", "control"]
    ) == 5
    assert resolve_xgb_max_used_lag(
        pool, max_lag=2, available_columns=["target", "control", "x"]
    ) == 2


def test_gap_equals_resolved_max_used_lag_and_partitions_do_not_overlap():
    pool = pd.DataFrame([{"variable": "x", "screening_lag": 2}])
    max_used_lag = resolve_xgb_max_used_lag(pool, max_lag=5)
    splits = build_expanding_time_splits(700, gap=max_used_lag)

    assert all(split.gap == max_used_lag for split in splits)
    assert all(split.train_slice.stop < split.validation_slice.start for split in splits)
    assert all(split.validation_slice.stop < split.test_slice.start for split in splits)
    assert splits[0].train_slice.start == 0
    assert splits[-1].test_slice.stop == 700


@pytest.mark.parametrize("mode", ["lowpass", "lowpass_detrend", "lowpass_diff"])
def test_validation_and_test_are_independent_of_train_tail(
    tmp_path, monkeypatch, mode: str
):
    baseline = _frames()
    changed = baseline.copy(deep=True)
    changed.iloc[:100, changed.columns.get_loc("x")] = 1e12

    _, baseline_calls = _run_fold_safe(
        tmp_path / "base", baseline, monkeypatch, preprocess_mode=mode
    )
    _, changed_calls = _run_fold_safe(
        tmp_path / "changed", changed, monkeypatch, preprocess_mode=mode
    )

    assert len(baseline_calls) == len(changed_calls)
    for left, right in zip(baseline_calls, changed_calls):
        assert left["fold"] == right["fold"]
        assert left["model_name"] == right["model_name"]
        pd.testing.assert_frame_equal(left["X_valid"], right["X_valid"])
        pd.testing.assert_frame_equal(left["X_test"], right["X_test"])


def test_diff_partition_first_row_does_not_borrow_previous_partition():
    frame = _frames(40)
    train = frame.iloc[:20]
    gap = frame.iloc[20:25]
    validation = frame.iloc[25:35]

    def prep(partition):
        cleaned = preprocess_frame_causal(
            partition, "target", None, max_forward_fill_gap_points=2
        )
        return transform_frame_causal(
            cleaned, "lowpass_diff", detrend_window=8, diff_interval_minutes=5.0
        )

    separate = prep(validation)
    combined = prep(pd.concat([gap, validation]))
    combined_validation = combined.loc[validation.index.intersection(combined.index)]

    assert validation.index[0] not in separate.index
    assert validation.index[0] in combined_validation.index


def test_forward_fill_does_not_cross_partition_boundary():
    frame = _frames(30)
    gap = frame.iloc[20:25].copy()
    validation = frame.iloc[25:30].copy()
    validation.iloc[0, validation.columns.get_loc("x")] = np.nan
    gap.iloc[-1, gap.columns.get_loc("x")] = 999.0

    cleaned = preprocess_frame_causal(
        validation, "target", None, max_forward_fill_gap_points=5, min_rows=0
    )

    assert pd.isna(cleaned.iloc[0]["x"])


def test_physical_gap_still_resets_state_within_partition():
    complete = _frames(40)
    partition = complete.drop(index=complete.index[20]).iloc[:30]
    after_gap = complete.index[21]

    differenced = transform_frame_causal(partition, "lowpass_diff", detrend_window=8)
    detrended = transform_frame_causal(partition, "lowpass_detrend", detrend_window=8)

    assert after_gap not in differenced.index
    assert after_gap not in detrended.index


def test_raw_fold_path_still_uses_gap_and_positive_lag(tmp_path, monkeypatch):
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "XGBRegressor", object)
    monkeypatch.setattr(runner, "train_xgb_fold", _fake_trainer(captured))

    result = runner.run_xgb_validation_fold_safe(
        run_dir=tmp_path,
        target="target",
        data=_frames(),
        final_review_summary=_final_review(),
        ranked_features=_ranked(),
        control_columns=["control"],
        max_lag=5,
        preprocess_mode="raw",
    )

    assert result.status == "success"
    assert captured
    assert {"M0", "M1", "M2"}.issubset({call["model_name"] for call in captured})
    m2_calls = [call for call in captured if call["model_name"] == "M2"]
    assert any(
        column.startswith("x__lag_")
        for call in m2_calls
        for column in call["X_test"].columns
    )
    for call in captured:
        assert list(call["X_test"].columns)[:1] == ["target__lag_1"]


def test_m1_baseline_trained_once_per_fold_not_once_per_candidate(
    tmp_path, monkeypatch
):
    data = _frames().assign(y=np.arange(700, dtype=float) * 4.0)
    final = pd.DataFrame(
        [
            {"variable": "x", "final_rank": 1, "final_recommendation": "priority_review", "screening_lag": 2},
            {"variable": "y", "final_rank": 2, "final_recommendation": "priority_review", "screening_lag": 2},
        ]
    )
    ranked = pd.DataFrame(
        [
            {"variable": "x", "lag": 2, "candidate_class": "upstream_driver_candidate", "risk_flags": "", "recommended_use": "strong_screening_candidate"},
            {"variable": "y", "lag": 2, "candidate_class": "upstream_driver_candidate", "risk_flags": "", "recommended_use": "strong_screening_candidate"},
        ]
    )
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "XGBRegressor", object)
    monkeypatch.setattr(runner, "train_xgb_fold", _fake_trainer(captured))

    result = runner.run_xgb_validation_fold_safe(
        run_dir=tmp_path,
        target="target",
        data=data,
        final_review_summary=final,
        ranked_features=ranked,
        control_columns=["control"],
        max_lag=5,
        preprocess_mode="raw",
    )

    assert result.status == "success"
    assert sum(1 for call in captured if call["model_name"] == "M1") == 3
    assert sum(1 for call in captured if call["model_name"] == "CANDIDATE") == 6
