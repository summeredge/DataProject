from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from chem_ts_corr.xgb_validation import (
    CandidateUpliftMetric,
    XGBTimeSplit,
    XGB_FOLD_CONTEXT_COLUMNS,
    build_candidate_fold_metrics,
    build_xgb_fold_context,
)


def _irregular_fold_indexes() -> dict[int, tuple[pd.Index, pd.Index, pd.Index]]:
    return {
        0: (
            pd.DatetimeIndex(
                [
                    "2025-01-01 10:00",
                    "2025-01-01 10:01",
                    "2025-01-01 10:02",
                    "2025-01-01 10:30",
                    "2025-01-01 10:31",
                ]
            ),
            pd.date_range("2025-01-01 11:00", periods=3, freq="min"),
            pd.date_range("2025-01-01 12:00", periods=3, freq="min"),
        )
    }


def test_fold_context_uses_actual_timestamps_and_direct_gap_values():
    splits = [XGBTimeSplit(0, slice(0, 5), slice(5, 8), slice(8, 11), 2)]

    context = build_xgb_fold_context(
        _irregular_fold_indexes(), splits, max_used_lag=4
    )

    assert context.columns.tolist() == list(XGB_FOLD_CONTEXT_COLUMNS)
    row = context.iloc[0]
    assert row["train_start"] == pd.Timestamp("2025-01-01 10:00")
    assert row["train_end"] == pd.Timestamp("2025-01-01 10:31")
    assert row["validation_start"] == pd.Timestamp("2025-01-01 11:00")
    assert row["validation_end"] == pd.Timestamp("2025-01-01 11:02")
    assert row["test_start"] == pd.Timestamp("2025-01-01 12:00")
    assert row["test_end"] == pd.Timestamp("2025-01-01 12:02")
    assert row["train_rows"] == 5
    assert row["validation_rows"] == 3
    assert row["test_rows"] == 3
    assert row["train_duration_minutes"] == 31.0
    assert row["validation_duration_minutes"] == 2.0
    assert row["test_duration_minutes"] == 2.0
    assert row["sampling_interval_minutes"] == 1.0
    assert row["gap_rows"] == 2
    assert row["gap_duration_minutes"] == 2.0
    assert row["max_used_lag"] == 4
    assert row["max_used_lag_duration_minutes"] == 4.0
    assert row["train_duration_minutes"] != row["train_rows"]


def test_candidate_fold_metrics_reuses_fold_context_coverage():
    indexes = _irregular_fold_indexes()
    splits = [XGBTimeSplit(0, slice(0, 5), slice(5, 8), slice(8, 11), 2)]
    context = build_xgb_fold_context(indexes, splits, max_used_lag=4)
    metrics = pd.DataFrame(
        [
            asdict(
                CandidateUpliftMetric(
                    variable="x",
                    fold=0,
                    train_rows=5,
                    validation_rows=3,
                    test_rows=3,
                    rmse=0.9,
                    mae=0.8,
                    r2=0.5,
                    baseline_rmse=1.0,
                    baseline_mae=1.0,
                    rmse_improvement_pct=10.0,
                    mae_improvement_pct=20.0,
                    best_iteration=4,
                )
            )
        ]
    )

    details = build_candidate_fold_metrics(metrics, indexes, fold_context=context)

    for field in (
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
        "train_rows",
        "validation_rows",
        "test_rows",
        "train_duration_minutes",
        "validation_duration_minutes",
        "test_duration_minutes",
        "sampling_interval_minutes",
        "gap_rows",
        "gap_duration_minutes",
    ):
        assert details.loc[0, field] == context.loc[0, field]


def test_fold_context_preserves_missing_time_semantics():
    indexes = {
        0: (pd.Index([0, 1, 2]), pd.Index([3, 4]), pd.Index([5, 6]))
    }
    context = build_xgb_fold_context(
        indexes,
        [XGBTimeSplit(0, slice(0, 3), slice(3, 5), slice(5, 7), 2)],
        max_used_lag=4,
    )

    row = context.iloc[0]
    assert pd.isna(row["train_duration_minutes"])
    assert pd.isna(row["validation_duration_minutes"])
    assert pd.isna(row["test_duration_minutes"])
    assert pd.isna(row["sampling_interval_minutes"])
    assert row["gap_rows"] == 2
    assert pd.isna(row["gap_duration_minutes"])
    assert pd.isna(row["max_used_lag_duration_minutes"])

    single_row_partitions = {
        0: (
            pd.DatetimeIndex(["2025-01-01 10:00"]),
            pd.DatetimeIndex(["2025-01-01 10:30"]),
            pd.DatetimeIndex(["2025-01-01 11:00"]),
        )
    }
    single_row_context = build_xgb_fold_context(
        single_row_partitions,
        [XGBTimeSplit(0, slice(0, 1), slice(1, 2), slice(2, 3), 2)],
        max_used_lag=4,
    )
    assert pd.isna(single_row_context.loc[0, "sampling_interval_minutes"])
    assert pd.isna(single_row_context.loc[0, "gap_duration_minutes"])
    assert pd.isna(single_row_context.loc[0, "max_used_lag_duration_minutes"])
