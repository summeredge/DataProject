from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr.preprocess import (
    operating_segment_mask,
    preprocess_frame_causal,
    transform_frame_causal,
)
from chem_ts_corr.xgb_validation import (
    DEFAULT_CANDIDATE_LAG_RADIUS,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_XGB_MIN_TEST_ROWS,
    DEFAULT_XGB_MIN_TRAIN_ROWS,
    DEFAULT_XGB_MIN_VALIDATION_ROWS,
    DEFAULT_XGB_TOP_N,
    DEFAULT_XGB_PARAMS,
    CandidateUpliftMetric,
    CandidateUpliftSummary,
    XGBFoldMetric,
    XGBRegressor,
    _improvement_pct,
    _insufficient_uplift_summary,
    _summarize_xgb_metrics,
    _xgb_data_fingerprint,
    build_expanding_time_splits,
    build_xgb_candidate_pool,
    build_xgb_feature_sets,
    resolve_xgb_max_used_lag,
    run_candidate_uplift_validation,
    run_xgb_time_validation,
    summarize_candidate_uplift,
    train_xgb_fold,
    validate_xgb_max_lag,
    validate_xgb_top_n,
)


XGB_OUTPUT_FILES = (
    "xgb_fold_metrics.csv",
    "xgb_model_summary.csv",
    "xgb_candidate_uplift.csv",
    "xgb_predictions.csv",
    "xgb_validation_summary.json",
)
XGB_RUN_STATUSES = frozenset({"success", "missing_dependency", "invalid_input", "failed"})
_MISSING_DEPENDENCY_MESSAGE = (
    "xgboost is not installed. Install optional dependency: pip install -e '.[xgb]'"
)


@dataclass(frozen=True)
class XGBRunResult:
    status: str
    output_files: tuple[str, ...]
    fold_metrics_path: str | None
    summary_path: str | None
    candidate_uplift_path: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        if self.status not in XGB_RUN_STATUSES:
            raise ValueError("unknown XGB run status")


def run_xgb_validation(
    *,
    run_dir: str | Path,
    target: str,
    data: pd.DataFrame,
    final_review_summary: pd.DataFrame,
    ranked_features: pd.DataFrame | None = None,
    control_columns: list[str] | None = None,
    whitelist: list[str] | None = None,
    top_n: int = DEFAULT_XGB_TOP_N,
    max_lag: int | None = None,
    target_mask: pd.Series | None = None,
) -> XGBRunResult:
    total_started_at = time.perf_counter()
    timings = {
        name: 0.0
        for name in (
            "input_validation",
            "candidate_pool",
            "feature_build",
            "split_build",
            "model_validation",
            "candidate_uplift",
            "write_outputs",
            "total",
        )
    }
    stage_started_at = time.perf_counter()
    input_error = _input_error(run_dir, target, data, final_review_summary, ranked_features)
    if input_error:
        return _error_result("invalid_input", input_error)
    try:
        resolved_top_n = validate_xgb_top_n(top_n)
    except ValueError as exc:
        return _error_result("invalid_input", str(exc))

    try:
        resolved_max_lag = _resolve_max_lag(max_lag, final_review_summary, ranked_features)
    except (TypeError, ValueError) as exc:
        return _error_result("invalid_input", str(exc))

    try:
        output_dir = Path(run_dir) / "xgb_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise OSError(f"XGB output path is not a directory: {output_dir}")
    except (OSError, TypeError, ValueError) as exc:
        return _error_result("invalid_input", f"run_dir is not writable: {exc}")

    timings["input_validation"] = _elapsed_seconds(stage_started_at)

    stage_started_at = time.perf_counter()
    try:
        candidate_pool = build_xgb_candidate_pool(
            final_review_summary,
            ranked_features,
            target=target,
            top_n=resolved_top_n,
            whitelist=whitelist,
            control_columns=control_columns,
        )
        timings["candidate_pool"] = _elapsed_seconds(stage_started_at)
    except (TypeError, ValueError) as exc:
        return _error_result("invalid_input", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    if XGBRegressor is None:
        return _error_result("missing_dependency", _MISSING_DEPENDENCY_MESSAGE)

    stage_started_at = time.perf_counter()
    try:
        feature_sets = build_xgb_feature_sets(
            data,
            target,
            candidate_pool,
            control_columns=control_columns,
            max_lag=resolved_max_lag,
            target_mask=target_mask,
        )
        timings["feature_build"] = _elapsed_seconds(stage_started_at)
    except (TypeError, ValueError) as exc:
        return _error_result("invalid_input", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    stage_started_at = time.perf_counter()
    try:
        splits = build_expanding_time_splits(
            len(feature_sets.features), gap=feature_sets.max_used_lag
        )
        timings["split_build"] = _elapsed_seconds(stage_started_at)
    except (TypeError, ValueError) as exc:
        return _error_result("invalid_input", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    stage_started_at = time.perf_counter()
    try:
        model_result = run_xgb_time_validation(feature_sets, splits)
        timings["model_validation"] = _elapsed_seconds(stage_started_at)
    except RuntimeError as exc:
        if "xgboost is not installed" in str(exc).lower():
            return _error_result("missing_dependency", _MISSING_DEPENDENCY_MESSAGE)
        return _error_result("failed", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    stage_started_at = time.perf_counter()
    try:
        _, candidate_summary = run_candidate_uplift_validation(
            feature_sets,
            splits,
            candidate_pool,
            baseline_result=model_result,
        )
        timings["candidate_uplift"] = _elapsed_seconds(stage_started_at)
    except RuntimeError as exc:
        if "xgboost is not installed" in str(exc).lower():
            return _error_result("missing_dependency", _MISSING_DEPENDENCY_MESSAGE)
        return _error_result("failed", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    try:
        paths = {name: output_dir / name for name in XGB_OUTPUT_FILES}
        provenance = model_result.provenance
        summary_payload = {
            "status": "success",
            "target": target,
            "candidate_count": int(len(candidate_summary)),
            "candidate_pool_count": int(len(candidate_pool)),
            "fold_count": int(len(splits)),
            "row_count": int(len(feature_sets.features)),
            "m0_feature_count": int(len(feature_sets.m0_features)),
            "m1_feature_count": int(len(feature_sets.m1_features)),
            "m2_feature_count": int(len(feature_sets.m2_features)),
            "max_used_lag": int(feature_sets.max_used_lag),
            "resolved_max_lag": int(resolved_max_lag),
            "top_n": int(resolved_top_n),
            "data_fingerprint": (
                provenance.data_fingerprint if provenance is not None else ""
            ),
            "early_stopping_rounds": DEFAULT_EARLY_STOPPING_ROUNDS,
            "model_parameters": dict(DEFAULT_XGB_PARAMS),
            "timings_seconds": timings,
            "files": list(XGB_OUTPUT_FILES),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_outputs_transactionally(
            output_dir,
            {
                "xgb_fold_metrics.csv": model_result.fold_metrics,
                "xgb_model_summary.csv": model_result.summary,
                "xgb_candidate_uplift.csv": candidate_summary,
                "xgb_predictions.csv": model_result.predictions,
            },
            summary_payload,
            timings=timings,
            total_started_at=total_started_at,
            write_started_at=time.perf_counter(),
        )
    except OSError as exc:
        return _error_result("invalid_input", f"run_dir is not writable: {exc}")
    except Exception as exc:
        return _error_result("failed", str(exc))

    return XGBRunResult(
        status="success",
        output_files=XGB_OUTPUT_FILES,
        fold_metrics_path=str(paths["xgb_fold_metrics.csv"]),
        summary_path=str(paths["xgb_model_summary.csv"]),
        candidate_uplift_path=str(paths["xgb_candidate_uplift.csv"]),
        error_message=None,
    )


def run_xgb_validation_fold_safe(
    *,
    run_dir: str | Path,
    target: str,
    data: pd.DataFrame,
    final_review_summary: pd.DataFrame,
    ranked_features: pd.DataFrame | None = None,
    control_columns: list[str] | None = None,
    whitelist: list[str] | None = None,
    top_n: int = DEFAULT_XGB_TOP_N,
    max_lag: int | None = None,
    preprocess_mode: str = "raw",
    lowpass_tau_minutes: float = 5.0,
    diff_interval_minutes: float | None = None,
    detrend_window: int = 24,
    resample_rule: str | None = None,
    max_interpolate_gap_points: int = 5,
    segment_column: str | None = None,
    segment_mode: str = "all",
    segment_min: float | None = None,
    segment_max: float | None = None,
) -> XGBRunResult:
    """Run XGB time validation with preprocessing state isolated per fold.

    Unlike the legacy ``run_xgb_validation``, this formal path establishes a
    single split-base time axis first (resampling and target-missing handling
    only), then preprocesses each ``train`` / ``gap_1`` / ``validation`` /
    ``gap_2`` / ``test`` partition independently. The positive-lag features are
    built only after the independent transforms, so ``gap`` keeps providing
    lag history while lowpass / detrend / diff / forward-fill state never
    crosses a fold boundary.
    """
    total_started_at = time.perf_counter()
    timings = {
        name: 0.0
        for name in (
            "input_validation",
            "candidate_pool",
            "feature_build",
            "split_build",
            "model_validation",
            "candidate_uplift",
            "write_outputs",
            "total",
        )
    }

    stage_started_at = time.perf_counter()
    input_error = _input_error(run_dir, target, data, final_review_summary, ranked_features)
    if input_error:
        return _error_result("invalid_input", input_error)
    try:
        resolved_top_n = validate_xgb_top_n(top_n)
    except ValueError as exc:
        return _error_result("invalid_input", str(exc))
    try:
        resolved_max_lag = _resolve_max_lag(max_lag, final_review_summary, ranked_features)
    except (TypeError, ValueError) as exc:
        return _error_result("invalid_input", str(exc))

    try:
        output_dir = Path(run_dir) / "xgb_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise OSError(f"XGB output path is not a directory: {output_dir}")
    except (OSError, TypeError, ValueError) as exc:
        return _error_result("invalid_input", f"run_dir is not writable: {exc}")
    timings["input_validation"] = _elapsed_seconds(stage_started_at)

    stage_started_at = time.perf_counter()
    try:
        candidate_pool = build_xgb_candidate_pool(
            final_review_summary,
            ranked_features,
            target=target,
            top_n=resolved_top_n,
            whitelist=whitelist,
            control_columns=control_columns,
        )
        timings["candidate_pool"] = _elapsed_seconds(stage_started_at)
    except (TypeError, ValueError) as exc:
        return _error_result("invalid_input", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    if XGBRegressor is None:
        return _error_result("missing_dependency", _MISSING_DEPENDENCY_MESSAGE)

    stage_started_at = time.perf_counter()
    try:
        split_base = _fold_safe_split_base(data, target, resample_rule)
        target_mask = _fold_safe_target_mask(
            split_base,
            segment_column=segment_column,
            segment_mode=segment_mode,
            segment_min=segment_min,
            segment_max=segment_max,
        )
        max_used_lag = resolve_xgb_max_used_lag(
            candidate_pool,
            max_lag=resolved_max_lag,
            available_columns=split_base.columns,
        )
        timings["feature_build"] = _elapsed_seconds(stage_started_at)
    except (TypeError, ValueError) as exc:
        return _error_result("invalid_input", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    stage_started_at = time.perf_counter()
    try:
        splits = build_expanding_time_splits(len(split_base), gap=max_used_lag)
        timings["split_build"] = _elapsed_seconds(stage_started_at)
    except (TypeError, ValueError) as exc:
        return _error_result("invalid_input", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    stage_started_at = time.perf_counter()
    try:
        fold_data, fold_metric_rows, prediction_rows, m0, m1, m2, candidate_map = (
            _fold_safe_run_models(
                split_base,
                splits,
                target=target,
                candidate_pool=candidate_pool,
                control_columns=control_columns,
                max_lag=resolved_max_lag,
                target_mask=target_mask,
                preprocess_mode=preprocess_mode,
                lowpass_tau_minutes=lowpass_tau_minutes,
                diff_interval_minutes=diff_interval_minutes,
                detrend_window=detrend_window,
                max_interpolate_gap_points=max_interpolate_gap_points,
            )
        )
        timings["model_validation"] = _elapsed_seconds(stage_started_at)
    except RuntimeError as exc:
        if "xgboost is not installed" in str(exc).lower():
            return _error_result("missing_dependency", _MISSING_DEPENDENCY_MESSAGE)
        return _error_result("failed", str(exc))
    except (TypeError, ValueError) as exc:
        return _error_result("invalid_input", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    stage_started_at = time.perf_counter()
    try:
        valid_candidates, invalid_variables = _fold_safe_candidate_map(candidate_pool, candidate_map)
        candidate_metric_rows = _fold_safe_run_candidates(
            fold_data,
            valid_candidates,
            m1,
        )
        timings["candidate_uplift"] = _elapsed_seconds(stage_started_at)
    except RuntimeError as exc:
        if "xgboost is not installed" in str(exc).lower():
            return _error_result("missing_dependency", _MISSING_DEPENDENCY_MESSAGE)
        return _error_result("failed", str(exc))
    except (TypeError, ValueError) as exc:
        return _error_result("invalid_input", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    try:
        metric_columns = list(XGBFoldMetric.__dataclass_fields__)
        fold_metrics = pd.DataFrame(fold_metric_rows, columns=metric_columns)
        fold_metrics["_model_order"] = fold_metrics["model_name"].map(
            {"M0": 0, "M1": 1, "M2": 2}
        )
        fold_metrics = fold_metrics.sort_values(
            ["fold", "_model_order"], kind="mergesort"
        ).drop(columns="_model_order").reset_index(drop=True)

        summary = _summarize_xgb_metrics(fold_metrics)
        predictions = pd.concat(prediction_rows, ignore_index=True).loc[
            :, ["fold", "timestamp_index", "y_true", "M0_prediction", "M1_prediction", "M2_prediction"]
        ]

        candidate_columns = list(CandidateUpliftMetric.__dataclass_fields__)
        candidate_metrics = pd.DataFrame(candidate_metric_rows, columns=candidate_columns)
        candidate_summary = summarize_candidate_uplift(candidate_metrics)
        if invalid_variables:
            candidate_summary = pd.DataFrame(
                [
                    *candidate_summary.to_dict("records"),
                    *(asdict(_insufficient_uplift_summary(variable)) for variable in invalid_variables),
                ],
                columns=list(CandidateUpliftSummary.__dataclass_fields__),
            )
        candidates = list(candidate_pool["variable"])
        if not candidate_summary.empty:
            candidate_summary["_candidate_order"] = candidate_summary["variable"].map(
                {variable: order for order, variable in enumerate(candidates)}
            )
            candidate_summary = candidate_summary.sort_values(
                ["median_rmse_improvement_pct", "_candidate_order"],
                ascending=[False, True],
                kind="mergesort",
                na_position="last",
            ).drop(columns="_candidate_order").reset_index(drop=True)

        data_fingerprint = (
            _fold_safe_data_fingerprint(fold_data, m2) if fold_data else ""
        )
        paths = {name: output_dir / name for name in XGB_OUTPUT_FILES}
        summary_payload = {
            "status": "success",
            "target": target,
            "candidate_count": int(len(candidate_summary)),
            "candidate_pool_count": int(len(candidate_pool)),
            "fold_count": int(len(splits)),
            "row_count": int(len(predictions)),
            "m0_feature_count": int(len(m0)),
            "m1_feature_count": int(len(m1)),
            "m2_feature_count": int(len(m2)),
            "max_used_lag": int(max_used_lag),
            "resolved_max_lag": int(resolved_max_lag),
            "top_n": int(resolved_top_n),
            "data_fingerprint": data_fingerprint,
            "early_stopping_rounds": DEFAULT_EARLY_STOPPING_ROUNDS,
            "model_parameters": dict(DEFAULT_XGB_PARAMS),
            "preprocess_mode": preprocess_mode,
            "fold_preprocessing_isolated": True,
            "timings_seconds": timings,
            "files": list(XGB_OUTPUT_FILES),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_outputs_transactionally(
            output_dir,
            {
                "xgb_fold_metrics.csv": fold_metrics,
                "xgb_model_summary.csv": summary,
                "xgb_candidate_uplift.csv": candidate_summary,
                "xgb_predictions.csv": predictions,
            },
            summary_payload,
            timings=timings,
            total_started_at=total_started_at,
            write_started_at=time.perf_counter(),
        )
    except OSError as exc:
        return _error_result("invalid_input", f"run_dir is not writable: {exc}")
    except Exception as exc:
        return _error_result("failed", str(exc))

    return XGBRunResult(
        status="success",
        output_files=XGB_OUTPUT_FILES,
        fold_metrics_path=str(paths["xgb_fold_metrics.csv"]),
        summary_path=str(paths["xgb_model_summary.csv"]),
        candidate_uplift_path=str(paths["xgb_candidate_uplift.csv"]),
        error_message=None,
    )


def _fold_safe_split_base(
    data: pd.DataFrame,
    target: str,
    resample_rule: str | None,
) -> pd.DataFrame:
    """Establish one resampled, target-complete time axis before folding."""
    return preprocess_frame_causal(
        data,
        target,
        resample_rule,
        max_forward_fill_gap_points=0,
    )


def _fold_safe_target_mask(
    split_base: pd.DataFrame,
    *,
    segment_column: str | None,
    segment_mode: str,
    segment_min: float | None,
    segment_max: float | None,
) -> pd.Series | None:
    if not segment_column or segment_column not in split_base.columns or segment_mode == "all":
        return None
    mask = operating_segment_mask(
        split_base,
        segment_column,
        segment_mode,
        segment_min,
        segment_max,
    )
    resolved = mask.reindex(split_base.index).fillna(False).astype(bool)
    return None if bool(resolved.all()) else resolved


def _fold_safe_preprocess_partition(
    partition: pd.DataFrame,
    *,
    target: str,
    preprocess_mode: str,
    detrend_window: int,
    lowpass_tau_minutes: float,
    diff_interval_minutes: float | None,
    max_interpolate_gap_points: int,
    min_rows: int,
) -> pd.DataFrame:
    cleaned = preprocess_frame_causal(
        partition,
        target,
        None,
        max_forward_fill_gap_points=max_interpolate_gap_points,
        min_rows=min_rows,
    )
    return transform_frame_causal(
        cleaned,
        preprocess_mode,
        detrend_window,
        lowpass_tau_minutes=lowpass_tau_minutes,
        diff_interval_minutes=diff_interval_minutes,
    )


def _fold_safe_partition_features(
    source: pd.DataFrame,
    *,
    target: str,
    candidate_pool: pd.DataFrame,
    control_columns: list[str] | None,
    max_lag: int,
    target_mask: pd.Series | None,
):
    return build_xgb_feature_sets(
        source,
        target,
        candidate_pool,
        control_columns=control_columns,
        max_lag=max_lag,
        target_mask=target_mask,
    )


def _require_fold_safe_effective_rows(
    *,
    fold: int,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
) -> None:
    """Reject a fold whose real model input falls below the fixed minimums.

    These rows are the effective sample after preprocessing, target mask, lag
    feature alignment and complete-case dropna; the split-base slice lengths
    are only the initial fold geometry.
    """
    if train_rows < DEFAULT_XGB_MIN_TRAIN_ROWS:
        raise ValueError(
            f"fold {fold} effective train rows {train_rows} are below "
            f"min_train_rows {DEFAULT_XGB_MIN_TRAIN_ROWS}"
        )
    if validation_rows < DEFAULT_XGB_MIN_VALIDATION_ROWS:
        raise ValueError(
            f"fold {fold} effective validation rows {validation_rows} are below "
            f"min_validation_rows {DEFAULT_XGB_MIN_VALIDATION_ROWS}"
        )
    if test_rows < DEFAULT_XGB_MIN_TEST_ROWS:
        raise ValueError(
            f"fold {fold} effective test rows {test_rows} are below "
            f"min_test_rows {DEFAULT_XGB_MIN_TEST_ROWS}"
        )


def _fold_safe_data_fingerprint(
    fold_data: list[dict[str, object]],
    m2: tuple[str, ...],
) -> str:
    """Hash every fold's actual train / validation / test model input.

    The description frame carries fold id, partition type, the original time
    index, target values and the full M2 feature columns (M2 includes every
    candidate's features), so any real model-input change is detected while
    unused unrelated columns are ignored. It contains no paths, timestamps or
    random values.
    """
    frames: list[pd.DataFrame] = []
    for entry in fold_data:
        fold = int(entry["fold"])
        partitions = (
            ("train", entry["train_fs"].features, entry["train_fs"].target),
            ("validation", entry["validation_features"], entry["validation_target"]),
            ("test", entry["test_features"], entry["test_target"]),
        )
        for partition, features, target in partitions:
            if features.empty:
                continue
            frame = features.copy(deep=True)
            frame["fold"] = fold
            frame["partition"] = partition
            frame["__target__"] = target.to_numpy()
            frames.append(frame)
    if not frames:
        return ""
    combined = pd.concat(frames, axis=0)
    ordered_columns = [
        "fold",
        "partition",
        *(column for column in m2 if column in combined.columns),
        "__target__",
    ]
    combined = combined.loc[:, ordered_columns]
    features_frame = combined.drop(columns="__target__")
    target_series = combined["__target__"].rename("target")
    return _xgb_data_fingerprint(features_frame, target_series)


def _fold_safe_run_models(
    split_base: pd.DataFrame,
    splits,
    *,
    target: str,
    candidate_pool: pd.DataFrame,
    control_columns: list[str] | None,
    max_lag: int,
    target_mask: pd.Series | None,
    preprocess_mode: str,
    lowpass_tau_minutes: float,
    diff_interval_minutes: float | None,
    detrend_window: int,
    max_interpolate_gap_points: int,
):
    fold_data: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    canonical = None

    for split in splits:
        partitions = {
            "train": split_base.iloc[split.train_slice],
            "gap_1": split_base.iloc[slice(split.train_slice.stop, split.validation_slice.start)],
            "validation": split_base.iloc[split.validation_slice],
            "gap_2": split_base.iloc[slice(split.validation_slice.stop, split.test_slice.start)],
            "test": split_base.iloc[split.test_slice],
        }

        transformed = {
            name: _fold_safe_preprocess_partition(
                frame,
                target=target,
                preprocess_mode=preprocess_mode,
                detrend_window=detrend_window,
                lowpass_tau_minutes=lowpass_tau_minutes,
                diff_interval_minutes=diff_interval_minutes,
                max_interpolate_gap_points=max_interpolate_gap_points,
                min_rows=0 if name.startswith("gap_") else 10,
            )
            for name, frame in partitions.items()
        }

        train_fs = _fold_safe_partition_features(
            transformed["train"],
            target=target,
            candidate_pool=candidate_pool,
            control_columns=control_columns,
            max_lag=max_lag,
            target_mask=target_mask,
        )
        if canonical is None:
            if not train_fs.m2_features:
                raise ValueError("No valid XGB features available")
            canonical = (
                train_fs.m0_features,
                train_fs.m1_features,
                train_fs.m2_features,
                train_fs.candidate_feature_map,
            )
        m0, m1, m2, candidate_map = canonical

        validation_source = pd.concat([transformed["gap_1"], transformed["validation"]])
        validation_fs = _fold_safe_partition_features(
            validation_source,
            target=target,
            candidate_pool=candidate_pool,
            control_columns=control_columns,
            max_lag=max_lag,
            target_mask=target_mask,
        )
        validation_keep = validation_fs.features.index.intersection(
            transformed["validation"].index
        )
        validation_features = validation_fs.features.loc[validation_keep]
        validation_target = validation_fs.target.loc[validation_keep]

        test_source = pd.concat([transformed["gap_2"], transformed["test"]])
        test_fs = _fold_safe_partition_features(
            test_source,
            target=target,
            candidate_pool=candidate_pool,
            control_columns=control_columns,
            max_lag=max_lag,
            target_mask=target_mask,
        )
        test_keep = test_fs.features.index.intersection(transformed["test"].index)
        test_features = test_fs.features.loc[test_keep]
        test_target = test_fs.target.loc[test_keep]

        _require_fold_safe_effective_rows(
            fold=split.fold,
            train_rows=len(train_fs.features),
            validation_rows=len(validation_features),
            test_rows=len(test_features),
        )

        model_features = {"M0": m0, "M1": m1, "M2": m2}
        fold_predictions = {
            "fold": split.fold,
            "timestamp_index": test_features.index,
            "y_true": test_target.to_numpy(),
        }
        m1_metric = None
        for model_name in ("M0", "M1", "M2"):
            columns = tuple(model_features[model_name])
            metric, prediction = train_xgb_fold(
                train_fs.features.loc[:, list(columns)],
                train_fs.target,
                validation_features.loc[:, list(columns)],
                validation_target,
                test_features.loc[:, list(columns)],
                test_target,
                fold=split.fold,
                model_name=model_name,
            )
            fold_metric_rows.append(asdict(metric))
            fold_predictions[f"{model_name}_prediction"] = prediction
            if model_name == "M1":
                m1_metric = metric

        prediction_rows.append(pd.DataFrame(fold_predictions))
        fold_data.append(
            {
                "fold": split.fold,
                "train_fs": train_fs,
                "validation_features": validation_features,
                "validation_target": validation_target,
                "test_features": test_features,
                "test_target": test_target,
                "m1_metric": m1_metric,
            }
        )

    m0, m1, m2, candidate_map = canonical
    return fold_data, fold_metric_rows, prediction_rows, m0, m1, m2, candidate_map


def _fold_safe_candidate_map(
    candidate_pool: pd.DataFrame,
    candidate_map: dict[str, tuple[str, ...]],
) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    valid: dict[str, tuple[str, ...]] = {}
    invalid: list[str] = []
    for variable in candidate_pool["variable"].astype(str):
        added = tuple(candidate_map.get(variable, ()))
        if added:
            valid[variable] = added
        else:
            invalid.append(variable)
    return valid, invalid


def _fold_safe_run_candidates(
    fold_data: list[dict[str, object]],
    valid_candidates: dict[str, tuple[str, ...]],
    m1: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in fold_data:
        baseline = fold["m1_metric"]
        for variable, added in valid_candidates.items():
            columns = (*m1, *added)
            metric, _ = train_xgb_fold(
                fold["train_fs"].features.loc[:, list(columns)],
                fold["train_fs"].target,
                fold["validation_features"].loc[:, list(columns)],
                fold["validation_target"],
                fold["test_features"].loc[:, list(columns)],
                fold["test_target"],
                fold=fold["fold"],
                model_name="CANDIDATE",
            )
            rows.append(
                asdict(
                    CandidateUpliftMetric(
                        variable=variable,
                        fold=fold["fold"],
                        train_rows=metric.train_rows,
                        validation_rows=metric.validation_rows,
                        test_rows=metric.test_rows,
                        rmse=metric.rmse,
                        mae=metric.mae,
                        r2=metric.r2,
                        baseline_rmse=baseline.rmse,
                        baseline_mae=baseline.mae,
                        rmse_improvement_pct=_improvement_pct(baseline.rmse, metric.rmse),
                        mae_improvement_pct=_improvement_pct(baseline.mae, metric.mae),
                        best_iteration=metric.best_iteration,
                    )
                )
            )
    return rows


def _write_outputs_transactionally(
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    summary_payload: dict[str, object],
    *,
    timings: dict[str, float] | None = None,
    total_started_at: float | None = None,
    write_started_at: float | None = None,
) -> None:
    with tempfile.TemporaryDirectory(prefix=".xgb-stage-", dir=output_dir.parent) as temp_name:
        transaction_dir = Path(temp_name)
        staged_dir = transaction_dir / "staged"
        backup_dir = transaction_dir / "backup"
        staged_dir.mkdir()
        backup_dir.mkdir()

        for name in XGB_OUTPUT_FILES[:-1]:
            frames[name].to_csv(staged_dir / name, index=False, encoding="utf-8-sig")
        summary_name = XGB_OUTPUT_FILES[-1]
        (staged_dir / summary_name).write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        for name in XGB_OUTPUT_FILES:
            destination = output_dir / name
            if destination.exists():
                shutil.copy2(destination, backup_dir / name)

        committed: list[str] = []
        try:
            for name in XGB_OUTPUT_FILES:
                (staged_dir / name).replace(output_dir / name)
                committed.append(name)

            # Measure after one complete five-file commit; the final replace persists the timings.
            if timings is not None and write_started_at is not None:
                timings["write_outputs"] = _elapsed_seconds(write_started_at)
            if timings is not None and total_started_at is not None:
                timings["total"] = _elapsed_seconds(total_started_at)
            if timings is not None:
                summary_payload["timings_seconds"] = dict(timings)
                final_summary = staged_dir / ".final-summary.json"
                final_summary.write_text(
                    json.dumps(summary_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                final_summary.replace(output_dir / summary_name)
        except Exception as commit_error:
            rollback_errors: list[str] = []
            for name in reversed(committed):
                destination = output_dir / name
                backup = backup_dir / name
                try:
                    if backup.exists():
                        shutil.copy2(backup, destination)
                    elif destination.exists():
                        destination.unlink()
                except Exception as rollback_error:
                    rollback_errors.append(f"{name}: {rollback_error}")
            if rollback_errors:
                details = "; ".join(rollback_errors)
                raise OSError(f"XGB output commit failed and rollback was incomplete: {details}") from commit_error
            raise


def _input_error(
    run_dir: object,
    target: object,
    data: object,
    final_review_summary: object,
    ranked_features: object,
) -> str | None:
    if run_dir is None or str(run_dir).strip() == "":
        return "missing run_dir"
    if not isinstance(target, str) or not target.strip():
        return "missing target"
    if not isinstance(data, pd.DataFrame) or data.empty:
        return "missing data"
    if target not in data.columns:
        return f"target column not found: {target}"
    if not isinstance(final_review_summary, pd.DataFrame) or final_review_summary.empty:
        return "missing final_review_summary"
    for column in ["variable", "final_recommendation"]:
        if column not in final_review_summary.columns:
            return f"final_review_summary missing column: {column}"
    if ranked_features is not None and not isinstance(ranked_features, pd.DataFrame):
        return "ranked_features must be a DataFrame"
    return None


def _resolve_max_lag(
    max_lag: int | None,
    final_review_summary: pd.DataFrame,
    ranked_features: pd.DataFrame | None,
) -> int:
    if max_lag is not None:
        return validate_xgb_max_lag(max_lag)

    lag_values: list[pd.Series] = []
    if "screening_lag" in final_review_summary.columns:
        lag_values.append(pd.to_numeric(final_review_summary["screening_lag"], errors="coerce"))
    if ranked_features is not None and "lag" in ranked_features.columns:
        lag_values.append(pd.to_numeric(ranked_features["lag"], errors="coerce"))
    if not lag_values:
        return 1
    positive = pd.concat(lag_values, ignore_index=True)
    positive = positive[np.isfinite(positive) & positive.gt(0)]
    if positive.empty:
        return 1
    inferred = max(1, int(np.ceil(float(positive.max()))) + DEFAULT_CANDIDATE_LAG_RADIUS)
    return validate_xgb_max_lag(inferred)


def _elapsed_seconds(started_at: float) -> float:
    return round(max(0.0, time.perf_counter() - started_at), 6)


def _error_result(status: str, message: str) -> XGBRunResult:
    return XGBRunResult(
        status=status,
        output_files=(),
        fold_metrics_path=None,
        summary_path=None,
        candidate_uplift_path=None,
        error_message=message,
    )
