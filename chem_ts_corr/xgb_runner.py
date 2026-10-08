"""XGBoost time-ordered holdout validation for downstream manual review.

This module implements the fourth-layer temporal holdout validation. Its
outputs are prediction-increment evidence only: they do not prove causality,
confirm a process root cause, or define a variable ranking. The runner keeps
the formal screening artefacts isolated, so XGBoost results cannot change
``final_score``, ``ranked_features.csv``, Top-K, or any earlier-layer result.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chem_ts_corr.preprocess import (
    operating_segment_mask,
    preprocess_frame_causal,
    transform_frame_causal,
)
from chem_ts_corr.xgb_validation import (
    CANDIDATE_FOLD_METRICS_COLUMNS,
    XGB_FOLD_CONTEXT_COLUMNS,
    XGB_PREDICTION_COLUMNS,
    DEFAULT_CANDIDATE_LAG_RADIUS,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_XGB_MIN_TEST_ROWS,
    DEFAULT_XGB_MIN_TRAIN_ROWS,
    DEFAULT_XGB_MIN_VALIDATION_ROWS,
    DEFAULT_XGB_TOP_N,
    DEFAULT_XGB_PARAMS,
    build_candidate_fold_metrics,
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
    build_xgb_fold_context,
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
    "xgb_fold_context.csv",
    "xgb_model_summary.csv",
    "xgb_candidate_uplift.csv",
    "xgb_candidate_fold_metrics.csv",
    "xgb_predictions.csv",
    "xgb_validation_summary.json",
)
XGB_RUN_STATUSES = frozenset({"success", "missing_dependency", "invalid_input", "failed"})
_MISSING_DEPENDENCY_MESSAGE = (
    "xgboost is not installed. Install optional dependency: pip install -e '.[xgb]'"
)


def read_xgb_execution_state(run_dir: str | Path) -> dict:
    """Separate the latest attempt from the retained seven-file success bundle."""
    directory = Path(run_dir)
    state_path = directory / "xgb_execution_state.json"
    try:
        summary = json.loads((directory / "xgb_validation/xgb_validation_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {}
    if not isinstance(summary, dict):
        summary = {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
            "status": summary.get("status", "not_run"), "error_message": None,
        }
        if (not isinstance(state, dict) or not isinstance(state.get("status"), str)
                or state["status"] not in XGB_RUN_STATUSES | {"not_run"}):
            raise ValueError("invalid execution state")
    except (OSError, ValueError):
        state = {"status": "failed", "error_message": "XGB execution state is unreadable"}
    valid = summary.get("status") == "success"
    valid = valid and all((directory / "xgb_validation" / name).is_file() for name in XGB_OUTPUT_FILES)
    try:
        model = pd.read_csv(directory / "xgb_validation/xgb_model_summary.csv", encoding="utf-8-sig")
        uplift = pd.read_csv(directory / "xgb_validation/xgb_candidate_uplift.csv", encoding="utf-8-sig", dtype={"variable": str})
        valid = valid and _validate_xgb_model_summary(model) and _validate_xgb_candidate_uplift(uplift, summary)
        valid = valid and _validate_xgb_fold_outputs(directory / "xgb_validation", model, uplift, summary)
    except (OSError, ValueError):
        valid = False
    state = dict(state)
    state["historical_result_available"] = bool(valid and state["status"] != "success")
    state["current_result_available"] = bool(valid and state["status"] == "success")
    if state["status"] == "success" and not valid:
        state["status"] = "incomplete_outputs"
    return state


def _validate_xgb_fold_outputs(
    directory: Path, model: pd.DataFrame, uplift: pd.DataFrame, summary: dict,
) -> bool:
    context = pd.read_csv(directory / "xgb_fold_context.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(directory / "xgb_fold_metrics.csv", encoding="utf-8-sig")
    details = pd.read_csv(directory / "xgb_candidate_fold_metrics.csv", encoding="utf-8-sig", dtype={"variable": str})
    for frame, columns in (
        (context, XGB_FOLD_CONTEXT_COLUMNS),
        (metrics, XGBFoldMetric.__dataclass_fields__),
        (details, CANDIDATE_FOLD_METRICS_COLUMNS),
    ):
        if not set(columns).issubset(frame.columns):
            return False
        frame["fold"] = frame["fold"].map(_validated_nonnegative_integer)
        if frame["fold"].isna().any():
            return False
    fold_count = _validated_nonnegative_integer(summary.get("fold_count"))
    if not fold_count or len(context) != fold_count or context["fold"].duplicated().any():
        return False
    folds = set(context["fold"])
    if (len(metrics) != fold_count * 3 or metrics.duplicated(["fold", "model_name"]).any()
            or set(metrics["fold"]) != folds or set(metrics["model_name"]) != {"M0", "M1", "M2"}
            or not pd.to_numeric(model["fold_count"], errors="coerce").eq(fold_count).all()):
        return False
    for frame, columns in (
        (metrics, ("rmse", "mae", "r2", "best_iteration")),
        (details, ("baseline_rmse", "candidate_rmse", "rmse_improvement_pct", "baseline_mae",
                   "candidate_mae", "mae_improvement_pct", "candidate_r2", "best_iteration")),
    ):
        values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
        if ((frame[list(columns)].notna() & values.isna()).any().any()
                or np.isinf(values.to_numpy(dtype=float)).any()):
            return False
    errors = metrics[["rmse", "mae"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(errors.to_numpy()).all() or errors.lt(0).any().any():
        return False
    errors = details[["candidate_rmse", "candidate_mae"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(errors.to_numpy(dtype=float)).all() or errors.lt(0).any().any():
        return False
    for column in ("train_rows", "validation_rows", "test_rows"):
        context[column] = context[column].map(_validated_nonnegative_integer)
        if context[column].isna().any() or not context[column].gt(0).all():
            return False
        expected = metrics["fold"].map(context.set_index("fold")[column])
        if not pd.to_numeric(metrics[column], errors="coerce").eq(expected).all():
            return False
    for partition in ("train", "validation", "test"):
        start = pd.to_datetime(context[f"{partition}_start"], format="mixed", errors="coerce", utc=True)
        end = pd.to_datetime(context[f"{partition}_end"], format="mixed", errors="coerce", utc=True)
        if ((context[f"{partition}_start"].notna() & start.isna()).any()
                or (context[f"{partition}_end"].notna() & end.isna()).any()
                or (end < start).any()):
            return False
    if (details.duplicated(["variable", "fold"]).any()
            or not set(details["fold"]).issubset(folds)
            or not set(details["variable"]).issubset(set(uplift["variable"]))):
        return False
    for column in set(CANDIDATE_FOLD_METRICS_COLUMNS).intersection(XGB_FOLD_CONTEXT_COLUMNS) - {"fold"}:
        expected = details["fold"].map(context.set_index("fold")[column])
        if not (details[column].eq(expected) | (details[column].isna() & expected.isna())).all():
            return False
    baseline = metrics[metrics["model_name"].eq("M1")].set_index("fold")
    for column, metric in (("baseline_rmse", "rmse"), ("baseline_mae", "mae")):
        if not np.allclose(pd.to_numeric(details[column], errors="coerce"),
                           pd.to_numeric(details["fold"].map(baseline[metric]), errors="coerce"),
                           equal_nan=False):
            return False
    computed = summarize_candidate_uplift(details).set_index("variable")
    for row in uplift.to_dict("records"):
        variable = row["variable"]
        if row["validation_status"] == "insufficient_features":
            if variable in computed.index:
                return False
            continue
        if variable not in computed.index:
            return False
        if row["validation_status"] != computed.loc[variable, "validation_status"]:
            return False
        for column in CandidateUpliftSummary.__dataclass_fields__:
            if column in {"variable", "validation_status"}:
                continue
            actual, expected = row[column], computed.loc[variable, column]
            if _is_missing_value(actual) and _is_missing_value(expected):
                continue
            value = _finite_number(actual)
            if value is None or not np.isclose(value, expected, rtol=1e-7, atol=1e-10):
                return False
    prediction_counts = pd.Series(0, index=context["fold"], dtype="int64")
    for chunk in pd.read_csv(directory / "xgb_predictions.csv", encoding="utf-8-sig", chunksize=100_000):
        if not set(XGB_PREDICTION_COLUMNS).issubset(chunk.columns):
            return False
        prediction_folds = pd.to_numeric(chunk["fold"], errors="coerce")
        if not prediction_folds.isin(folds).all() or chunk["timestamp_index"].isna().any():
            return False
        values = chunk[list(XGB_PREDICTION_COLUMNS[2:])].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy()).all():
            return False
        prediction_counts += prediction_folds.value_counts().reindex(prediction_counts.index, fill_value=0)
    if not prediction_counts.eq(context.set_index("fold")["test_rows"]).all():
        return False
    if summary.get("fold_preprocessing_isolated") is True:
        return _validated_nonnegative_integer(summary.get("row_count")) == int(prediction_counts.sum())
    return True


def record_xgb_execution(run_dir: str | Path, status: str, error_message: str | None = None) -> None:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "error_message": error_message,
               "updated_at": datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                     prefix=".xgb-state-", delete=False) as stream:
        temp_path = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False)
    try:
        temp_path.replace(directory / "xgb_execution_state.json")
    finally:
        temp_path.unlink(missing_ok=True)


def persist_xgb_execution(function):
    @wraps(function)
    def run(*args, **kwargs):
        run_dir = kwargs.get("run_dir", args[0] if args else None)
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            if run_dir is not None and str(run_dir).strip():
                record_xgb_execution(run_dir, "invalid_input" if isinstance(exc, ValueError) else "failed", str(exc))
            raise
        status = result["status"] if isinstance(result, dict) else result.status
        message = result.get("error_message") if isinstance(result, dict) else result.error_message
        if run_dir is not None and str(run_dir).strip():
            try:
                record_xgb_execution(run_dir, status, message)
            except OSError:
                if status == "success":
                    raise
        return result
    return run


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


@persist_xgb_execution
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
    """Run legacy time-ordered XGBoost validation for manual review evidence.

    Candidate uplift compares each candidate model with the same M1 baseline:
    target history plus configured control-variable history. A positive uplift
    means the candidate supplied additional out-of-time predictive information;
    it is not a causal contribution or a ranking/scoring signal.
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
        candidate_metrics, candidate_summary = run_candidate_uplift_validation(
            feature_sets,
            splits,
            candidate_pool,
            baseline_result=model_result,
        )
        fold_indices = {
            split.fold: (
                feature_sets.features.iloc[split.train_slice].index,
                feature_sets.features.iloc[split.validation_slice].index,
                feature_sets.features.iloc[split.test_slice].index,
            )
            for split in splits
        }
        fold_context = build_xgb_fold_context(
            fold_indices,
            splits,
            max_used_lag=feature_sets.max_used_lag,
            sampling_source=data,
        )
        candidate_fold_metrics = build_candidate_fold_metrics(
            candidate_metrics,
            fold_indices,
            fold_context=fold_context,
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
                "xgb_fold_context.csv": fold_context,
                "xgb_model_summary.csv": model_result.summary,
                "xgb_candidate_uplift.csv": candidate_summary,
                "xgb_candidate_fold_metrics.csv": candidate_fold_metrics,
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


@persist_xgb_execution
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
    """Run fourth-layer temporal holdout validation with fold isolation.

    The result is candidate-variable prediction-increment evidence for manual
    review only. It cannot change ``final_score``, ``ranked_features.csv``,
    Top-K, second-layer ``validation_summary``, or third-layer review results.
    Candidate uplift compares each candidate model with the same M1 baseline:
    target history plus configured control-variable history.

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
        fold_indices = {
            int(entry["fold"]): (
                entry["train_fs"].features.index,
                entry["validation_features"].index,
                entry["test_features"].index,
            )
            for entry in fold_data
        }
        actual_max_used_lag = int(fold_data[0]["train_fs"].max_used_lag)
        fold_context = build_xgb_fold_context(
            fold_indices,
            splits,
            max_used_lag=actual_max_used_lag,
            sampling_source=split_base,
        )
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
            :, list(XGB_PREDICTION_COLUMNS)
        ]

        candidate_columns = list(CandidateUpliftMetric.__dataclass_fields__)
        candidate_metrics = pd.DataFrame(candidate_metric_rows, columns=candidate_columns)
        candidate_fold_metrics = build_candidate_fold_metrics(
            candidate_metrics,
            fold_indices,
            fold_context=fold_context,
        )
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
                "xgb_fold_context.csv": fold_context,
                "xgb_model_summary.csv": summary,
                "xgb_candidate_uplift.csv": candidate_summary,
                "xgb_candidate_fold_metrics.csv": candidate_fold_metrics,
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

            # Measure after one complete seven-file commit; the final replace persists the timings.
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


def _validate_xgb_model_summary(frame: pd.DataFrame) -> bool:
    required = {
        "model_name", "mean_rmse", "median_rmse", "mean_mae", "median_mae",
        "mean_r2", "fold_count",
    }
    if frame.empty or not required.issubset(frame.columns):
        return False
    names = frame["model_name"]
    if names.isna().any() or names.astype(str).str.strip().eq("").any():
        return False
    normalized_names = names.astype(str).str.strip()
    if normalized_names.duplicated().any() or not {"M0", "M1", "M2"}.issubset(
        set(normalized_names)
    ):
        return False
    error_columns = ["mean_rmse", "median_rmse", "mean_mae", "median_mae"]
    for model_name in ("M0", "M1", "M2"):
        row = frame.loc[normalized_names.eq(model_name)].iloc[0]
        fold_count = _validated_nonnegative_integer(row["fold_count"])
        if fold_count is None or fold_count == 0:
            return False
        for column in error_columns:
            value = _finite_number(row[column])
            if value is None or value < 0:
                return False
        mean_r2 = row["mean_r2"]
        if not _is_missing_value(mean_r2) and _finite_number(mean_r2) is None:
            return False
    return True


def _validate_xgb_candidate_uplift(
    frame: pd.DataFrame, summary: dict[str, Any]
) -> bool:
    required = {
        "variable", "fold_count", "positive_rmse_fold_count", "positive_mae_fold_count",
        "positive_rmse_fold_ratio", "median_rmse_improvement_pct",
        "median_mae_improvement_pct", "mean_rmse_improvement_pct",
        "mean_mae_improvement_pct", "worst_fold_rmse_improvement_pct", "validation_status",
    }
    if not required.issubset(frame.columns):
        return False
    if "candidate_count" not in summary:
        return False
    candidate_count = _validated_nonnegative_integer(summary["candidate_count"])
    if candidate_count is None or len(frame) != candidate_count:
        return False
    if candidate_count == 0:
        return True
    variables = frame["variable"]
    statuses = frame["validation_status"]
    if (
        variables.isna().any()
        or variables.astype(str).str.strip().eq("").any()
        or statuses.isna().any()
        or statuses.astype(str).str.strip().eq("").any()
    ):
        return False
    allowed_statuses = {
        "validated_incremental_signal", "weak_incremental_value", "redundant_with_baseline",
        "unstable_out_of_time", "insufficient_features",
    }
    if not set(statuses.astype(str).str.strip()).issubset(allowed_statuses):
        return False
    if variables.duplicated().any():
        return False
    for row in frame.to_dict("records"):
        fold_count = _validated_nonnegative_integer(row["fold_count"])
        insufficient = row["validation_status"] == "insufficient_features"
        if fold_count is None or insufficient != (fold_count == 0):
            return False
        for column in ("positive_rmse_fold_count", "positive_mae_fold_count"):
            count = _validated_nonnegative_integer(row[column])
            if count is None or count > fold_count:
                return False
        ratio = row["positive_rmse_fold_ratio"]
        if insufficient and _is_missing_value(ratio):
            continue
        value = _finite_number(ratio)
        if value is None or not 0 <= value <= 1:
            return False
    return True


def _validated_nonnegative_integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        return None
    return int(number)


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_missing_value(value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
