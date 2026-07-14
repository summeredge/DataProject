from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr.xgb_validation import (
    DEFAULT_CANDIDATE_LAG_RADIUS,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_XGB_TOP_N,
    DEFAULT_XGB_PARAMS,
    XGBRegressor,
    build_expanding_time_splits,
    build_xgb_candidate_pool,
    build_xgb_feature_sets,
    run_candidate_uplift_validation,
    run_xgb_time_validation,
    validate_xgb_max_lag,
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
    if (
        isinstance(top_n, bool)
        or not isinstance(top_n, Integral)
        or not 1 <= int(top_n) <= DEFAULT_XGB_TOP_N
    ):
        return _error_result("invalid_input", "top_n must be an integer between 1 and 8")
    resolved_top_n = int(top_n)

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

    if XGBRegressor is None:
        return _error_result("missing_dependency", _MISSING_DEPENDENCY_MESSAGE)
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

    stage_started_at = time.perf_counter()
    try:
        feature_sets = build_xgb_feature_sets(
            data,
            target,
            candidate_pool,
            control_columns=control_columns,
            max_lag=resolved_max_lag,
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
