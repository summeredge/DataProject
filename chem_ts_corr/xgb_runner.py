from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr.xgb_validation import (
    DEFAULT_CANDIDATE_LAG_RADIUS,
    DEFAULT_XGB_TOP_N,
    XGBRegressor,
    build_expanding_time_splits,
    build_xgb_candidate_pool,
    build_xgb_feature_sets,
    run_candidate_uplift_validation,
    run_xgb_time_validation,
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
    input_error = _input_error(run_dir, target, data, final_review_summary, ranked_features)
    if input_error:
        return _error_result("invalid_input", input_error)
    try:
        resolved_top_n = int(top_n)
    except (TypeError, ValueError, OverflowError):
        return _error_result("invalid_input", "top_n must be an integer between 0 and 8")
    if resolved_top_n != top_n or not 0 <= resolved_top_n <= DEFAULT_XGB_TOP_N:
        return _error_result("invalid_input", "top_n must be an integer between 0 and 8")

    try:
        output_dir = Path(run_dir) / "xgb_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise OSError(f"XGB output path is not a directory: {output_dir}")
    except (OSError, TypeError, ValueError) as exc:
        return _error_result("invalid_input", f"run_dir is not writable: {exc}")

    if XGBRegressor is None:
        return _error_result("missing_dependency", _MISSING_DEPENDENCY_MESSAGE)

    try:
        resolved_max_lag = _resolve_max_lag(max_lag, final_review_summary, ranked_features)
        candidate_pool = build_xgb_candidate_pool(
            final_review_summary,
            ranked_features,
            target=target,
            top_n=resolved_top_n,
            whitelist=whitelist,
            control_columns=control_columns,
        )
        feature_sets = build_xgb_feature_sets(
            data,
            target,
            candidate_pool,
            control_columns=control_columns,
            max_lag=resolved_max_lag,
        )
        splits = build_expanding_time_splits(
            len(feature_sets.features), gap=feature_sets.max_used_lag
        )
    except (TypeError, ValueError) as exc:
        return _error_result("invalid_input", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    try:
        model_result = run_xgb_time_validation(feature_sets, splits)
        _, candidate_summary = run_candidate_uplift_validation(
            feature_sets,
            splits,
            candidate_pool,
            baseline_result=model_result,
        )
    except RuntimeError as exc:
        if "xgboost is not installed" in str(exc).lower():
            return _error_result("missing_dependency", _MISSING_DEPENDENCY_MESSAGE)
        return _error_result("failed", str(exc))
    except Exception as exc:
        return _error_result("failed", str(exc))

    try:
        paths = {name: output_dir / name for name in XGB_OUTPUT_FILES}
        summary_payload = {
            "status": "success",
            "target": target,
            "candidate_count": int(len(candidate_summary)),
            "candidate_pool_count": int(len(candidate_pool)),
            "fold_count": int(len(splits)),
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
) -> None:
    with tempfile.TemporaryDirectory(prefix=".xgb-stage-", dir=output_dir.parent) as temp_name:
        transaction_dir = Path(temp_name)
        staged_dir = transaction_dir / "staged"
        backup_dir = transaction_dir / "backup"
        staged_dir.mkdir()
        backup_dir.mkdir()

        for name in XGB_OUTPUT_FILES[:-1]:
            frames[name].to_csv(staged_dir / name, index=False, encoding="utf-8-sig")
        (staged_dir / XGB_OUTPUT_FILES[-1]).write_text(
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
        try:
            resolved = int(max_lag)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_lag must be a positive integer") from exc
        if resolved < 1 or float(max_lag) != resolved:
            raise ValueError("max_lag must be a positive integer")
        return resolved

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
    return max(1, int(np.ceil(float(positive.max()))) + DEFAULT_CANDIDATE_LAG_RADIUS)


def _error_result(status: str, message: str) -> XGBRunResult:
    return XGBRunResult(
        status=status,
        output_files=(),
        fold_metrics_path=None,
        summary_path=None,
        candidate_uplift_path=None,
        error_message=message,
    )
