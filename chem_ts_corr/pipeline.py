from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.data import (
    drop_excluded_columns,
    load_timeseries_csv,
    select_numeric_frame,
)
from chem_ts_corr.preprocess import preprocess_frame, resolve_diff_interval
from chem_ts_corr.report import write_outputs
from chem_ts_corr.service import analyze_initial_screening_branch_frame, analyze_numeric_frame


SCREENING_BRANCH_NAMES = frozenset({"raw", "processed"})
PROCESSED_BRANCH_PREPROCESS_MODES = frozenset(
    {"lowpass", "lowpass_detrend", "lowpass_diff"}
)
WORKFLOW_PREPROCESS_MODES = frozenset(
    {"raw", "lowpass", "lowpass_detrend", "lowpass_diff"}
)
CONTEXT_FILENAME = "preprocessing_context.json"
DOWNSTREAM_LOCK_FILENAME = "screening_downstream.lock"
DOWNSTREAM_LOCK_CONTENT = "downstream-locked"
CONTEXT_FIELDS = [
    "selected_preprocessing_mode",
    "active_screening_branch",
    "active_preprocessing_mode",
    "lowpass_tau_minutes",
    "requested_diff_interval_minutes",
    "effective_diff_points",
    "effective_diff_interval_minutes",
    "resample_rule",
    "branch_selection_status",
]
REQUIRED_FORMAL_SCREENING_FILES = [
    "ranked_features.csv",
    "recommended_candidates.csv",
    "causal_review_candidates.csv",
    "lag_scores.csv",
    "near_miss_candidates.csv",
    "diagnostics.csv",
    "risk_flags.csv",
    "lag_peak_quality.csv",
    "summary.md",
]
OPTIONAL_FORMAL_SCREENING_FILES = ["residual_corr_scores.csv"]
FORMAL_SCREENING_FILES = (
    REQUIRED_FORMAL_SCREENING_FILES + OPTIONAL_FORMAL_SCREENING_FILES
)
DOWNSTREAM_FORMAL_INPUT_FILES = [
    "ranked_features.csv",
    "recommended_candidates.csv",
]
MODEL_DOWNSTREAM_FORMAL_INPUT_FILES = [
    "ranked_features.csv",
    "recommended_candidates.csv",
    "risk_flags.csv",
]
CAUSAL_REVIEW_FORMAL_INPUT_FILES = [
    "ranked_features.csv",
    "recommended_candidates.csv",
    "causal_review_candidates.csv",
    "risk_flags.csv",
]
XGB_FORMAL_INPUT_FILES = [
    "ranked_features.csv",
    "final_review_summary.csv",
]


def run_analysis(config: AnalysisConfig, progress_callback=None) -> dict[str, float]:
    pipeline_started = time.perf_counter()
    _progress(progress_callback, "读取数据中")
    read_started = time.perf_counter()
    raw = load_timeseries_csv(config.input_path, config.time_column, encoding=config.encoding)
    raw = drop_excluded_columns(
        raw,
        config.excluded_columns,
        protected_columns=[
            config.time_column,
            config.target,
            config.segment_column,
            *(config.capacity_columns or []),
            *(config.residual_control_columns or []),
            *(config.force_include_variables or []),
        ],
    )
    read_data_seconds = time.perf_counter() - read_started

    analysis_started = time.perf_counter()
    tables = analyze_numeric_frame(raw, config, progress_callback=progress_callback)
    analysis_core_seconds = time.perf_counter() - analysis_started

    _progress(progress_callback, "正在写出结果文件")
    write_started = time.perf_counter()
    write_outputs(
        config.output_dir,
        target=config.target,
        ranked_features=tables.ranked_features,
        recommended_candidates=getattr(tables, "recommended_candidates", None),
        lag_scores=tables.lag_scores,
        granger_tests=tables.granger_tests,
        importance=tables.importance,
        metrics=tables.metrics,
        diagnostics=tables.diagnostics,
        residual_corr_scores=tables.residual_corr_scores,
        regime_scores=tables.regime_scores,
        risk_flags=tables.risk_flags,
        model_lift_scores=tables.model_lift_scores,
        lag_peak_quality=tables.lag_peak_quality,
        rolling_corr_scores=tables.rolling_corr_scores,
    )
    write_outputs_seconds = time.perf_counter() - write_started
    _progress(progress_callback, "分析完成")
    return {
        "read_data_seconds": read_data_seconds,
        "analysis_core_seconds": analysis_core_seconds,
        "write_outputs_seconds": write_outputs_seconds,
        "pipeline_total_seconds": time.perf_counter() - pipeline_started,
    }


def run_initial_screening_branch(
    config: AnalysisConfig,
    *,
    branch: str,
    progress_callback=None,
) -> dict[str, float]:
    """Run exactly one initial-screening branch into an isolated subdirectory.

    ``branch`` must be "raw" or "processed" and must match
    ``config.preprocess_mode``; no silent correction or extra branch is run.
    All screening outputs are written to
    ``config.output_dir / "screening_branches" / branch`` and are never
    published to the run root.
    """
    _validate_screening_branch(branch, config.preprocess_mode)
    branch_dir = config.output_dir / "screening_branches" / branch
    _clear_branch_formal_outputs(branch_dir)
    pipeline_started = time.perf_counter()
    _progress(progress_callback, "读取数据中")
    read_started = time.perf_counter()
    raw = load_timeseries_csv(config.input_path, config.time_column, encoding=config.encoding)
    raw = drop_excluded_columns(
        raw,
        config.excluded_columns,
        protected_columns=[
            config.time_column,
            config.target,
            config.segment_column,
            *(config.capacity_columns or []),
            *(config.residual_control_columns or []),
            *(config.force_include_variables or []),
        ],
    )
    read_data_seconds = time.perf_counter() - read_started

    analysis_started = time.perf_counter()
    tables = analyze_initial_screening_branch_frame(
        raw, config, progress_callback=progress_callback
    )
    analysis_core_seconds = time.perf_counter() - analysis_started

    _progress(progress_callback, "正在写出结果文件")
    write_started = time.perf_counter()
    write_outputs(
        branch_dir,
        target=config.target,
        ranked_features=tables.ranked_features,
        recommended_candidates=getattr(tables, "recommended_candidates", None),
        lag_scores=tables.lag_scores,
        granger_tests=tables.granger_tests,
        importance=tables.importance,
        metrics=tables.metrics,
        diagnostics=tables.diagnostics,
        residual_corr_scores=tables.residual_corr_scores,
        regime_scores=tables.regime_scores,
        risk_flags=tables.risk_flags,
        model_lift_scores=tables.model_lift_scores,
        lag_peak_quality=tables.lag_peak_quality,
        rolling_corr_scores=tables.rolling_corr_scores,
    )
    write_outputs_seconds = time.perf_counter() - write_started
    _progress(progress_callback, "分析完成")
    return {
        "read_data_seconds": read_data_seconds,
        "analysis_core_seconds": analysis_core_seconds,
        "write_outputs_seconds": write_outputs_seconds,
        "pipeline_total_seconds": time.perf_counter() - pipeline_started,
    }


def _validate_screening_branch(branch: str, preprocess_mode: str) -> None:
    if branch not in SCREENING_BRANCH_NAMES:
        raise ValueError(
            f"Unknown screening branch: {branch!r}; expected one of "
            f"{sorted(SCREENING_BRANCH_NAMES)}"
        )
    if branch == "raw":
        if preprocess_mode != "raw":
            raise ValueError(
                f"Raw branch requires preprocess_mode='raw', got {preprocess_mode!r}"
            )
        return
    if preprocess_mode not in PROCESSED_BRANCH_PREPROCESS_MODES:
        raise ValueError(
            "Processed branch requires preprocess_mode in "
            f"{sorted(PROCESSED_BRANCH_PREPROCESS_MODES)}, got {preprocess_mode!r}"
        )


COMPARISON_COLUMNS = [
    "variable",
    "processed_mode",
    "raw_available",
    "processed_available",
    "raw_final_score",
    "processed_final_score",
    "final_score_delta",
    "raw_rank",
    "processed_rank",
    "rank_delta",
    "raw_pearson",
    "processed_pearson",
    "raw_spearman",
    "processed_spearman",
    "raw_best_lag",
    "processed_best_lag",
    "lag_direction_changed",
    "raw_in_top_k",
    "processed_in_top_k",
    "raw_candidate",
    "processed_candidate",
    "raw_risk_tags",
    "processed_risk_tags",
]


def build_preprocessing_comparison(
    raw_ranked: pd.DataFrame,
    processed_ranked: pd.DataFrame,
    raw_candidates: pd.DataFrame,
    processed_candidates: pd.DataFrame,
    *,
    processed_mode: str,
    top_k: int,
) -> pd.DataFrame:
    """Build the frozen-field comparison of two screening branches.

    The table only mirrors already-produced branch results: it never re-runs
    screening, never computes a new score, and never recommends a branch.
    Missing evidence stays missing (NaN/NA), never 0.0/false/"".
    """
    raw_lookup = _ranked_lookup(raw_ranked)
    processed_lookup = _ranked_lookup(processed_ranked)
    raw_variables = list(raw_lookup)
    processed_variables = list(processed_lookup)
    raw_set = set(raw_variables)
    variables = [
        *raw_variables,
        *(variable for variable in processed_variables if variable not in raw_set),
    ]
    raw_candidate_set = _variable_set(raw_candidates)
    processed_candidate_set = _variable_set(processed_candidates)

    rows: list[dict[str, object]] = []
    for variable in variables:
        raw_row = raw_lookup.get(variable)
        processed_row = processed_lookup.get(variable)
        raw_available = raw_row is not None
        processed_available = processed_row is not None
        raw_final = _numeric_cell(raw_row, "final_score")
        processed_final = _numeric_cell(processed_row, "final_score")
        final_score_delta = (
            processed_final - raw_final
            if pd.notna(raw_final) and pd.notna(processed_final)
            else np.nan
        )
        raw_rank = _integer_cell(raw_row, "driver_rank")
        processed_rank = _integer_cell(processed_row, "driver_rank")
        rank_delta = (
            raw_rank - processed_rank
            if raw_rank is not None and processed_rank is not None
            else None
        )
        raw_lag = _integer_cell(raw_row, "lag")
        processed_lag = _integer_cell(processed_row, "lag")
        rows.append(
            {
                "variable": variable,
                "processed_mode": processed_mode,
                "raw_available": raw_available,
                "processed_available": processed_available,
                "raw_final_score": raw_final,
                "processed_final_score": processed_final,
                "final_score_delta": final_score_delta,
                "raw_rank": raw_rank,
                "processed_rank": processed_rank,
                "rank_delta": rank_delta,
                "raw_pearson": _numeric_cell(raw_row, "pearson"),
                "processed_pearson": _numeric_cell(processed_row, "pearson"),
                "raw_spearman": _numeric_cell(raw_row, "spearman"),
                "processed_spearman": _numeric_cell(processed_row, "spearman"),
                "raw_best_lag": raw_lag,
                "processed_best_lag": processed_lag,
                "lag_direction_changed": _lag_direction_changed(
                    raw_lag, processed_lag
                ),
                "raw_in_top_k": (
                    raw_available and raw_rank is not None and raw_rank <= top_k
                ),
                "processed_in_top_k": (
                    processed_available
                    and processed_rank is not None
                    and processed_rank <= top_k
                ),
                "raw_candidate": variable in raw_candidate_set,
                "processed_candidate": variable in processed_candidate_set,
                "raw_risk_tags": _risk_tags_cell(raw_row),
                "processed_risk_tags": _risk_tags_cell(processed_row),
            }
        )

    comparison = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    for column in (
        "raw_rank",
        "processed_rank",
        "rank_delta",
        "raw_best_lag",
        "processed_best_lag",
    ):
        comparison[column] = comparison[column].astype("Int64")
    comparison["lag_direction_changed"] = comparison[
        "lag_direction_changed"
    ].astype("boolean")
    return comparison


def _ranked_lookup(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if frame is None or frame.empty or "variable" not in frame.columns:
        return {}
    return {str(row["variable"]): row for row in frame.to_dict(orient="records")}


def _variable_set(frame: pd.DataFrame) -> set[str]:
    if frame is None or frame.empty or "variable" not in frame.columns:
        return set()
    return {str(value) for value in frame["variable"].tolist()}


def _numeric_cell(row: dict[str, object] | None, column: str) -> float:
    if row is None:
        return np.nan
    value = row.get(column, np.nan)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def _integer_cell(row: dict[str, object] | None, column: str) -> int | None:
    if row is None:
        return None
    value = row.get(column, np.nan)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _lag_direction_changed(raw_lag: int | None, processed_lag: int | None) -> object:
    if raw_lag is None or processed_lag is None:
        return pd.NA
    return bool(np.sign(raw_lag) != np.sign(processed_lag))


def _risk_tags_cell(row: dict[str, object] | None) -> object:
    if row is None:
        return pd.NA
    value = row.get("risk_flags", pd.NA)
    return pd.NA if pd.isna(value) else str(value)


def run_initial_screening_comparison(
    config: AnalysisConfig,
    *,
    progress_callback=None,
) -> dict[str, object]:
    """Run raw + selected processed branches and write the comparison CSV.

    Only non-raw processed modes are allowed and the caller's config is never
    modified. ``preprocessing_comparison.csv`` is written only after both
    branches succeed and never recommends a branch. Any previous comparison
    file is cleared before branches start so a failed re-run never leaves
    stale results behind.
    """
    _validate_comparison_mode(config.preprocess_mode)
    comparison_path = config.output_dir / "preprocessing_comparison.csv"
    comparison_path.unlink(missing_ok=True)
    raw_config = replace(config, preprocess_mode="raw")
    processed_config = replace(config, preprocess_mode=config.preprocess_mode)
    raw_timings = run_initial_screening_branch(
        raw_config, branch="raw", progress_callback=progress_callback
    )
    processed_timings = run_initial_screening_branch(
        processed_config, branch="processed", progress_callback=progress_callback
    )
    branches_root = config.output_dir / "screening_branches"
    comparison = build_preprocessing_comparison(
        pd.read_csv(
            branches_root / "raw" / "ranked_features.csv",
            encoding="utf-8-sig",
            keep_default_na=False,
        ),
        pd.read_csv(
            branches_root / "processed" / "ranked_features.csv",
            encoding="utf-8-sig",
            keep_default_na=False,
        ),
        pd.read_csv(
            branches_root / "raw" / "recommended_candidates.csv", encoding="utf-8-sig"
        ),
        pd.read_csv(
            branches_root / "processed" / "recommended_candidates.csv",
            encoding="utf-8-sig",
        ),
        processed_mode=config.preprocess_mode,
        top_k=config.top_k,
    )
    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN",
    )
    return {
        "raw": raw_timings,
        "processed": processed_timings,
        "comparison_path": comparison_path,
    }


def _validate_comparison_mode(preprocess_mode: str) -> None:
    if preprocess_mode not in PROCESSED_BRANCH_PREPROCESS_MODES:
        raise ValueError(
            "run_initial_screening_comparison requires preprocess_mode in "
            f"{sorted(PROCESSED_BRANCH_PREPROCESS_MODES)}, got {preprocess_mode!r}"
        )


def run_initial_screening_workflow(
    config: AnalysisConfig,
    *,
    progress_callback=None,
) -> dict[str, object]:
    """Run the unified initial-screening workflow with branch state.

    ``raw`` runs only the raw branch, transactionally promotes it to the
    formal root and writes a ``not_required`` context. Each ``lowpass*`` mode
    runs raw + selected processed branches and the comparison CSV via
    ``run_initial_screening_comparison()``, then writes an
    ``awaiting_confirmation`` context while leaving the formal root empty.
    The mode is validated and locked runs are rejected before any previous
    formal state is cleared.
    """
    _validate_workflow_mode(config.preprocess_mode)
    _reject_locked_run(config.output_dir)
    _clear_previous_formal_state(
        config.output_dir, preprocess_mode=config.preprocess_mode
    )
    if config.preprocess_mode == "raw":
        timings = run_initial_screening_branch(
            config, branch="raw", progress_callback=progress_callback
        )
        context = _build_preprocessing_context(
            config,
            active_screening_branch="raw",
            active_preprocessing_mode="raw",
            branch_selection_status="not_required",
        )
        _promote_screening_branch(
            config.output_dir, branch="raw", new_context=context
        )
        return {
            "branch": "raw",
            "timings": timings,
            "context_path": Path(config.output_dir) / CONTEXT_FILENAME,
        }
    comparison = run_initial_screening_comparison(
        config, progress_callback=progress_callback
    )
    context = _build_preprocessing_context(
        config,
        active_screening_branch=None,
        active_preprocessing_mode=None,
        branch_selection_status="awaiting_confirmation",
    )
    context_path = _write_context(config.output_dir, context)
    return {**comparison, "context_path": context_path}


def confirm_initial_screening_branch(
    run_dir: Path,
    *,
    branch: str,
) -> None:
    """Publish an existing branch as the formal screening result.

    Confirmation never re-runs screening: it reads the existing
    ``preprocessing_context.json``, validates the selected branch outputs and
    transactionally promotes them to the run root. Confirming the active
    branch is an idempotent no-op; switching to the other branch is allowed
    only until a downstream stage has started.
    """
    run_dir = Path(run_dir)
    if branch not in SCREENING_BRANCH_NAMES:
        raise ValueError(
            f"Unknown screening branch: {branch!r}; expected one of "
            f"{sorted(SCREENING_BRANCH_NAMES)}"
        )
    context = _read_preprocessing_context(run_dir)
    if (run_dir / DOWNSTREAM_LOCK_FILENAME).exists():
        if context["active_screening_branch"] == branch:
            return
        raise ValueError(
            "initial_screening_branch_locked: cannot switch the active "
            "screening branch after a downstream stage has started"
        )

    status = context["branch_selection_status"]
    if status == "confirmed":
        if context["active_screening_branch"] == branch:
            return
        active_mode = _active_mode_for_branch(context, branch)
        _promote_screening_branch(
            run_dir,
            branch=branch,
            new_context=_confirmed_context(context, branch, active_mode),
        )
        return
    if status == "not_required":
        if branch == "raw":
            return
        raise ValueError(
            "initial_screening_branch_unavailable: the processed branch was "
            "not run in the raw-only workflow"
        )
    if status == "awaiting_confirmation":
        active_mode = _active_mode_for_branch(context, branch)
        _promote_screening_branch(
            run_dir,
            branch=branch,
            new_context=_confirmed_context(context, branch, active_mode),
        )
        return
    raise ValueError(
        f"initial_screening_context_invalid: unknown branch_selection_status "
        f"{status!r}"
    )


def begin_downstream_stage(run_dir: Path) -> None:
    """Gate a downstream stage behind a confirmed screening branch.

    Reads ``preprocessing_context.json`` and creates
    ``screening_downstream.lock`` on first success. The lock only prevents
    switching the formal screening branch after downstream has started; it
    never enters scoring, CSV, reports or the comparison.
    """
    run_dir = Path(run_dir)
    context = _read_preprocessing_context(run_dir)
    status = context["branch_selection_status"]
    if status == "awaiting_confirmation":
        raise ValueError(
            "initial_screening_branch_not_confirmed: a downstream stage "
            "cannot start until raw or processed is confirmed"
        )
    if status not in {"confirmed", "not_required"}:
        raise ValueError(
            f"initial_screening_context_invalid: unknown branch_selection_status "
            f"{status!r}"
        )
    (run_dir / DOWNSTREAM_LOCK_FILENAME).write_text(
        DOWNSTREAM_LOCK_CONTENT, encoding="utf-8"
    )


def _validate_workflow_mode(preprocess_mode: str) -> None:
    if preprocess_mode not in WORKFLOW_PREPROCESS_MODES:
        raise ValueError(
            "run_initial_screening_workflow requires preprocess_mode in "
            f"{sorted(WORKFLOW_PREPROCESS_MODES)}, got {preprocess_mode!r}"
        )


def _reject_locked_run(run_dir: Path) -> None:
    if (Path(run_dir) / DOWNSTREAM_LOCK_FILENAME).exists():
        raise ValueError(
            "initial_screening_run_locked: the screening branch is locked "
            "because a downstream stage has started; use a new analysis/run "
            "directory"
        )


def _clear_previous_formal_state(run_dir: Path, *, preprocess_mode: str) -> None:
    run_dir = Path(run_dir)
    for name in FORMAL_SCREENING_FILES:
        (run_dir / name).unlink(missing_ok=True)
    (run_dir / CONTEXT_FILENAME).unlink(missing_ok=True)
    if preprocess_mode == "raw":
        (run_dir / "preprocessing_comparison.csv").unlink(missing_ok=True)


def _clear_branch_formal_outputs(branch_dir: Path) -> None:
    """Remove a branch's previous formal screening outputs before a re-run.

    Only files in the formal screening set inside the current branch
    directory are removed; the branch directory itself, other branches and
    run-root files are never touched.
    """
    for name in FORMAL_SCREENING_FILES:
        (branch_dir / name).unlink(missing_ok=True)


def _build_preprocessing_context(
    config: AnalysisConfig,
    *,
    active_screening_branch: str | None,
    active_preprocessing_mode: str | None,
    branch_selection_status: str,
) -> dict[str, object]:
    mode = config.preprocess_mode
    requested_diff_interval_minutes = None
    effective_diff_points = None
    effective_diff_interval_minutes = None
    if mode == "lowpass_diff":
        if config.diff_interval_minutes is not None:
            requested_diff_interval_minutes = float(config.diff_interval_minutes)
        effective_diff_points, effective_diff_interval_minutes = (
            _effective_diff_params(config)
        )
    return {
        "selected_preprocessing_mode": mode,
        "active_screening_branch": active_screening_branch,
        "active_preprocessing_mode": active_preprocessing_mode,
        "lowpass_tau_minutes": (
            float(config.lowpass_tau_minutes)
            if mode in PROCESSED_BRANCH_PREPROCESS_MODES
            else None
        ),
        "requested_diff_interval_minutes": requested_diff_interval_minutes,
        "effective_diff_points": effective_diff_points,
        "effective_diff_interval_minutes": effective_diff_interval_minutes,
        "resample_rule": config.resample_rule,
        "branch_selection_status": branch_selection_status,
    }


def _effective_diff_params(config: AnalysisConfig) -> tuple[int, float]:
    frame = _load_cleaned_frame(config)
    effective_points, effective_interval_minutes = resolve_diff_interval(
        frame,
        config.diff_interval_minutes,
    )
    return int(effective_points), float(effective_interval_minutes)


def _load_cleaned_frame(config: AnalysisConfig) -> pd.DataFrame:
    raw = load_timeseries_csv(
        config.input_path, config.time_column, encoding=config.encoding
    )
    raw = drop_excluded_columns(
        raw,
        config.excluded_columns,
        protected_columns=[
            config.time_column,
            config.target,
            config.segment_column,
            *(config.capacity_columns or []),
            *(config.residual_control_columns or []),
            *(config.force_include_variables or []),
        ],
    )
    numeric = select_numeric_frame(raw, config.target)
    protected = [
        config.target,
        config.segment_column,
        *(config.capacity_columns or []),
        *(config.residual_control_columns or []),
        *(config.force_include_variables or []),
    ]
    return preprocess_frame(
        numeric,
        config.target,
        config.resample_rule,
        config.min_valid_ratio,
        protected_columns=[column for column in protected if column],
        max_interpolate_gap_points=config.max_interpolate_gap_points,
        interpolate_limit_area=config.interpolate_limit_area,
    )


def _write_context(run_dir: Path, context: dict[str, object]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / CONTEXT_FILENAME
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=".preprocessing_context_", suffix=".tmp", dir=str(run_dir)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(context, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return target


def _read_preprocessing_context(run_dir: Path) -> dict[str, object]:
    path = Path(run_dir) / CONTEXT_FILENAME
    if not path.exists():
        raise ValueError(f"initial_screening_context_missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"initial_screening_context_invalid: {path}"
        ) from exc
    _validate_context(data)
    return data


def _validate_context(data: object) -> None:
    prefix = "initial_screening_context_invalid: preprocessing_context.json"
    if not isinstance(data, dict):
        raise ValueError(prefix)
    missing = [field for field in CONTEXT_FIELDS if field not in data]
    if missing:
        raise ValueError(f"{prefix} missing fields {missing}")

    status = data["branch_selection_status"]
    selected = data["selected_preprocessing_mode"]
    active_branch = data["active_screening_branch"]
    active_mode = data["active_preprocessing_mode"]
    if status not in {"not_required", "awaiting_confirmation", "confirmed"}:
        raise ValueError(f"{prefix} unknown branch_selection_status {status!r}")
    if selected not in WORKFLOW_PREPROCESS_MODES:
        raise ValueError(
            f"{prefix} unknown selected_preprocessing_mode {selected!r}"
        )
    if active_branch is not None and active_branch not in SCREENING_BRANCH_NAMES:
        raise ValueError(
            f"{prefix} unknown active_screening_branch {active_branch!r}"
        )
    if active_mode is not None and active_mode not in WORKFLOW_PREPROCESS_MODES:
        raise ValueError(
            f"{prefix} unknown active_preprocessing_mode {active_mode!r}"
        )
    if status == "awaiting_confirmation":
        if (
            selected not in PROCESSED_BRANCH_PREPROCESS_MODES
            or active_branch is not None
            or active_mode is not None
        ):
            raise ValueError(
                f"{prefix} awaiting_confirmation requires a lowpass* selected "
                "mode and null active fields"
            )
    elif status == "not_required":
        if selected != "raw" or active_branch != "raw" or active_mode != "raw":
            raise ValueError(
                f"{prefix} not_required must describe the raw-only workflow"
            )
    elif active_branch is None or active_mode is None:
        raise ValueError(f"{prefix} confirmed requires active branch and mode")
    elif active_branch == "raw" and active_mode != "raw":
        raise ValueError(f"{prefix} confirmed raw branch must use raw mode")
    elif active_branch == "processed" and (
        selected not in PROCESSED_BRANCH_PREPROCESS_MODES
        or active_mode != selected
    ):
        raise ValueError(
            f"{prefix} confirmed processed branch must use the selected "
            "lowpass* mode"
        )

    for field in (
        "lowpass_tau_minutes",
        "requested_diff_interval_minutes",
        "effective_diff_interval_minutes",
    ):
        value = data[field]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not float(value) > 0
        ):
            raise ValueError(f"{prefix} field {field} must be null or positive")
    value = data["effective_diff_points"]
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not float(value) > 0
        or float(value) != int(float(value))
    ):
        raise ValueError(
            f"{prefix} effective_diff_points must be null or a positive integer"
        )
    resample_rule = data["resample_rule"]
    if resample_rule is not None and not isinstance(resample_rule, str):
        raise ValueError(f"{prefix} resample_rule must be null or a string")


def _active_mode_for_branch(
    context: dict[str, object], branch: str
) -> str:
    if branch == "raw":
        return "raw"
    selected = context["selected_preprocessing_mode"]
    if selected not in PROCESSED_BRANCH_PREPROCESS_MODES:
        raise ValueError(
            "initial_screening_context_invalid: processed confirmation "
            f"requires a lowpass* selected mode, got {selected!r}"
        )
    return selected


def _confirmed_context(
    context: dict[str, object],
    branch: str,
    active_mode: str,
) -> dict[str, object]:
    updated = dict(context)
    updated["active_screening_branch"] = branch
    updated["active_preprocessing_mode"] = active_mode
    updated["branch_selection_status"] = "confirmed"
    return updated


def _validate_branch_output_complete(run_dir: Path, branch: str) -> None:
    branch_dir = Path(run_dir) / "screening_branches" / branch
    missing = [
        name
        for name in REQUIRED_FORMAL_SCREENING_FILES
        if not (branch_dir / name).exists()
    ]
    if missing:
        raise ValueError(
            "initial_screening_branch_output_incomplete: "
            f"branch={branch!r} missing {missing}"
        )


def _promote_screening_branch(
    run_dir: Path,
    *,
    branch: str,
    new_context: dict[str, object],
) -> None:
    """Transactionally publish one branch as the formal screening result.

    The branch is validated first, current root formal files are backed up,
    files are replaced via ``os.replace``, optional-file residue is removed,
    and only then is ``preprocessing_context.json`` updated. Any failure
    restores the previous root files and context so a mixed Raw/Processed
    root can never survive.
    """
    run_dir = Path(run_dir)
    _validate_branch_output_complete(run_dir, branch)
    branch_dir = run_dir / "screening_branches" / branch
    staged_names = [
        name
        for name in FORMAL_SCREENING_FILES
        if (branch_dir / name).exists()
    ]
    context_path = run_dir / CONTEXT_FILENAME
    original_context_bytes = (
        context_path.read_bytes() if context_path.exists() else None
    )
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".screening_promote_", dir=str(run_dir))
    )
    backup_dir = Path(
        tempfile.mkdtemp(prefix=".screening_backup_", dir=str(run_dir))
    )
    try:
        for name in staged_names:
            shutil.copy2(branch_dir / name, staging_dir / name)
        for name in FORMAL_SCREENING_FILES:
            root_file = run_dir / name
            if root_file.exists():
                shutil.copy2(root_file, backup_dir / name)
        for name in staged_names:
            os.replace(staging_dir / name, run_dir / name)
        for name in FORMAL_SCREENING_FILES:
            if name not in staged_names:
                (run_dir / name).unlink(missing_ok=True)
        _write_context(run_dir, new_context)
    except Exception:
        _restore_promotion(
            run_dir,
            backup_dir,
            staged_names=staged_names,
            original_context_bytes=original_context_bytes,
        )
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)


def _restore_promotion(
    run_dir: Path,
    backup_dir: Path,
    *,
    staged_names: list[str],
    original_context_bytes: bytes | None,
) -> None:
    for name in FORMAL_SCREENING_FILES:
        backup_file = backup_dir / name
        if backup_file.exists():
            shutil.copy2(backup_file, run_dir / name)
        elif name in staged_names:
            (run_dir / name).unlink(missing_ok=True)
    context_path = run_dir / CONTEXT_FILENAME
    if original_context_bytes is None:
        context_path.unlink(missing_ok=True)
        return
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=".preprocessing_context_restore_", suffix=".tmp", dir=str(run_dir)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(original_context_bytes)
        os.replace(tmp_name, context_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _progress(progress_callback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def prepare_downstream_analysis_context(
    run_dir: Path,
    *,
    base_config: AnalysisConfig | None = None,
    required_formal_files: list[str] | None = None,
) -> tuple[dict[str, object], AnalysisConfig]:
    """Resolve the frozen formal branch and preprocessing config.

    Reads ``preprocessing_context.json`` through the PR-8 reader (missing /
    invalid contexts raise their frozen errors), rejects
    ``awaiting_confirmation``, validates that the promoted formal root
    screening inputs exist, and builds a downstream ``AnalysisConfig`` whose
    preprocessing fields come exclusively from the context. Only then does it
    open the downstream stage via ``begin_downstream_stage()``, which creates
    ``screening_downstream.lock`` on first success. Returns the context and
    the downstream config.
    """
    run_dir = Path(run_dir)
    context = _read_preprocessing_context(run_dir)
    status = context["branch_selection_status"]
    if status == "awaiting_confirmation":
        raise ValueError(
            "initial_screening_branch_not_confirmed: a downstream stage "
            "cannot start until raw or processed is confirmed"
        )
    if status not in {"confirmed", "not_required"}:
        raise ValueError(
            "initial_screening_context_invalid: unknown "
            f"branch_selection_status {status!r}"
        )
    missing = _missing_formal_screening_outputs(
        run_dir, required_formal_files=required_formal_files
    )
    if missing:
        raise ValueError(
            "initial_screening_formal_output_missing: "
            f"{run_dir} is missing formal screening inputs {missing}"
        )
    base = _resolve_base_config(run_dir, base_config)
    downstream_config = _downstream_config_from_context(base, context)
    begin_downstream_stage(run_dir)
    return context, downstream_config


def run_enhanced_screening_for_active_branch(
    run_dir: Path,
    *,
    base_config: AnalysisConfig | None = None,
    progress_callback=None,
) -> dict[str, object]:
    """Run enhanced screening on the active formal screening branch.

    The formal ``preprocessing_context.json`` is the only source of truth for
    the branch and its preprocessing parameters; the existing enhanced
    screening helpers are reused unchanged and the promoted root screening
    files are never rewritten.
    """
    run_dir = Path(run_dir)
    context, config = prepare_downstream_analysis_context(
        run_dir,
        base_config=base_config,
        required_formal_files=DOWNSTREAM_FORMAL_INPUT_FILES,
    )
    _progress(progress_callback, "正在运行增强筛选（正式 branch）")
    from chem_ts_corr import web as web_module
    from chem_ts_corr.screening import (
        model_lift_scores,
        prepare_best_lag_evidence,
        rolling_corr_scores,
    )

    ranked = pd.read_csv(run_dir / "ranked_features.csv", encoding="utf-8-sig")
    variables = web_module._secondary_variables_from_ranked(ranked, config)
    web_module._save_secondary_candidate_context(run_dir, variables)
    if not variables:
        raise ValueError("ranked_features.csv 中没有可运行增强筛选的候选变量")

    scaled = web_module._scaled_frame_for_secondary_causal(config)
    target_mask = web_module._target_segment_mask(scaled)
    variables = [
        variable
        for variable in variables
        if variable in scaled.columns and variable != config.target
    ]
    if not variables:
        raise ValueError(
            "二次验证候选变量在预处理后的数据中不存在，请检查 TopK、白名单和二次验证重采样设置。"
        )

    best_lag_evidence, _ = prepare_best_lag_evidence(
        scaled,
        config.target,
        variables,
        config.max_lag,
        ranked=ranked,
        allow_ranked_reuse=False,
        target_mask=target_mask,
    )
    best_lags = {
        variable: evidence["best_lag"]
        for variable, evidence in best_lag_evidence.items()
        if evidence["best_lag"] is not None
    }
    lift = model_lift_scores(
        scaled,
        config.target,
        variables,
        config.max_lag,
        best_lags=best_lags,
        target_mask=target_mask,
    )
    rolling = rolling_corr_scores(
        scaled,
        config.target,
        variables,
        config.max_lag,
        best_lag_evidence=best_lag_evidence,
        target_mask=target_mask,
    )
    enhanced = web_module._enhanced_validation_summary(ranked, lift, rolling)

    lift.to_csv(run_dir / "model_lift_scores.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(
        run_dir / "rolling_corr_scores.csv", index=False, encoding="utf-8-sig"
    )
    enhanced.to_csv(
        run_dir / "enhanced_validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    _progress(progress_callback, "增强筛选完成")
    return {
        "run_dir": run_dir,
        "active_screening_branch": context["active_screening_branch"],
        "active_preprocessing_mode": context["active_preprocessing_mode"],
        "config": config,
        "model_lift_scores_path": run_dir / "model_lift_scores.csv",
        "rolling_corr_scores_path": run_dir / "rolling_corr_scores.csv",
        "enhanced_validation_summary_path": run_dir / "enhanced_validation_summary.csv",
    }


def run_granger_for_active_branch(
    run_dir: Path,
    *,
    base_config: AnalysisConfig | None = None,
    progress_callback=None,
) -> dict[str, object]:
    """Run ordinary bivariate Granger on the active formal screening branch.

    The formal ``preprocessing_context.json`` is the only source of truth for
    the branch and its preprocessing parameters; the existing ordinary Granger
    service is reused unchanged and the promoted root screening files are
    never rewritten.
    """
    run_dir = Path(run_dir)
    context, config = prepare_downstream_analysis_context(
        run_dir,
        base_config=base_config,
        required_formal_files=DOWNSTREAM_FORMAL_INPUT_FILES,
    )
    _progress(progress_callback, "正在运行普通 Granger（正式 branch）")
    from chem_ts_corr import causality as causality_module
    from chem_ts_corr import web as web_module

    ranked = pd.read_csv(run_dir / "ranked_features.csv", encoding="utf-8-sig")
    variables = web_module._secondary_variables_from_ranked(ranked, config)
    web_module._save_secondary_candidate_context(run_dir, variables)
    if not variables:
        raise ValueError("ranked_features.csv 中没有可运行 Granger 的候选变量")

    scaled = web_module._scaled_frame_for_secondary_causal(config)
    target_mask = web_module._target_segment_mask(scaled)
    variables = [
        variable
        for variable in variables
        if variable in scaled.columns and variable != config.target
    ]
    if not variables:
        raise ValueError(
            "二次验证候选变量在预处理后的数据中不存在，请检查 TopK、白名单和二次验证重采样设置。"
        )

    granger = causality_module.run_granger_tests(
        scaled,
        target=config.target,
        variables=variables,
        maxlag=max(1, config.max_lag),
        target_mask=target_mask,
    )
    granger.to_csv(run_dir / "granger_tests.csv", index=False, encoding="utf-8-sig")
    _progress(progress_callback, "普通 Granger 完成")
    return {
        "run_dir": run_dir,
        "active_screening_branch": context["active_screening_branch"],
        "active_preprocessing_mode": context["active_preprocessing_mode"],
        "config": config,
        "granger_tests_path": run_dir / "granger_tests.csv",
    }


def run_model_for_active_branch(
    run_dir: Path,
    *,
    base_config: AnalysisConfig | None = None,
    progress_callback=None,
) -> dict[str, object]:
    """Run RF / SHAP model explanation on the active formal screening branch.

    The formal ``preprocessing_context.json`` is the only source of truth for
    the branch and its preprocessing parameters; the existing model helpers are
    reused unchanged and the promoted root screening files are never rewritten.
    This stage only produces the three model-explanation outputs and never
    starts enhanced screening, Granger, conditional Granger, causal/final
    review or XGBoost.
    """
    run_dir = Path(run_dir)
    context, config = prepare_downstream_analysis_context(
        run_dir,
        base_config=base_config,
        required_formal_files=MODEL_DOWNSTREAM_FORMAL_INPUT_FILES,
    )
    _progress(progress_callback, "正在运行模型解释（正式 branch）")
    from chem_ts_corr import web as web_module
    from chem_ts_corr.model_discovery import (
        build_model_discovered_candidates,
        build_model_variable_importance,
    )
    from chem_ts_corr.modeling import fit_explainable_model

    ranked = pd.read_csv(run_dir / "ranked_features.csv", encoding="utf-8-sig")
    variables = web_module._secondary_variables_from_ranked(ranked, config)
    if not variables:
        raise ValueError("ranked_features.csv 中没有可运行模型解释的候选变量")

    scaled = web_module._scaled_frame_for_secondary_causal(config)
    target_mask = web_module._target_segment_mask(scaled)
    variables = [
        variable
        for variable in variables
        if variable in scaled.columns and variable != config.target
    ]
    if not variables:
        raise ValueError(
            "二次验证候选变量在预处理后的数据中不存在，请检查 TopK、白名单和二次验证重采样设置。"
        )

    best_lags = _causal_best_lags(
        scaled,
        config.target,
        variables,
        config.max_lag,
        target_mask=target_mask,
    )

    importance, metrics = fit_explainable_model(
        scaled,
        target=config.target,
        max_lag=config.max_lag,
        candidate_variables=variables,
        max_features=config.max_model_features,
        random_state=config.random_state,
        best_lags=best_lags,
        lag_mode="best_only",
        target_mask=target_mask,
    )
    risk = pd.read_csv(run_dir / "risk_flags.csv", encoding="utf-8-sig")
    model_variable_importance = build_model_variable_importance(
        importance, ranked, risk_flags=risk
    )
    model_discovered = build_model_discovered_candidates(
        importance,
        ranked,
        risk_flags=risk,
        screening_top_n=config.top_k,
        max_lag=config.max_lag,
    )
    importance.to_csv(
        run_dir / "shap_or_importance.csv", index=False, encoding="utf-8-sig"
    )
    model_variable_importance.to_csv(
        run_dir / "model_variable_importance.csv", index=False, encoding="utf-8-sig"
    )
    model_discovered.to_csv(
        run_dir / "model_discovered_candidates.csv", index=False, encoding="utf-8-sig"
    )
    _progress(progress_callback, "模型解释完成")
    return {
        "run_dir": run_dir,
        "active_screening_branch": context["active_screening_branch"],
        "active_preprocessing_mode": context["active_preprocessing_mode"],
        "config": config,
        "shap_or_importance_path": run_dir / "shap_or_importance.csv",
        "model_variable_importance_path": run_dir / "model_variable_importance.csv",
        "model_discovered_candidates_path": run_dir / "model_discovered_candidates.csv",
        "model_metrics": metrics,
    }


def run_causal_review_for_active_branch(
    run_dir: Path,
    *,
    base_config: AnalysisConfig | None = None,
    control_columns: list[str] | None = None,
    maxlag: int | None = None,
    min_rows: int = 60,
    top_n: int | None = None,
    conditional_lag_mode: str = "ranked_window",
    conditional_lag_window: int = 5,
    conditional_fallback_maxlag: int = 24,
    conditional_baseline_maxlag: int | None = 24,
    progress_callback=None,
) -> dict[str, object]:
    """Run the three-tier causal review on the active formal screening branch.

    The formal ``preprocessing_context.json`` is the only source of truth for
    the branch and its preprocessing parameters. The existing
    ``run_causal_review_stage()`` is reused unchanged and the formal review
    candidates come exclusively from the promoted root
    ``causal_review_candidates.csv``; the promoted root screening files are
    never rewritten.
    """
    run_dir = Path(run_dir)
    context, config = prepare_downstream_analysis_context(
        run_dir,
        base_config=base_config,
        required_formal_files=CAUSAL_REVIEW_FORMAL_INPUT_FILES,
    )
    _progress(progress_callback, "正在运行三级复核（正式 branch）")
    from chem_ts_corr import web as web_module
    from chem_ts_corr.causal_review_runner import run_causal_review_stage

    ranked = pd.read_csv(run_dir / "ranked_features.csv", encoding="utf-8-sig")
    causal_candidates = pd.read_csv(
        run_dir / "causal_review_candidates.csv", encoding="utf-8-sig"
    )
    risk = pd.read_csv(run_dir / "risk_flags.csv", encoding="utf-8-sig")
    resolved_control_columns = (
        control_columns
        if control_columns is not None
        else config.residual_control_columns
        or config.capacity_columns
        or []
    )
    web_module._ensure_columns_not_excluded(
        config, resolved_control_columns, "三层复核控制列"
    )
    resolved_maxlag = (
        config.resolved_granger_maxlag() if maxlag is None else maxlag
    )
    if control_columns is not None:
        scaled = web_module._scaled_frame_for_secondary_causal(
            config, protected_columns=resolved_control_columns
        )
    else:
        scaled = web_module._scaled_frame_for_secondary_causal(config)
    target_mask = web_module._target_segment_mask(scaled)
    causal_ranked = _causal_ranked_lag_view(
        scaled,
        ranked,
        target=config.target,
        max_lag=resolved_maxlag,
        target_mask=target_mask,
    )
    result = run_causal_review_stage(
        frame=scaled,
        target=config.target,
        ranked_features=causal_ranked,
        causal_review_candidates=causal_candidates,
        risk_flags=risk,
        output_dir=run_dir,
        control_columns=resolved_control_columns,
        maxlag=resolved_maxlag,
        min_rows=min_rows,
        top_n=top_n,
        conditional_lag_mode=conditional_lag_mode,
        conditional_lag_window=conditional_lag_window,
        conditional_fallback_maxlag=conditional_fallback_maxlag,
        conditional_baseline_maxlag=conditional_baseline_maxlag,
        target_mask=target_mask,
        prefer_ranked_lag=True,
    )
    result["conditional_granger_scores"].to_csv(
        run_dir / "conditional_granger_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result["causal_review_report"].to_csv(
        run_dir / "causal_review_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result["causal_review_evidence"].to_csv(
        run_dir / "causal_review_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result["final_review_summary"].to_csv(
        run_dir / "final_review_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _progress(progress_callback, "三级复核完成")
    return {
        "run_dir": run_dir,
        "active_screening_branch": context["active_screening_branch"],
        "active_preprocessing_mode": context["active_preprocessing_mode"],
        "config": config,
        "conditional_granger_scores_path": run_dir / "conditional_granger_scores.csv",
        "causal_review_report_path": run_dir / "causal_review_report.csv",
        "causal_review_evidence_path": run_dir / "causal_review_evidence.csv",
        "final_review_summary_path": run_dir / "final_review_summary.csv",
    }


def run_xgb_for_active_branch(
    run_dir: Path,
    *,
    base_config: AnalysisConfig | None = None,
    control_columns: list[str] | None = None,
    whitelist: list[str] | None = None,
    top_n: int | None = None,
    max_lag: int | None = None,
    progress_callback=None,
) -> dict[str, object]:
    """Run the fold-safe XGB validation on the active formal screening branch.

    The formal ``preprocessing_context.json`` is the only source of truth for
    the branch and its preprocessing parameters. XGB requires the promoted
    ``ranked_features.csv`` and the PR-11 ``final_review_summary.csv``; it
    never re-runs the three-tier review and never reads branch directories or
    preprocessing comparisons. The existing fold-safe backend keeps
    lowpass/detrend/diff/forward-fill state isolated per fold.
    """
    run_dir = Path(run_dir)
    context, config = prepare_downstream_analysis_context(
        run_dir,
        base_config=base_config,
        required_formal_files=XGB_FORMAL_INPUT_FILES,
    )
    _progress(progress_callback, "正在运行 XGB 四级验证（正式 branch）")
    from chem_ts_corr import web as web_module
    from chem_ts_corr.xgb_runner import run_xgb_validation_fold_safe

    ranked = pd.read_csv(run_dir / "ranked_features.csv", encoding="utf-8-sig")
    final_review_summary = pd.read_csv(
        run_dir / "final_review_summary.csv", encoding="utf-8-sig"
    )

    resolved_control_columns = (
        control_columns
        if control_columns is not None
        else config.residual_control_columns
        or config.capacity_columns
        or []
    )
    resolved_whitelist = (
        whitelist if whitelist is not None else list(config.xgb_whitelist or [])
    )
    web_module._ensure_columns_not_excluded(
        config, resolved_control_columns, "XGBoost 控制列"
    )
    web_module._ensure_columns_not_excluded(
        config, resolved_whitelist, "XGBoost 白名单"
    )

    resolved_top_n = config.xgb_top_n if top_n is None else top_n
    resolved_max_lag = config.xgb_max_lag if max_lag is None else max_lag

    numeric = web_module._numeric_frame(
        config, protected_columns=resolved_control_columns
    )

    result = run_xgb_validation_fold_safe(
        run_dir=run_dir,
        target=config.target,
        data=numeric,
        final_review_summary=final_review_summary,
        ranked_features=ranked,
        control_columns=resolved_control_columns,
        whitelist=resolved_whitelist,
        top_n=resolved_top_n,
        max_lag=resolved_max_lag,
        preprocess_mode=config.preprocess_mode,
        lowpass_tau_minutes=config.lowpass_tau_minutes,
        diff_interval_minutes=config.diff_interval_minutes,
        detrend_window=config.detrend_window,
        resample_rule=config.resample_rule,
        max_interpolate_gap_points=config.max_interpolate_gap_points,
        segment_column=config.segment_column,
        segment_mode=config.segment_mode,
        segment_min=config.segment_min,
        segment_max=config.segment_max,
    )

    _progress(progress_callback, "XGB 四级验证完成")
    return {
        "run_dir": run_dir,
        "active_screening_branch": context["active_screening_branch"],
        "active_preprocessing_mode": context["active_preprocessing_mode"],
        "config": config,
        "status": result.status,
        "error_message": result.error_message,
        "output_files": result.output_files,
        "fold_metrics_path": result.fold_metrics_path,
        "summary_path": result.summary_path,
        "candidate_uplift_path": result.candidate_uplift_path,
    }


def _resolve_base_config(
    run_dir: Path, base_config: AnalysisConfig | None
) -> AnalysisConfig:
    if base_config is not None:
        return base_config
    return _load_run_config(run_dir)


def _causal_best_lags(
    frame: pd.DataFrame,
    target: str,
    variables: list[str],
    max_lag: int,
    *,
    target_mask: pd.Series | None,
) -> dict[str, int]:
    from chem_ts_corr import web as web_module

    return web_module._secondary_best_lags_for_missing_variables(
        frame,
        target,
        variables,
        {},
        max_lag,
        recompute_limit=None,
        target_mask=target_mask,
    )


def _causal_ranked_lag_view(
    frame: pd.DataFrame,
    ranked_features: pd.DataFrame,
    *,
    target: str,
    max_lag: int,
    target_mask: pd.Series | None,
) -> pd.DataFrame:
    causal_ranked = ranked_features.copy(deep=True)
    if "variable" not in causal_ranked.columns:
        return causal_ranked
    variables = causal_ranked["variable"].dropna().astype(str).tolist()
    causal_lags = _causal_best_lags(
        frame,
        target,
        variables,
        max_lag,
        target_mask=target_mask,
    )
    causal_ranked["lag"] = causal_ranked["variable"].astype(str).map(causal_lags)
    return causal_ranked


def _load_run_config(run_dir: Path) -> AnalysisConfig:
    path = Path(run_dir) / "run_config.json"
    if not path.exists():
        raise ValueError(f"initial_screening_config_missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"initial_screening_config_invalid: {path}") from exc
    data["input_path"] = Path(data["input_path"])
    data["output_dir"] = Path(run_dir)
    data["roles_path"] = (
        Path(data["roles_path"]) if data.get("roles_path") else None
    )
    data.pop("file_id", None)
    return AnalysisConfig(**data)


def _missing_formal_screening_outputs(
    run_dir: Path,
    *,
    required_formal_files: list[str] | None,
) -> list[str]:
    names = list(required_formal_files or DOWNSTREAM_FORMAL_INPUT_FILES)
    return [name for name in names if not (Path(run_dir) / name).exists()]


def _downstream_config_from_context(
    base_config: AnalysisConfig,
    context: dict[str, object],
) -> AnalysisConfig:
    lowpass_tau_minutes = context["lowpass_tau_minutes"]
    requested_diff_interval_minutes = context["requested_diff_interval_minutes"]
    return replace(
        base_config,
        preprocess_mode=str(context["active_preprocessing_mode"]),
        lowpass_tau_minutes=(
            float(lowpass_tau_minutes)
            if lowpass_tau_minutes is not None
            else base_config.lowpass_tau_minutes
        ),
        diff_interval_minutes=(
            float(requested_diff_interval_minutes)
            if requested_diff_interval_minutes is not None
            else None
        ),
        resample_rule=context["resample_rule"],
    )
