from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
import pandas as pd

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.data import drop_excluded_columns, load_timeseries_csv
from chem_ts_corr.report import write_outputs
from chem_ts_corr.service import analyze_initial_screening_branch_frame, analyze_numeric_frame


SCREENING_BRANCH_NAMES = frozenset({"raw", "processed"})
PROCESSED_BRANCH_PREPROCESS_MODES = frozenset(
    {"lowpass", "lowpass_detrend", "lowpass_diff"}
)


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
    branches succeed and never recommends a branch.
    """
    _validate_comparison_mode(config.preprocess_mode)
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
    comparison_path = config.output_dir / "preprocessing_comparison.csv"
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


def _progress(progress_callback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
