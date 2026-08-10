from __future__ import annotations

import time

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


def _progress(progress_callback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
