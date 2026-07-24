from __future__ import annotations

import time

from chem_ts_corr.auto_closed_loop import build_auto_closed_loop_diagnosis
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.data import drop_excluded_columns, load_timeseries_csv
from chem_ts_corr.report import write_outputs
from chem_ts_corr.service import analyze_numeric_frame


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
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables.closed_loop_evidence.to_csv(
        config.output_dir / "closed_loop_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_auto_closed_loop_diagnosis(
        tables.ranked_features,
        tables.risk_flags,
        tables.lag_peak_quality,
        tables.rolling_corr_scores,
        tables.model_lift_scores,
        config.target,
    ).to_csv(
        config.output_dir / "auto_closed_loop_diagnosis.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_outputs_seconds = time.perf_counter() - write_started
    _progress(progress_callback, "分析完成")
    return {
        "read_data_seconds": read_data_seconds,
        "analysis_core_seconds": analysis_core_seconds,
        "write_outputs_seconds": write_outputs_seconds,
        "pipeline_total_seconds": time.perf_counter() - pipeline_started,
    }


def _progress(progress_callback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
