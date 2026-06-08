from __future__ import annotations

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.data import load_timeseries_csv
from chem_ts_corr.report import write_outputs
from chem_ts_corr.service import analyze_numeric_frame


def run_analysis(config: AnalysisConfig, progress_callback=None) -> None:
    _progress(progress_callback, "读取数据中")
    raw = load_timeseries_csv(config.input_path, config.time_column, encoding=config.encoding)
    tables = analyze_numeric_frame(raw, config, progress_callback=progress_callback)

    _progress(progress_callback, "正在写出结果文件")
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
    _progress(progress_callback, "分析完成")


def _progress(progress_callback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
