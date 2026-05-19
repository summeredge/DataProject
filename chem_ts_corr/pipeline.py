from __future__ import annotations

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.data import load_timeseries_csv
from chem_ts_corr.report import write_outputs
from chem_ts_corr.service import analyze_numeric_frame


def run_analysis(config: AnalysisConfig) -> None:
    raw = load_timeseries_csv(config.input_path, config.time_column, encoding=config.encoding)
    tables = analyze_numeric_frame(raw, config)

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
    )
