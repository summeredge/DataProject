from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chem_ts_corr.causality import run_granger_tests
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.data import select_numeric_frame
from chem_ts_corr.lag import build_lag_peak_quality, compute_lag_scores, summarize_best_lags
from chem_ts_corr.modeling import fit_explainable_model
from chem_ts_corr.preprocess import preprocess_frame, segment_by_load, standardize_frame, transform_frame


@dataclass(frozen=True)
class AnalysisTables:
    ranked_features: pd.DataFrame
    lag_scores: pd.DataFrame
    granger_tests: pd.DataFrame
    importance: pd.DataFrame
    diagnostics: pd.DataFrame
    residual_corr_scores: pd.DataFrame
    regime_scores: pd.DataFrame
    risk_flags: pd.DataFrame
    model_lift_scores: pd.DataFrame
    lag_peak_quality: pd.DataFrame
    rolling_corr_scores: pd.DataFrame
    metrics: dict[str, float | str]


def _candidate_list(top: list[str], forced: list[str] | None, columns: list[str]) -> tuple[list[str], list[str]]:
    ordered = list(dict.fromkeys(top + (forced or [])))
    valid = [v for v in ordered if v in columns]
    warnings = [v for v in (forced or []) if v not in columns]
    return valid, warnings


def analyze_numeric_frame(frame: pd.DataFrame, config: AnalysisConfig) -> AnalysisTables:
    from chem_ts_corr.screening import (
        apply_ignore_roles,
        diagnostics,
        final_ranked_features,
        load_roles,
        model_lift_scores,
        regime_scores,
        residual_corr_scores,
        risk_flags,
        rolling_corr_scores,
    )

    numeric = select_numeric_frame(frame, config.target)
    roles = load_roles(config, list(numeric.columns))
    numeric = apply_ignore_roles(numeric, roles, config.target)
    diag = diagnostics(numeric, roles)
    segmented = segment_by_load(numeric, config.segment_column, config.segment_mode, config.segment_min, config.segment_max)
    protected = [config.target, config.segment_column, *(config.capacity_columns or []), *(config.residual_control_columns or []), *(config.force_include_variables or [])]
    cleaned = preprocess_frame(
        segmented,
        config.target,
        config.resample_rule,
        config.min_valid_ratio,
        protected_columns=[c for c in protected if c],
        max_interpolate_gap_points=config.max_interpolate_gap_points,
        interpolate_limit_area=config.interpolate_limit_area,
    )
    scaled = standardize_frame(transform_frame(cleaned, config.preprocess_mode, config.detrend_window))

    lag_scores = compute_lag_scores(scaled, config.target, config.max_lag)
    raw_ranked = summarize_best_lags(lag_scores)
    topk = raw_ranked.head(config.top_k)["variable"].tolist() if not raw_ranked.empty else []
    candidate_variables, missing_forced = _candidate_list(topk, config.force_include_variables, list(scaled.columns))
    best_lags = _best_lag_map(raw_ranked)
    residual_controls = config.residual_control_columns or config.capacity_columns

    residual = residual_corr_scores(scaled, config.target, residual_controls, config.max_lag)
    regime, stability = regime_scores(scaled, config.target, config.segment_column, config.max_lag)
    regime_output = regime.merge(stability, on="variable", how="left") if not regime.empty else stability
    lag_peak = build_lag_peak_quality(lag_scores, config.max_lag)
    lift = model_lift_scores(scaled, config.target, candidate_variables, config.max_lag, best_lags=best_lags)
    rolling = rolling_corr_scores(scaled, config.target, candidate_variables, config.max_lag)
    risks = risk_flags(raw_ranked, residual, stability, diag, roles, residual_controls, lag_peak, rolling, lift)
    ranked = final_ranked_features(
        raw_ranked,
        residual,
        stability,
        lift,
        risks,
        lag_peak,
        rolling,
        force_include_variables=config.force_include_variables,
        top_k=config.top_k,
    )
    candidate_variables = ranked["variable"].tolist() if "variable" in ranked.columns else []

    if config.enable_model:
        importance, metrics = fit_explainable_model(scaled, config.target, config.max_lag, candidate_variables, config.max_model_features, config.random_state, best_lags=best_lags)
    else:
        importance, metrics = pd.DataFrame(), {"model_status": "skipped: enable model analysis"}

    if config.enable_granger:
        granger = run_granger_tests(scaled, config.target, variables=candidate_variables[: config.top_k], maxlag=config.resolved_granger_maxlag())
    else:
        granger = pd.DataFrame([{"status": "skipped: enable Granger analysis", "variable": "", "min_p_value": None}])

    metrics.update({"rows_after_segment": float(len(segmented)), "rows_after_preprocess": float(len(scaled)), "variables": float(len(scaled.columns)), "missing_force_include": ",".join(missing_forced), "protected_low_variance_columns": ",".join(cleaned.attrs.get("protected_low_variance_columns", []))})

    return AnalysisTables(ranked, lag_scores, granger, importance, diag, residual, regime_output, risks, lift, lag_peak, rolling, metrics)


def _best_lag_map(ranked: pd.DataFrame) -> dict[str, int]:
    if ranked.empty or not {"variable", "lag"}.issubset(ranked.columns):
        return {}
    return {str(row["variable"]): int(row["lag"]) for _, row in ranked[["variable", "lag"]].dropna().iterrows()}
