from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chem_ts_corr.causality import run_granger_tests
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.data import select_numeric_frame
from chem_ts_corr.lag import build_lag_peak_quality, compute_lag_scores, summarize_best_lags
from chem_ts_corr.modeling import fit_explainable_model
from chem_ts_corr.preprocess import preprocess_frame, segment_by_load, standardize_frame, transform_frame
from chem_ts_corr.xgb_runner import XGBRunResult, run_xgb_validation


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


def run_xgb_analysis(
    *,
    run_dir: str | Path,
    data: pd.DataFrame,
    target: str,
    final_review_summary: pd.DataFrame,
    ranked_features: pd.DataFrame | None = None,
    control_columns: list[str] | None = None,
    whitelist: list[str] | None = None,
    top_n: int = 8,
    max_lag: int | None = None,
) -> XGBRunResult:
    return run_xgb_validation(
        run_dir=run_dir,
        data=data,
        target=target,
        final_review_summary=final_review_summary,
        ranked_features=ranked_features,
        control_columns=control_columns,
        whitelist=whitelist,
        top_n=top_n,
        max_lag=max_lag,
    )


def _candidate_list(top: list[str], forced: list[str] | None, columns: list[str], excluded: set[str] | None = None) -> tuple[list[str], list[str]]:
    excluded = excluded or set()
    top_filtered = [v for v in top if v not in excluded]
    ordered = list(dict.fromkeys(top_filtered + (forced or [])))
    valid = [v for v in ordered if v in columns]
    warnings = [v for v in (forced or []) if v not in columns]
    return valid, warnings


def analyze_numeric_frame(frame: pd.DataFrame, config: AnalysisConfig, progress_callback=None) -> AnalysisTables:
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

    _progress(progress_callback, "预处理中")
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
    scaled = standardize_frame(
        transform_frame(
            cleaned,
            config.preprocess_mode,
            config.detrend_window,
            max_interpolate_gap_points=config.max_interpolate_gap_points,
            interpolate_limit_area=config.interpolate_limit_area,
        )
    )

    _progress(progress_callback, "正在计算滞后相关")
    lag_scores = compute_lag_scores(scaled, config.target, config.max_lag)
    raw_ranked = summarize_best_lags(lag_scores)
    excluded_controls = set()
    if config.exclude_control_columns_from_candidates:
        excluded_controls = set(config.residual_control_columns or []) | set(config.capacity_columns or [])
    topk = raw_ranked.head(config.top_k)["variable"].tolist() if not raw_ranked.empty else []
    candidate_variables, missing_forced = _candidate_list(topk, config.force_include_variables, list(scaled.columns), excluded_controls)
    best_lags = _best_lag_map(raw_ranked)
    residual_controls = config.residual_control_columns or config.capacity_columns

    _progress(progress_callback, "正在计算残差相关")
    residual = residual_corr_scores(scaled, config.target, residual_controls, config.max_lag)
    _progress(progress_callback, "正在计算工况稳定性")
    regime, stability = regime_scores(scaled, config.target, config.segment_column, config.max_lag)
    regime_output = regime.merge(stability, on="variable", how="left") if not regime.empty else stability
    lag_peak = build_lag_peak_quality(lag_scores, config.max_lag)
    if config.skip_model_lift:
        _progress(progress_callback, "已跳过模型提升评分")
        lift = _skipped_model_lift_scores(candidate_variables)
    else:
        _progress(progress_callback, "正在计算模型提升评分")
        lift = model_lift_scores(scaled, config.target, candidate_variables, config.max_lag, best_lags=best_lags)
    if config.skip_rolling_corr:
        _progress(progress_callback, "已跳过滚动稳定性评分")
        rolling = _skipped_rolling_corr_scores(candidate_variables)
    else:
        _progress(progress_callback, "正在计算滚动稳定性")
        rolling = rolling_corr_scores(scaled, config.target, candidate_variables, config.max_lag)
    _progress(progress_callback, "正在生成候选排序")
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
        control_columns=list(excluded_controls),
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

    metrics.update({"rows_after_segment": float(len(segmented)), "rows_after_preprocess": float(len(scaled)), "variables": float(len(scaled.columns)), "max_lag": float(config.max_lag), "top_k": float(config.top_k), "skip_model_lift": str(config.skip_model_lift), "skip_rolling_corr": str(config.skip_rolling_corr), "missing_force_include": ",".join(missing_forced), "protected_low_variance_columns": ",".join(cleaned.attrs.get("protected_low_variance_columns", []))})

    return AnalysisTables(ranked, lag_scores, granger, importance, diag, residual, regime_output, risks, lift, lag_peak, rolling, metrics)


def _best_lag_map(ranked: pd.DataFrame) -> dict[str, int]:
    if ranked.empty or not {"variable", "lag"}.issubset(ranked.columns):
        return {}
    return {str(row["variable"]): int(row["lag"]) for _, row in ranked[["variable", "lag"]].dropna().iterrows()}


def _progress(progress_callback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _skipped_model_lift_scores(candidate_variables: list[str]) -> pd.DataFrame:
    cols = ["variable", "status", "ar_baseline_rmse", "candidate_rmse", "model_lift"]
    return pd.DataFrame(
        [
            {
                "variable": variable,
                "status": "skipped: user disabled model lift scoring",
                "ar_baseline_rmse": pd.NA,
                "candidate_rmse": pd.NA,
                "model_lift": pd.NA,
            }
            for variable in candidate_variables
        ],
        columns=cols,
    )


def _skipped_rolling_corr_scores(candidate_variables: list[str]) -> pd.DataFrame:
    cols = [
        "variable",
        "best_lag",
        "best_score",
        "rolling_corr_median",
        "rolling_abs_corr_median",
        "rolling_corr_iqr",
        "rolling_sign_consistency",
        "valid_window_count",
        "rolling_stability",
    ]
    return pd.DataFrame(
        [
            {
                "variable": variable,
                "best_lag": pd.NA,
                "best_score": pd.NA,
                "rolling_corr_median": pd.NA,
                "rolling_abs_corr_median": pd.NA,
                "rolling_corr_iqr": pd.NA,
                "rolling_sign_consistency": pd.NA,
                "valid_window_count": 0,
                "rolling_stability": pd.NA,
            }
            for variable in candidate_variables
        ],
        columns=cols,
    )
