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
    segmented = segment_by_load(
        numeric,
        segment_column=config.segment_column,
        segment_mode=config.segment_mode,
        segment_min=config.segment_min,
        segment_max=config.segment_max,
    )
    cleaned = preprocess_frame(
        segmented,
        target=config.target,
        resample_rule=config.resample_rule,
        min_valid_ratio=config.min_valid_ratio,
    )
    transformed = transform_frame(cleaned, config.preprocess_mode, config.detrend_window)
    scaled = standardize_frame(transformed)

    lag_scores = compute_lag_scores(scaled, config.target, config.max_lag)
    raw_ranked = summarize_best_lags(lag_scores)
    candidate_variables = raw_ranked.head(config.top_k)["variable"].tolist()
    best_lags = _best_lag_map(raw_ranked)
    residual_controls = config.residual_control_columns or config.capacity_columns
    residual = residual_corr_scores(scaled, config.target, residual_controls, config.max_lag)
    regime, stability = regime_scores(scaled, config.target, config.segment_column, config.max_lag)
    regime_output = regime.merge(stability, on="variable", how="left") if not regime.empty else stability
    lag_peak = build_lag_peak_quality(lag_scores, config.max_lag)
    lift = model_lift_scores(
        scaled,
        config.target,
        candidate_variables,
        config.max_lag,
        best_lags=best_lags,
    )
    rolling = rolling_corr_scores(scaled, config.target, list(dict.fromkeys(candidate_variables + (config.force_include_variables or []))))
    risks = risk_flags(raw_ranked, residual, stability, diag, roles, residual_controls)
    ranked = final_ranked_features(raw_ranked, residual, stability, lift, risks)
    ranked = ranked.merge(lag_peak[["variable","lag_quality","lag_boundary_flag"]], on="variable", how="left")
    ranked = ranked.merge(rolling[["variable","rolling_stability"]], on="variable", how="left")
    ranked["rolling_stability"] = ranked["rolling_stability"].fillna(0.5)
    ranked["lag_quality"] = ranked["lag_quality"].fillna(0.5)
    ranked["model_lift_score"] = ranked["model_lift"].fillna(0.0)
    ranked["risk_penalty"] = ranked["risk_count"].fillna(0)
    ranked["raw_corr_score"] = ranked["raw_corr"].fillna(0)
    ranked["residual_corr_score"] = ranked["residual_corr"].fillna(ranked["raw_corr_score"])
    ranked["regime_stability_final"] = ranked["regime_stability"].fillna(0.5)
    ranked["final_score"] = (0.25*ranked["raw_corr_score"] +0.25*ranked["residual_corr_score"] +0.15*ranked["regime_stability_final"] +0.15*ranked["rolling_stability"] +0.10*ranked["lag_quality"] +0.10*ranked["model_lift_score"] -0.10*ranked["risk_penalty"]).clip(lower=0,upper=1)
    ranked = ranked.sort_values("final_score", ascending=False).head(config.top_k)
    candidate_variables = ranked["variable"].tolist()

    if config.enable_model:
        importance, metrics = fit_explainable_model(
            scaled,
            target=config.target,
            max_lag=config.max_lag,
            candidate_variables=candidate_variables,
            max_features=config.max_model_features,
            random_state=config.random_state,
            best_lags=best_lags,
        )
    else:
        importance, metrics = _skipped_model_result()

    if config.enable_granger:
        granger = run_granger_tests(
            scaled,
            target=config.target,
            variables=candidate_variables[: min(len(candidate_variables), config.top_k)],
            maxlag=config.resolved_granger_maxlag(),
        )
    else:
        granger = _skipped_granger_result()

    metrics.update(
        {
            "rows_after_segment": float(len(segmented)),
            "rows_after_preprocess": float(len(scaled)),
            "variables": float(len(scaled.columns)),
            "preprocess_mode": config.preprocess_mode,
            "detrend_window": float(config.detrend_window),
            "segment": _segment_label(config),
        }
    )

    return AnalysisTables(
        ranked_features=ranked,
        lag_scores=lag_scores,
        granger_tests=granger,
        importance=importance,
        diagnostics=diag,
        residual_corr_scores=residual,
        regime_scores=regime_output,
        risk_flags=risks,
        model_lift_scores=lift,
        lag_peak_quality=lag_peak,
        rolling_corr_scores=rolling,
        metrics=metrics,
    )


def _skipped_model_result() -> tuple[pd.DataFrame, dict[str, str]]:
    return pd.DataFrame(), {"model_status": "skipped: enable model analysis"}


def _skipped_granger_result() -> pd.DataFrame:
    return pd.DataFrame(
        [{"status": "skipped: enable Granger analysis", "variable": "", "min_p_value": None}]
    )


def _segment_label(config: AnalysisConfig) -> str:
    if not config.segment_column or config.segment_mode == "all":
        return "all"
    if config.segment_mode == "custom":
        lower = "-inf" if config.segment_min is None else str(config.segment_min)
        upper = "+inf" if config.segment_max is None else str(config.segment_max)
        return f"{config.segment_column}: custom [{lower}, {upper}]"
    return f"{config.segment_column}: {config.segment_mode}"


def _best_lag_map(ranked: pd.DataFrame) -> dict[str, int]:
    if ranked.empty or not {"variable", "lag"}.issubset(ranked.columns):
        return {}
    return {
        str(row["variable"]): int(row["lag"])
        for _, row in ranked[["variable", "lag"]].dropna().iterrows()
    }
