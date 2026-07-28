from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.data import select_numeric_frame
from chem_ts_corr.lag import build_lag_peak_quality, compute_lag_scores, summarize_best_lags
from chem_ts_corr.preprocess import (
    difference_by_contiguous_segment,
    operating_segment_mask,
    preprocess_frame,
    standardize_frame,
    transform_frame,
)
from chem_ts_corr.xgb_runner import XGBRunResult, run_xgb_validation


INNOVATION_LAG_RADIUS = 2
INNOVATION_COLUMNS = [
    "variable",
    "innovation_score",
    "innovation_lag",
    "innovation_direction",
    "innovation_sign",
    "innovation_status",
]


@dataclass(frozen=True)
class AnalysisTables:
    ranked_features: pd.DataFrame
    recommended_candidates: pd.DataFrame
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
    target_mask: pd.Series | None = None,
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
        target_mask=target_mask,
    )


def analyze_numeric_frame(frame: pd.DataFrame, config: AnalysisConfig, progress_callback=None) -> AnalysisTables:
    from chem_ts_corr.screening import (
        apply_ignore_roles,
        diagnostics,
        build_recommended_candidates,
        final_ranked_features,
        load_roles,
        residual_corr_scores,
        risk_flags,
    )

    _progress(progress_callback, "预处理中")
    numeric = select_numeric_frame(frame, config.target)
    roles = load_roles(config, list(numeric.columns))
    numeric = apply_ignore_roles(numeric, roles, config.target)
    diag = diagnostics(numeric, roles)
    raw_segment_mask = operating_segment_mask(
        numeric,
        config.segment_column,
        config.segment_mode,
        config.segment_min,
        config.segment_max,
    )
    protected = [config.target, config.segment_column, *(config.capacity_columns or []), *(config.residual_control_columns or []), *(config.force_include_variables or [])]
    cleaned = preprocess_frame(
        numeric,
        config.target,
        config.resample_rule,
        config.min_valid_ratio,
        protected_columns=[c for c in protected if c],
        max_interpolate_gap_points=config.max_interpolate_gap_points,
        interpolate_limit_area=config.interpolate_limit_area,
    )
    segment_mask = operating_segment_mask(
        cleaned,
        config.segment_column,
        config.segment_mode,
        config.segment_min,
        config.segment_max,
    )
    transformed = transform_frame(
        cleaned,
        config.preprocess_mode,
        config.detrend_window,
        max_interpolate_gap_points=config.max_interpolate_gap_points,
        interpolate_limit_area=config.interpolate_limit_area,
    )
    target_mask = segment_mask.reindex(transformed.index).fillna(False).astype(bool)
    analysis_target_mask = None if bool(target_mask.all()) else target_mask
    scaled = standardize_frame(transformed, fit_mask=analysis_target_mask)

    _progress(progress_callback, "正在计算滞后相关")
    lag_scores = compute_lag_scores(
        scaled,
        config.target,
        config.max_lag,
        target_mask=analysis_target_mask,
    )
    raw_ranked = summarize_best_lags(lag_scores)
    _progress(progress_callback, "正在计算变化量关联")
    innovation_ranked = _innovation_evidence(
        scaled,
        config.target,
        config.max_lag,
        raw_ranked,
        config.preprocess_mode,
        target_mask=analysis_target_mask,
    )
    raw_ranked = raw_ranked.merge(
        innovation_ranked,
        on="variable",
        how="left",
    )
    missing_forced = [value for value in (config.force_include_variables or []) if value not in scaled.columns]
    residual_controls = config.residual_control_columns or config.capacity_columns
    lag_peak = build_lag_peak_quality(lag_scores, config.max_lag)
    residual_output = pd.DataFrame(columns=["variable"])
    if residual_controls:
        _progress(progress_callback, "正在计算负荷控制后的残差关联")
        best_lags = raw_ranked.set_index("variable")["lag"].to_dict()
        residual_output = residual_corr_scores(
            scaled,
            config.target,
            residual_controls,
            config.max_lag,
            best_lags=best_lags,
            target_mask=analysis_target_mask,
        )
    # PR-2 output is intentionally isolated from screening scores and risks.
    residual_for_scoring = pd.DataFrame(columns=["variable"])
    regime_output = pd.DataFrame(columns=["variable"])
    lift = pd.DataFrame(columns=["variable"])
    rolling = pd.DataFrame(columns=["variable"])
    _progress(progress_callback, "正在生成候选排序")
    risks = risk_flags(
        raw_ranked,
        residual_for_scoring,
        regime_output,
        diag,
        roles,
        residual_controls,
        lag_peak,
        rolling,
        lift,
        frame=scaled,
        target=config.target,
    )
    ranked = final_ranked_features(
        raw_ranked,
        residual_for_scoring,
        regime_output,
        lift,
        risks,
        lag_peak,
        rolling,
        force_include_variables=config.force_include_variables,
        control_columns=config.residual_control_columns,
        capacity_columns=config.capacity_columns,
        segment_column=config.segment_column,
    )
    recommended = build_recommended_candidates(
        ranked,
        config.top_k,
        config.force_include_variables,
        config.exclude_control_columns_from_candidates,
    )
    importance = pd.DataFrame()
    granger = pd.DataFrame()
    metrics: dict[str, float | str] = {}

    metrics.update({"rows_after_segment": float(raw_segment_mask.sum()), "rows_after_preprocess": float(target_mask.sum()), "variables": float(len(scaled.columns)), "effective_variables": float(len(ranked)), "recommended_candidate_count": float(len(recommended)), "control_reference_count": float(ranked[["is_residual_control", "is_capacity_reference", "is_segment_reference"]].any(axis=1).sum()), "max_lag": float(config.max_lag), "top_k": float(config.top_k), "missing_force_include": ",".join(missing_forced), "protected_low_variance_columns": ",".join(cleaned.attrs.get("protected_low_variance_columns", []))})

    return AnalysisTables(ranked, recommended, lag_scores, granger, importance, diag, residual_output, regime_output, risks, lift, lag_peak, rolling, metrics)


def _innovation_evidence(
    frame: pd.DataFrame,
    target: str,
    max_lag: int,
    raw_ranked: pd.DataFrame,
    preprocess_mode: str,
    target_mask: pd.Series | None = None,
) -> pd.DataFrame:
    if raw_ranked.empty:
        return pd.DataFrame(columns=INNOVATION_COLUMNS)

    rows: list[dict[str, object]] = []
    already_differenced = preprocess_mode in {"diff", "detrend_diff"}
    innovation_frame = (
        frame
        if already_differenced
        else difference_by_contiguous_segment(frame).dropna()
    )
    for _, raw_row in raw_ranked.iterrows():
        variable = str(raw_row["variable"])
        raw_lag = int(raw_row["lag"])
        if already_differenced:
            innovation_row = raw_row
        else:
            lower = max(-max_lag, raw_lag - INNOVATION_LAG_RADIUS)
            upper = min(max_lag, raw_lag + INNOVATION_LAG_RADIUS)
            scores = compute_lag_scores(
                innovation_frame[[target, variable]],
                target,
                max_lag,
                lag_values=range(lower, upper + 1),
                target_mask=target_mask,
            )
            best = summarize_best_lags(scores)
            if best.empty:
                rows.append(
                    {
                        "variable": variable,
                        "innovation_score": np.nan,
                        "innovation_lag": pd.NA,
                        "innovation_direction": pd.NA,
                        "innovation_sign": pd.NA,
                        "innovation_status": "not_computed",
                    }
                )
                continue
            innovation_row = best.iloc[0]

        innovation_lag = int(innovation_row["lag"])
        innovation_sign = _correlation_sign(innovation_row)
        raw_sign = _correlation_sign(raw_row)
        lag_consistent = (
            abs(innovation_lag - raw_lag) <= INNOVATION_LAG_RADIUS
            and np.sign(innovation_lag) == np.sign(raw_lag)
        )
        if already_differenced:
            status = "innovation_verified"
        elif not lag_consistent:
            status = "innovation_lag_conflict"
        elif raw_sign is None or innovation_sign is None:
            status = "innovation_sign_unknown"
        elif innovation_sign != raw_sign:
            status = "innovation_sign_conflict"
        else:
            status = "innovation_verified"
        rows.append(
            {
                "variable": variable,
                "innovation_score": (
                    float(innovation_row["score"])
                    if status == "innovation_verified"
                    else np.nan
                ),
                "innovation_lag": innovation_lag,
                "innovation_direction": innovation_row.get("direction", pd.NA),
                "innovation_sign": innovation_sign if innovation_sign is not None else pd.NA,
                "innovation_status": status,
            }
        )
    return pd.DataFrame(rows, columns=INNOVATION_COLUMNS)


def _correlation_sign(row: pd.Series) -> int | None:
    method = str(row.get("method", ""))
    value = row.get(method, np.nan) if method in {"pearson", "spearman"} else np.nan
    if pd.isna(value):
        candidates = [
            candidate
            for candidate in [row.get("pearson", np.nan), row.get("spearman", np.nan)]
            if pd.notna(candidate)
        ]
        if not candidates:
            return None
        value = max(candidates, key=lambda candidate: abs(float(candidate)))
    return int(np.sign(float(value)))


def _progress(progress_callback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _skipped_model_lift_scores(candidate_variables: list[str]) -> pd.DataFrame:
    cols = [
        "variable", "status", "ar_baseline_rmse", "candidate_rmse", "model_lift",
        "median_fold_lift", "positive_fold_ratio", "model_lift_score",
    ]
    return pd.DataFrame(
        [
            {
                "variable": variable,
                "status": "skipped: user disabled model lift scoring",
                "ar_baseline_rmse": pd.NA,
                "candidate_rmse": pd.NA,
                "model_lift": pd.NA,
                "median_fold_lift": pd.NA,
                "positive_fold_ratio": pd.NA,
                "model_lift_score": pd.NA,
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
