from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr.config import AnalysisConfig, NOT_WIRED_ANALYSIS_PREPROCESS_MODES
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
    shadow_regime_stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    shadow_rolling_corr_scores: pd.DataFrame = field(default_factory=pd.DataFrame)


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
    if config.preprocess_mode in NOT_WIRED_ANALYSIS_PREPROCESS_MODES:
        raise ValueError(
            f"Preprocess mode {config.preprocess_mode!r} is not wired into the "
            "analysis/screening flow yet"
        )
    return _analyze_numeric_frame_core(frame, config, progress_callback=progress_callback)


def analyze_initial_screening_branch_frame(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    progress_callback=None,
) -> AnalysisTables:
    """Branch-runner-only internal entry that shares the screening core.

    Only the single-branch runner calls this after it has validated the
    branch/mode pair. The formal ``analyze_numeric_frame()`` guard remains
    enforced for all other callers.
    """
    return _analyze_numeric_frame_core(
        frame,
        config,
        progress_callback=progress_callback,
    )


def _analyze_numeric_frame_core(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    progress_callback=None,
) -> AnalysisTables:
    from chem_ts_corr.screening import (
        apply_ignore_roles,
        diagnostics,
        build_recommended_candidates,
        final_ranked_features,
        load_roles,
        prioritize_recommended_candidates,
        residual_corr_scores,
        risk_flags,
    )

    _progress(progress_callback, "预处理中")
    numeric = select_numeric_frame(frame, config.target)
    numeric.attrs = dict(frame.attrs)
    roles = load_roles(config, list(numeric.columns))
    numeric = apply_ignore_roles(numeric, roles, config.target)
    numeric.attrs = dict(frame.attrs)
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
        lowpass_tau_minutes=config.lowpass_tau_minutes,
        diff_interval_minutes=config.diff_interval_minutes,
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
        residual_output = residual_corr_scores(
            scaled,
            config.target,
            residual_controls,
            config.max_lag,
            target_mask=analysis_target_mask,
        )
    regime_output = pd.DataFrame(columns=["variable"])
    lift = pd.DataFrame(columns=["variable"])
    rolling = pd.DataFrame(columns=["variable"])
    _progress(progress_callback, "正在计算 V5 支持证据")
    try:
        v5_regime_stability, v5_rolling = _v5_shadow_stability_evidence(
            scaled,
            config,
            raw_ranked,
            analysis_target_mask,
            regime_level_frame=_v5_shadow_regime_level_frame(
                cleaned,
                transformed,
                scaled,
                config,
            ),
        )
    except Exception:
        variables = (
            raw_ranked["variable"].astype(str).tolist()
            if "variable" in raw_ranked.columns
            else []
        )
        v5_regime_stability = _v5_shadow_regime_status_frame(
            variables,
            "calculation_failed",
        )
        v5_rolling = _v5_shadow_rolling_status_frame(
            variables,
            "calculation_failed",
        )
    _progress(progress_callback, "正在生成候选排序")
    risks = risk_flags(
        raw_ranked,
        residual_output,
        v5_regime_stability,
        diag,
        roles,
        residual_controls,
        lag_peak,
        v5_rolling,
        lift,
        frame=scaled,
        target=config.target,
    )
    ranked = final_ranked_features(
        raw_ranked,
        residual_output,
        v5_regime_stability,
        lift,
        risks,
        lag_peak,
        v5_rolling,
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
        residual_corr_scores=residual_output,
        residual_top_k=config.top_k,
    )
    recommended = prioritize_recommended_candidates(recommended, residual_output)
    importance = pd.DataFrame()
    granger = pd.DataFrame()
    metrics: dict[str, float | str] = {}

    metrics.update({"rows_after_segment": float(raw_segment_mask.sum()), "rows_after_preprocess": float(target_mask.sum()), "variables": float(len(scaled.columns)), "effective_variables": float(len(ranked)), "raw_candidate_count": float(recommended["selected_by_raw"].sum()), "residual_candidate_count": float(recommended["selected_by_residual"].sum()), "candidate_overlap_count": float((recommended["selected_by_raw"] & recommended["selected_by_residual"]).sum()), "forced_only_candidate_count": float((recommended["candidate_source"] == "force_included").sum() + ((recommended["candidate_source"] == "control_reference") & recommended["force_included"]).sum()), "recommended_candidate_count": float(len(recommended)), "control_reference_count": float(ranked["is_control_reference"].sum()), "max_lag": float(config.max_lag), "top_k": float(config.top_k), "missing_force_include": ",".join(missing_forced), "protected_low_variance_columns": ",".join(cleaned.attrs.get("protected_low_variance_columns", []))})

    return AnalysisTables(
        ranked,
        recommended,
        lag_scores,
        granger,
        importance,
        diag,
        residual_output,
        regime_output,
        risks,
        lift,
        lag_peak,
        rolling,
        metrics,
        shadow_regime_stability=v5_regime_stability,
        shadow_rolling_corr_scores=v5_rolling,
    )


def _v5_shadow_stability_evidence(
    scaled: pd.DataFrame,
    config: AnalysisConfig,
    raw_ranked: pd.DataFrame,
    target_mask: pd.Series | None,
    *,
    regime_level_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the pure consistency evidence used by formal V5 scoring.

    The returned frames are not passed to ``risk_flags`` and do not alter the
    existing public stability output tables.
    """
    from chem_ts_corr.screening import (
        prepare_best_lag_evidence,
        regime_scores,
        rolling_corr_scores,
    )

    variables = (
        raw_ranked["variable"].astype(str).tolist()
        if not raw_ranked.empty and "variable" in raw_ranked.columns
        else []
    )
    if not variables:
        return pd.DataFrame(), pd.DataFrame()

    try:
        best_lag_evidence, _ = prepare_best_lag_evidence(
            scaled,
            config.target,
            variables,
            config.max_lag,
            ranked=raw_ranked,
            allow_ranked_reuse=True,
            ranked_source_frame=scaled,
            target_mask=target_mask,
        )
    except Exception:
        best_lag_evidence = {}

    if config.skip_rolling_corr:
        shadow_rolling = _v5_shadow_rolling_status_frame(
            variables,
            "not_computed",
        )
    else:
        try:
            rolling = rolling_corr_scores(
                scaled,
                config.target,
                variables,
                config.max_lag,
                best_lag_evidence=best_lag_evidence,
                target_mask=target_mask,
            )
        except Exception:
            shadow_rolling = _v5_shadow_rolling_status_frame(
                variables,
                "calculation_failed",
            )
        else:
            shadow_rolling = _v5_normalize_shadow_rolling(rolling, variables)

    basis_frame = scaled if regime_level_frame is None else regime_level_frame
    regime_column = _v5_shadow_regime_column(basis_frame, config)
    if regime_column is None:
        shadow_regime = _v5_shadow_regime_status_frame(
            variables,
            "no_regime_basis",
        )
        return shadow_regime, shadow_rolling

    best_lags = {
        variable: evidence["best_lag"]
        for variable, evidence in best_lag_evidence.items()
        if evidence["best_lag"] is not None
    }
    try:
        _, regime = regime_scores(
            scaled,
            config.target,
            regime_column,
            config.max_lag,
            best_lags=best_lags,
            target_mask=target_mask,
            regime_basis=basis_frame,
        )
    except Exception:
        shadow_regime = _v5_shadow_regime_status_frame(
            variables,
            "calculation_failed",
        )
    else:
        shadow_regime = _v5_normalize_shadow_regime(regime, variables)
    return shadow_regime, shadow_rolling


def _v5_shadow_regime_column(
    frame: pd.DataFrame,
    config: AnalysisConfig,
) -> str | None:
    candidates = [
        config.segment_column,
        *(config.capacity_columns or []),
        *(config.residual_control_columns or []),
    ]
    for candidate in candidates:
        if candidate and candidate != config.target and candidate in frame.columns:
            return candidate
    return None


def _v5_shadow_regime_level_frame(
    cleaned: pd.DataFrame,
    transformed: pd.DataFrame,
    scaled: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame | None:
    level_frame = transformed if config.preprocess_mode == "lowpass" else cleaned
    column = _v5_shadow_regime_column(level_frame, config)
    if column is None:
        return None
    basis = level_frame[[column]].reindex(scaled.index)
    basis.attrs = dict(level_frame.attrs)
    return basis


def _v5_shadow_rolling_status_frame(
    candidate_variables: list[str],
    status: str,
) -> pd.DataFrame:
    frame = _skipped_rolling_corr_scores(candidate_variables)
    frame["rolling_support_status"] = status
    return frame


def _v5_normalize_shadow_rolling(
    result: pd.DataFrame,
    candidate_variables: list[str],
) -> pd.DataFrame:
    defaults = _skipped_rolling_corr_scores(candidate_variables)
    result_lookup = _v5_shadow_lookup(result)
    rows: list[dict[str, object]] = []
    value_columns = [column for column in defaults.columns if column != "variable"]
    for variable in candidate_variables:
        row = defaults.loc[defaults["variable"].eq(variable)].iloc[0].to_dict()
        source = result_lookup.get(variable)
        if source is None:
            row["rolling_support_status"] = "insufficient_data"
        else:
            for column in value_columns:
                if column in source.index and not _v5_shadow_missing(source[column]):
                    row[column] = source[column]
            source_status = source.get("rolling_support_status")
            if source_status in {
                "not_computed",
                "calculation_failed",
                "insufficient_data",
            }:
                row["rolling_support_status"] = source_status
            elif _v5_shadow_finite(row["rolling_sign_consistency"]) and _v5_shadow_finite(
                row["rolling_corr_iqr"]
            ):
                row["rolling_support_status"] = "ok"
            else:
                row["rolling_support_status"] = "insufficient_data"
        rows.append(row)
    return pd.DataFrame(rows, columns=defaults.columns)


def _v5_shadow_regime_status_frame(
    candidate_variables: list[str],
    status: str,
) -> pd.DataFrame:
    from chem_ts_corr.screening import REGIME_STABILITY_COLUMNS

    columns = [*REGIME_STABILITY_COLUMNS, "regime_support_status"]
    rows: list[dict[str, object]] = []
    for variable in candidate_variables:
        row = {column: np.nan for column in columns}
        row["variable"] = variable
        row["regime_evidence_status"] = status
        row["regime_support_status"] = status
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _v5_normalize_shadow_regime(
    result: pd.DataFrame,
    candidate_variables: list[str],
) -> pd.DataFrame:
    from chem_ts_corr.screening import REGIME_STABILITY_COLUMNS

    if result is None or result.empty:
        return _v5_shadow_regime_status_frame(
            candidate_variables,
            "insufficient_regimes",
        )
    defaults = _v5_shadow_regime_status_frame(
        candidate_variables,
        "insufficient_metrics",
    )
    result_lookup = _v5_shadow_lookup(result)
    value_columns = [column for column in REGIME_STABILITY_COLUMNS if column != "variable"]
    rows: list[dict[str, object]] = []
    metric_columns = [
        "regime_coverage",
        "regime_strength_consistency",
        "regime_sign_consistency",
        "regime_lag_consistency",
    ]
    for variable in candidate_variables:
        row = defaults.loc[defaults["variable"].eq(variable)].iloc[0].to_dict()
        source = result_lookup.get(variable)
        if source is not None:
            for column in value_columns:
                if column in source.index and not _v5_shadow_missing(source[column]):
                    row[column] = source[column]
            raw_status = source.get(
                "regime_support_status",
                source.get("regime_evidence_status"),
            )
            status = _v5_shadow_regime_status(raw_status)
            if status is None:
                status = (
                    "ok"
                    if all(_v5_shadow_finite(row[column]) for column in metric_columns)
                    else "insufficient_metrics"
                )
            elif status == "ok" and not all(
                _v5_shadow_finite(row[column]) for column in metric_columns
            ):
                status = "insufficient_metrics"
            row["regime_support_status"] = status
        rows.append(row)
    return pd.DataFrame(rows, columns=[*REGIME_STABILITY_COLUMNS, "regime_support_status"])


def _v5_shadow_lookup(frame: pd.DataFrame | None) -> dict[str, pd.Series]:
    if frame is None or frame.empty or "variable" not in frame.columns:
        return {}
    prepared = frame.copy(deep=True)
    prepared["variable"] = prepared["variable"].astype(str)
    return {
        str(row["variable"]): row
        for _, row in prepared.drop_duplicates("variable").iterrows()
    }


def _v5_shadow_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _v5_shadow_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _v5_shadow_regime_status(value: object) -> str | None:
    if _v5_shadow_missing(value):
        return None
    status = str(value)
    if status in {
        "ok",
        "no_regime_basis",
        "insufficient_regimes",
        "insufficient_metrics",
        "calculation_failed",
    }:
        return status
    if status in {"full_coverage", "partial_coverage"}:
        return "ok"
    if status in {"not_computed", "unavailable"}:
        return "no_regime_basis"
    if status == "fit_failed":
        return "calculation_failed"
    return "calculation_failed"


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
    already_differenced = preprocess_mode in {
        "diff",
        "detrend_diff",
        "lowpass_diff",
    }
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
        "rolling_support_status",
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
                "rolling_support_status": "not_computed",
            }
            for variable in candidate_variables
        ],
        columns=cols,
    )
