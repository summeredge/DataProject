from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from chem_ts_corr.common import to_float

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.feature_alignment import fit_linear_model, predict_linear_model
from chem_ts_corr.lag import build_lag_peak_quality, compute_lag_scores, summarize_best_lags
from chem_ts_corr.time_axis import lagged_series, sample_period_ns

ROLES = {"TIME", "Y", "CAPACITY", "MV", "PV", "DV", "IGNORE"}
RISK_RELATIVE_PENALTY_WEIGHTS = {
    "formula_like": 0.00,
    "strong_formula_leakage": 0.50,
    "common_capacity_driver": 0.00,
    "target_leads_variable": 0.00,
    "unstable_across_regimes": 0.00,
    "unstable_over_time": 0.00,
    "lag_boundary": 0.00,
    "low_model_lift": 0.00,
    "poor_data_quality": 0.00,
    "residual_collinearity": 0.10,
    "redundant_proxy": 0.00,
}
EVIDENCE_SCORE_CAPS = {
    "strong_formula_leakage": 0.25,
    "poor_data_quality": 0.44,
}
CLASS_PRIORITY_FACTORS = {
    "upstream_driver_candidate": 1.00,
    "synchronous_association": 0.90,
    "downstream_response": 0.45,
    "capacity_driven": 0.75,
    "formula_or_derived": 0.25,
    "poor_quality": 0.35,
    "uncertain_candidate": 0.80,
}
INDUSTRIAL_SCORE_COMPONENTS = ("association", "prediction", "stability", "lag_quality")


def _industrial_score_weight_profiles() -> tuple[dict[str, float], ...]:
    profiles: list[dict[str, float]] = []
    for association in range(10, 41, 5):
        for prediction in range(10, 41, 5):
            for stability in range(10, 41, 5):
                lag_quality = 100 - association - prediction - stability
                if not 10 <= lag_quality <= 40:
                    continue
                profiles.append({
                    "association": association / 100,
                    "prediction": prediction / 100,
                    "stability": stability / 100,
                    "lag_quality": lag_quality / 100,
                })
    return tuple(profiles)


INDUSTRIAL_SCORE_WEIGHT_PROFILES = _industrial_score_weight_profiles()


def _available_weight_profile_scores(
    components: pd.DataFrame, profiles: Sequence[Mapping[str, float]]
) -> pd.DataFrame:
    component_values = components.to_numpy(dtype=float)
    available = ~pd.isna(component_values)
    profile_weights = pd.DataFrame(profiles, columns=components.columns).to_numpy(dtype=float)
    weighted_sum = np.where(available, component_values, 0.0) @ profile_weights.T
    available_weight = available.astype(float) @ profile_weights.T
    profile_scores = np.divide(
        weighted_sum,
        available_weight,
        out=np.full_like(weighted_sum, np.nan),
        where=available_weight > 0,
    )
    return pd.DataFrame(profile_scores, index=components.index)


REGIME_NAMES = ("low", "mid", "high")
MIN_REGIMES_FOR_STABILITY = 2
REGIME_UNSTABLE_THRESHOLD = 0.50
EVIDENCE_SEPARATION_MARGIN = 0.05
REGIME_CONSISTENCY_WEIGHTS = {
    "strength": 0.60,
    "lag": 0.40,
}
REGIME_STABILITY_COLUMNS = [
    "variable",
    "regime_stability_final",
    "regime_consistency_score",
    "regime_coverage",
    "regime_strength_consistency",
    "regime_sign_consistency",
    "regime_lag_consistency",
    "regime_score_cv",
    "regime_count",
    "regime_evidence_status",
    "regime_sign_reversal_flag",
]
PRIMARY_RANK_COLUMN = "final_score"
PRIMARY_SCORE_COLUMN = "final_score"


class BestLagEvidence(TypedDict):
    best_lag: int | None
    best_score: float | None
    max_lag: int
    pair_alignment_key: str
    source: str
    status: Literal["ok", "scanned_no_result"]


def load_roles(config: AnalysisConfig, columns: list[str]) -> dict[str, str]:
    roles = {column: "PV" for column in columns}
    roles[config.target] = "Y"
    if config.segment_column and config.segment_column in roles:
        roles[config.segment_column] = "CAPACITY"
    for column in config.capacity_columns or []:
        if column in roles:
            roles[column] = "CAPACITY"
    for column in config.residual_control_columns or []:
        if column in roles:
            roles[column] = "CAPACITY"

    if config.roles_path:
        role_frame = pd.read_csv(config.roles_path)
        if not {"variable", "role"}.issubset(role_frame.columns):
            raise ValueError("roles file must contain columns: variable, role")
        for _, row in role_frame.iterrows():
            variable = str(row["variable"])
            role = str(row["role"]).upper()
            if variable in roles and role in ROLES:
                roles[variable] = role
    return roles


def apply_ignore_roles(frame: pd.DataFrame, roles: dict[str, str], target: str) -> pd.DataFrame:
    ignored = [column for column, role in roles.items() if role == "IGNORE" and column != target]
    return frame.drop(columns=[column for column in ignored if column in frame.columns], errors="ignore")


def diagnostics(frame: pd.DataFrame, roles: dict[str, str]) -> pd.DataFrame:
    columns = [
        "variable", "role", "missing_rate", "longest_missing_run", "duplicate_timestamps",
        "sampling_period_seconds", "constant_run_max", "abnormal_jump_count", "abnormal_jump_ratio", "robust_outlier_ratio", "saturation_ratio",
    ]
    rows: list[dict[str, object]] = []
    duplicate_timestamps = int(frame.attrs.get("duplicate_timestamps", 0))
    sampling_period = _sampling_period_seconds(frame.index)
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        diffs = series.diff().abs()
        non_na = series.dropna()
        q1 = non_na.quantile(0.25) if len(non_na) else np.nan
        q3 = non_na.quantile(0.75) if len(non_na) else np.nan
        iqr = q3 - q1 if pd.notna(q1) and pd.notna(q3) else np.nan
        jump_threshold = 10 * iqr if pd.notna(iqr) and iqr > 0 else np.nan
        abnormal_jump_count = int((diffs > jump_threshold).sum()) if pd.notna(jump_threshold) else 0
        valid_diff_count = int(diffs.notna().sum())
        rows.append({
            "variable": column,
            "role": roles.get(column, "PV"),
            "missing_rate": float(series.isna().mean()),
            "longest_missing_run": int(_longest_run(series.isna())),
            "duplicate_timestamps": duplicate_timestamps,
            "sampling_period_seconds": sampling_period,
            "constant_run_max": int(_longest_constant_run(series)),
            "abnormal_jump_count": abnormal_jump_count,
            "abnormal_jump_ratio": abnormal_jump_count / valid_diff_count if valid_diff_count else 0.0,
            "robust_outlier_ratio": float(_robust_outlier_ratio(non_na)),
            "saturation_ratio": float(_saturation_ratio(series)),
        })
    return pd.DataFrame(rows, columns=columns)


RESIDUAL_SCORE_COLUMNS = [
    "variable", "residual_pearson", "residual_spearman", "residual_signed_corr",
    "residual_corr", "residual_method", "residual_lag", "residual_direction",
    "residual_n", "residual_lag_quality", "residual_lag_boundary_flag",
    "residual_status", "requested_control_columns", "effective_control_columns",
    "control_count", "control_matrix_rank", "control_condition_number",
]


def residual_corr_scores(
    frame: pd.DataFrame,
    target: str,
    capacity_columns: list[str] | None,
    max_lag: int,
    best_lags: Mapping[str, int] | None = None,
    target_mask: pd.Series | None = None,
) -> pd.DataFrame:
    requested_controls = list(dict.fromkeys(str(column) for column in (capacity_columns or [])))
    available_controls = [
        column for column in requested_controls
        if column in frame.columns and column != target
    ]
    if not requested_controls:
        return pd.DataFrame(columns=RESIDUAL_SCORE_COLUMNS)
    period_ns = sample_period_ns(frame)
    aligned_controls = pd.DataFrame(
        {
            column: lagged_series(
                frame[column],
                frame.index,
                int((best_lags or {}).get(column, 0) or 0),
                period_ns=period_ns,
            )
            for column in available_controls
        },
        index=frame.index,
    )
    resolved_mask = (
        target_mask.reindex(frame.index).fillna(False).astype(bool)
        if target_mask is not None
        else pd.Series(True, index=frame.index, dtype=bool)
    )
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        if column == target:
            continue
        base = _residual_result_row(column, requested_controls)
        if column in available_controls:
            base["residual_status"] = "control_reference_not_residualized"
            rows.append(base)
            continue
        pair = pd.concat([frame[[target, column]], aligned_controls], axis=1).loc[resolved_mask]
        usable_controls = [
            control for control in available_controls
            if pair[control].notna().any() and pair[control].nunique(dropna=True) > 1
        ]
        if not usable_controls:
            base["residual_status"] = "no_valid_controls"
            rows.append(base)
            continue
        pair = pair[[target, column, *usable_controls]].dropna()
        usable_controls = [
            control for control in usable_controls if pair[control].nunique(dropna=True) > 1
        ]
        if not usable_controls:
            base["residual_status"] = "no_valid_controls"
            rows.append(base)
            continue
        pair = pair[[target, column, *usable_controls]].dropna()
        if len(pair) < max(10, max_lag + 5):
            base.update({
                "effective_control_columns": ",".join(usable_controls),
                "control_count": len(usable_controls),
                "residual_n": len(pair),
                "residual_status": "insufficient_joint_samples",
            })
            rows.append(base)
            continue
        control_matrix = np.column_stack([np.ones(len(pair)), pair[usable_controls].to_numpy(dtype=float)])
        matrix_rank = int(np.linalg.matrix_rank(control_matrix))
        condition_number = float(np.linalg.cond(control_matrix))
        target_residual, method, _, _ = _residualize(pair[target], pair[usable_controls])
        candidate_residual, _, _, _ = _residualize(pair[column], pair[usable_controls])
        residual_pair = pd.DataFrame({target: target_residual, column: candidate_residual}, index=pair.index)
        best = _best_lag_review_scores(
            residual_pair, target, max_lag, (best_lags or {}).get(column)
        )
        base.update({
            "effective_control_columns": ",".join(usable_controls),
            "control_count": len(usable_controls),
            "control_matrix_rank": matrix_rank,
            "control_condition_number": condition_number,
            "residual_method": "ols_rank_deficient" if matrix_rank < control_matrix.shape[1] else method,
            "residual_status": "rank_deficient" if matrix_rank < control_matrix.shape[1] else "ok",
        })
        if best.empty:
            base["residual_status"] = "no_valid_residual_lag" if matrix_rank == control_matrix.shape[1] else "rank_deficient_no_valid_residual_lag"
            rows.append(base)
            continue
        best_row = best.iloc[0]
        # Lag quality is assessed against the full physical-time lag curve.
        lag_scores = compute_lag_scores(residual_pair, target, max_lag)
        quality = build_lag_peak_quality(lag_scores, max_lag)
        quality_row = quality.loc[quality["variable"] == column]
        method_name = str(best_row["method"])
        signed = best_row[method_name]
        base.update({
            "residual_pearson": best_row["pearson"],
            "residual_spearman": best_row["spearman"],
            "residual_signed_corr": signed,
            "residual_corr": best_row["score"],
            "residual_lag": int(best_row["lag"]),
            "residual_direction": best_row["direction"],
            "residual_n": int(best_row["n"]),
            "residual_lag_boundary_flag": bool(best_row["lag_boundary_flag"]),
            "residual_lag_quality": (
                quality_row.iloc[0]["lag_quality"] if not quality_row.empty else np.nan
            ),
        })
        rows.append(base)
    return pd.DataFrame(rows, columns=RESIDUAL_SCORE_COLUMNS)


def _residual_result_row(variable: str, requested_controls: list[str]) -> dict[str, object]:
    row: dict[str, object] = {column: np.nan for column in RESIDUAL_SCORE_COLUMNS}
    row.update({
        "variable": variable,
        "requested_control_columns": ",".join(requested_controls),
        "effective_control_columns": "",
        "control_count": 0,
        "control_matrix_rank": np.nan,
        "control_condition_number": np.nan,
    })
    return row


def regime_scores(
    frame: pd.DataFrame,
    target: str,
    capacity_column: str | None,
    max_lag: int,
    best_lags: Mapping[str, int] | None = None,
    target_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = ["variable", "regime", "regime_row_count", "score", "signed_corr", "lag", "direction", "p_value", "r2"]
    if not capacity_column or capacity_column not in frame.columns:
        return pd.DataFrame(columns=score_cols), pd.DataFrame(columns=REGIME_STABILITY_COLUMNS)

    capacity = pd.to_numeric(frame[capacity_column], errors="coerce")
    resolved_mask = (
        target_mask.reindex(frame.index).fillna(False).astype(bool)
        if target_mask is not None
        else pd.Series(True, index=frame.index, dtype=bool)
    )
    selected_capacity = capacity.where(resolved_mask)
    q1 = selected_capacity.quantile(1 / 3)
    q2 = selected_capacity.quantile(2 / 3)
    regimes = dict(zip(REGIME_NAMES, [
        resolved_mask & (capacity <= q1),
        resolved_mask & (capacity > q1) & (capacity <= q2),
        resolved_mask & (capacity > q2),
    ]))

    all_rows: list[pd.DataFrame] = []
    for name, regime_mask in regimes.items():
        if int(regime_mask.sum()) < max(10, max_lag + 5):
            continue
        regime_rows: list[pd.DataFrame] = []
        for column in frame.columns:
            if column == target:
                continue
            best = _best_lag_review_scores(
                frame[[target, column]],
                target,
                max_lag,
                (best_lags or {}).get(column),
                target_mask=regime_mask,
            )
            if best.empty:
                continue
            best["signed_corr"] = np.where(best["method"].eq("pearson"), best["pearson"], best["spearman"])
            best = best.assign(regime=name, regime_row_count=best["n"].astype(int))
            regime_rows.append(best[score_cols])
        if regime_rows:
            all_rows.append(
                pd.concat(regime_rows, ignore_index=True)
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )

    if not all_rows:
        return pd.DataFrame(columns=score_cols), pd.DataFrame(columns=REGIME_STABILITY_COLUMNS)

    scores = pd.concat(all_rows, ignore_index=True)
    return scores, _summarize_regime_robustness(scores, max_lag)


def _best_lag_review_scores(
    pair: pd.DataFrame,
    target: str,
    max_lag: int,
    primary_best_lag,
    target_mask: pd.Series | None = None,
) -> pd.DataFrame:
    def scan(lag_values=None) -> pd.DataFrame:
        if target_mask is None:
            return compute_lag_scores(pair, target, max_lag, lag_values=lag_values)
        return compute_lag_scores(
            pair,
            target,
            max_lag,
            lag_values=lag_values,
            target_mask=target_mask,
        )

    limit = max(0, int(max_lag))
    if limit == 0:
        return summarize_best_lags(scan([0]))

    primary_lag = _valid_primary_lag(primary_best_lag, limit)
    if primary_lag is None or abs(primary_lag) == limit:
        return summarize_best_lags(scan())

    radius = min(limit, max(3, int(np.ceil(limit * 0.05))))
    lower = max(-limit, primary_lag - radius)
    upper = min(limit, primary_lag + radius)
    best = summarize_best_lags(scan(range(lower, upper + 1)))
    if best.empty:
        return summarize_best_lags(scan())

    best_lag = int(best.iloc[0]["lag"])
    touches_local_boundary = (
        (best_lag == lower and lower != -limit)
        or (best_lag == upper and upper != limit)
    )
    if touches_local_boundary:
        return summarize_best_lags(scan())
    return best


def _valid_primary_lag(value, max_lag: int) -> int | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    lag = int(numeric)
    return lag if abs(lag) <= max_lag else None


def _summarize_regime_robustness(scores: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    if scores.empty or not {"variable", "regime"}.issubset(scores.columns):
        return pd.DataFrame(columns=REGIME_STABILITY_COLUMNS)

    cleaned = scores.copy(deep=True)
    cleaned = cleaned[cleaned["regime"].isin(REGIME_NAMES)]
    cleaned = cleaned.drop_duplicates(subset=["variable", "regime"], keep="first")
    if cleaned.empty:
        return pd.DataFrame(columns=REGIME_STABILITY_COLUMNS)
    for column in ["score", "signed_corr", "lag"]:
        cleaned[column] = pd.to_numeric(cleaned.get(column), errors="coerce")

    rows: list[dict[str, object]] = []
    for variable, group in cleaned.groupby("variable", sort=False):
        regime_count = int(group["regime"].nunique())
        coverage = float(np.clip(regime_count / len(REGIME_NAMES), 0.0, 1.0))
        valid_scores = group["score"].dropna().clip(0, 1)
        valid_signed = group["signed_corr"].dropna()
        valid_lags = group["lag"].dropna()
        score_mean = float(valid_scores.mean()) if not valid_scores.empty else np.nan
        score_cv = (
            float(valid_scores.std(ddof=1) / abs(score_mean))
            if len(valid_scores) >= 2 and abs(score_mean) > 1e-12
            else np.nan
        )
        enough_metrics = all(
            len(values) >= MIN_REGIMES_FOR_STABILITY
            for values in [valid_scores, valid_signed, valid_lags]
        )

        reversal = bool((valid_signed > 0).any() and (valid_signed < 0).any())
        strength = sign = lag_consistency = consistency = final_score = np.nan
        if regime_count < MIN_REGIMES_FOR_STABILITY:
            status = "insufficient_regimes"
        elif not enough_metrics:
            status = "insufficient_metrics"
        else:
            score_max = float(valid_scores.max())
            score_min = float(valid_scores.min())
            strength = float(np.clip(score_min / score_max if score_max > 1e-12 else 0.0, 0, 1))
            signed_strength = float(valid_signed.abs().sum())
            sign = float(
                np.clip(
                    abs(float(valid_signed.sum())) / signed_strength
                    if signed_strength > 1e-12
                    else 0.0,
                    0,
                    1,
                )
            )
            lag_std = float(np.std(valid_lags, ddof=0))
            lag_consistency = float(np.clip(1.0 - lag_std / max(1.0, float(max_lag)), 0, 1))
            shape = (
                REGIME_CONSISTENCY_WEIGHTS["strength"] * strength
                + REGIME_CONSISTENCY_WEIGHTS["lag"] * lag_consistency
            )
            consistency = float(np.clip(sign * shape, 0, 1))
            final_score = float(np.clip(coverage * consistency, 0, 1))
            status = "full_coverage" if regime_count == len(REGIME_NAMES) else "partial_coverage"

        rows.append({
            "variable": variable,
            "regime_stability_final": final_score,
            "regime_consistency_score": consistency,
            "regime_coverage": coverage,
            "regime_strength_consistency": strength,
            "regime_sign_consistency": sign,
            "regime_lag_consistency": lag_consistency,
            "regime_score_cv": score_cv,
            "regime_count": regime_count,
            "regime_evidence_status": status,
            "regime_sign_reversal_flag": reversal,
        })
    return pd.DataFrame(rows, columns=REGIME_STABILITY_COLUMNS)


def model_lift_scores(frame: pd.DataFrame, target: str, candidate_variables: list[str], max_lag: int, n_splits: int = 4, best_lags: dict[str, int] | None = None, target_mask: pd.Series | None = None) -> pd.DataFrame:
    cols = [
        "variable", "status", "ar_baseline_rmse", "candidate_rmse", "model_lift",
        "median_fold_lift", "positive_fold_ratio", "model_lift_score",
    ]
    rows: list[dict[str, object]] = []
    ar_lags = list(range(1, min(max_lag, 6) + 1))
    period_ns = sample_period_ns(frame)
    for variable in candidate_variables:
        if variable == target or variable not in frame.columns:
            continue
        best_lag = best_lags.get(variable) if best_lags else None
        if best_lag is not None and pd.notna(best_lag) and int(best_lag) <= 0:
            rows.append({"variable": variable, "status": "non_predictive_lag", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": np.nan, "median_fold_lift": np.nan, "positive_fold_ratio": np.nan, "model_lift_score": np.nan})
            continue
        candidate_lags = [lag for lag in _nearby_lags(best_lag, max_lag) if lag >= 1]
        if not candidate_lags:
            rows.append({"variable": variable, "status": "non_predictive_lag", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": np.nan, "median_fold_lift": np.nan, "positive_fold_ratio": np.nan, "model_lift_score": np.nan})
            continue
        dataset = pd.DataFrame(index=frame.index)
        dataset[target] = frame[target]
        for lag in ar_lags:
            dataset[f"{target}__lag_{lag}"] = lagged_series(
                frame[target], frame.index, lag, period_ns=period_ns
            )
        for lag in candidate_lags:
            lagged_candidate = lagged_series(
                frame[variable], frame.index, lag, period_ns=period_ns
            )
            dataset[f"{variable}__lag_{lag}"] = lagged_candidate
            # PR-8C nonlinear_stable_driver: expose a quadratic incremental basis.
            dataset[f"{variable}__lag_{lag}__squared"] = lagged_candidate.pow(2)
        if target_mask is not None:
            dataset = dataset.loc[target_mask.reindex(dataset.index).fillna(False).astype(bool)]
        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
        if len(dataset) < 60:
            rows.append({"variable": variable, "status": "skipped: insufficient rows", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": np.nan, "median_fold_lift": np.nan, "positive_fold_ratio": np.nan, "model_lift_score": np.nan})
            continue
        base_cols = [f"{target}__lag_{lag}" for lag in ar_lags]
        full_cols = base_cols + [
            feature
            for lag in candidate_lags
            for feature in [f"{variable}__lag_{lag}", f"{variable}__lag_{lag}__squared"]
        ]
        base_errors: list[float] = []
        full_errors: list[float] = []
        splits = _time_series_splits(len(dataset), n_splits)
        if not splits:
            rows.append({"variable": variable, "status": "skipped: no valid time series split", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": np.nan, "median_fold_lift": np.nan, "positive_fold_ratio": np.nan, "model_lift_score": np.nan})
            continue
        for train_idx, test_idx in splits:
            y_train = dataset.iloc[train_idx][target].to_numpy()
            y_test = dataset.iloc[test_idx][target].to_numpy()
            base_pred = _linear_predict(dataset.iloc[train_idx][base_cols], y_train, dataset.iloc[test_idx][base_cols])
            full_pred = _linear_predict(dataset.iloc[train_idx][full_cols], y_train, dataset.iloc[test_idx][full_cols])
            base_errors.append(_rmse(y_test, base_pred))
            full_errors.append(_rmse(y_test, full_pred))
        base_rmse = float(np.mean(base_errors))
        full_rmse = float(np.mean(full_errors))
        if np.isnan(base_rmse) or np.isnan(full_rmse):
            rows.append({"variable": variable, "status": "skipped: no valid time series split", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": np.nan, "median_fold_lift": np.nan, "positive_fold_ratio": np.nan, "model_lift_score": np.nan})
            continue
        lift = max(0.0, (base_rmse - full_rmse) / base_rmse) if base_rmse > 0 else 0.0
        fold_lifts = np.array([
            (base - full) / base if base > 0 else 0.0
            for base, full in zip(base_errors, full_errors)
        ])
        median_fold_lift = float(np.median(fold_lifts))
        positive_fold_ratio = float(np.mean(fold_lifts > 0))
        lift_strength = float(np.clip(max(0.0, median_fold_lift) / 0.05, 0.0, 1.0))
        model_lift_score = lift_strength * positive_fold_ratio
        rows.append({"variable": variable, "status": "ok", "ar_baseline_rmse": base_rmse, "candidate_rmse": full_rmse, "model_lift": lift, "median_fold_lift": median_fold_lift, "positive_fold_ratio": positive_fold_ratio, "model_lift_score": model_lift_score})
    return pd.DataFrame(rows, columns=cols)


def pair_alignment_key(pair: pd.DataFrame) -> str:
    index_hashes = pd.util.hash_pandas_object(pair.index, index=False).to_numpy(
        dtype=np.uint64,
        copy=False,
    )
    digest = hashlib.sha256()
    digest.update(str(len(pair)).encode("utf-8"))
    digest.update(str(pair.index.dtype).encode("utf-8"))
    digest.update(index_hashes.tobytes())
    return digest.hexdigest()[:24]


def prepare_best_lag_evidence(
    frame: pd.DataFrame,
    target: str,
    candidate_variables: list[str],
    max_lag: int,
    ranked: pd.DataFrame | None = None,
    allow_ranked_reuse: bool = True,
    ranked_source_frame: pd.DataFrame | None = None,
    target_mask: pd.Series | None = None,
) -> tuple[dict[str, BestLagEvidence], dict[str, int]]:
    evidence: dict[str, BestLagEvidence] = {}
    diagnostics = {
        "reused_evidence_count": 0,
        "recomputed_evidence_count": 0,
        "invalid_evidence_count": 0,
    }
    if target not in frame.columns or max_lag < 0:
        return evidence, diagnostics

    for variable in dict.fromkeys(candidate_variables):
        if variable == target or variable not in frame.columns:
            continue
        current_pair = frame[[target, variable]].dropna()
        current_alignment_key = pair_alignment_key(current_pair)
        ranked_row = _ranked_row(ranked, variable)
        if allow_ranked_reuse and ranked_row is not None:
            source_columns_available = (
                ranked_source_frame is not None
                and target in ranked_source_frame.columns
                and variable in ranked_source_frame.columns
            )
            if source_columns_available:
                source_pair = ranked_source_frame[[target, variable]].dropna()
                source_alignment_key = pair_alignment_key(source_pair)
                source_matches_current = (
                    source_alignment_key == current_alignment_key
                    and source_pair.index.equals(current_pair.index)
                )
                if source_matches_current:
                    candidate = _evidence_from_ranked_row(
                        ranked_row,
                        max_lag,
                        source_alignment_key,
                    )
                    if _validated_best_lag_evidence(candidate, source_pair, max_lag) is not None:
                        evidence[variable] = candidate
                        diagnostics["reused_evidence_count"] += 1
                        continue
            diagnostics["invalid_evidence_count"] += 1
        if len(current_pair) < max(10, max_lag + 5):
            continue
        scores = (
            compute_lag_scores(current_pair, target, max_lag)
            if target_mask is None
            else compute_lag_scores(current_pair, target, max_lag, target_mask=target_mask)
        )
        best = summarize_best_lags(scores)
        diagnostics["recomputed_evidence_count"] += 1
        if best.empty:
            evidence[variable] = {
                "best_lag": None,
                "best_score": None,
                "max_lag": int(max_lag),
                "pair_alignment_key": current_alignment_key,
                "source": "recomputed",
                "status": "scanned_no_result",
            }
            continue
        best_row = best.iloc[0]
        evidence[variable] = {
            "best_lag": int(best_row["lag"]),
            "best_score": float(best_row["score"]),
            "max_lag": int(max_lag),
            "pair_alignment_key": current_alignment_key,
            "source": "recomputed",
            "status": "ok",
        }
    return evidence, diagnostics


def _ranked_row(ranked: pd.DataFrame | None, variable: str) -> pd.Series | None:
    if ranked is None or ranked.empty or not {"variable", "lag"}.issubset(ranked.columns):
        return None
    matches = ranked.loc[ranked["variable"].astype(str).eq(variable)]
    return None if matches.empty else matches.iloc[0]


def _evidence_from_ranked_row(
    row: pd.Series,
    max_lag: int,
    alignment_key: str,
) -> BestLagEvidence:
    score = row.get("score", row.get("raw_corr", np.nan))
    return {
        "best_lag": row.get("lag"),
        "best_score": score,
        "max_lag": int(max_lag),
        "pair_alignment_key": alignment_key,
        "source": "ranked",
        "status": "ok",
    }


def _is_scanned_no_result_evidence(
    evidence: Mapping[str, object] | None,
    pair: pd.DataFrame,
    max_lag: int,
) -> bool:
    if not isinstance(evidence, Mapping) or evidence.get("status") != "scanned_no_result":
        return False
    try:
        evidence_max_lag = float(evidence.get("max_lag"))
    except (TypeError, ValueError):
        return False
    return (
        np.isfinite(evidence_max_lag)
        and evidence_max_lag.is_integer()
        and int(evidence_max_lag) == max_lag
        and evidence.get("best_lag") is None
        and evidence.get("best_score") is None
        and evidence.get("source") == "recomputed"
        and evidence.get("pair_alignment_key") == pair_alignment_key(pair)
    )


def _validated_best_lag_evidence(
    evidence: Mapping[str, object] | None,
    pair: pd.DataFrame,
    max_lag: int,
) -> tuple[int, float] | None:
    if not isinstance(evidence, Mapping):
        return None
    try:
        lag_value = float(evidence.get("best_lag"))
        score = float(evidence.get("best_score"))
        evidence_max_lag = float(evidence.get("max_lag"))
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in [lag_value, score, evidence_max_lag]):
        return None
    if not lag_value.is_integer() or not evidence_max_lag.is_integer():
        return None
    lag = int(lag_value)
    if int(evidence_max_lag) != max_lag or not -max_lag <= lag <= max_lag:
        return None
    if not 0.0 <= score <= 1.0 or not str(evidence.get("source", "")).strip():
        return None
    if evidence.get("pair_alignment_key") != pair_alignment_key(pair):
        return None
    return lag, score


def rolling_corr_scores(
    frame: pd.DataFrame,
    target: str,
    candidate_variables: list[str],
    max_lag: int,
    window: int | None = None,
    min_periods: int | None = None,
    best_lag_evidence: Mapping[str, Mapping[str, object]] | None = None,
    target_mask: pd.Series | None = None,
) -> pd.DataFrame:
    cols = ["variable", "best_lag", "best_score", "rolling_corr_median", "rolling_abs_corr_median", "rolling_corr_iqr", "rolling_sign_consistency", "valid_window_count", "rolling_stability"]
    if target not in frame.columns or not candidate_variables:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, object]] = []
    window_size = max(12, int(window or min(len(frame), max(24, max_lag * 4))))
    min_points = max(6, int(min_periods or window_size // 2))
    for variable in candidate_variables:
        if variable == target or variable not in frame.columns:
            continue
        pair = frame[[target, variable]].dropna()
        if len(pair) < max(window_size, max_lag + 10):
            continue
        candidate_evidence = best_lag_evidence.get(variable) if best_lag_evidence else None
        if _is_scanned_no_result_evidence(candidate_evidence, pair, max_lag):
            continue
        prepared = _validated_best_lag_evidence(
            candidate_evidence,
            pair,
            max_lag,
        )
        if prepared is None:
            scores = (
                compute_lag_scores(pair, target, max_lag)
                if target_mask is None
                else compute_lag_scores(pair, target, max_lag, target_mask=target_mask)
            )
            best = summarize_best_lags(scores)
            if best.empty:
                continue
            best_row = best.iloc[0]
            best_lag = int(best_row["lag"])
            best_score = float(best_row.get("score", 0.0) or 0.0)
        else:
            best_lag, best_score = prepared
        shifted = lagged_series(
            pair[variable], pair.index, best_lag, period_ns=sample_period_ns(frame)
        )
        rolling = shifted.rolling(window=window_size, min_periods=min_points).corr(pair[target])
        if target_mask is not None:
            rolling = rolling.where(target_mask.reindex(rolling.index).fillna(False).astype(bool))
        rolling = rolling.replace([np.inf, -np.inf], np.nan).dropna()
        if rolling.empty:
            continue
        sign_consistency = rolling.apply(lambda value: 1 if value >= 0 else -1).value_counts(normalize=True).max()
        iqr = float(rolling.quantile(0.75) - rolling.quantile(0.25))
        abs_median = float(rolling.abs().median())
        stability = max(0.0, min(1.0, abs_median * float(sign_consistency) * (1.0 - min(1.0, iqr))))
        rows.append({"variable": variable, "best_lag": best_lag, "best_score": best_score, "rolling_corr_median": float(rolling.median()), "rolling_abs_corr_median": abs_median, "rolling_corr_iqr": iqr, "rolling_sign_consistency": float(sign_consistency), "valid_window_count": int(len(rolling)), "rolling_stability": stability})
    return pd.DataFrame(rows, columns=cols)


def _safe_float(value: object, default: float = 0.0) -> float:
    return to_float(value, default)


def _risk_token_set(value: object) -> set[str]:
    if value is None:
        return set()
    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        return set()
    return {token.strip() for token in str(value).split(";") if token.strip()}


def _risk_adjustment(value: object) -> tuple[float, float, str]:
    tokens = _risk_token_set(value)
    penalty_rate = min(0.80, sum(RISK_RELATIVE_PENALTY_WEIGHTS.get(token, 0.0) for token in tokens))
    cap = 1.0
    reason = ""
    for token, token_cap in EVIDENCE_SCORE_CAPS.items():
        if token in tokens and token_cap < cap:
            cap = token_cap
            reason = token
    return float(penalty_rate), float(cap), reason


def classify_candidate(row: pd.Series) -> str:
    flags = _risk_token_set(row.get("risk_flags", ""))
    for token, candidate_class in [
        ("strong_formula_leakage", "formula_or_derived"),
        ("poor_data_quality", "poor_quality"),
        ("target_leads_variable", "downstream_response"),
        ("common_capacity_driver", "capacity_driven"),
        ("redundant_proxy", "uncertain_candidate"),
    ]:
        if token in flags:
            return candidate_class

    lag_value = row.get("best_lag", row.get("lag", pd.NA))
    try:
        if pd.isna(lag_value):
            return "uncertain_candidate"
    except (TypeError, ValueError):
        return "uncertain_candidate"
    lag = _safe_float(lag_value, default=np.nan)
    if np.isnan(lag):
        return "uncertain_candidate"
    if lag > 0:
        return "upstream_driver_candidate"
    if lag == 0:
        return "synchronous_association"
    return "uncertain_candidate"


def _combine_correlation_evidence(
    association_score: pd.Series,
    independent_signal_score: pd.Series,
    innovation_score: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    association = pd.to_numeric(association_score, errors="coerce").astype(float)
    independent = pd.to_numeric(independent_signal_score, errors="coerce").astype(float)
    innovation = pd.to_numeric(innovation_score, errors="coerce").astype(float)
    innovation_verified = innovation.notna()
    independent_verified = independent.notna()
    available = pd.DataFrame(
        {
            "association": association.clip(0, 1),
            "innovation": innovation.clip(0, 1),
            "independent": independent.clip(0, 1),
        }
    )
    available_count = available.notna().sum(axis=1)
    combined = available.prod(axis=1, skipna=True).pow(
        1.0 / available_count.where(available_count.gt(0))
    )
    combined = combined.where(association.notna())
    status = pd.Series("association_only", index=association_score.index, dtype=object)
    status.loc[innovation_verified] = "innovation_verified"
    status.loc[independent_verified & ~innovation_verified] = "independent_verified"
    status.loc[independent_verified & innovation_verified] = "innovation_and_independent_verified"
    return combined.clip(0, 1), status


def _data_quality_score(diag: Mapping[str, object]) -> float:
    rates = np.array(
        [
            max(0.0, _safe_float(diag.get("missing_rate", 0.0))),
            max(0.0, _safe_float(diag.get("saturation_ratio", 0.0))),
            max(0.0, _safe_float(diag.get("abnormal_jump_ratio", 0.0))),
            max(0.0, _safe_float(diag.get("robust_outlier_ratio", 0.0))),
        ]
    )
    reference_rates = np.array([0.20, 0.20, 0.01, 0.01])
    quality_components = np.exp(-np.log(2) * rates / reference_rates)
    return float(
        np.clip(
            np.prod(quality_components) ** (1 / len(quality_components)), 0.0, 1.0
        )
    )


def _redundant_proxy_variables(
    frame: pd.DataFrame | None,
    ranked: pd.DataFrame,
    target: str | None,
    *,
    residual_map: Mapping[str, object] | None = None,
    stability_map: Mapping[str, Mapping[str, object]] | None = None,
    diag_map: Mapping[str, Mapping[str, object]] | None = None,
    lag_map: Mapping[str, Mapping[str, object]] | None = None,
    rolling_map: Mapping[str, Mapping[str, object]] | None = None,
    lift_map: Mapping[str, Mapping[str, object]] | None = None,
) -> set[str]:
    """Resolve redundant positive-lag candidate groups from computed evidence."""
    if frame is None or not target or target not in frame.columns:
        return set()
    period_ns = sample_period_ns(frame)
    residual_map = residual_map or {}
    stability_map = stability_map or {}
    diag_map = diag_map or {}
    lag_map = lag_map or {}
    rolling_map = rolling_map or {}
    lift_map = lift_map or {}
    candidates = [
        str(row["variable"])
        for _, row in ranked.iterrows()
        if str(row.get("variable", "")) in frame.columns
        and str(row.get("variable", "")) != target
        and _safe_float(row.get("lag", 0), default=0.0) > 0
    ]
    candidate_lags = {
        str(row["variable"]): int(_safe_float(row.get("lag", 0), default=0.0))
        for _, row in ranked.iterrows()
    }
    adjacency = {variable: set() for variable in candidates}
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            pair = pd.DataFrame(
                {
                    left: lagged_series(frame[left], frame.index, candidate_lags[left], period_ns=period_ns),
                    right: lagged_series(frame[right], frame.index, candidate_lags[right], period_ns=period_ns),
                }
            ).dropna()
            if len(pair) < 30 or abs(float(pair[left].corr(pair[right]))) < 0.995:
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)

    profiles = {
        str(row["variable"]): _redundancy_evidence_profile(
            row,
            residual_map=residual_map,
            stability_map=stability_map,
            diag_map=diag_map,
            lag_map=lag_map,
            rolling_map=rolling_map,
            lift_map=lift_map,
        )
        for _, row in ranked.iterrows()
        if str(row.get("variable", "")) in adjacency
    }
    redundant: set[str] = set()
    visited: set[str] = set()
    for variable in candidates:
        if variable in visited:
            continue
        stack = [variable]
        group: set[str] = set()
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            stack.extend(adjacency[current] - group)
        visited.update(group)
        if len(group) < 2:
            continue
        representatives = [
            candidate
            for candidate in group
            if all(
                candidate == other
                or _evidence_clearly_exceeds(
                    profiles[candidate], profiles[other]
                )
                for other in group
            )
        ]
        if len(representatives) == 1:
            redundant.update(group - set(representatives))
        else:
            redundant.update(group)
    return redundant


def _redundancy_evidence_profile(
    row: pd.Series,
    *,
    residual_map: Mapping[str, object],
    stability_map: Mapping[str, Mapping[str, object]],
    diag_map: Mapping[str, Mapping[str, object]],
    lag_map: Mapping[str, Mapping[str, object]],
    rolling_map: Mapping[str, Mapping[str, object]],
    lift_map: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    variable = str(row["variable"])
    stability = stability_map.get(variable, {})
    rolling = rolling_map.get(variable, {})
    lift = lift_map.get(variable, {})
    return {
        "independence": _safe_float(residual_map.get(variable, np.nan), default=np.nan),
        "prediction": _safe_float(
            lift.get("model_lift_score", lift.get("model_lift", np.nan)), default=np.nan
        ),
        "data_quality": _data_quality_score(diag_map.get(variable, {})),
        "stability": _safe_float(
            stability.get("regime_stability_final", rolling.get("rolling_stability", np.nan)),
            default=np.nan,
        ),
        "lag_quality": _safe_float(lag_map.get(variable, {}).get("lag_quality", np.nan), default=np.nan),
        "association": _safe_float(
            row.get("association_score", row.get("score", np.nan)), default=np.nan
        ),
    }


def _evidence_clearly_exceeds(
    candidate: Mapping[str, float], other: Mapping[str, float]
) -> bool:
    differences = [
        candidate[field] - other[field]
        for field in candidate
        if np.isfinite(candidate[field]) and np.isfinite(other.get(field, np.nan))
    ]
    return bool(
        any(difference >= EVIDENCE_SEPARATION_MARGIN for difference in differences)
        and not any(difference <= -EVIDENCE_SEPARATION_MARGIN for difference in differences)
    )


def risk_flags(ranked: pd.DataFrame, residual: pd.DataFrame, stability: pd.DataFrame, diag: pd.DataFrame, roles: dict[str, str], control_columns: list[str] | None, lag_peak_quality: pd.DataFrame | None = None, rolling_corr_scores: pd.DataFrame | None = None, model_lift_scores: pd.DataFrame | None = None, *, frame: pd.DataFrame | None = None, target: str | None = None) -> pd.DataFrame:
    cols = ["variable", "formula_like_flag", "strong_formula_leakage_flag", "common_capacity_driver_flag", "redundant_proxy_flag", "target_leads_variable_flag", "unstable_across_regimes_flag", "unstable_over_time_flag", "lag_boundary_flag", "low_model_lift_flag", "poor_data_quality_flag", "residual_collinearity_flag", "data_quality_score", "risk_flags", "risk_count", "strong_risk_count", "weak_risk_count", "risk_level", "human_reason"]
    if ranked.empty:
        return pd.DataFrame(columns=cols)

    residual_map = residual.set_index("variable")["residual_corr"].to_dict() if not residual.empty and "residual_corr" in residual.columns else {}
    residual_cond_map = residual.set_index("variable")["condition_number"].to_dict() if not residual.empty and "condition_number" in residual.columns else {}
    stability_map = stability.set_index("variable").to_dict("index") if not stability.empty else {}
    diag_map = diag.set_index("variable").to_dict("index") if not diag.empty else {}
    lag_map = lag_peak_quality.set_index("variable").to_dict("index") if lag_peak_quality is not None and not lag_peak_quality.empty else {}
    roll_map = rolling_corr_scores.set_index("variable").to_dict("index") if rolling_corr_scores is not None and not rolling_corr_scores.empty else {}
    lift_map = model_lift_scores.set_index("variable").to_dict("index") if model_lift_scores is not None and not model_lift_scores.empty else {}
    redundant_variables = _redundant_proxy_variables(
        frame,
        ranked,
        target,
        residual_map=residual_map,
        stability_map=stability_map,
        diag_map=diag_map,
        lag_map=lag_map,
        rolling_map=roll_map,
        lift_map=lift_map,
    )

    rows = []
    for _, row in ranked.iterrows():
        variable = str(row.get("variable", ""))
        raw_corr = _safe_float(row.get("score", 0), default=0.0)
        residual_corr = _safe_float(residual_map.get(variable, raw_corr), default=raw_corr)
        regime_info = stability_map.get(variable, {})
        regime_stability = regime_info.get("regime_stability_final", np.nan)
        regime_status = str(regime_info.get("regime_evidence_status", ""))
        d = diag_map.get(variable, {})
        data_quality_score = _data_quality_score(d)
        poor_quality = (
            _safe_float(d.get("missing_rate", 0), default=0.0) > 0.2
            or _safe_float(d.get("saturation_ratio", 0), default=0.0) > 0.2
            or _safe_float(d.get("abnormal_jump_ratio", 0), default=0.0) > 0.01
            or _safe_float(d.get("robust_outlier_ratio", 0), default=0.0) > 0.01
        )
        lag_value = int(_safe_float(row.get("lag", 0), default=0.0))
        formula_like = _looks_like_formula_variable(variable)
        strong_formula = formula_like and raw_corr > 0.98 and lag_value == 0
        common_capacity = bool(control_columns) and raw_corr >= 0.5 and residual_corr < raw_corr * 0.65
        target_leads = lag_value < 0
        regime_evaluated = regime_status in {"partial_coverage", "full_coverage"}
        unstable_reg = (
            regime_evaluated
            and pd.notna(regime_stability)
            and float(regime_stability) < REGIME_UNSTABLE_THRESHOLD
        )
        lift_info = lift_map.get(variable, {})
        model_supported = str(lift_info.get("status", "")).startswith("ok") and _safe_float(
            lift_info.get("model_lift", 0.0), default=0.0
        ) >= 0.05
        unstable_time = _safe_float(
            roll_map.get(variable, {}).get("rolling_stability", 1.0), default=1.0
        ) < 0.35 and not (model_supported and raw_corr < 0.2)
        lag_boundary = bool(lag_map.get(variable, {}).get("lag_boundary_flag", False))
        low_lift = str(lift_info.get("status", "")).startswith("ok") and _safe_float(
            lift_info.get("model_lift", 0.0), default=0.0
        ) < 0.01
        residual_collinearity = _safe_float(residual_cond_map.get(variable, 0), default=0.0) > 1e8

        flags = [name for name, active in [
            ("formula_like", formula_like),
            ("strong_formula_leakage", strong_formula),
            ("common_capacity_driver", common_capacity),
            ("redundant_proxy", variable in redundant_variables),
            ("target_leads_variable", target_leads),
            ("unstable_across_regimes", unstable_reg),
            ("unstable_over_time", unstable_time),
            ("lag_boundary", lag_boundary),
            ("low_model_lift", low_lift),
            ("poor_data_quality", poor_quality),
            ("residual_collinearity", residual_collinearity),
        ] if active]

        strong_risks = [f for f in flags if f in {"strong_formula_leakage", "common_capacity_driver", "poor_data_quality"}]
        weak_risks = [f for f in flags if f not in set(strong_risks)]
        level = "none" if not flags else ("strong" if len(strong_risks) >= 2 else ("medium" if strong_risks else "weak"))
        reason_map = {
            "formula_like": "疑似公式类变量",
            "strong_formula_leakage": "强公式泄漏风险",
            "common_capacity_driver": "疑似共同负荷驱动",
            "target_leads_variable": "目标领先变量",
            "unstable_across_regimes": "跨工况不稳定",
            "unstable_over_time": "随时间不稳定",
            "lag_boundary": "滞后触边界",
            "low_model_lift": "模型增益偏低",
            "poor_data_quality": "数据质量较差",
            "residual_collinearity": "残差控制共线性高",
            "redundant_proxy": "与其他候选变量高度冗余，独立信息不足",
        }
        reason = "；".join(reason_map.get(flag, flag) for flag in flags)
        rows.append({"variable": variable, "formula_like_flag": formula_like, "strong_formula_leakage_flag": strong_formula, "common_capacity_driver_flag": common_capacity, "redundant_proxy_flag": variable in redundant_variables, "target_leads_variable_flag": target_leads, "unstable_across_regimes_flag": unstable_reg, "unstable_over_time_flag": unstable_time, "lag_boundary_flag": lag_boundary, "low_model_lift_flag": low_lift, "poor_data_quality_flag": poor_quality, "residual_collinearity_flag": residual_collinearity, "data_quality_score": data_quality_score, "risk_flags": ";".join(flags), "risk_count": len(flags), "strong_risk_count": len(strong_risks), "weak_risk_count": len(weak_risks), "risk_level": level, "human_reason": reason})
    return pd.DataFrame(rows, columns=cols)


def final_ranked_features(ranked: pd.DataFrame, residual: pd.DataFrame, stability: pd.DataFrame, model_lift: pd.DataFrame, risks: pd.DataFrame, lag_peak_quality: pd.DataFrame, rolling_corr_scores: pd.DataFrame, force_include_variables: list[str] | None = None, top_k: int | None = None, control_columns: list[str] | None = None, capacity_columns: list[str] | None = None, segment_column: str | None = None) -> pd.DataFrame:
    cols = [
        "variable", "lag", "direction", "pearson", "spearman", "method",
        "pearson_p", "spearman_p", "pearson_q", "spearman_q", "corr_q_value",
        "pearson_r2", "spearman_r2", "n", "raw_corr", "association_score",
        "innovation_score", "innovation_lag", "innovation_direction", "innovation_sign",
        "innovation_status", "correlation_evidence_score", "correlation_evidence_status",
        "lag_quality", "lag_quality_status", "lag_boundary_flag", "data_quality_score",
        "evidence_strength", "evidence_available_count", "evidence_completeness",
        "evidence_confidence", "evidence_score_low", "evidence_score_high", "score_method",
        "risk_count", "strong_risk_count", "weak_risk_count", "risk_level", "human_reason",
        "risk_flags", "evidence_score", "risk_penalty_rate", "risk_penalty", "risk_score_cap",
        "risk_cap_reason", "final_score", "association_rank", "candidate_class",
        "driver_priority_factor", "driver_priority_score", "driver_rank", "candidate_grade",
        "recommended_use", "recommended_action", "force_included", "engineering_context",
        "is_residual_control", "is_capacity_reference", "is_segment_reference", "variable_role",
    ]
    if ranked.empty:
        return pd.DataFrame(columns=cols)
    final = ranked.rename(columns={"score": "raw_corr"}).copy()
    residual_source = residual[[c for c in ["variable", "residual_corr"] if c in residual.columns]].copy()
    if "variable" not in residual_source.columns:
        residual_source = pd.DataFrame(columns=["variable"])
    final = final.merge(residual_source, on="variable", how="left")
    stability_columns = list(REGIME_STABILITY_COLUMNS)
    stability_source = stability[[c for c in stability_columns if c in stability.columns]].copy()
    if "variable" not in stability_source.columns:
        stability_source = pd.DataFrame(columns=["variable"])
    # Upstream callers may already carry selected regime fields.  Keep the
    # freshly computed stability evidence as the single source for this stage.
    final = final.drop(columns=[c for c in stability_source.columns if c != "variable" and c in final.columns])
    final = final.merge(stability_source, on="variable", how="left")
    model_columns = [c for c in ["variable", "model_lift", "model_lift_score", "status"] if c in model_lift.columns]
    model_source = model_lift[model_columns].copy()
    if "status" in model_source.columns:
        model_source = model_source.rename(columns={"status": "_model_lift_source_status"})
    final = final.merge(model_source, on="variable", how="left")
    risk_columns = ["variable", "risk_flags", "risk_count", "strong_risk_count", "weak_risk_count", "risk_level", "human_reason", "data_quality_score"]
    risk_source = risks[[c for c in risk_columns if c in risks.columns]].copy()
    if "variable" not in risk_source.columns:
        risk_source = pd.DataFrame(columns=["variable"])
    final = final.merge(risk_source, on="variable", how="left")
    lag_peak_columns = ["variable", "lag_quality"]
    if "lag_boundary_flag" not in final.columns:
        lag_peak_columns.append("lag_boundary_flag")
    final = final.merge(
        lag_peak_quality[[c for c in lag_peak_columns if c in lag_peak_quality.columns]],
        on="variable",
        how="left",
    )
    final = final.merge(rolling_corr_scores[[c for c in ["variable", "rolling_stability"] if c in rolling_corr_scores.columns]], on="variable", how="left")
    # Some legacy callers pre-merge regime evidence.  Pandas then suffixes the
    # duplicate flag; normalize it before the layer-status contract is built.
    if "regime_sign_reversal_flag" not in final.columns:
        for suffix in ("_y", "_x"):
            legacy_flag = f"regime_sign_reversal_flag{suffix}"
            if legacy_flag in final.columns:
                final["regime_sign_reversal_flag"] = final[legacy_flag]
                break

    residual_raw = pd.to_numeric(final["residual_corr"], errors="coerce") if "residual_corr" in final.columns else pd.Series(np.nan, index=final.index, dtype=float)
    regime_raw = pd.to_numeric(final["regime_stability_final"], errors="coerce") if "regime_stability_final" in final.columns else pd.Series(np.nan, index=final.index, dtype=float)
    rolling_raw = pd.to_numeric(final["rolling_stability"], errors="coerce") if "rolling_stability" in final.columns else pd.Series(np.nan, index=final.index, dtype=float)
    lagq_raw = pd.to_numeric(final["lag_quality"], errors="coerce") if "lag_quality" in final.columns else pd.Series(np.nan, index=final.index, dtype=float)
    innovation_raw = pd.to_numeric(final["innovation_score"], errors="coerce") if "innovation_score" in final.columns else pd.Series(np.nan, index=final.index, dtype=float)
    if "model_lift_score" in final.columns:
        lift_raw = final["model_lift_score"]
    elif "model_lift" in final.columns:
        lift_raw = final["model_lift"]
    else:
        lift_raw = pd.Series(np.nan, index=final.index, dtype=float)
    lift_raw = pd.to_numeric(lift_raw, errors="coerce")
    final["residual_status"] = np.where(residual_raw.notna(), "ok", "not_computed")
    regime_evidence_status = final.get("regime_evidence_status", pd.Series(np.nan, index=final.index))
    final["regime_status"] = regime_evidence_status.where(
        regime_evidence_status.notna(), np.where(regime_raw.notna(), "ok", "not_computed")
    )
    final["rolling_status"] = np.where(rolling_raw.notna(), "ok", "not_computed")
    model_source_status = final.get("_model_lift_source_status", pd.Series(np.nan, index=final.index))
    final["model_lift_status"] = model_source_status.where(
        model_source_status.notna(), np.where(lift_raw.notna(), "ok", "not_computed")
    )
    final["lag_quality_status"] = np.where(lagq_raw.notna(), "ok", "not_computed")
    final["association_score"] = pd.to_numeric(final["raw_corr"], errors="coerce").fillna(0.0).clip(0, 1)
    final["innovation_score"] = pd.to_numeric(innovation_raw, errors="coerce").clip(0, 1)
    final["regime_stability_final"] = regime_raw.clip(0,1)
    final["rolling_stability"] = rolling_raw.clip(0,1)
    final["lag_quality"] = lagq_raw.clip(0,1)
    final["model_lift_score"] = lift_raw.clip(0,1)
    model_lift_computed = final["model_lift_status"].astype(str).str.startswith("ok")
    final["prediction_score"] = final["model_lift_score"].where(model_lift_computed)
    final["independent_signal_score"] = pd.to_numeric(residual_raw, errors="coerce").clip(0, 1)
    (
        final["correlation_evidence_score"],
        final["correlation_evidence_status"],
    ) = _combine_correlation_evidence(
        final["association_score"], final["independent_signal_score"], final["innovation_score"]
    )
    both_stability = final["regime_stability_final"].notna() & final["rolling_stability"].notna()
    final["stability_score"] = final["rolling_stability"].where(
        final["rolling_stability"].notna(), final["regime_stability_final"]
    ).astype(float)
    final.loc[both_stability, "stability_score"] = np.sqrt(
        final.loc[both_stability, "rolling_stability"]
        * final.loc[both_stability, "regime_stability_final"]
    )

    correlation_completeness = 0.5 + 0.5 * final["innovation_score"].notna().astype(float)
    final["evidence_completeness"] = (
        correlation_completeness
        + final["prediction_score"].notna().astype(float)
        + final["stability_score"].notna().astype(float)
        + final["lag_quality"].notna().astype(float)
    ) / 4.0
    data_quality_raw = final.get("data_quality_score", pd.Series(1.0, index=final.index))
    final["data_quality_score"] = pd.to_numeric(data_quality_raw, errors="coerce").fillna(1.0).clip(0, 1)
    # Coverage remains an output contract.  Only measured data quality adjusts
    # the score; unavailable optional evidence is omitted by profile reweighting.
    final["evidence_confidence"] = final["data_quality_score"]
    evidence_items = {
        "innovation_score": "变化量验证",
        "prediction_score": "模型提升",
        "stability_score": "稳定性验证",
        "lag_quality": "滞后质量",
    }
    missing_evidence = final[list(evidence_items)].isna()
    final["evidence_missing_items"] = missing_evidence.apply(
        lambda row: "；".join(
            label for field, label in evidence_items.items() if row[field]
        ),
        axis=1,
    )
    missing_count = missing_evidence.sum(axis=1)
    final["evidence_coverage_status"] = np.select(
        [missing_count.eq(0), missing_count.eq(1)],
        ["完整", "部分完整"],
        default="证据不足",
    )

    components = pd.DataFrame(
        {
            "association": final["correlation_evidence_score"],
            "prediction": final["prediction_score"],
            "stability": final["stability_score"],
            "lag_quality": final["lag_quality"],
        },
        index=final.index,
    )
    final["evidence_available_count"] = components.notna().sum(axis=1).astype(int)
    profile_scores = _available_weight_profile_scores(
        components, INDUSTRIAL_SCORE_WEIGHT_PROFILES
    )
    final["evidence_strength"] = profile_scores.median(axis=1).clip(0, 1)
    final["evidence_score_low"] = (
        profile_scores.quantile(0.10, axis=1) * final["evidence_confidence"]
    ).clip(0, 1)
    final["evidence_score_high"] = (
        profile_scores.quantile(0.90, axis=1) * final["evidence_confidence"]
    ).clip(0, 1)
    final["evidence_score"] = (
        final["evidence_strength"] * final["evidence_confidence"]
    ).clip(0, 1)
    final["score_method"] = "industrial_robust_v3"
    risk_values = final.get("risk_flags", pd.Series("", index=final.index)).map(_risk_adjustment)
    final[["risk_penalty_rate", "risk_score_cap", "risk_cap_reason"]] = pd.DataFrame(
        risk_values.tolist(), index=final.index
    )
    final["risk_penalty"] = final["evidence_score"] * final["risk_penalty_rate"]
    penalized_score = (final["evidence_score"] - final["risk_penalty"]).clip(0, 1)
    final["final_score"] = np.minimum(penalized_score, final["risk_score_cap"]).clip(0, 1)
    final["association_rank"] = final["evidence_score"].rank(
        method="first", ascending=False
    ).astype(int)
    final["candidate_class"] = final.apply(classify_candidate, axis=1)
    # These columns are retained only for readers of historical CSV files. They
    # are aliases of the statistical score and never affect screening behavior.
    final["driver_priority_factor"] = 1.0
    final["driver_priority_score"] = final["final_score"]
    final = _finalize_driver_ranking(
        final,
        force_include_variables=force_include_variables,
        control_columns=control_columns,
        capacity_columns=capacity_columns,
        segment_column=segment_column,
        primary_rank_column=PRIMARY_RANK_COLUMN,
    )
    final = order_initial_candidates(final)
    final["driver_rank"] = np.arange(1, len(final) + 1)
    for c in cols:
        if c not in final.columns:
            final[c] = np.nan
    return final.reset_index(drop=True)[cols]


def _finalize_driver_ranking(
    final: pd.DataFrame,
    force_include_variables: list[str] | None = None,
    top_k: int | None = None,
    control_columns: list[str] | None = None,
    capacity_columns: list[str] | None = None,
    segment_column: str | None = None,
    primary_rank_column: str = PRIMARY_RANK_COLUMN,
) -> pd.DataFrame:
    final = final.copy()
    final["driver_priority_score"] = final["final_score"]
    forced = set(force_include_variables or [])
    final["force_included"] = final["variable"].astype(str).isin(forced)
    final["candidate_grade"] = final.apply(_grade_candidate, axis=1)
    final["recommended_use"] = final.apply(_recommend_use, axis=1)
    variables = final["variable"].astype(str)
    residual_set = {str(value) for value in (control_columns or [])}
    capacity_set = {str(value) for value in (capacity_columns or [])}
    final["is_residual_control"] = variables.isin(residual_set)
    final["is_capacity_reference"] = variables.isin(capacity_set)
    final["is_segment_reference"] = bool(segment_column) & variables.eq(str(segment_column))
    final["variable_role"] = np.select(
        [final["is_residual_control"], final["is_capacity_reference"], final["is_segment_reference"]],
        ["residual_control", "capacity_reference", "segment_reference"],
        default="candidate",
    )
    final["recommended_action"] = final.apply(_recommended_action, axis=1)
    final = order_initial_candidates(final)
    final["driver_rank"] = np.arange(1, len(final) + 1)
    return final


def build_recommended_candidates(
    ranked_features: pd.DataFrame,
    top_k: int | None,
    force_include_variables: list[str] | None = None,
    exclude_control_columns: bool = True,
) -> pd.DataFrame:
    """Build the downstream candidate pool from the complete initial ranking."""
    if ranked_features.empty:
        return ranked_features.copy(deep=True)
    frame = ranked_features.copy(deep=True)
    forced = {str(value) for value in (force_include_variables or [])}
    variable = frame["variable"].astype(str)
    reference_columns = ["is_residual_control", "is_capacity_reference", "is_segment_reference"]
    references = frame.reindex(columns=reference_columns, fill_value=False).fillna(False).astype(bool).any(axis=1)
    eligible = ~references if exclude_control_columns else pd.Series(True, index=frame.index)
    top = order_initial_candidates(frame.loc[eligible]).head(top_k) if top_k is not None else order_initial_candidates(frame.loc[eligible])
    forced_rows = frame.loc[variable.isin(forced)]
    return order_initial_candidates(
        pd.concat([top, forced_rows], ignore_index=True).drop_duplicates(subset=["variable"], keep="first")
    ).reset_index(drop=True)


def order_initial_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Order initial candidates by score, evidence, lag quality, then variable."""
    if frame.empty:
        return frame.copy()
    ordered = frame.copy()
    ordered["_initial_final_score"] = pd.to_numeric(
        ordered.get("final_score", pd.Series(np.nan, index=ordered.index)), errors="coerce"
    ).fillna(-np.inf)
    ordered["_initial_association_score"] = pd.to_numeric(
        ordered.get("association_score", pd.Series(np.nan, index=ordered.index)), errors="coerce"
    ).fillna(-np.inf)
    ordered["_initial_lag_quality"] = pd.to_numeric(
        ordered.get("lag_quality", pd.Series(np.nan, index=ordered.index)), errors="coerce"
    ).fillna(-np.inf)
    ordered["_initial_variable"] = ordered.get("variable", pd.Series("", index=ordered.index)).astype(str)
    return ordered.sort_values(
        ["_initial_final_score", "_initial_association_score", "_initial_lag_quality", "_initial_variable"],
        ascending=[False, False, False, True],
        kind="stable",
    ).drop(columns=["_initial_final_score", "_initial_association_score", "_initial_lag_quality", "_initial_variable"])


def _grade_candidate(row: pd.Series) -> str:
    score = _safe_float(row.get("final_score", 0), default=0.0)
    grade = "A" if score >= 0.75 else "B" if score >= 0.6 else "C" if score >= 0.45 else "D" if score >= 0.3 else "E"
    flags = _risk_token_set(row.get("risk_flags", ""))
    # PR-8C downstream_response/lag_boundary: explicit temporal conflict is
    # retained for explanation but cannot receive an upstream high-confidence grade.
    if flags.intersection({"target_leads_variable", "lag_boundary"}):
        return max(grade, "C", key=lambda value: "ABCDE".index(value))
    # PR-8C nonlinear_stable_driver: validated incremental prediction can earn
    # a reviewable grade only when no explicit temporal, quality, or regime conflict exists.
    if (
        _safe_float(row.get("prediction_score", 0), default=0.0) > 0.05
        and _safe_float(row.get("lag", 0), default=0.0) > 0
        and not flags.intersection(
            {
                "strong_formula_leakage",
                "poor_data_quality",
                "unstable_across_regimes",
                "unstable_over_time",
            }
        )
    ):
        return min(grade, "C", key=lambda value: "ABCDE".index(value))
    return grade


def _recommend_use(row: pd.Series) -> str:
    flags = _risk_token_set(row.get("risk_flags", ""))
    grade = str(row.get("candidate_grade", "E"))
    if "poor_data_quality" in flags:
        return "poor_quality_variable"
    if "common_capacity_driver" in flags:
        return "capacity_driven"
    raw_corr = _safe_float(row.get("raw_corr", 0), default=0.0)
    lag = int(_safe_float(row.get("lag", 0), default=0.0))
    has_formula = "formula_like" in flags
    has_strong_formula = "strong_formula_leakage" in flags
    has_common = "common_capacity_driver" in flags
    if has_strong_formula or (has_formula and has_common) or (has_formula and lag == 0 and raw_corr >= 0.95):
        return "formula_coupled_reference"
    if "unstable_across_regimes" in flags or "unstable_over_time" in flags:
        return "unstable_candidate"
    if lag < 0:
        return "state_indicator"
    if grade == "A":
        return "strong_screening_candidate"
    if grade == "B" and _safe_float(row.get("model_lift_score", 0), default=0.0) > 0.05:
        return "prediction_candidate"
    return "manual_review_required"


def _recommended_action(row: pd.Series) -> str:
    use_value = row.get("recommended_use", "manual_review_required")
    use = "manual_review_required" if pd.isna(use_value) else str(use_value)
    mapping = {
        "strong_screening_candidate": "优先进入机理复核",
        "prediction_candidate": "可作为预测候选",
        "capacity_driven": "疑似共同负荷驱动",
        "formula_coupled_reference": "疑似公式耦合，仅参考",
        "unstable_candidate": "跨工况/时间不稳定，建议复核",
        "poor_quality_variable": "数据质量风险，建议剔除",
        "state_indicator": "更可能是状态指示量",
        "control_variable_reference": "残差/负荷控制变量，仅作控制基准参考。",
    }
    return mapping.get(use, "建议人工工艺复核")


def _sampling_period_seconds(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return np.nan
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    return float(diffs.median()) if len(diffs) else np.nan


def _longest_run(mask: pd.Series) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def _longest_constant_run(series: pd.Series) -> int:
    values = series.dropna()
    if values.empty:
        return 0
    best = current = 1
    previous = values.iloc[0]
    for value in values.iloc[1:]:
        current = current + 1 if value == previous else 1
        previous = value
        best = max(best, current)
    return best


def _saturation_ratio(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return 0.0
    counts = values.value_counts(normalize=True)
    return float(counts.iloc[0]) if len(counts) else 0.0


def _robust_outlier_ratio(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 10:
        return 0.0
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad <= 1e-12:
        return 0.0
    robust_z = 0.6745 * (values - median).abs() / mad
    return float((robust_z > 6.0).mean())


def _residualize(
    y: pd.Series,
    x: pd.DataFrame,
    fit_mask: pd.Series | None = None,
) -> tuple[pd.Series, str, float, list[str]]:
    application_data = pd.concat([y, x], axis=1).dropna()
    fit_data = application_data
    if fit_mask is not None:
        resolved_mask = fit_mask.reindex(application_data.index).fillna(False).astype(bool)
        fit_data = application_data.loc[resolved_mask]
    fit_x = fit_data.iloc[:, 1:]
    usable_columns = [column for column in fit_x.columns if fit_x[column].nunique() > 1]
    if len(fit_data) < 5 or not usable_columns:
        return y - fit_data.iloc[:, 0].mean(), "demean", np.nan, []
    fit_features = fit_x[usable_columns]
    fit_matrix = np.column_stack([np.ones(len(fit_data)), fit_features.to_numpy(dtype=float)])
    cond = float(np.linalg.cond(fit_matrix))
    coefficients, _, _, _ = np.linalg.lstsq(
        fit_matrix, fit_data.iloc[:, 0].to_numpy(dtype=float), rcond=None
    )
    application_matrix = np.column_stack(
        [np.ones(len(application_data)), application_data[usable_columns].to_numpy(dtype=float)]
    )
    fitted = np.dot(application_matrix, coefficients)
    residual = pd.Series(
        index=application_data.index,
        data=application_data.iloc[:, 0].to_numpy() - fitted,
    )
    return residual.reindex(y.index), "ols", cond, usable_columns


def _nearby_lags(best_lag: int | None, max_lag: int, radius: int = 2) -> list[int]:
    if best_lag is None or pd.isna(best_lag):
        return list(range(0, min(max_lag, 6) + 1))
    center = int(best_lag)
    if center <= 0:
        return [0]
    lower = max(0, center - radius)
    upper = min(max_lag, center + radius)
    return list(range(lower, upper + 1))


def _time_series_splits(n_rows: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    test_size = max(5, n_rows // (n_splits + 1))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for split in range(n_splits):
        train_end = n_rows - test_size * (n_splits - split)
        test_start = train_end
        test_end = test_start + test_size
        if train_end <= test_size or test_end > n_rows:
            continue
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


def _linear_predict(x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    model = fit_linear_model(x_train, y_train)
    return predict_linear_model(model, x_test)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _looks_like_formula_variable(name: str) -> bool:
    lower = name.lower()
    tokens = ["单耗", "消耗", "比值", "ratio", "rate", "%", "百分比", "折算", "累计", "平均", "total", "consumption", "specific"]
    return any(token in lower for token in tokens)
