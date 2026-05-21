from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags

ROLES = {"TIME", "Y", "CAPACITY", "MV", "PV", "DV", "IGNORE"}


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
        "sampling_period_seconds", "constant_run_max", "abnormal_jump_count", "abnormal_jump_ratio", "saturation_ratio",
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
            "saturation_ratio": float(_saturation_ratio(series)),
        })
    return pd.DataFrame(rows, columns=columns)


def residual_corr_scores(frame: pd.DataFrame, target: str, capacity_columns: list[str] | None, max_lag: int) -> pd.DataFrame:
    out_cols = ["variable", "lag", "residual_corr", "residual_p_value", "residual_r2", "direction", "residual_method", "condition_number", "used_control_columns"]
    capacity_columns = [col for col in (capacity_columns or []) if col in frame.columns]
    if not capacity_columns:
        return pd.DataFrame(columns=out_cols)
    target_residual, t_method, t_cond, used_cols = _residualize(frame[target], frame[capacity_columns])
    all_scores: list[pd.DataFrame] = []
    for column in frame.columns:
        if column == target or column in capacity_columns:
            continue
        candidate_residual, c_method, c_cond, c_used_cols = _residualize(frame[column], frame[capacity_columns])
        pair = pd.DataFrame({target: target_residual, column: candidate_residual}).dropna()
        if len(pair) < max(10, max_lag + 5):
            continue
        scores_for_candidate = compute_lag_scores(pair, target, max_lag)
        if not scores_for_candidate.empty:
            all_scores.append(scores_for_candidate)
    if not all_scores:
        return pd.DataFrame(columns=out_cols)
    scores = summarize_best_lags(pd.concat(all_scores, ignore_index=True))
    if scores.empty:
        return pd.DataFrame(columns=out_cols)
    out = scores.rename(columns={"score": "residual_corr", "p_value": "residual_p_value", "r2": "residual_r2"})
    out["residual_method"] = t_method
    out["condition_number"] = t_cond
    out["used_control_columns"] = ",".join(used_cols)
    for col in out_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[out_cols]


def regime_scores(frame: pd.DataFrame, target: str, capacity_column: str | None, max_lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = ["variable", "regime", "regime_row_count", "score", "signed_corr", "lag", "direction", "p_value", "r2"]
    stability_cols = ["variable", "regime_stability_final", "regime_sign_consistency", "regime_lag_consistency", "regime_score_cv", "regime_count"]
    if not capacity_column or capacity_column not in frame.columns:
        return pd.DataFrame(columns=score_cols), pd.DataFrame(columns=stability_cols)

    capacity = pd.to_numeric(frame[capacity_column], errors="coerce")
    q1 = capacity.quantile(1 / 3)
    q2 = capacity.quantile(2 / 3)
    regimes = {
        "low": frame.loc[capacity <= q1],
        "mid": frame.loc[(capacity > q1) & (capacity <= q2)],
        "high": frame.loc[capacity > q2],
    }

    all_rows: list[pd.DataFrame] = []
    for name, subset in regimes.items():
        if len(subset) < max(10, max_lag + 5):
            continue
        best = summarize_best_lags(compute_lag_scores(subset, target, max_lag))
        if best.empty:
            continue
        best["signed_corr"] = np.where(best["method"].eq("pearson"), best["pearson"], best["spearman"])
        best = best.assign(regime=name, regime_row_count=len(subset))
        all_rows.append(best[score_cols])

    if not all_rows:
        return pd.DataFrame(columns=score_cols), pd.DataFrame(columns=stability_cols)

    scores = pd.concat(all_rows, ignore_index=True)
    stability = scores.groupby("variable").agg(
        regime_score_std=("score", "std"),
        regime_score_mean=("score", "mean"),
        regime_count=("regime", "count"),
        regime_lag_std=("lag", "std"),
    ).reset_index()
    stability["regime_score_std"] = stability["regime_score_std"].fillna(0)
    stability["regime_score_cv"] = (stability["regime_score_std"] / stability["regime_score_mean"].abs().replace(0, np.nan)).fillna(1.0)
    sign_consistency = scores.assign(sign=np.sign(scores["signed_corr"])).groupby("variable")["sign"].agg(lambda v: float(v.value_counts(normalize=True).iloc[0]) if len(v) else 0.0)
    lag_consistency = (1 - stability["regime_lag_std"].fillna(0) / max(1, max_lag)).clip(lower=0, upper=1)
    strength_stability = (1 - stability["regime_score_cv"]).clip(lower=0, upper=1)
    stability["regime_sign_consistency"] = stability["variable"].map(sign_consistency).fillna(0)
    stability["regime_lag_consistency"] = lag_consistency
    stability["regime_stability_final"] = (
        0.5 * strength_stability + 0.25 * stability["regime_sign_consistency"] + 0.25 * stability["regime_lag_consistency"]
    ).clip(lower=0, upper=1)
    return scores, stability[stability_cols]


def model_lift_scores(frame: pd.DataFrame, target: str, candidate_variables: list[str], max_lag: int, n_splits: int = 4, best_lags: dict[str, int] | None = None) -> pd.DataFrame:
    cols = ["variable", "status", "ar_baseline_rmse", "candidate_rmse", "model_lift"]
    rows: list[dict[str, object]] = []
    ar_lags = list(range(1, min(max_lag, 6) + 1))
    for variable in candidate_variables:
        if variable == target or variable not in frame.columns:
            continue
        candidate_lags = _nearby_lags(best_lags.get(variable) if best_lags else None, max_lag)
        dataset = pd.DataFrame(index=frame.index)
        dataset[target] = frame[target]
        for lag in ar_lags:
            dataset[f"{target}__lag_{lag}"] = frame[target].shift(lag)
        for lag in candidate_lags:
            dataset[f"{variable}__lag_{lag}"] = frame[variable].shift(lag)
        dataset = dataset.dropna()
        if len(dataset) < 60:
            rows.append({"variable": variable, "status": "skipped: insufficient rows", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": 0.0})
            continue
        base_cols = [f"{target}__lag_{lag}" for lag in ar_lags]
        full_cols = base_cols + [f"{variable}__lag_{lag}" for lag in candidate_lags]
        base_errors: list[float] = []
        full_errors: list[float] = []
        splits = _time_series_splits(len(dataset), n_splits)
        if not splits:
            rows.append({"variable": variable, "status": "skipped: no valid time series split", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": 0.0})
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
            rows.append({"variable": variable, "status": "skipped: no valid time series split", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": 0.0})
            continue
        lift = max(0.0, (base_rmse - full_rmse) / base_rmse) if base_rmse > 0 else 0.0
        rows.append({"variable": variable, "status": "ok", "ar_baseline_rmse": base_rmse, "candidate_rmse": full_rmse, "model_lift": lift})
    return pd.DataFrame(rows, columns=cols)


def rolling_corr_scores(frame: pd.DataFrame, target: str, candidate_variables: list[str], max_lag: int, window: int | None = None, min_periods: int | None = None) -> pd.DataFrame:
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
        best = summarize_best_lags(compute_lag_scores(pair, target, max_lag))
        if best.empty:
            continue
        best_row = best.iloc[0]
        best_lag = int(best_row["lag"])
        shifted = pair[variable].shift(best_lag)
        rolling = shifted.rolling(window=window_size, min_periods=min_points).corr(pair[target]).dropna()
        if rolling.empty:
            continue
        sign_consistency = rolling.apply(lambda value: 1 if value >= 0 else -1).value_counts(normalize=True).max()
        iqr = float(rolling.quantile(0.75) - rolling.quantile(0.25))
        abs_median = float(rolling.abs().median())
        stability = max(0.0, min(1.0, abs_median * float(sign_consistency) * (1.0 - min(1.0, iqr))))
        rows.append({"variable": variable, "best_lag": best_lag, "best_score": float(best_row.get("score", 0.0) or 0.0), "rolling_corr_median": float(rolling.median()), "rolling_abs_corr_median": abs_median, "rolling_corr_iqr": iqr, "rolling_sign_consistency": float(sign_consistency), "valid_window_count": int(len(rolling)), "rolling_stability": stability})
    return pd.DataFrame(rows, columns=cols)


def risk_flags(ranked: pd.DataFrame, residual: pd.DataFrame, stability: pd.DataFrame, diag: pd.DataFrame, roles: dict[str, str], control_columns: list[str] | None, lag_peak_quality: pd.DataFrame | None = None, rolling_corr_scores: pd.DataFrame | None = None, model_lift_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    cols = ["variable", "formula_like_flag", "strong_formula_leakage_flag", "common_capacity_driver_flag", "closed_loop_suspect_flag", "target_leads_variable_flag", "unstable_across_regimes_flag", "unstable_over_time_flag", "lag_boundary_flag", "low_model_lift_flag", "poor_data_quality_flag", "residual_collinearity_flag", "risk_flags", "risk_count", "strong_risk_count", "weak_risk_count", "risk_level", "human_reason"]
    if ranked.empty:
        return pd.DataFrame(columns=cols)

    residual_map = residual.set_index("variable")["residual_corr"].to_dict() if not residual.empty and "residual_corr" in residual.columns else {}
    residual_cond_map = residual.set_index("variable")["condition_number"].to_dict() if not residual.empty and "condition_number" in residual.columns else {}
    stability_map = stability.set_index("variable").to_dict("index") if not stability.empty else {}
    diag_map = diag.set_index("variable").to_dict("index") if not diag.empty else {}
    lag_map = lag_peak_quality.set_index("variable").to_dict("index") if lag_peak_quality is not None and not lag_peak_quality.empty else {}
    roll_map = rolling_corr_scores.set_index("variable").to_dict("index") if rolling_corr_scores is not None and not rolling_corr_scores.empty else {}
    lift_map = model_lift_scores.set_index("variable").to_dict("index") if model_lift_scores is not None and not model_lift_scores.empty else {}

    rows = []
    for _, row in ranked.iterrows():
        variable = str(row.get("variable", ""))
        raw_corr = float(row.get("score", 0) or 0)
        residual_corr = float(residual_map.get(variable, raw_corr))
        regime_stability = float(stability_map.get(variable, {}).get("regime_stability_final", 1.0) or 1.0)
        d = diag_map.get(variable, {})
        poor_quality = (
            float(d.get("missing_rate", 0) or 0) > 0.2
            or float(d.get("saturation_ratio", 0) or 0) > 0.2
            or float(d.get("abnormal_jump_ratio", 0) or 0) > 0.01
        )
        formula_like = _looks_like_formula_variable(variable)
        strong_formula = formula_like and raw_corr > 0.98 and int(row.get("lag", 0) or 0) == 0
        common_capacity = bool(control_columns) and raw_corr >= 0.5 and residual_corr < raw_corr * 0.65
        closed_loop = roles.get(variable) == "MV" and int(row.get("lag", 0) or 0) < 0
        target_leads = int(row.get("lag", 0) or 0) < 0
        unstable_reg = regime_stability < 0.5
        unstable_time = float(roll_map.get(variable, {}).get("rolling_stability", 1.0) or 1.0) < 0.35
        lag_boundary = bool(lag_map.get(variable, {}).get("lag_boundary_flag", False))
        lift_info = lift_map.get(variable, {})
        low_lift = str(lift_info.get("status", "")).startswith("ok") and float(lift_info.get("model_lift", 0.0) or 0.0) < 0.01
        residual_collinearity = float(residual_cond_map.get(variable, 0) or 0) > 1e8

        flags = [name for name, active in [
            ("formula_like", formula_like),
            ("strong_formula_leakage", strong_formula),
            ("common_capacity_driver", common_capacity),
            ("closed_loop_suspect", closed_loop),
            ("target_leads_variable", target_leads),
            ("unstable_across_regimes", unstable_reg),
            ("unstable_over_time", unstable_time),
            ("lag_boundary", lag_boundary),
            ("low_model_lift", low_lift),
            ("poor_data_quality", poor_quality),
        ] if active]

        strong_risks = [f for f in flags if f in {"strong_formula_leakage", "common_capacity_driver", "closed_loop_suspect", "poor_data_quality"}]
        weak_risks = [f for f in flags if f not in set(strong_risks)]
        level = "none" if not flags else ("strong" if len(strong_risks) >= 2 else ("medium" if strong_risks else "weak"))
        reason = "；".join(flags)
        rows.append({"variable": variable, "formula_like_flag": formula_like, "strong_formula_leakage_flag": strong_formula, "common_capacity_driver_flag": common_capacity, "closed_loop_suspect_flag": closed_loop, "target_leads_variable_flag": target_leads, "unstable_across_regimes_flag": unstable_reg, "unstable_over_time_flag": unstable_time, "lag_boundary_flag": lag_boundary, "low_model_lift_flag": low_lift, "poor_data_quality_flag": poor_quality, "residual_collinearity_flag": residual_collinearity, "risk_flags": ";".join(flags), "risk_count": len(flags), "strong_risk_count": len(strong_risks), "weak_risk_count": len(weak_risks), "risk_level": level, "human_reason": reason})
    return pd.DataFrame(rows, columns=cols)


def final_ranked_features(ranked: pd.DataFrame, residual: pd.DataFrame, stability: pd.DataFrame, model_lift: pd.DataFrame, risks: pd.DataFrame, lag_peak_quality: pd.DataFrame, rolling_corr_scores: pd.DataFrame, force_include_variables: list[str] | None = None, top_k: int | None = None) -> pd.DataFrame:
    cols = ["variable", "lag", "direction", "raw_corr", "residual_corr", "regime_stability_final", "rolling_stability", "lag_quality", "lag_boundary_flag", "model_lift_score", "risk_penalty", "final_score", "candidate_grade", "recommended_use", "recommended_action", "risk_flags", "risk_count", "force_included"]
    if ranked.empty:
        return pd.DataFrame(columns=cols)
    final = ranked.rename(columns={"score": "raw_corr"}).copy()
    final = final.merge(residual[[c for c in ["variable", "residual_corr"] if c in residual.columns]], on="variable", how="left")
    final = final.merge(stability[[c for c in ["variable", "regime_stability_final"] if c in stability.columns]], on="variable", how="left")
    final = final.merge(model_lift[[c for c in ["variable", "model_lift", "status"] if c in model_lift.columns]], on="variable", how="left")
    final = final.merge(risks[[c for c in ["variable", "risk_flags", "risk_count"] if c in risks.columns]], on="variable", how="left")
    final = final.merge(lag_peak_quality[[c for c in ["variable", "lag_quality", "lag_boundary_flag"] if c in lag_peak_quality.columns]], on="variable", how="left")
    final = final.merge(rolling_corr_scores[[c for c in ["variable", "rolling_stability"] if c in rolling_corr_scores.columns]], on="variable", how="left")

    residual_raw = final["residual_corr"] if "residual_corr" in final.columns else pd.Series(np.nan, index=final.index)
    regime_raw = final["regime_stability_final"] if "regime_stability_final" in final.columns else pd.Series(np.nan, index=final.index)
    rolling_raw = final["rolling_stability"] if "rolling_stability" in final.columns else pd.Series(np.nan, index=final.index)
    lagq_raw = final["lag_quality"] if "lag_quality" in final.columns else pd.Series(np.nan, index=final.index)
    lift_raw = final["model_lift"] if "model_lift" in final.columns else pd.Series(np.nan, index=final.index)
    final["residual_status"] = np.where(residual_raw.notna(), "ok", "not_computed")
    final["regime_status"] = np.where(regime_raw.notna(), "ok", "not_computed")
    final["rolling_status"] = np.where(rolling_raw.notna(), "ok", "not_computed")
    final["model_lift_status"] = np.where(lift_raw.notna(), "ok", "not_computed")
    final["lag_quality_status"] = np.where(lagq_raw.notna(), "ok", "not_computed")
    final["raw_corr_score"] = final["raw_corr"].fillna(0).clip(0, 1)
    final["residual_corr_score"] = residual_raw.clip(0,1)
    final["regime_stability_final"] = regime_raw.clip(0,1)
    final["rolling_stability"] = rolling_raw.clip(0,1)
    final["lag_quality"] = lagq_raw.clip(0,1)
    final["model_lift_score"] = lift_raw.clip(0,1)
    display_residual = residual_raw.fillna(final["raw_corr_score"]).clip(0,1)
    display_regime = regime_raw.fillna(0.5).clip(0,1)
    display_rolling = rolling_raw.fillna(0.5).clip(0,1)
    display_lagq = lagq_raw.fillna(0.5).clip(0,1)
    display_lift = lift_raw.fillna(0.0).clip(0,1)
    final["risk_penalty"] = final.get("risk_count", pd.Series(index=final.index, dtype=float)).fillna(0).clip(0, 5)
    parts = {"raw": (final["raw_corr_score"], 0.25), "residual": (final["residual_corr_score"], 0.25), "regime": (final["regime_stability_final"], 0.15), "rolling": (final["rolling_stability"], 0.15), "lagq": (final["lag_quality"], 0.10), "lift": (final["model_lift_score"], 0.10)}
    num = 0
    den = 0
    for series, w in parts.values():
        valid = series.notna()
        num = num + series.fillna(0) * w
        den = den + valid.astype(float) * w
    final["final_score"] = (num / den.replace(0, np.nan)).fillna(0) - 0.10 * final["risk_penalty"]
    final["final_score"] = final["final_score"].clip(lower=0, upper=1)
    forced = set(force_include_variables or [])
    final["force_included"] = final["variable"].astype(str).isin(forced)
    final["residual_corr_score"] = display_residual
    final["regime_stability_final"] = display_regime
    final["rolling_stability"] = display_rolling
    final["lag_quality"] = display_lagq
    final["model_lift_score"] = display_lift
    final["candidate_grade"] = final.apply(_grade_candidate, axis=1)
    final["recommended_use"] = final.apply(_recommend_use, axis=1)
    final["recommended_action"] = final.apply(_recommended_action, axis=1)

    if top_k is not None:
        top = final.sort_values("final_score", ascending=False).head(top_k)
        forced_rows = final[final["force_included"]]
        final = pd.concat([top, forced_rows], ignore_index=True).drop_duplicates(subset=["variable"], keep="first")
    else:
        final = final.sort_values("final_score", ascending=False)

    for c in cols:
        if c not in final.columns:
            final[c] = np.nan
    return final.reset_index(drop=True)[cols]


def _grade_candidate(row: pd.Series) -> str:
    score = float(row.get("final_score", 0) or 0)
    if score >= 0.75:
        return "A"
    if score >= 0.6:
        return "B"
    if score >= 0.45:
        return "C"
    if score >= 0.3:
        return "D"
    return "E"


def _recommend_use(row: pd.Series) -> str:
    flags = str(row.get("risk_flags", "") or "")
    grade = str(row.get("candidate_grade", "E"))
    if "poor_data_quality" in flags:
        return "poor_quality_variable"
    if "closed_loop_suspect" in flags:
        return "closed_loop_suspect"
    if "common_capacity_driver" in flags:
        return "capacity_driven"
    raw_corr = float(row.get("raw_corr", 0) or 0)
    lag = int(row.get("lag", 0) or 0)
    has_formula = "formula_like" in flags
    has_strong_formula = "strong_formula_leakage" in flags
    has_common = "common_capacity_driver" in flags
    if has_strong_formula or (has_formula and has_common) or (has_formula and lag == 0 and raw_corr >= 0.95):
        return "formula_coupled_reference"
    if "unstable_across_regimes" in flags or "unstable_over_time" in flags:
        return "unstable_candidate"
    if grade == "A":
        return "strong_screening_candidate"
    if grade == "B" and float(row.get("model_lift_score", 0) or 0) > 0.05:
        return "prediction_candidate"
    if int(row.get("lag", 0) or 0) < 0:
        return "state_indicator"
    return "manual_review_required"


def _recommended_action(row: pd.Series) -> str:
    use = str(row.get("recommended_use", "manual_review_required"))
    mapping = {
        "strong_screening_candidate": "优先进入机理复核",
        "prediction_candidate": "可作为预测候选",
        "capacity_driven": "疑似共同负荷驱动",
        "closed_loop_suspect": "疑似闭环反馈",
        "formula_coupled_reference": "疑似公式耦合，仅参考",
        "unstable_candidate": "跨工况/时间不稳定，建议复核",
        "poor_quality_variable": "数据质量风险，建议剔除",
        "state_indicator": "更可能是状态指示量",
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


def _residualize(y: pd.Series, x: pd.DataFrame) -> tuple[pd.Series, str, float, list[str]]:
    data = pd.concat([y, x], axis=1).dropna()
    x_data = data.iloc[:, 1:]
    usable_columns = [column for column in x_data.columns if x_data[column].nunique() > 1]
    if len(data) < 5 or not usable_columns:
        return y - y.mean(), "demean", np.nan, []
    x_matrix = np.column_stack([np.ones(len(data)), x_data[usable_columns].to_numpy(dtype=float)])
    cond = float(np.linalg.cond(x_matrix))
    method = "ols"
    if cond > 1e8:
        method = "ridge"
        alpha = 1e-3
        penalty = alpha * np.eye(x_matrix.shape[1])
        penalty[0, 0] = 0.0
        xtx = x_matrix.T @ x_matrix + penalty
        coef = np.linalg.solve(xtx, x_matrix.T @ data.iloc[:, 0].to_numpy())
    else:
        coef, *_ = np.linalg.lstsq(x_matrix, data.iloc[:, 0].to_numpy(), rcond=None)
    fitted = x_matrix @ coef
    residual = pd.Series(index=data.index, data=data.iloc[:, 0].to_numpy() - fitted)
    return residual.reindex(y.index), method, cond, usable_columns


def _nearby_lags(best_lag: int | None, max_lag: int, radius: int = 2) -> list[int]:
    if best_lag is None or pd.isna(best_lag):
        return list(range(0, min(max_lag, 6) + 1))
    center = max(0, int(abs(best_lag)))
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
    train = np.column_stack([np.ones(len(x_train)), x_train.to_numpy()])
    test = np.column_stack([np.ones(len(x_test)), x_test.to_numpy()])
    coef, *_ = np.linalg.lstsq(train, y_train, rcond=None)
    return test @ coef


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _looks_like_formula_variable(name: str) -> bool:
    lower = name.lower()
    tokens = ["单耗", "消耗", "比值", "ratio", "rate", "%", "百分比", "折算", "累计", "平均", "total", "consumption", "specific"]
    return any(token in lower for token in tokens)
