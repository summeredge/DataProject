from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.lag import build_lag_peak_quality, compute_lag_scores, summarize_best_lags


ROLES = {"TIME", "Y", "CAPACITY", "MV", "PV", "DV", "IGNORE"}


def load_roles(config: AnalysisConfig, columns: list[str]) -> dict[str, str]:
    roles = {column: "PV" for column in columns}
    roles[config.target] = "Y"
    if config.segment_column and config.segment_column in roles:
        roles[config.segment_column] = "CAPACITY"
    for column in config.capacity_columns or []:
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
        rows.append(
            {
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
            }
        )
    return pd.DataFrame(rows)


def residual_corr_scores(
    frame: pd.DataFrame,
    target: str,
    capacity_columns: list[str] | None,
    max_lag: int,
) -> pd.DataFrame:
    capacity_columns = [col for col in (capacity_columns or []) if col in frame.columns]
    if not capacity_columns:
        return pd.DataFrame(
            columns=["variable", "lag", "residual_corr", "residual_p_value", "residual_r2"]
        )

    target_residual = _residualize(frame[target], frame[capacity_columns])
    all_scores: list[pd.DataFrame] = []
    for column in frame.columns:
        if column == target or column in capacity_columns:
            continue
        candidate_residual = _residualize(frame[column], frame[capacity_columns])
        pair = pd.DataFrame({target: target_residual, column: candidate_residual}).dropna()
        if len(pair) < max(10, max_lag + 5):
            continue
        scores_for_candidate = compute_lag_scores(pair, target, max_lag)
        if not scores_for_candidate.empty:
            all_scores.append(scores_for_candidate)

    if not all_scores:
        return pd.DataFrame(
            columns=["variable", "lag", "residual_corr", "residual_p_value", "residual_r2"]
        )
    scores = summarize_best_lags(pd.concat(all_scores, ignore_index=True))
    if scores.empty:
        return scores
    return scores.rename(
        columns={"score": "residual_corr", "p_value": "residual_p_value", "r2": "residual_r2"}
    )[["variable", "lag", "residual_corr", "residual_p_value", "residual_r2", "direction"]]


def regime_scores(
    frame: pd.DataFrame,
    target: str,
    capacity_column: str | None,
    max_lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_columns = ["variable", "regime", "score", "signed_corr", "lag", "direction", "p_value", "r2"]
    stability_columns = [
        "variable",
        "regime_stability",
        "regime_sign_consistency",
        "regime_lag_consistency",
        "regime_count",
    ]
    if not capacity_column or capacity_column not in frame.columns:
        empty = pd.DataFrame(columns=score_columns)
        stability = pd.DataFrame(columns=stability_columns)
        return empty, stability

    capacity = pd.to_numeric(frame[capacity_column], errors="coerce")
    q1 = capacity.quantile(1 / 3)
    q2 = capacity.quantile(2 / 3)
    regimes = {
        "low": frame.loc[capacity <= q1],
        "mid": frame.loc[(capacity >= q1) & (capacity <= q2)],
        "high": frame.loc[capacity >= q2],
    }

    all_rows: list[pd.DataFrame] = []
    for name, subset in regimes.items():
        if len(subset) < max(10, max_lag + 5):
            continue
        best = summarize_best_lags(compute_lag_scores(subset, target, max_lag))
        if best.empty:
            continue
        best["signed_corr"] = np.where(
            best["method"].eq("pearson"),
            best["pearson"],
            best["spearman"],
        )
        best = best.assign(regime=name)
        all_rows.append(
            best[["variable", "regime", "score", "signed_corr", "lag", "direction", "p_value", "r2"]]
        )

    if not all_rows:
        empty = pd.DataFrame(columns=score_columns)
        stability = pd.DataFrame(columns=stability_columns)
        return empty, stability

    scores = pd.concat(all_rows, ignore_index=True)
    stability = (
        scores.groupby("variable")
        .agg(
            regime_score_std=("score", "std"),
            regime_score_mean=("score", "mean"),
            regime_count=("regime", "count"),
            regime_lag_std=("lag", "std"),
        )
        .reset_index()
    )
    stability["regime_score_std"] = stability["regime_score_std"].fillna(0)
    strength_stability = (
        1
        - stability["regime_score_std"]
        / stability["regime_score_mean"].abs().replace(0, np.nan)
    ).clip(lower=0, upper=1).fillna(0)
    sign_consistency = scores.assign(sign=np.sign(scores["signed_corr"])).groupby("variable")["sign"].agg(
        lambda values: float(values.value_counts(normalize=True).iloc[0]) if len(values) else 0.0
    )
    lag_consistency = (1 - stability["regime_lag_std"].fillna(0) / max(1, max_lag)).clip(lower=0, upper=1)
    stability["regime_sign_consistency"] = stability["variable"].map(sign_consistency).fillna(0)
    stability["regime_lag_consistency"] = lag_consistency
    stability["regime_stability"] = (
        0.5 * strength_stability
        + 0.25 * stability["regime_sign_consistency"]
        + 0.25 * stability["regime_lag_consistency"]
    ).clip(lower=0, upper=1)
    return scores, stability[
        [
            "variable",
            "regime_stability",
            "regime_sign_consistency",
            "regime_lag_consistency",
            "regime_count",
        ]
    ]


def model_lift_scores(
    frame: pd.DataFrame,
    target: str,
    candidate_variables: list[str],
    max_lag: int,
    n_splits: int = 4,
    best_lags: dict[str, int] | None = None,
) -> pd.DataFrame:
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
            rows.append({"variable": variable, "status": "skipped: insufficient rows", "model_lift": 0.0})
            continue

        base_cols = [f"{target}__lag_{lag}" for lag in ar_lags]
        full_cols = base_cols + [f"{variable}__lag_{lag}" for lag in candidate_lags]
        base_errors: list[float] = []
        full_errors: list[float] = []
        for train_idx, test_idx in _time_series_splits(len(dataset), n_splits):
            y_train = dataset.iloc[train_idx][target].to_numpy()
            y_test = dataset.iloc[test_idx][target].to_numpy()
            base_pred = _linear_predict(dataset.iloc[train_idx][base_cols], y_train, dataset.iloc[test_idx][base_cols])
            full_pred = _linear_predict(dataset.iloc[train_idx][full_cols], y_train, dataset.iloc[test_idx][full_cols])
            base_errors.append(_rmse(y_test, base_pred))
            full_errors.append(_rmse(y_test, full_pred))

        base_rmse = float(np.mean(base_errors))
        full_rmse = float(np.mean(full_errors))
        lift = max(0.0, (base_rmse - full_rmse) / base_rmse) if base_rmse > 0 else 0.0
        rows.append(
            {
                "variable": variable,
                "status": "ok",
                "ar_baseline_rmse": base_rmse,
                "candidate_rmse": full_rmse,
                "model_lift": lift,
            }
        )
    return pd.DataFrame(rows)




def rolling_corr_scores(
    frame: pd.DataFrame,
    target: str,
    candidate_variables: list[str],
    max_lag: int,
    window: int | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    columns = [
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
    if target not in frame.columns or not candidate_variables:
        return pd.DataFrame(columns=columns)

    target_series = frame[target]
    window_size = max(12, int(window or min(len(frame), max(24, max_lag * 4))))
    min_points = max(6, int(min_periods or window_size // 2))

    rows: list[dict[str, object]] = []
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
        rows.append(
            {
                "variable": variable,
                "best_lag": best_lag,
                "best_score": float(best_row.get("score", 0.0) or 0.0),
                "rolling_corr_median": float(rolling.median()),
                "rolling_abs_corr_median": abs_median,
                "rolling_corr_iqr": iqr,
                "rolling_sign_consistency": float(sign_consistency),
                "valid_window_count": int(len(rolling)),
                "rolling_stability": stability,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]
def risk_flags(
    ranked: pd.DataFrame,
    residual: pd.DataFrame,
    stability: pd.DataFrame,
    diag: pd.DataFrame,
    roles: dict[str, str],
    capacity_columns: list[str] | None,
) -> pd.DataFrame:
    residual_map = residual.set_index("variable")["residual_corr"].to_dict() if not residual.empty else {}
    stability_map = stability.set_index("variable")["regime_stability"].to_dict() if not stability.empty else {}
    diag_map = diag.set_index("variable").to_dict(orient="index") if not diag.empty else {}
    rows: list[dict[str, object]] = []
    for _, row in ranked.iterrows():
        variable = row["variable"]
        raw_corr = float(row.get("score", 0) or 0)
        residual_corr = float(residual_map.get(variable, raw_corr))
        regime_stability = float(stability_map.get(variable, 1.0))
        d = diag_map.get(variable, {})
        poor_quality = (
            float(d.get("missing_rate", 0) or 0) > 0.2
            or float(d.get("saturation_ratio", 0) or 0) > 0.2
            or float(d.get("abnormal_jump_ratio", 0) or 0) > 0.01
        )
        formula_leakage = (
            raw_corr > 0.98 and int(row.get("lag", 0)) == 0
        ) or _looks_like_formula_variable(str(variable))
        common_capacity = bool(capacity_columns) and raw_corr >= 0.5 and residual_corr < raw_corr * 0.65
        closed_loop = roles.get(variable) == "MV" and int(row.get("lag", 0)) < 0
        target_leads = int(row.get("lag", 0)) < 0
        unstable = regime_stability < 0.5
        flags = [
            name
            for name, active in [
                ("formula_like", formula_leakage),
                ("common_capacity_driver", common_capacity),
                ("closed_loop_suspect", closed_loop),
                ("target_leads_variable", target_leads),
                ("unstable_across_regimes", unstable),
                ("poor_data_quality", poor_quality),
            ]
            if active
        ]
        rows.append(
            {
                "variable": variable,
                "formula_like_flag": formula_leakage,
                "common_capacity_driver_flag": common_capacity,
                "closed_loop_suspect_flag": closed_loop,
                "target_leads_variable_flag": target_leads,
                "unstable_across_regimes_flag": unstable,
                "poor_data_quality_flag": poor_quality,
                "risk_flags": ";".join(flags),
                "risk_count": len(flags),
            }
        )
    return pd.DataFrame(rows)


def final_ranked_features(
    ranked: pd.DataFrame,
    residual: pd.DataFrame,
    stability: pd.DataFrame,
    model_lift: pd.DataFrame,
    risks: pd.DataFrame,
) -> pd.DataFrame:
    final = ranked.rename(columns={"score": "raw_corr"}).copy()
    final = final.merge(residual[["variable", "residual_corr"]], on="variable", how="left")
    final = final.merge(stability[["variable", "regime_stability"]], on="variable", how="left")
    final = final.merge(model_lift[["variable", "model_lift"]], on="variable", how="left")
    final = final.merge(risks[["variable", "risk_flags", "risk_count"]], on="variable", how="left")
    final["residual_corr"] = final["residual_corr"].fillna(final["raw_corr"])
    final["regime_stability"] = final["regime_stability"].fillna(1.0)
    final["model_lift"] = final["model_lift"].fillna(0.0)
    final["risk_count"] = final["risk_count"].fillna(0)
    final["lead_lag_value"] = np.where(final["lag"] > 0, 1.0, np.where(final["lag"] == 0, 0.75, 0.35))
    final["final_score"] = (
        0.35 * final["raw_corr"].clip(0, 1)
        + 0.25 * final["residual_corr"].clip(0, 1)
        + 0.15 * final["regime_stability"].clip(0, 1)
        + 0.15 * final["lead_lag_value"]
        + 0.10 * final["model_lift"].clip(0, 1)
        - 0.08 * final["risk_count"]
    ).clip(lower=0)
    final["recommended_use"] = np.where(
        final["risk_count"] >= 2,
        "not_recommended_as_causal",
        np.where(final["model_lift"] > 0.05, "prediction_candidate", "correlation_lead"),
    )
    final["recommended_action"] = final.apply(_recommended_action, axis=1)
    final["candidate_grade"] = final["final_score"].map(_grade_candidate)
    final["recommended_use"] = final.apply(_recommended_use_v2, axis=1)
    return final.sort_values("final_score", ascending=False).reset_index(drop=True)


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


def _residualize(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    data = pd.concat([y, x], axis=1).dropna()
    x_data = data.iloc[:, 1:]
    usable_columns = [column for column in x_data.columns if x_data[column].nunique() > 1]
    if len(data) < 5 or not usable_columns:
        return y - y.mean()
    x_matrix = np.column_stack([np.ones(len(data)), x_data[usable_columns].to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(x_matrix, data.iloc[:, 0].to_numpy(), rcond=None)
    fitted = x_matrix @ coef
    residual = pd.Series(index=data.index, data=data.iloc[:, 0].to_numpy() - fitted)
    return residual.reindex(y.index)


def _nearby_lags(best_lag: int | None, max_lag: int, radius: int = 2) -> list[int]:
    if best_lag is None or pd.isna(best_lag):
        return list(range(0, min(max_lag, 6) + 1))
    center = max(0, int(abs(best_lag)))
    lower = max(0, center - radius)
    upper = min(max_lag, center + radius)
    return list(range(lower, upper + 1))


def _looks_like_formula_variable(name: str) -> bool:
    lower = name.lower()
    tokens = ["单耗", "ratio", "rate", "百分比", "%", "percent", "占比", "比率", "效率"]
    return any(token in lower for token in tokens)


def _recommended_action(row: pd.Series) -> str:
    flags = str(row.get("risk_flags", "") or "")
    if "common_capacity_driver" in flags:
        return "疑似共同负荷驱动"
    if "closed_loop_suspect" in flags or "target_leads_variable" in flags:
        return "疑似闭环反馈"
    if int(row.get("risk_count", 0) or 0) >= 2 or "formula_leakage" in flags:
        return "建议人工工艺复核"
    if float(row.get("model_lift", 0) or 0) > 0.05:
        return "可作为预测候选"
    return "仅作相关性参考"


def _time_series_splits(n_rows: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    test_size = max(5, n_rows // (n_splits + 1))
    splits = []
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
