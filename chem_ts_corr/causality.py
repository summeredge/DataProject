from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f

from chem_ts_corr.common import benjamini_hochberg


_GRANGER_COLUMNS = [
    "variable",
    "status",
    "best_granger_lag",
    "min_p_value",
    "f_statistic",
    "predictive_contribution",
    "interpretation",
    "fdr_q_value",
]


def run_granger_tests(
    frame: pd.DataFrame,
    target: str,
    variables: list[str],
    maxlag: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | None]] = []
    for variable in variables:
        if variable == target:
            continue

        try:
            pair = frame[[target, variable]].dropna()
        except Exception as exc:
            rows.append({"variable": variable, "status": f"failed: {exc}", "min_p_value": None})
            continue

        if maxlag <= 0:
            rows.append({"variable": variable, "status": "skipped: no valid lag tests", "min_p_value": None})
            continue

        if len(pair) < max(30, maxlag * 5):
            rows.append(
                {"variable": variable, "status": "skipped: insufficient rows", "min_p_value": None}
            )
            continue

        try:
            lag_results = _fast_granger_ssr_ftests(pair, target, variable, maxlag)
        except Exception as exc:
            rows.append({"variable": variable, "status": f"failed: {exc}", "min_p_value": None})
            continue

        if not lag_results:
            rows.append(
                {"variable": variable, "status": "skipped: no valid lag tests", "min_p_value": None}
            )
            continue

        best_lag = min(lag_results, key=lambda lag: lag_results[lag][1])
        f_statistic, min_p_value = lag_results[best_lag]
        rows.append(
            {
                "variable": variable,
                "status": "ok",
                "best_granger_lag": best_lag,
                "min_p_value": min_p_value,
                "f_statistic": f_statistic,
                "predictive_contribution": _predictive_contribution(pair[target], pair[variable], best_lag),
                "interpretation": "predictive validation only; not a causal conclusion",
            }
        )

    result_frame = pd.DataFrame(rows)
    if "min_p_value" in result_frame.columns:
        result_frame["fdr_q_value"] = benjamini_hochberg(result_frame["min_p_value"])
        result_frame = result_frame.sort_values("fdr_q_value", na_position="last")
    return result_frame.reindex(columns=_GRANGER_COLUMNS)


def _fast_granger_ssr_ftests(
    pair: pd.DataFrame,
    target: str,
    variable: str,
    maxlag: int,
) -> dict[int, tuple[float, float]]:
    if maxlag <= 0:
        return {}
    if target not in pair.columns or variable not in pair.columns:
        raise KeyError(f"missing required columns: {target}, {variable}")

    clean_pair = pair[[target, variable]].dropna()
    target_values = clean_pair[target].to_numpy(dtype=float)
    variable_values = clean_pair[variable].to_numpy(dtype=float)

    results: dict[int, tuple[float, float]] = {}
    for lag in range(1, maxlag + 1):
        try:
            y, y_lags, x_lags = _lagged_arrays(target_values, variable_values, lag)
            nobs = len(y)
            if nobs == 0:
                continue

            ssr_r, restricted_rank = _ols_ssr_and_rank(y_lags, y)
            unrestricted_x = np.column_stack([y_lags, x_lags])
            ssr_u, unrestricted_rank = _ols_ssr_and_rank(unrestricted_x, y)
            df_num = unrestricted_rank - restricted_rank
            df_den = nobs - unrestricted_rank
            if df_num <= 0 or df_den <= 0:
                continue

            if (
                not np.isfinite(ssr_r)
                or not np.isfinite(ssr_u)
                or ssr_u <= 0
                or _is_near_perfect_fit(ssr_u, y)
            ):
                continue

            ssr_delta = max(0.0, ssr_r - ssr_u)
            f_statistic = (ssr_delta / df_num) / (ssr_u / df_den)
            p_value = float(f.sf(f_statistic, df_num, df_den))
            if not np.isfinite(f_statistic) or not np.isfinite(p_value):
                continue
            results[lag] = (float(f_statistic), p_value)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
    return results


def _lagged_arrays(
    target_values: np.ndarray,
    variable_values: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lag <= 0:
        raise ValueError("lag must be positive")
    n = len(target_values)
    if n <= lag:
        return (
            np.empty(0, dtype=float),
            np.empty((0, lag), dtype=float),
            np.empty((0, lag), dtype=float),
        )

    y = target_values[lag:]
    y_lags = np.column_stack([target_values[lag - i : n - i] for i in range(1, lag + 1)])
    x_lags = np.column_stack([variable_values[lag - i : n - i] for i in range(1, lag + 1)])
    return y, y_lags, x_lags


def _ols_ssr_and_rank(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    matrix = _add_intercept(x)
    coef, _residuals, rank, _singular_values = np.linalg.lstsq(matrix, y, rcond=None)
    residual = y - matrix @ coef
    ssr = float(np.dot(residual, residual))
    return ssr, int(rank)


def _add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def _is_near_perfect_fit(ssr: float, y: np.ndarray) -> bool:
    centered = y - np.mean(y)
    target_variation = float(np.dot(centered, centered))
    scale = max(target_variation, 1.0)
    return ssr <= np.finfo(float).eps * scale


def _predictive_contribution(target: pd.Series, variable: pd.Series, lag: int) -> float:
    data = pd.DataFrame({"target": target, "candidate": variable.shift(lag), "target_lag_1": target.shift(1)}).dropna()
    if len(data) < 10:
        return 0.0
    y = data["target"].to_numpy()
    base = _linear_rmse(data[["target_lag_1"]], y)
    full = _linear_rmse(data[["target_lag_1", "candidate"]], y)
    return max(0.0, (base - full) / base) if base > 0 else 0.0


def _linear_rmse(x: pd.DataFrame, y: object) -> float:
    matrix = _add_intercept(x.to_numpy())
    coef, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    pred = matrix @ coef
    return float(np.sqrt(np.mean((y - pred) ** 2)))
