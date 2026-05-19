from __future__ import annotations

import pandas as pd


def run_granger_tests(
    frame: pd.DataFrame,
    target: str,
    variables: list[str],
    maxlag: int,
) -> pd.DataFrame:
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception:
        return pd.DataFrame(
            [{"status": "skipped: statsmodels is not installed", "variable": "", "min_p_value": None}]
        )

    rows: list[dict[str, float | int | str | None]] = []
    for variable in variables:
        if variable == target:
            continue

        pair = frame[[target, variable]].dropna()
        if len(pair) < max(30, maxlag * 5):
            rows.append(
                {"variable": variable, "status": "skipped: insufficient rows", "min_p_value": None}
            )
            continue

        try:
            result = grangercausalitytests(pair[[target, variable]], maxlag=maxlag, verbose=False)
        except Exception as exc:
            rows.append({"variable": variable, "status": f"failed: {exc}", "min_p_value": None})
            continue

        lag_p_values = {lag: float(test_result[0]["ssr_ftest"][1]) for lag, test_result in result.items()}
        lag_f_values = {lag: float(test_result[0]["ssr_ftest"][0]) for lag, test_result in result.items()}
        best_lag = min(lag_p_values, key=lag_p_values.get)
        rows.append(
            {
                "variable": variable,
                "status": "ok",
                "best_granger_lag": best_lag,
                "min_p_value": lag_p_values[best_lag],
                "f_statistic": lag_f_values[best_lag],
                "predictive_contribution": _predictive_contribution(pair[target], pair[variable], best_lag),
                "interpretation": "predictive validation only; not a causal conclusion",
            }
        )

    frame = pd.DataFrame(rows)
    if "min_p_value" in frame.columns:
        frame["fdr_q_value"] = _benjamini_hochberg(frame["min_p_value"])
        frame = frame.sort_values("fdr_q_value", na_position="last")
    return frame


def _predictive_contribution(target: pd.Series, variable: pd.Series, lag: int) -> float:
    data = pd.DataFrame({"target": target, "candidate": variable.shift(lag), "target_lag_1": target.shift(1)}).dropna()
    if len(data) < 10:
        return 0.0
    y = data["target"].to_numpy()
    base = _linear_rmse(data[["target_lag_1"]], y)
    full = _linear_rmse(data[["target_lag_1", "candidate"]], y)
    return max(0.0, (base - full) / base) if base > 0 else 0.0


def _linear_rmse(x: pd.DataFrame, y: object) -> float:
    import numpy as np

    matrix = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coef, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    pred = matrix @ coef
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    import numpy as np

    pvals = pd.to_numeric(values, errors="coerce")
    qvals = pd.Series(np.nan, index=values.index, dtype=float)
    valid = pvals.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return qvals
    running = 1.0
    ranked_items = list(valid.items())
    for rank in range(m, 0, -1):
        original_idx, original_p = ranked_items[rank - 1]
        value = min(running, float(original_p) * m / rank)
        running = value
        qvals.loc[original_idx] = value
    return qvals
