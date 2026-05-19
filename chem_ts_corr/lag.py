from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def _safe_corr_stats(x: pd.Series, y: pd.Series, method: str) -> dict[str, float | int]:
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 5:
        return {"r": np.nan, "p_value": np.nan, "r2": np.nan, "n": len(aligned)}
    if aligned.iloc[:, 0].nunique() <= 1 or aligned.iloc[:, 1].nunique() <= 1:
        return {"r": np.nan, "p_value": np.nan, "r2": np.nan, "n": len(aligned)}

    corr_frame = aligned if method == "pearson" else aligned.rank(method="average")
    r = float(corr_frame.iloc[:, 0].corr(corr_frame.iloc[:, 1], method="pearson"))
    return {"r": r, "p_value": _corr_p_value(r, len(aligned)), "r2": r * r, "n": len(aligned)}


def _corr_p_value(r: float, n: int) -> float:
    if n < 3 or np.isnan(r):
        return np.nan
    abs_r = min(abs(r), 0.999999999999)
    if abs_r >= 0.999999999999:
        return 0.0
    df = n - 2
    t_stat = abs_r * np.sqrt(df / max(1e-15, 1 - abs_r * abs_r))
    try:
        from scipy.stats import t as student_t  # type: ignore

        return float(2 * student_t.sf(t_stat, df))
    except Exception:
        return float(2 * (1 - NormalDist().cdf(t_stat)))


def compute_lag_scores(frame: pd.DataFrame, target: str, max_lag: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    target_series = frame[target]

    for variable in frame.columns:
        if variable == target:
            continue

        series = frame[variable]
        for lag in range(-max_lag, max_lag + 1):
            shifted = series.shift(lag)
            pearson = _safe_corr_stats(shifted, target_series, "pearson")
            spearman = _safe_corr_stats(shifted, target_series, "spearman")
            pearson_r = float(pearson["r"])
            spearman_r = float(spearman["r"])
            rows.append(
                {
                    "variable": variable,
                    "lag": lag,
                    "pearson": pearson_r,
                    "pearson_p": pearson["p_value"],
                    "pearson_r2": pearson["r2"],
                    "spearman": spearman_r,
                    "spearman_p": spearman["p_value"],
                    "spearman_r2": spearman["r2"],
                    "n": pearson["n"],
                    "effective_n": pearson["n"],
                    "abs_pearson": abs(pearson_r) if not np.isnan(pearson_r) else np.nan,
                    "abs_spearman": abs(spearman_r) if not np.isnan(spearman_r) else np.nan,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        use_pearson = result["abs_pearson"] >= result["abs_spearman"]
        result["p_value"] = np.where(use_pearson, result["pearson_p"], result["spearman_p"])
        result["corr_fdr_q_value"] = _benjamini_hochberg(result["p_value"])
    return result


def summarize_best_lags(lag_scores: pd.DataFrame) -> pd.DataFrame:
    if lag_scores.empty:
        return lag_scores

    ranked = lag_scores.assign(score=lag_scores[["abs_pearson", "abs_spearman"]].max(axis=1))
    idx = ranked.groupby("variable")["score"].idxmax()
    best = ranked.loc[idx].sort_values("score", ascending=False).reset_index(drop=True)

    use_pearson = best["abs_pearson"] >= best["abs_spearman"]
    best["method"] = np.where(use_pearson, "pearson", "spearman")
    best["p_value"] = np.where(use_pearson, best["pearson_p"], best["spearman_p"])
    best["r2"] = np.where(use_pearson, best["pearson_r2"], best["spearman_r2"])
    if "corr_fdr_q_value" in best.columns:
        best["corr_fdr_q_value"] = best["corr_fdr_q_value"]
    best["direction"] = best["lag"].map(describe_lag_direction)
    return best


def describe_lag_direction(lag: int) -> str:
    if lag > 0:
        return "变量领先目标"
    if lag < 0:
        return "变量滞后目标"
    return "同步变化"


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    pvals = pd.to_numeric(values, errors="coerce")
    qvals = pd.Series(np.nan, index=values.index, dtype=float)
    valid = pvals.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return qvals
    ranked_items = list(valid.items())
    running = 1.0
    for rank in range(m, 0, -1):
        original_idx, original_p = ranked_items[rank - 1]
        value = min(running, float(original_p) * m / rank)
        running = value
        qvals.loc[original_idx] = value
    return qvals
