from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from chem_ts_corr.common import benjamini_hochberg


LAG_PEAK_QUALITY_COLUMNS = [
    "variable",
    "best_lag",
    "best_score",
    "nearby_score_mean",
    "peak_sharpness",
    "second_peak_score",
    "peak_prominence",
    "local_sharpness",
    "shape_quality",
    "lag_boundary_flag",
    "lag_quality",
]


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
    except Exception:
        return np.nan
    return float(2 * student_t.sf(t_stat, df))


def compute_lag_scores(
    frame: pd.DataFrame,
    target: str,
    max_lag: int,
    lag_values: Iterable[int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    target_series = frame[target]
    scan_lags = tuple(range(-max_lag, max_lag + 1)) if lag_values is None else tuple(lag_values)

    for variable in frame.columns:
        if variable == target:
            continue
        series = frame[variable]
        for lag in scan_lags:
            lag = int(lag)
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
                    "abs_pearson": abs(pearson_r) if not np.isnan(pearson_r) else np.nan,
                    "abs_spearman": abs(spearman_r) if not np.isnan(spearman_r) else np.nan,
                    "lag_boundary_flag": abs(lag) == max_lag,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["pearson_q"] = benjamini_hochberg(result["pearson_p"])
        result["spearman_q"] = benjamini_hochberg(result["spearman_p"])
        use_pearson = result["abs_pearson"] >= result["abs_spearman"]
        result["p_value"] = np.where(use_pearson, result["pearson_p"], result["spearman_p"])
        result["corr_q_value"] = np.where(use_pearson, result["pearson_q"], result["spearman_q"])
        result["p_value_status"] = np.where(result["p_value"].isna(), "scipy_unavailable_or_invalid", "ok")
    return result


def summarize_best_lags(lag_scores: pd.DataFrame) -> pd.DataFrame:
    if lag_scores.empty:
        return lag_scores
    ranked = lag_scores.assign(score=lag_scores[["abs_pearson", "abs_spearman"]].max(axis=1))
    ranked = ranked.dropna(subset=["score"])
    if ranked.empty:
        return pd.DataFrame(columns=list(lag_scores.columns) + ["score", "method", "p_value", "r2", "corr_q_value", "direction"])
    idx = ranked.groupby("variable")["score"].idxmax()
    best = ranked.loc[idx].sort_values("score", ascending=False).reset_index(drop=True)
    use_pearson = best["abs_pearson"] >= best["abs_spearman"]
    best["method"] = np.where(use_pearson, "pearson", "spearman")
    best["p_value"] = np.where(use_pearson, best["pearson_p"], best["spearman_p"])
    best["r2"] = np.where(use_pearson, best["pearson_r2"], best["spearman_r2"])
    best["corr_q_value"] = np.where(use_pearson, best["pearson_q"], best["spearman_q"])
    best["direction"] = best["lag"].map(describe_lag_direction)
    return best


def build_lag_peak_quality(lag_scores: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    if lag_scores.empty:
        return pd.DataFrame(columns=LAG_PEAK_QUALITY_COLUMNS)
    ranked = lag_scores.assign(score=lag_scores[["abs_pearson", "abs_spearman"]].max(axis=1))
    rows = []
    epsilon = 1e-12
    for var, g in ranked.groupby("variable", sort=False):
        g = g.dropna(subset=["score"])
        if g.empty:
            continue
        best = g.loc[g["score"].idxmax()]
        bl = int(best["lag"])
        best_score = float(best["score"])

        nearby_scores = g.loc[g["lag"].sub(bl).abs().eq(1), "score"]
        nearby = float(nearby_scores.mean()) if not nearby_scores.empty else np.nan
        peak_sharpness = max(0.0, best_score - nearby) if pd.notna(nearby) else 0.0
        local_sharpness = float(np.clip(peak_sharpness / max(best_score, epsilon), 0.0, 1.0))

        competing_scores = g.loc[g["lag"].sub(bl).abs().gt(1), "score"]
        second_peak_score = float(competing_scores.max()) if not competing_scores.empty else np.nan
        peak_prominence = (
            float(np.clip((best_score - second_peak_score) / max(best_score, epsilon), 0.0, 1.0))
            if pd.notna(second_peak_score)
            else 0.0
        )
        shape_quality = float(
            np.clip(
                0.70 * peak_prominence
                + 0.30 * min(peak_prominence, local_sharpness),
                0.0,
                1.0,
            )
        )
        boundary = abs(bl) == max_lag
        lag_quality = float(np.clip(shape_quality - (0.25 if boundary else 0.0), 0.0, 1.0))
        rows.append(
            {
                "variable": var,
                "best_lag": bl,
                "best_score": best_score,
                "nearby_score_mean": nearby,
                "peak_sharpness": peak_sharpness,
                "second_peak_score": second_peak_score,
                "peak_prominence": peak_prominence,
                "local_sharpness": local_sharpness,
                "shape_quality": shape_quality,
                "lag_boundary_flag": boundary,
                "lag_quality": lag_quality,
            }
        )
    return pd.DataFrame(rows, columns=LAG_PEAK_QUALITY_COLUMNS)


def describe_lag_direction(lag: int) -> str:
    if lag > 0:
        return "变量领先目标"
    if lag < 0:
        return "变量滞后目标"
    return "同步变化"
