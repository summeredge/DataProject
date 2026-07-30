from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from chem_ts_corr.common import benjamini_hochberg
from chem_ts_corr.time_axis import lagged_series, sample_period_ns

NEAR_PEAK_SCORE_RATIO = 0.95
TEMPORAL_NEUTRAL_BAND_POINTS = 1

LAG_SCORE_COLUMNS = [
    "variable", "lag", "pearson", "pearson_p", "pearson_r2", "spearman",
    "spearman_p", "spearman_r2", "n", "abs_pearson", "abs_spearman",
    "lag_boundary_flag", "pearson_q", "spearman_q", "p_value",
    "corr_q_value", "p_value_status",
]
BEST_LAG_COLUMNS = LAG_SCORE_COLUMNS + ["score", "method", "r2", "direction"]


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
    "near_peak_lag_min",
    "near_peak_lag_max",
    "near_peak_lag_count",
    "temporal_direction_status",
]


def _aligned_corr_stats(x: pd.Series, y: pd.Series) -> dict[str, dict[str, float | int]]:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    valid = ~np.isnan(x_values) & ~np.isnan(y_values)
    x_valid = x_values[valid]
    y_valid = y_values[valid]
    n = len(x_valid)
    empty = {"r": np.nan, "p_value": np.nan, "r2": np.nan, "n": n}
    if n < 5 or len(np.unique(x_valid)) <= 1 or len(np.unique(y_valid)) <= 1:
        return {"pearson": empty.copy(), "spearman": empty.copy()}

    pearson_r = float(np.corrcoef(x_valid, y_valid)[0, 1])
    x_ranked = pd.Series(x_valid).rank(method="average").to_numpy(dtype=float)
    y_ranked = pd.Series(y_valid).rank(method="average").to_numpy(dtype=float)
    spearman_r = float(np.corrcoef(x_ranked, y_ranked)[0, 1])
    return {
        "pearson": _stats_from_correlation(pearson_r, n),
        "spearman": _stats_from_correlation(spearman_r, n),
    }


def _stats_from_correlation(r: float, n: int) -> dict[str, float | int]:
    return {"r": r, "p_value": _corr_p_value(r, n), "r2": r * r, "n": n}


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
    target_mask: pd.Series | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    target_series = frame[target]
    scan_lags = tuple(range(-max_lag, max_lag + 1)) if lag_values is None else tuple(lag_values)
    period_ns = sample_period_ns(frame)
    resolved_mask = (
        target_mask.reindex(frame.index).fillna(False).astype(bool)
        if target_mask is not None
        else None
    )

    for variable in frame.columns:
        if variable == target:
            continue
        series = frame[variable]
        for lag in scan_lags:
            lag = int(lag)
            shifted = lagged_series(series, target_series.index, lag, period_ns=period_ns)
            if resolved_mask is not None:
                shifted = shifted.where(resolved_mask)
            stats = _aligned_corr_stats(shifted, target_series)
            pearson = stats["pearson"]
            spearman = stats["spearman"]
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
                    "lag_boundary_flag": lag == -max_lag or lag == max_lag,
                }
            )

    result = pd.DataFrame(rows, columns=LAG_SCORE_COLUMNS)
    if not result.empty:
        family: list[float] = []
        positions: list[tuple[str, object]] = []
        for method in ("pearson", "spearman"):
            for index, p_value in result[f"{method}_p"].items():
                if pd.notna(p_value):
                    family.append(float(p_value))
                    positions.append((method, index))
        result["pearson_q"] = np.nan
        result["spearman_q"] = np.nan
        for (method, index), q_value in zip(positions, benjamini_hochberg(family)):
            result.loc[index, f"{method}_q"] = q_value
        use_pearson = result["abs_pearson"] >= result["abs_spearman"]
        result["p_value"] = np.where(use_pearson, result["pearson_p"], result["spearman_p"])
        result["corr_q_value"] = np.where(use_pearson, result["pearson_q"], result["spearman_q"])
        result["p_value_status"] = np.where(result["p_value"].isna(), "scipy_unavailable_or_invalid", "ok")
    return result


def summarize_best_lags(lag_scores: pd.DataFrame) -> pd.DataFrame:
    if lag_scores.empty:
        return pd.DataFrame(columns=BEST_LAG_COLUMNS)
    ranked = lag_scores.assign(score=lag_scores[["abs_pearson", "abs_spearman"]].max(axis=1))
    ranked = ranked.dropna(subset=["score"])
    if ranked.empty:
        return pd.DataFrame(columns=BEST_LAG_COLUMNS)
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
        near_peak = g.loc[g["score"].ge(best_score * NEAR_PEAK_SCORE_RATIO)]
        near_peak_lag_min = int(near_peak["lag"].min())
        near_peak_lag_max = int(near_peak["lag"].max())
        if near_peak_lag_min > TEMPORAL_NEUTRAL_BAND_POINTS:
            temporal_direction_status = "variable_leads_supported"
        elif near_peak_lag_max < -TEMPORAL_NEUTRAL_BAND_POINTS:
            temporal_direction_status = "target_leads_supported"
        elif near_peak_lag_min == 0 and near_peak_lag_max == 0:
            temporal_direction_status = "synchronous"
        else:
            temporal_direction_status = "direction_unresolved"

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
        boundary = bl == -max_lag or bl == max_lag
        lag_quality = float(np.clip(shape_quality * (0.75 if boundary else 1.0), 0.0, 1.0))
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
                "near_peak_lag_min": near_peak_lag_min,
                "near_peak_lag_max": near_peak_lag_max,
                "near_peak_lag_count": int(len(near_peak)),
                "temporal_direction_status": temporal_direction_status,
            }
        )
    return pd.DataFrame(rows, columns=LAG_PEAK_QUALITY_COLUMNS)


def describe_lag_direction(lag: int) -> str:
    if lag > 0:
        return "变量领先目标"
    if lag < 0:
        return "变量滞后目标"
    return "同步变化"
