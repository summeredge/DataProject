from __future__ import annotations

import argparse
import json
import time
import tracemalloc

import numpy as np
import pandas as pd

from chem_ts_corr.common import benjamini_hochberg
from chem_ts_corr.lag import (
    LAG_SCORE_COLUMNS,
    _corr_p_value,
    compute_lag_scores,
    summarize_best_lags,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark lag-score alignment reuse")
    parser.add_argument("--rows", type=int, default=4_000)
    parser.add_argument("--variables", type=int, default=14)
    parser.add_argument("--max-lag", type=int, default=72)
    return parser


def _legacy_stats(x: pd.Series, y: pd.Series, method: str) -> dict[str, float | int]:
    aligned = pd.concat([x, y], axis=1).dropna()
    n = len(aligned)
    if n < 5 or aligned.iloc[:, 0].nunique() <= 1 or aligned.iloc[:, 1].nunique() <= 1:
        return {"r": np.nan, "p_value": np.nan, "r2": np.nan, "n": n}
    corr_frame = aligned if method == "pearson" else aligned.rank(method="average")
    r = float(corr_frame.iloc[:, 0].corr(corr_frame.iloc[:, 1], method="pearson"))
    return {"r": r, "p_value": _corr_p_value(r, n), "r2": r * r, "n": n}


def _legacy_compute_lag_scores(frame: pd.DataFrame, target: str, max_lag: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in frame.columns:
        if variable == target:
            continue
        for lag in range(-max_lag, max_lag + 1):
            shifted = frame[variable].shift(lag)
            pearson = _legacy_stats(shifted, frame[target], "pearson")
            spearman = _legacy_stats(shifted, frame[target], "spearman")
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
    result = pd.DataFrame(rows, columns=LAG_SCORE_COLUMNS)
    family = pd.concat([result["pearson_p"], result["spearman_p"]], ignore_index=True)
    q_values = benjamini_hochberg(family)
    result["pearson_q"] = q_values.iloc[: len(result)].to_numpy()
    result["spearman_q"] = q_values.iloc[len(result) :].to_numpy()
    use_pearson = result["abs_pearson"] >= result["abs_spearman"]
    result["p_value"] = np.where(use_pearson, result["pearson_p"], result["spearman_p"])
    result["corr_q_value"] = np.where(use_pearson, result["pearson_q"], result["spearman_q"])
    result["p_value_status"] = np.where(
        result["p_value"].isna(), "scipy_unavailable_or_invalid", "ok"
    )
    return result


def _measure(function, frame: pd.DataFrame, max_lag: int) -> tuple[pd.DataFrame, float, float]:
    tracemalloc.start()
    started = time.perf_counter()
    result = function(frame, "target", max_lag)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def _frame(rows: int, variables: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    target = rng.normal(size=rows)
    data = {"target": target}
    for number in range(variables):
        values = rng.normal(size=rows)
        values[number :: 211] = np.nan
        data[f"x{number + 1}"] = values
    return pd.DataFrame(data)


def main() -> int:
    args = _parser().parse_args()
    frame = _frame(args.rows, args.variables)
    legacy, legacy_seconds, legacy_peak = _measure(
        _legacy_compute_lag_scores, frame, args.max_lag
    )
    optimized, optimized_seconds, optimized_peak = _measure(
        compute_lag_scores, frame, args.max_lag
    )
    numeric = [
        "pearson", "pearson_p", "pearson_r2", "spearman", "spearman_p",
        "spearman_r2", "n", "abs_pearson", "abs_spearman", "pearson_q",
        "spearman_q", "p_value", "corr_q_value",
    ]
    difference = np.abs(
        legacy[numeric].to_numpy(dtype=float) - optimized[numeric].to_numpy(dtype=float)
    )
    legacy_best = summarize_best_lags(legacy)[["variable", "lag", "method"]]
    optimized_best = summarize_best_lags(optimized)[["variable", "lag", "method"]]
    payload = {
        "rows": args.rows,
        "variables": args.variables,
        "max_lag": args.max_lag,
        "legacy_elapsed_seconds": legacy_seconds,
        "optimized_elapsed_seconds": optimized_seconds,
        "speedup": legacy_seconds / optimized_seconds,
        "legacy_peak_memory_mb": legacy_peak,
        "optimized_peak_memory_mb": optimized_peak,
        "legacy_alignment_calls": args.variables * (2 * args.max_lag + 1) * 2,
        "optimized_alignment_calls": args.variables * (2 * args.max_lag + 1),
        "max_absolute_error": float(np.nanmax(difference)),
        "best_lag_and_method_identical": legacy_best.equals(optimized_best),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
