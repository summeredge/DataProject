from __future__ import annotations

import numpy as np
import pandas as pd

from chem_ts_corr.common import benjamini_hochberg
from chem_ts_corr.feature_alignment import fit_named_matrix_model, predict_named_matrix_model
from chem_ts_corr.time_axis import lagged_series, sample_period_ns


OUT_COLS = [
    "variable",
    "status",
    "best_lag",
    "min_p_value",
    "fdr_q_value",
    "baseline_rmse",
    "full_rmse",
    "predictive_contribution",
    "base_condition_number",
    "full_condition_number",
    "condition_number",
    "control_columns",
    "n_rows",
    "tested_lags",
    "lag_mode",
    "lag_window",
    "fallback_maxlag",
    "baseline_maxlag",
    "interpretation",
]


def run_conditional_granger_tests(
    frame: pd.DataFrame,
    target: str,
    variables: list[str],
    control_columns: list[str] | None = None,
    maxlag: int = 12,
    min_rows: int = 60,
    candidate_lags: dict[str, list[int]] | None = None,
    candidate_lag_status: dict[str, str] | None = None,
    baseline_maxlag: int | None = None,
    lag_mode: str | None = None,
    lag_window: int | None = None,
    fallback_maxlag: int | None = None,
    target_mask: pd.Series | None = None,
) -> pd.DataFrame:
    if target not in frame.columns:
        raise ValueError(f"target column not found: {target}")

    controls = _normalized_controls(control_columns, frame.columns, target)
    baseline_lag_limit = maxlag if baseline_maxlag is None else min(maxlag, max(1, int(baseline_maxlag)))
    rows: list[dict[str, object]] = []
    period_ns = sample_period_ns(frame)
    resolved_target_mask = (
        target_mask.reindex(frame.index).fillna(False).astype(bool)
        if target_mask is not None
        else None
    )

    scipy_f = None
    scipy_available = True
    try:
        from scipy.stats import f as scipy_f  # type: ignore
    except Exception:
        scipy_available = False

    for variable in variables:
        effective_controls = _effective_controls(controls, variable)
        base_row = {
            "variable": variable,
            "status": "",
            "best_lag": np.nan,
            "min_p_value": np.nan,
            "fdr_q_value": np.nan,
            "baseline_rmse": np.nan,
            "full_rmse": np.nan,
            "predictive_contribution": 0.0,
            "base_condition_number": np.nan,
            "full_condition_number": np.nan,
            "condition_number": np.nan,
            "control_columns": ",".join(str(control) for control in effective_controls),
            "n_rows": 0,
            "tested_lags": "",
            "lag_mode": lag_mode or "",
            "lag_window": np.nan if lag_window is None else int(lag_window),
            "fallback_maxlag": np.nan if fallback_maxlag is None else int(fallback_maxlag),
            "baseline_maxlag": baseline_lag_limit,
            "interpretation": "predictive validation only; not a causal conclusion",
        }
        if variable not in frame.columns:
            base_row["status"] = "skipped: variable not found"
            rows.append(base_row)
            continue
        if variable == target:
            base_row["status"] = "skipped: same as target"
            rows.append(base_row)
            continue

        candidates: list[dict[str, object]] = []
        y_series = pd.to_numeric(frame[target], errors="coerce")
        x_series = pd.to_numeric(frame[variable], errors="coerce")
        control_series = {c: pd.to_numeric(frame[c], errors="coerce") for c in effective_controls}
        lag_values = _candidate_lags_for_variable(variable, maxlag, candidate_lags)
        lag_status = (candidate_lag_status or {}).get(str(variable), "")
        base_row["tested_lags"] = ",".join(str(lag) for lag in lag_values)
        if not lag_values:
            if lag_status == "non_positive_screening_lag":
                base_row["status"] = "skipped: non-positive screening lag"
            elif lag_status == "ranked_lag_outside_maxlag":
                base_row["status"] = "skipped: ranked lag outside maxlag"
            elif lag_status == "invalid_screening_lag":
                base_row["status"] = "skipped: invalid screening lag"
            elif (lag_mode or "") in {"ranked_window", "best_only"} and candidate_lags is not None and str(variable) in candidate_lags:
                base_row["status"] = "skipped: non-positive screening lag"
            else:
                base_row["status"] = "skipped: no candidate lags"
            rows.append(base_row)
            continue

        base_df = pd.DataFrame(index=frame.index)
        base_df["__target_current"] = y_series
        # baseline model: target lags + control lags. Build these once per
        # variable because they do not change across candidate x lags.
        y_lag_cols = []
        for lag in range(1, baseline_lag_limit + 1):
            col = f"__target_lag_{lag}"
            base_df[col] = lagged_series(y_series, frame.index, lag, period_ns=period_ns)
            y_lag_cols.append(col)
        control_lag_cols = []
        for c, series in control_series.items():
            for lag in range(1, baseline_lag_limit + 1):
                col = f"__control__{len(control_lag_cols)}__lag_{lag}"
                base_df[col] = lagged_series(series, frame.index, lag, period_ns=period_ns)
                control_lag_cols.append(col)
        base_cols = y_lag_cols + control_lag_cols

        for lag in lag_values:
            # full model adds exactly the candidate lag being tested.
            x_lag_col = f"__candidate_lag_{lag}"
            df = base_df.assign(
                **{x_lag_col: lagged_series(x_series, frame.index, lag, period_ns=period_ns)}
            )
            if resolved_target_mask is not None:
                df = df.loc[resolved_target_mask]
            df = df.dropna()
            n = len(df)
            if n < min_rows:
                continue

            y = df["__target_current"].to_numpy(dtype=float)
            full_cols = base_cols + [x_lag_col]

            x_base = np.column_stack([np.ones(n), df[base_cols].to_numpy(dtype=float)])
            x_full = np.column_stack([np.ones(n), df[full_cols].to_numpy(dtype=float)])
            base_condition_number = _condition_number(x_base)
            full_condition_number = _condition_number(x_full)
            condition_number = max(base_condition_number, full_condition_number)
            collinearity_status = (
                "high_collinearity_risk"
                if (not np.isfinite(condition_number) or condition_number > 1e8)
                else "ok"
            )

            try:
                base_feature_names = ["__intercept__", *base_cols]
                full_feature_names = ["__intercept__", *full_cols]
                base_model, _ = fit_named_matrix_model(x_base, y, base_feature_names)
                full_model, _ = fit_named_matrix_model(x_full, y, full_feature_names)
            except Exception:
                continue

            resid_b = y - predict_named_matrix_model(
                base_model, x_base, base_feature_names
            )
            resid_f = y - predict_named_matrix_model(
                full_model, x_full, full_feature_names
            )
            rss_b = float(np.sum(resid_b * resid_b))
            rss_f = float(np.sum(resid_f * resid_f))
            rmse_b = float(np.sqrt(np.mean(resid_b * resid_b)))
            rmse_f = float(np.sqrt(np.mean(resid_f * resid_f)))
            pred_contrib = max(0.0, (rmse_b - rmse_f) / rmse_b) if rmse_b > 0 else 0.0

            p = x_base.shape[1]
            q = x_full.shape[1]
            df_num = q - p
            df_den = n - q
            if df_num <= 0 or df_den <= 0:
                continue

            p_value = np.nan
            if collinearity_status == "ok" and rss_f > 0 and rss_b >= rss_f:
                f_stat = ((rss_b - rss_f) / df_num) / (rss_f / df_den)
                if scipy_available and scipy_f is not None and np.isfinite(f_stat):
                    p_value = float(scipy_f.sf(max(0.0, f_stat), df_num, df_den))

            candidate = {
                "lag": lag,
                "n_rows": n,
                "p_value": p_value,
                "baseline_rmse": rmse_b,
                "full_rmse": rmse_f,
                "predictive_contribution": pred_contrib,
                "base_condition_number": base_condition_number,
                "full_condition_number": full_condition_number,
                "condition_number": condition_number,
                "collinearity_status": collinearity_status,
                "q_value": np.nan,
            }
            candidates.append(candidate)

        if not candidates:
            base_row["status"] = "skipped: insufficient rows"
            rows.append(base_row)
            continue
        base_row["_candidates"] = candidates
        base_row["_lag_status"] = lag_status
        rows.append(base_row)

    family = [
        candidate
        for row in rows
        for candidate in row.get("_candidates", [])
        if np.isfinite(candidate["p_value"])
    ]
    q_values = benjamini_hochberg([candidate["p_value"] for candidate in family])
    for candidate, q_value in zip(family, q_values):
        candidate["q_value"] = float(q_value)

    for row in rows:
        candidates = row.pop("_candidates", None)
        lag_status = str(row.pop("_lag_status", ""))
        if not isinstance(candidates, list):
            continue
        best = min(candidates, key=_conditional_candidate_key)
        status = (
            "high_collinearity_risk"
            if best["collinearity_status"] == "high_collinearity_risk"
            else ("ok" if scipy_available else "ok: scipy unavailable, p_value is NaN")
        )
        if lag_status == "fallback_missing_ranked_lag" and status.startswith("ok"):
            status = status.replace("ok", "ok: fallback_missing_ranked_lag", 1)
        row.update(
            {
                "status": status,
                "best_lag": int(best["lag"]),
                "min_p_value": best["p_value"],
                "fdr_q_value": best["q_value"],
                "baseline_rmse": best["baseline_rmse"],
                "full_rmse": best["full_rmse"],
                "predictive_contribution": best["predictive_contribution"],
                "base_condition_number": best["base_condition_number"],
                "full_condition_number": best["full_condition_number"],
                "condition_number": best["condition_number"],
                "n_rows": int(best["n_rows"]),
                "interpretation": (
                    "predictive validation only; not a causal conclusion; analytic p/q values "
                    "do not fully remove industrial time-series autocorrelation effects"
                ),
            }
        )

    out = pd.DataFrame(rows, columns=OUT_COLS)
    return out


def _conditional_candidate_key(candidate: dict[str, object]) -> tuple[float, ...]:
    q_value = float(candidate["q_value"])
    p_value = float(candidate["p_value"])
    contribution = float(candidate["predictive_contribution"])
    lag = float(candidate["lag"])
    if np.isfinite(q_value):
        return (0.0, q_value, p_value, -contribution, lag)
    return (1.0, float("inf"), float("inf"), -contribution, lag)


def build_candidate_lag_windows(
    ranked_features: pd.DataFrame,
    variables: list[str],
    maxlag: int,
    window: int = 5,
    fallback_maxlag: int = 24,
) -> dict[str, list[int]]:
    return {
        variable: values["lags"]
        for variable, values in build_candidate_lag_window_status(
            ranked_features=ranked_features,
            variables=variables,
            maxlag=maxlag,
            window=window,
            fallback_maxlag=fallback_maxlag,
        ).items()
    }


def build_candidate_lag_window_status(
    ranked_features: pd.DataFrame,
    variables: list[str],
    maxlag: int,
    window: int = 5,
    fallback_maxlag: int = 24,
) -> dict[str, dict[str, object]]:
    ranked_lags: dict[str, object] = {}
    if not ranked_features.empty and {"variable", "lag"}.issubset(ranked_features.columns):
        for _, row in ranked_features.iterrows():
            variable = row.get("variable")
            if pd.isna(variable):
                continue
            name = str(variable)
            if name not in ranked_lags:
                ranked_lags[name] = row.get("lag")

    out: dict[str, dict[str, object]] = {}
    safe_maxlag = int(maxlag)
    safe_window = max(0, int(window))
    safe_fallback_maxlag = max(1, int(fallback_maxlag))
    for variable in variables:
        name = str(variable)
        has_ranked_lag = name in ranked_lags
        raw_lag = ranked_lags.get(name)
        parsed_lag = _parse_lag(raw_lag)
        center = _valid_lag_center(raw_lag)
        status = "ranked_lag_window"
        if center is None:
            if has_ranked_lag and parsed_lag is not None and parsed_lag <= 0:
                lags = range(0)
                status = "non_positive_screening_lag"
            elif has_ranked_lag and parsed_lag is None and not pd.isna(raw_lag):
                lags = range(0)
                status = "invalid_screening_lag"
            else:
                end = min(safe_maxlag, safe_fallback_maxlag)
                lags = range(1, end + 1) if end >= 1 else range(0)
                status = "fallback_missing_ranked_lag"
        else:
            start = max(1, center - safe_window)
            end = min(safe_maxlag, center + safe_window)
            if end < start:
                lags = range(0)
                status = "ranked_lag_outside_maxlag"
            else:
                lags = range(start, end + 1)
        out[name] = {
            "lags": sorted(set(int(lag) for lag in lags if 1 <= int(lag) <= safe_maxlag)),
            "status": status,
        }
    return out


def _normalized_controls(control_columns: list[str] | None, frame_columns: pd.Index, target: str) -> list[object]:
    existing_by_text = {str(column): column for column in frame_columns}
    target_text = str(target)
    controls: list[object] = []
    seen: set[str] = set()
    for column in control_columns or []:
        text = str(column)
        if text == target_text or text in seen or text not in existing_by_text:
            continue
        controls.append(existing_by_text[text])
        seen.add(text)
    return controls


def _effective_controls(controls: list[object], variable: str) -> list[object]:
    variable_text = str(variable)
    return [control for control in controls if str(control) != variable_text]


def _candidate_lags_for_variable(
    variable: str,
    maxlag: int,
    candidate_lags: dict[str, list[int]] | None,
) -> list[int]:
    if candidate_lags is not None and variable in candidate_lags:
        valid_lags = []
        for lag in candidate_lags[variable]:
            parsed = _valid_positive_lag(lag)
            if parsed is not None and 1 <= parsed <= maxlag:
                valid_lags.append(parsed)
        return sorted(set(valid_lags))
    return list(range(1, maxlag + 1))


def _valid_lag_center(value: object) -> int | None:
    lag = _parse_lag(value)
    if lag is None:
        return None
    return lag if lag >= 1 else None


def _valid_positive_lag(value: object) -> int | None:
    lag = _parse_lag(value)
    if lag is None:
        return None
    return lag if lag >= 1 else None


def _parse_lag(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _condition_number(matrix: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(matrix))
    except Exception:
        return float("inf")
