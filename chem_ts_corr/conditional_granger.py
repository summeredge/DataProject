from __future__ import annotations

import numpy as np
import pandas as pd


OUT_COLS = [
    "variable",
    "status",
    "best_lag",
    "min_p_value",
    "fdr_q_value",
    "baseline_rmse",
    "full_rmse",
    "predictive_contribution",
    "control_columns",
    "n_rows",
    "interpretation",
]


def run_conditional_granger_tests(
    frame: pd.DataFrame,
    target: str,
    variables: list[str],
    control_columns: list[str] | None = None,
    maxlag: int = 12,
    min_rows: int = 60,
) -> pd.DataFrame:
    if target not in frame.columns:
        raise ValueError(f"target column not found: {target}")

    controls = [c for c in (control_columns or []) if c in frame.columns and c != target]
    rows: list[dict[str, object]] = []

    scipy_f = None
    scipy_available = True
    try:
        from scipy.stats import f as scipy_f  # type: ignore
    except Exception:
        scipy_available = False

    for variable in variables:
        base_row = {
            "variable": variable,
            "status": "",
            "best_lag": np.nan,
            "min_p_value": np.nan,
            "fdr_q_value": np.nan,
            "baseline_rmse": np.nan,
            "full_rmse": np.nan,
            "predictive_contribution": 0.0,
            "control_columns": ",".join(controls),
            "n_rows": 0,
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

        best = None
        y_name = target
        for lag in range(1, maxlag + 1):
            df = pd.DataFrame(index=frame.index)
            df[y_name] = pd.to_numeric(frame[target], errors="coerce")
            # baseline model: target lags + control lags
            for l in range(1, maxlag + 1):
                df[f"y_lag_{l}"] = pd.to_numeric(frame[target], errors="coerce").shift(l)
            control_lag_cols = []
            for c in controls:
                for l in range(1, maxlag + 1):
                    col = f"{c}_lag_{l}"
                    df[col] = pd.to_numeric(frame[c], errors="coerce").shift(l)
                    control_lag_cols.append(col)
            # full model adds exactly the candidate lag being tested.
            candidate_lag_col = f"x_lag_{lag}"
            df[candidate_lag_col] = pd.to_numeric(frame[variable], errors="coerce").shift(lag)
            df = df.dropna()
            n = len(df)
            if n < min_rows:
                continue

            y = df[y_name].to_numpy(dtype=float)
            target_lag_cols = [f"y_lag_{l}" for l in range(1, maxlag + 1)]
            base_cols = target_lag_cols + control_lag_cols
            full_cols = base_cols + [candidate_lag_col]

            x_base = np.column_stack([np.ones(n), df[base_cols].to_numpy(dtype=float)])
            x_full = np.column_stack([np.ones(n), df[full_cols].to_numpy(dtype=float)])

            try:
                b_coef, *_ = np.linalg.lstsq(x_base, y, rcond=None)
                f_coef, *_ = np.linalg.lstsq(x_full, y, rcond=None)
            except Exception:
                continue

            resid_b = y - x_base @ b_coef
            resid_f = y - x_full @ f_coef
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
            if rss_f > 0 and rss_b >= rss_f:
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
            }
            if best is None:
                best = candidate
            else:
                # prefer lower p-value when available, otherwise higher contribution
                best_p = best["p_value"]
                cand_p = candidate["p_value"]
                if np.isnan(best_p) and not np.isnan(cand_p):
                    best = candidate
                elif (not np.isnan(cand_p)) and (not np.isnan(best_p)) and cand_p < best_p:
                    best = candidate
                elif np.isnan(cand_p) and np.isnan(best_p) and candidate["predictive_contribution"] > best["predictive_contribution"]:
                    best = candidate

        if best is None:
            base_row["status"] = "skipped: insufficient rows"
            rows.append(base_row)
            continue

        base_row.update(
            {
                "status": "ok" if scipy_available else "ok: scipy unavailable, p_value is NaN",
                "best_lag": int(best["lag"]),
                "min_p_value": best["p_value"],
                "baseline_rmse": best["baseline_rmse"],
                "full_rmse": best["full_rmse"],
                "predictive_contribution": best["predictive_contribution"],
                "n_rows": int(best["n_rows"]),
            }
        )
        rows.append(base_row)

    out = pd.DataFrame(rows, columns=OUT_COLS)
    if not out.empty:
        out["fdr_q_value"] = _benjamini_hochberg(out["min_p_value"])
    return out


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
        idx, p = ranked_items[rank - 1]
        value = min(running, float(p) * m / rank)
        running = value
        qvals.loc[idx] = value
    return qvals
