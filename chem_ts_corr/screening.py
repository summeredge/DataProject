from __future__ import annotations

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
    for column in (config.capacity_columns or []):
        if column in roles:
            roles[column] = "CAPACITY"
    return roles


def apply_ignore_roles(frame: pd.DataFrame, roles: dict[str, str], target: str) -> pd.DataFrame:
    ignored = [c for c, r in roles.items() if r == "IGNORE" and c != target]
    return frame.drop(columns=[c for c in ignored if c in frame.columns], errors="ignore")


def diagnostics(frame: pd.DataFrame, roles: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        rows.append({"variable": column, "role": roles.get(column, "PV"), "missing_rate": float(series.isna().mean())})
    return pd.DataFrame(rows, columns=["variable", "role", "missing_rate"])


def residual_corr_scores(frame: pd.DataFrame, target: str, control_columns: list[str] | None, max_lag: int) -> pd.DataFrame:
    out_cols = ["variable", "lag", "residual_corr", "residual_p_value", "residual_r2", "direction"]
    controls = [c for c in (control_columns or []) if c in frame.columns]
    if not controls or target not in frame.columns:
        return pd.DataFrame(columns=out_cols)
    target_residual = _residualize(frame[target], frame[controls])
    rows: list[pd.DataFrame] = []
    for column in frame.columns:
        if column == target or column in controls:
            continue
        candidate_residual = _residualize(frame[column], frame[controls])
        pair = pd.DataFrame({target: target_residual, column: candidate_residual}).dropna()
        if len(pair) < max(10, max_lag + 5):
            continue
        scores = summarize_best_lags(compute_lag_scores(pair, target, max_lag))
        if not scores.empty:
            rows.append(scores)
    if not rows:
        return pd.DataFrame(columns=out_cols)
    scores = pd.concat(rows, ignore_index=True)
    scores = scores.rename(columns={"score": "residual_corr", "p_value": "residual_p_value", "r2": "residual_r2"})
    for col in out_cols:
        if col not in scores.columns:
            scores[col] = np.nan
    return scores[out_cols]


def regime_scores(frame: pd.DataFrame, target: str, capacity_column: str | None, max_lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc = ["variable", "regime", "score", "signed_corr", "lag", "direction", "p_value", "r2"]
    st = ["variable", "regime_stability_final", "regime_sign_consistency", "regime_lag_consistency", "regime_score_cv", "regime_count"]
    return pd.DataFrame(columns=sc), pd.DataFrame(columns=st)


def model_lift_scores(frame: pd.DataFrame, target: str, candidate_variables: list[str], max_lag: int, n_splits: int = 4, best_lags: dict[str, int] | None = None) -> pd.DataFrame:
    cols = ["variable", "status", "ar_baseline_rmse", "candidate_rmse", "model_lift"]
    rows = [{"variable": v, "status": "skipped", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": 0.0} for v in candidate_variables if v != target and v in frame.columns]
    return pd.DataFrame(rows, columns=cols)


def rolling_corr_scores(frame: pd.DataFrame, target: str, candidate_variables: list[str], max_lag: int, window: int | None = None, min_periods: int | None = None) -> pd.DataFrame:
    cols = ["variable", "best_lag", "best_score", "rolling_corr_median", "rolling_abs_corr_median", "rolling_corr_iqr", "rolling_sign_consistency", "valid_window_count", "rolling_stability"]
    if target not in frame.columns or not candidate_variables:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, object]] = []
    w = max(12, int(window or min(len(frame), max(24, max_lag * 4))))
    mp = max(6, int(min_periods or w // 2))
    for v in candidate_variables:
        if v == target or v not in frame.columns:
            continue
        pair = frame[[target, v]].dropna()
        if len(pair) < max(w, max_lag + 10):
            continue
        best = summarize_best_lags(compute_lag_scores(pair, target, max_lag))
        if best.empty:
            continue
        b = best.iloc[0]
        lag = int(b["lag"])
        rc = pair[v].shift(lag).rolling(window=w, min_periods=mp).corr(pair[target]).dropna()
        if rc.empty:
            continue
        sign_cons = float(rc.apply(lambda x: 1 if x >= 0 else -1).value_counts(normalize=True).max())
        iqr = float(rc.quantile(0.75) - rc.quantile(0.25))
        abs_med = float(rc.abs().median())
        rows.append({"variable": v, "best_lag": lag, "best_score": float(b.get("score", 0) or 0), "rolling_corr_median": float(rc.median()), "rolling_abs_corr_median": abs_med, "rolling_corr_iqr": iqr, "rolling_sign_consistency": sign_cons, "valid_window_count": int(len(rc)), "rolling_stability": max(0.0, min(1.0, abs_med * sign_cons * (1 - min(1, iqr))))})
    return pd.DataFrame(rows, columns=cols)


def risk_flags(ranked: pd.DataFrame, residual: pd.DataFrame, stability: pd.DataFrame, diag: pd.DataFrame, roles: dict[str, str], control_columns: list[str] | None, lag_peak_quality: pd.DataFrame | None = None, rolling_corr_scores: pd.DataFrame | None = None, model_lift_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    cols = ["variable", "formula_like_flag", "strong_formula_leakage_flag", "common_capacity_driver_flag", "closed_loop_suspect_flag", "target_leads_variable_flag", "unstable_across_regimes_flag", "unstable_over_time_flag", "lag_boundary_flag", "low_model_lift_flag", "poor_data_quality_flag", "risk_flags", "risk_count"]
    if ranked.empty:
        return pd.DataFrame(columns=cols)
    lag_map = (lag_peak_quality or pd.DataFrame()).set_index("variable").to_dict("index") if lag_peak_quality is not None and not lag_peak_quality.empty else {}
    roll_map = (rolling_corr_scores or pd.DataFrame()).set_index("variable").to_dict("index") if rolling_corr_scores is not None and not rolling_corr_scores.empty else {}
    lift_map = (model_lift_scores or pd.DataFrame()).set_index("variable").to_dict("index") if model_lift_scores is not None and not model_lift_scores.empty else {}
    rows=[]
    for _,r in ranked.iterrows():
        v=str(r.get("variable",""))
        rf=[]
        formula_like=_looks_like_formula_variable(v)
        strong_formula=formula_like and float(r.get("score",0) or 0)>0.95
        common=bool(control_columns) and float(r.get("score",0) or 0)>=0.5
        closed=roles.get(v)=="MV" and int(r.get("lag",0) or 0)<0
        target_lead=int(r.get("lag",0) or 0)<0
        unstable_reg=False
        unstable_time=float(roll_map.get(v,{}).get("rolling_stability",1.0) or 1.0)<0.35
        lag_boundary=bool(lag_map.get(v,{}).get("lag_boundary_flag",False))
        low_lift=float(lift_map.get(v,{}).get("model_lift",0.0) or 0.0)<0.01
        poor=False
        for name,flag in [("formula_like",formula_like),("strong_formula_leakage",strong_formula),("common_capacity_driver",common),("closed_loop_suspect",closed),("target_leads_variable",target_lead),("unstable_across_regimes",unstable_reg),("unstable_over_time",unstable_time),("lag_boundary",lag_boundary),("low_model_lift",low_lift),("poor_data_quality",poor)]:
            if flag: rf.append(name)
        rows.append({"variable":v,"formula_like_flag":formula_like,"strong_formula_leakage_flag":strong_formula,"common_capacity_driver_flag":common,"closed_loop_suspect_flag":closed,"target_leads_variable_flag":target_lead,"unstable_across_regimes_flag":unstable_reg,"unstable_over_time_flag":unstable_time,"lag_boundary_flag":lag_boundary,"low_model_lift_flag":low_lift,"poor_data_quality_flag":poor,"risk_flags":";".join(rf),"risk_count":len(rf)})
    return pd.DataFrame(rows, columns=cols)


def final_ranked_features(ranked: pd.DataFrame, residual: pd.DataFrame, stability: pd.DataFrame, model_lift: pd.DataFrame, risks: pd.DataFrame, lag_peak_quality: pd.DataFrame, rolling_corr_scores: pd.DataFrame) -> pd.DataFrame:
    cols=["variable","lag","direction","raw_corr","residual_corr","regime_stability_final","rolling_stability","lag_quality","lag_boundary_flag","model_lift_score","risk_penalty","final_score","candidate_grade","recommended_use","recommended_action","risk_flags","risk_count"]
    if ranked.empty:
        return pd.DataFrame(columns=cols)
    final = ranked.rename(columns={"score": "raw_corr"}).copy()
    final = final.merge(residual[[c for c in ["variable","residual_corr"] if c in residual.columns]], on="variable", how="left")
    final = final.merge(stability[[c for c in ["variable","regime_stability_final"] if c in stability.columns]], on="variable", how="left")
    final = final.merge(model_lift[[c for c in ["variable","model_lift"] if c in model_lift.columns]], on="variable", how="left")
    final = final.merge(risks[[c for c in ["variable","risk_flags","risk_count"] if c in risks.columns]], on="variable", how="left")
    final = final.merge(lag_peak_quality[[c for c in ["variable","lag_quality","lag_boundary_flag"] if c in lag_peak_quality.columns]], on="variable", how="left")
    final = final.merge(rolling_corr_scores[[c for c in ["variable","rolling_stability"] if c in rolling_corr_scores.columns]], on="variable", how="left")
    final["raw_corr_score"] = final["raw_corr"].fillna(0).clip(0,1)
    final["residual_corr_score"] = final.get("residual_corr", pd.Series(index=final.index, dtype=float)).fillna(final["raw_corr_score"]).clip(0,1)
    final["regime_stability_final"] = final.get("regime_stability_final", pd.Series(index=final.index, dtype=float)).fillna(0.5).clip(0,1)
    final["rolling_stability"] = final.get("rolling_stability", pd.Series(index=final.index, dtype=float)).fillna(0.5).clip(0,1)
    final["lag_quality"] = final.get("lag_quality", pd.Series(index=final.index, dtype=float)).fillna(0.5).clip(0,1)
    final["model_lift_score"] = final.get("model_lift", pd.Series(index=final.index, dtype=float)).fillna(0.0).clip(0,1)
    final["risk_penalty"] = final.get("risk_count", pd.Series(index=final.index, dtype=float)).fillna(0).clip(0,5)
    final["final_score"]=(0.25*final["raw_corr_score"]+0.25*final["residual_corr_score"]+0.15*final["regime_stability_final"]+0.15*final["rolling_stability"]+0.10*final["lag_quality"]+0.10*final["model_lift_score"]-0.10*final["risk_penalty"]).clip(lower=0,upper=1)
    final["candidate_grade"] = final.apply(_grade_candidate, axis=1)
    final["recommended_use"] = final.apply(_recommend_use, axis=1)
    final["recommended_action"] = final.apply(_recommended_action, axis=1)
    for c in cols:
        if c not in final.columns:
            final[c]=np.nan
    return final.sort_values("final_score", ascending=False).reset_index(drop=True)[cols]


def _grade_candidate(row: pd.Series) -> str:
    score=float(row.get("final_score",0) or 0)
    if score>=0.75:return "A"
    if score>=0.6:return "B"
    if score>=0.45:return "C"
    if score>=0.3:return "D"
    return "E"


def _recommend_use(row: pd.Series) -> str:
    flags=str(row.get("risk_flags","") or "")
    grade=str(row.get("candidate_grade","E"))
    if "poor_data_quality" in flags:return "poor_quality_variable"
    if "closed_loop_suspect" in flags:return "closed_loop_suspect"
    if "common_capacity_driver" in flags:return "capacity_driven"
    if "formula_like" in flags:return "formula_coupled_reference"
    if "unstable_across_regimes" in flags or "unstable_over_time" in flags:return "unstable_candidate"
    if grade=="A":return "strong_screening_candidate"
    if grade=="B" and float(row.get("model_lift_score",0) or 0)>0.05:return "prediction_candidate"
    if int(row.get("lag",0) or 0)<0:return "state_indicator"
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


def _residualize(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    data = pd.concat([y, x], axis=1).dropna()
    if len(data) < 5:
        return y - y.mean()
    x_matrix = np.column_stack([np.ones(len(data)), data.iloc[:, 1:].to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(x_matrix, data.iloc[:, 0].to_numpy(), rcond=None)
    residual = pd.Series(index=data.index, data=data.iloc[:, 0].to_numpy() - x_matrix @ coef)
    return residual.reindex(y.index)


def _looks_like_formula_variable(name: str) -> bool:
    lower = name.lower()
    return any(t in lower for t in ["单耗", "消耗", "比值", "ratio", "rate", "%", "百分比", "折算", "累计", "平均", "total", "consumption", "specific"])
