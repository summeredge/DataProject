from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd


AUTO_CLOSED_LOOP_DIAGNOSIS_COLUMNS = [
    "mv_variable",
    "cv_variable",
    "diagnosis_status",
    "confidence_level",
    "evidence_items",
    "lag_information",
    "stability_information",
    "prediction_information",
    "created_time",
]


def build_auto_closed_loop_diagnosis(
    ranked_features: pd.DataFrame | None,
    risk_flags: pd.DataFrame | None,
    lag_peak_quality: pd.DataFrame | None,
    rolling_corr_scores: pd.DataFrame | None,
    model_lift_scores: pd.DataFrame | None,
    cv_variable: str,
    created_time: str | None = None,
) -> pd.DataFrame:
    """Build shadow-only MV-to-CV closed-loop diagnostics from persisted evidence."""
    ranked = ranked_features if ranked_features is not None else pd.DataFrame()
    if ranked.empty or "variable" not in ranked.columns:
        return pd.DataFrame(columns=AUTO_CLOSED_LOOP_DIAGNOSIS_COLUMNS)
    timestamp = created_time or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    risk = _indexed(risk_flags)
    lag = _indexed(lag_peak_quality)
    stability = _indexed(rolling_corr_scores)
    prediction = _indexed(model_lift_scores)
    rows: list[dict[str, object]] = []
    for _, candidate in ranked.drop_duplicates("variable").iterrows():
        variable = str(candidate["variable"])
        risk_row = risk.get(variable, {})
        lag_row = lag.get(variable, {})
        stability_row = stability.get(variable, {})
        prediction_row = prediction.get(variable, {})
        closed_loop_flag = _bool(risk_row.get("closed_loop_suspect_flag", False))
        best_lag = _number(lag_row.get("best_lag", candidate.get("lag")))
        lag_quality = _number(lag_row.get("lag_quality"))
        rolling_stability = _number(stability_row.get("rolling_stability", candidate.get("rolling_stability")))
        model_lift = _number(prediction_row.get("model_lift_score", candidate.get("model_lift_score")))
        stable = rolling_stability is not None and rolling_stability >= 0.6
        predictive = model_lift is not None and model_lift > 0.05
        lag_supported = best_lag is not None and lag_quality is not None and lag_quality >= 0.5
        if closed_loop_flag and stable and predictive:
            status, confidence = "confirmed", "high"
        elif closed_loop_flag or (best_lag == 0 and stable and predictive):
            status, confidence = "possible", "medium" if (stable or predictive) else "low"
        else:
            status, confidence = "not_supported", "low"
        evidence = []
        if closed_loop_flag:
            evidence.append("existing_closed_loop_indicator")
        if lag_supported:
            evidence.append("lag_quality_supported")
        if stable:
            evidence.append("rolling_stability_supported")
        if predictive:
            evidence.append("prediction_lift_supported")
        rows.append({
            "mv_variable": variable,
            "cv_variable": cv_variable,
            "diagnosis_status": status,
            "confidence_level": confidence,
            "evidence_items": json.dumps(evidence, ensure_ascii=False),
            "lag_information": json.dumps({"best_lag": best_lag, "lag_quality": lag_quality}, ensure_ascii=False),
            "stability_information": json.dumps({"rolling_stability": rolling_stability}, ensure_ascii=False),
            "prediction_information": json.dumps({"model_lift_score": model_lift}, ensure_ascii=False),
            "created_time": timestamp,
        })
    return pd.DataFrame(rows, columns=AUTO_CLOSED_LOOP_DIAGNOSIS_COLUMNS)


def _indexed(frame: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if frame is None or frame.empty or "variable" not in frame.columns:
        return {}
    return {str(row["variable"]): row.to_dict() for _, row in frame.drop_duplicates("variable").iterrows()}


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def _bool(value: object) -> bool:
    return bool(value) if pd.notna(value) else False
