from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


CLOSED_LOOP_RANKING_FUSION_COLUMNS = [
    "variable",
    "manual_status",
    "auto_probability",
    "automatic_closed_loop_risk",
    "final_closed_loop_status",
    "factor_adjustment",
    "closed_loop_ranking_reason",
    "timestamp",
]


def build_closed_loop_risk_context(
    closed_loop_evidence: pd.DataFrame | None,
    calibration_results: pd.DataFrame | None,
    *,
    medium_threshold: float = 0.3,
    high_threshold: float = 0.7,
    medium_factor: float = 0.8,
    high_factor: float = 0.55,
    timestamp: str | None = None,
) -> pd.DataFrame:
    """Build the auditable, ranking-only closed-loop context."""
    evidence = _indexed(closed_loop_evidence, "variable")
    calibration = _indexed(calibration_results, "variable")
    variables = list(dict.fromkeys([*evidence, *calibration]))
    created = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict[str, object]] = []
    for variable in variables:
        manual_status = str(evidence.get(variable, {}).get("manual_closed_loop_status", "unknown"))
        probability = _probability(calibration.get(variable, {}).get("auto_closed_loop_probability"))
        if manual_status == "confirmed_closed_loop":
            final_status, adjustment, reason = "manual_confirmed_closed_loop", high_factor, "manual_confirmed_closed_loop_limit"
        elif manual_status == "confirmed_not_closed_loop":
            final_status, adjustment, reason = "manual_confirmed_not_closed_loop", 1.0, "manual_non_closed_loop_override"
        elif probability >= high_threshold:
            final_status, adjustment, reason = "automatic_high_risk", high_factor, "automatic_closed_loop_high_risk_penalty"
        elif probability >= medium_threshold:
            final_status, adjustment, reason = "automatic_medium_risk", medium_factor, "automatic_closed_loop_medium_risk_penalty"
        else:
            final_status, adjustment, reason = "unknown", 1.0, "no_closed_loop_adjustment"
        rows.append({
            "variable": variable,
            "manual_status": manual_status,
            "auto_probability": probability,
            "automatic_closed_loop_risk": probability,
            "final_closed_loop_status": final_status,
            "factor_adjustment": adjustment,
            "closed_loop_ranking_reason": reason,
            "timestamp": created,
        })
    return pd.DataFrame(rows, columns=CLOSED_LOOP_RANKING_FUSION_COLUMNS)


def apply_closed_loop_risk_context(final: pd.DataFrame, context: pd.DataFrame | None) -> pd.DataFrame:
    """Apply manual-first factor limits without altering statistical evidence."""
    final = final.copy()
    prior_reason = final.get("closed_loop_ranking_reason", pd.Series("no_closed_loop_adjustment", index=final.index))
    final["closed_loop_ranking_reason"] = "no_closed_loop_adjustment"
    if context is None or context.empty or "variable" not in context.columns:
        return final
    columns = [column for column in ["variable", "manual_status", "factor_adjustment", "closed_loop_ranking_reason"] if column in context.columns]
    final = final.merge(context[columns].drop_duplicates("variable"), on="variable", how="left", suffixes=("", "_context"))
    reason = final["closed_loop_ranking_reason_context"].fillna("no_closed_loop_adjustment")
    manual = final.get("manual_status", pd.Series("unknown", index=final.index)).fillna("unknown")
    adjustment = pd.to_numeric(final.get("factor_adjustment", pd.Series(1.0, index=final.index)), errors="coerce").fillna(1.0)
    prior_adjustment = prior_reason.map({
        "automatic_closed_loop_high_risk_penalty": 0.55,
        "automatic_closed_loop_medium_risk_penalty": 0.8,
    }).fillna(1.0)
    final["driver_priority_factor"] = (final["driver_priority_factor"] / prior_adjustment).clip(0, 1)
    confirmed = manual.eq("confirmed_closed_loop")
    automatic = manual.eq("unknown") & adjustment.lt(1.0)
    final.loc[confirmed, "driver_priority_factor"] = final.loc[confirmed, "driver_priority_factor"].clip(upper=0.55)
    final.loc[automatic, "driver_priority_factor"] = (
        final.loc[automatic, "driver_priority_factor"] * adjustment.loc[automatic]
    ).clip(0, 1)
    final["closed_loop_ranking_reason"] = reason
    return final.drop(columns=["manual_status", "factor_adjustment", "closed_loop_ranking_reason_context"], errors="ignore")


def _indexed(frame: pd.DataFrame | None, key: str) -> dict[str, dict[str, object]]:
    if frame is None or frame.empty or key not in frame.columns:
        return {}
    return {str(row[key]): row.to_dict() for _, row in frame.drop_duplicates(key).iterrows()}


def _probability(value: object) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, probability)) if pd.notna(probability) else 0.0
