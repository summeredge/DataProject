from __future__ import annotations

import json

import pandas as pd


CLOSED_LOOP_EVIDENCE_COLUMNS = [
    "variable",
    "manual_closed_loop_status",
    "auto_closed_loop_status",
    "closed_loop_evidence_level",
    "closed_loop_evidence_source",
    "closed_loop_conflict",
    "closed_loop_reason",
]


def build_closed_loop_evidence(
    risk_flags: pd.DataFrame | None,
    manual_closed_loop_variables: list[str] | None = None,
    manual_non_closed_loop_variables: list[str] | None = None,
) -> pd.DataFrame:
    closed = set(manual_closed_loop_variables or [])
    non_closed = set(manual_non_closed_loop_variables or [])
    risk_by_variable = {
        str(risk["variable"]): risk
        for _, risk in (risk_flags if risk_flags is not None else pd.DataFrame()).iterrows()
    }
    variables = list(risk_by_variable)
    for variable in [*(manual_closed_loop_variables or []), *(manual_non_closed_loop_variables or [])]:
        if variable not in risk_by_variable and variable not in variables:
            variables.append(variable)
    rows: list[dict[str, object]] = []
    for variable in variables:
        risk = risk_by_variable.get(variable)
        manual = "confirmed_closed_loop" if variable in closed else "confirmed_not_closed_loop" if variable in non_closed else "unknown"
        automatic = "suspected_closed_loop" if risk is not None and bool(risk.get("closed_loop_suspect_flag", False)) else "unknown"
        conflict = manual == "confirmed_not_closed_loop" and automatic == "suspected_closed_loop"
        if manual == "confirmed_closed_loop":
            level, source = "confirmed", "manual_and_automatic" if automatic != "unknown" else "manual"
        elif conflict:
            level, source = "conflict", "manual_and_automatic"
        elif manual == "confirmed_not_closed_loop":
            level, source = "rejected", "manual"
        elif automatic == "suspected_closed_loop":
            level, source = "suspected", "automatic"
        else:
            level, source = "unknown", "none"
        reasons = []
        if manual == "confirmed_closed_loop":
            reasons.append("人工确认闭环控制变量")
        elif manual == "confirmed_not_closed_loop":
            reasons.append("人工确认非闭环")
        if automatic == "suspected_closed_loop":
            reasons.append("自动检测存在闭环嫌疑")
        elif risk is None:
            reasons.append("未获得自动闭环判断结果")
        rows.append({
            "variable": variable,
            "manual_closed_loop_status": manual,
            "auto_closed_loop_status": automatic,
            "closed_loop_evidence_level": level,
            "closed_loop_evidence_source": source,
            "closed_loop_conflict": conflict,
            "closed_loop_reason": json.dumps(reasons, ensure_ascii=False),
        })
    return pd.DataFrame(rows, columns=CLOSED_LOOP_EVIDENCE_COLUMNS)
