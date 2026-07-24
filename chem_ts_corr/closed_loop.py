from __future__ import annotations

import json

import pandas as pd


CLOSED_LOOP_EVIDENCE_COLUMNS = [
    "variable",
    "manual_closed_loop_status",
    "automatic_closed_loop_indicator",
    "closed_loop_context",
    "closed_loop_status",
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
        manual = "engineering_input_closed_loop" if variable in closed else "engineering_input_not_closed_loop" if variable in non_closed else "not_provided"
        automatic = "possible" if risk is not None and bool(risk.get("closed_loop_suspect_flag", False)) else "not_indicated"
        if manual != "not_provided" and automatic == "possible":
            context = "manual_engineering_input_and_automatic_indicator"
        elif manual != "not_provided":
            context = "manual_engineering_input"
        elif automatic == "possible":
            context = "automatic_indicator"
        else:
            context = "no_closed_loop_context"
        status = "possible_closed_loop_influence" if automatic == "possible" else "manual_context_requires_review" if manual != "not_provided" else "no_closed_loop_indicator"
        reasons = []
        if manual == "engineering_input_closed_loop":
            reasons.append("人工工程经验输入：可能与闭环控制相关，需人工复核")
        elif manual == "engineering_input_not_closed_loop":
            reasons.append("人工工程经验输入：未标记为闭环控制相关")
        if automatic == "possible":
            reasons.append("自动诊断指标提示可能存在闭环影响")
        elif risk is None:
            reasons.append("未获得自动闭环判断结果")
        rows.append({
            "variable": variable,
            "manual_closed_loop_status": manual,
            "automatic_closed_loop_indicator": automatic,
            "closed_loop_context": context,
            "closed_loop_status": status,
            "closed_loop_reason": json.dumps(reasons, ensure_ascii=False),
        })
    return pd.DataFrame(rows, columns=CLOSED_LOOP_EVIDENCE_COLUMNS)
