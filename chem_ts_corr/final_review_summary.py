from __future__ import annotations

import math
from typing import Any

import pandas as pd

from chem_ts_corr.common import to_float


INTERPRETATION = "final review summary only; not a causal conclusion"

SUMMARY_COLUMNS = [
    "final_rank",
    "variable",
    "final_recommendation",
    "data_priority",
    "evidence_level",
    "evidence_score",
    "statistical_limit_level",
    "risk_constraint_level",
    "key_reason",
    "suggested_next_action",
    "screening_grade",
    "screening_score",
    "screening_lag",
    "layer1_association_status",
    "layer2_temporal_status",
    "layer3_independence_status",
    "layer4_model_status",
    "stability_status",
    "data_quality_status",
    "evidence_support_items",
    "evidence_against_items",
    "evidence_missing_items",
    "evidence_conflict_items",
    "candidate_summary",
    "risk_flags",
    "conditional_status",
    "conditional_best_lag",
    "tested_lags",
    "lag_boundary_hint",
    "evidence_conflict_type",
    "evidence_conflict_reason",
    "interpretation",
]

_DECISION_ORDER = {
    "priority_review": 1,
    "priority_review_with_statistical_limit": 2,
    "secondary_review": 3,
    "secondary_review_with_statistical_limit": 4,
    "risk_limited_review": 5,
    "manual_review_only": 6,
    "insufficient_evidence": 7,
    "not_recommended": 8,
}
_DATA_PRIORITY_ORDER = {"high": 1, "medium": 2, "low": 3}

_KEY_REASONS = {
    "priority_review": "多证据支持，建议优先复核。",
    "priority_review_with_statistical_limit": "数据证据强，但统计检验受限，建议优先复核。",
    "secondary_review": "存在预测线索，建议二级复核。",
    "secondary_review_with_statistical_limit": "存在预测线索，但受统计限制，建议二级复核。",
    "risk_limited_review": "存在风险约束，仅限人工复核参考。",
    "manual_review_only": "证据不完整或风险较高，仅建议人工查看。",
    "insufficient_evidence": "当前证据不足，可增加数据后复核。",
    "not_recommended": "当前证据不足，不建议优先复核。",
}

_NEXT_ACTIONS = {
    "priority_review": "优先核查工艺机理、操作方向、滞后时间和上下游关系。",
    "priority_review_with_statistical_limit": "优先核查，但需重点检查共线性、共同负荷或滞后边界导致的统计限制。",
    "secondary_review": "作为二级候选，结合工艺流程和趋势图进一步确认。",
    "secondary_review_with_statistical_limit": "作为二级候选，优先检查统计限制来源，再判断是否保留。",
    "risk_limited_review": "仅作风险受限复核，需先排查共同负荷、稳定性或数据质量问题。",
    "manual_review_only": "仅人工查看，不建议直接进入优先候选。",
    "insufficient_evidence": "证据不足，可增加数据量或调整滞后参数后复核。",
    "not_recommended": "当前不建议优先复核。",
}

_ROLE_HINTS = {
    "MV": "疑似可操作变量，若数据证据强应优先复核。",
    "CAPACITY": "可能代表负荷或共同驱动，需要区分直接作用和共同变化。",
    "PV": "过程状态变量，需结合上下游关系判断。",
    "DV": "外部扰动变量，适合做扰动候选复核。",
}


def build_final_review_summary(
    causal_review_evidence: pd.DataFrame,
    conditional_granger_scores: pd.DataFrame | None = None,
    ranked_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build an engineer-facing manual review priority list from evidence rows."""
    evidence = causal_review_evidence.copy(deep=True) if causal_review_evidence is not None else pd.DataFrame()
    conditional = conditional_granger_scores.copy(deep=True) if conditional_granger_scores is not None else pd.DataFrame()
    ranked = ranked_features.copy(deep=True) if ranked_features is not None else pd.DataFrame()
    if evidence.empty or "variable" not in evidence.columns:
        return pd.DataFrame(columns=_columns_with_role(ranked))

    conditional_by_var = _index_by_variable(conditional)
    ranked_by_var = _index_by_variable(ranked)
    rows: list[dict[str, Any]] = []
    include_role = _has_role(ranked)

    for _, source in evidence.iterrows():
        variable = _text(source.get("variable"))
        cond = conditional_by_var.get(variable, {})
        rank_row = ranked_by_var.get(variable, {})
        decision = _text(source.get("integrated_review_decision")) or "not_recommended"
        tested_lags = _coalesce(cond.get("tested_lags"), source.get("tested_lags"))
        conflict_type, conflict_reason = _conflicts(source)
        row = {
            "variable": variable,
            "final_recommendation": decision,
            "data_priority": _text(source.get("data_priority")),
            "evidence_level": _text(source.get("evidence_level")),
            "evidence_score": source.get("evidence_score"),
            "statistical_limit_level": _text(source.get("statistical_limit_level")),
            "risk_constraint_level": _text(source.get("risk_constraint_level")),
            "key_reason": _key_reason(decision, source),
            "suggested_next_action": _NEXT_ACTIONS.get(decision, "当前不建议优先复核。"),
            "screening_grade": _coalesce(source.get("candidate_grade"), rank_row.get("candidate_grade")),
            "screening_score": _coalesce(source.get("final_score"), rank_row.get("final_score")),
            "screening_lag": _coalesce(source.get("lag"), rank_row.get("lag"), rank_row.get("best_lag")),
            "layer1_association_status": rank_row.get("layer1_association_status", ""),
            "layer2_temporal_status": rank_row.get("layer2_temporal_status", ""),
            "layer3_independence_status": rank_row.get("layer3_independence_status", ""),
            "layer4_model_status": rank_row.get("layer4_model_status", ""),
            "stability_status": rank_row.get("stability_status", ""),
            "data_quality_status": rank_row.get("data_quality_status", ""),
            "evidence_support_items": rank_row.get("evidence_support_items", ""),
            "evidence_against_items": rank_row.get("evidence_against_items", ""),
            "evidence_missing_items": rank_row.get("evidence_missing_items", ""),
            "evidence_conflict_items": rank_row.get("evidence_conflict_items", ""),
            "candidate_summary": rank_row.get("candidate_summary", ""),
            "risk_flags": rank_row.get("risk_flags", ""),
            "conditional_status": _coalesce(source.get("conditional_granger_status"), cond.get("status")),
            "conditional_best_lag": _coalesce(source.get("conditional_best_lag"), cond.get("best_lag")),
            "tested_lags": tested_lags,
            "lag_boundary_hint": _lag_boundary_hint(source, cond, tested_lags),
            "evidence_conflict_type": conflict_type,
            "evidence_conflict_reason": conflict_reason,
            "interpretation": INTERPRETATION,
        }
        if include_role:
            role = _role_for(rank_row)
            row["variable_role"] = role
            row["role_based_hint"] = _ROLE_HINTS.get(role, "")
        rows.append(row)

    out = pd.DataFrame(rows)
    out["_decision_order"] = out["final_recommendation"].map(_DECISION_ORDER).fillna(99)
    out["_priority_order"] = out["data_priority"].map(_DATA_PRIORITY_ORDER).fillna(99)
    out["_evidence_sort"] = pd.to_numeric(out["evidence_score"], errors="coerce").fillna(-math.inf)
    out["_screening_sort"] = pd.to_numeric(out["screening_score"], errors="coerce").fillna(-math.inf)
    out = out.sort_values(
        ["_decision_order", "_priority_order", "_evidence_sort", "_screening_sort"],
        ascending=[True, True, False, False],
        kind="mergesort",
    ).drop(columns=["_decision_order", "_priority_order", "_evidence_sort", "_screening_sort"])
    out.insert(0, "final_rank", range(1, len(out) + 1))
    return out[_columns_with_role(ranked) if include_role else SUMMARY_COLUMNS]


def _columns_with_role(ranked: pd.DataFrame) -> list[str]:
    return [*SUMMARY_COLUMNS, "variable_role", "role_based_hint"] if _has_role(ranked) else SUMMARY_COLUMNS


def _has_role(frame: pd.DataFrame) -> bool:
    return any(col in frame.columns for col in ["variable_role", "role"])


def _role_for(row: dict[str, Any]) -> str:
    return _text(_coalesce(row.get("variable_role"), row.get("role"))).upper()


def _index_by_variable(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "variable" not in frame.columns:
        return {}
    return {_text(row.get("variable")): row.to_dict() for _, row in frame.iterrows()}


def _key_reason(decision: str, row: pd.Series) -> str:
    base = _KEY_REASONS.get(decision, "当前证据不足，不建议优先复核。")
    details = [_text(row.get(col)) for col in ["integrated_review_reason", "evidence_reason", "statistical_limit_reason"]]
    details = [d for d in details if d]
    return base if not details else f"{base} 依据：{'; '.join(details[:2])}"


def _lag_boundary_hint(row: pd.Series, cond: dict[str, Any], tested_lags: Any) -> str:
    haystack = " ".join(
        _text(value).lower()
        for value in [
            row.get("risk_flags"),
            row.get("statistical_limit_reason"),
            row.get("evidence_reason"),
            row.get("integrated_review_reason"),
            row.get("conditional_granger_status"),
            cond.get("status"),
            tested_lags,
            row.get("lag"),
            cond.get("screening_lag"),
        ]
    )
    if "ranked lag outside maxlag" in haystack:
        return "主筛查滞后超出当前 maxlag，建议扩大 maxlag 或按工艺停留时间设定复核窗口。"
    if any(token in haystack for token in ["lag_boundary", "lag_boundary_risk", "滞后边界"]):
        return "命中滞后边界，建议扩大 max_lag 或结合工艺停留时间确认。"
    return ""


def _conflicts(row: pd.Series) -> tuple[str, str]:
    types: list[str] = []
    reasons: list[str] = []
    grade = _text(row.get("candidate_grade")).upper()
    cond = _text(row.get("conditional_granger_status")).lower()
    q = _number(row.get("conditional_fdr_q_value"))
    stat = _text(row.get("statistical_limit_level")).lower()
    flags = _text(row.get("risk_flags")).lower()
    if _text(row.get("data_priority")).lower() == "high" and stat in {"medium", "strong"}:
        types.append("strong_screening_but_statistical_limited")
        reasons.append("主筛查或模型证据较强，但统计检验受共线性或共同负荷限制。")
    if grade in {"A", "B"} and not (cond.startswith("ok") and q is not None and q <= 0.05):
        types.append("strong_screening_but_conditional_weak")
        reasons.append("主筛查较强，但条件 Granger 独立预测支持不足。")
    if cond.startswith("ok") and q is not None and q <= 0.05 and grade in {"D", "E"}:
        types.append("conditional_supported_but_screening_weak")
        reasons.append("条件 Granger 支持，但主筛查等级较低，需检查是否为局部、非线性或边界信号。")
    rank = _number(row.get("model_importance_rank"))
    if rank is not None and rank <= 5 and (not (cond.startswith("ok") and q is not None and q <= 0.05) or "high_collinearity_risk" in flags):
        types.append("model_supported_but_granger_weak")
        reasons.append("模型解释支持较强，但 Granger 验证不足或受限，可能是非线性或共线性关系。")
    text = " ".join(_text(row.get(col)).lower() for col in ["risk_flags", "statistical_limit_reason", "evidence_reason", "integrated_review_reason"])
    if any(token in text for token in ["lag_boundary", "lag_boundary_risk", "滞后边界"]):
        types.append("boundary_lag_uncertain")
        reasons.append("滞后命中边界，真实滞后可能超过当前搜索范围。")
    return ";".join(types), ";".join(reasons)


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value) != "nan":
            return value
    return ""


def _text(value: Any) -> str:
    value = _coalesce(value)
    return "" if value is None else str(value)


def _number(value: Any) -> float | None:
    numeric = to_float(value, default=float("nan"))
    return None if pd.isna(numeric) else numeric
