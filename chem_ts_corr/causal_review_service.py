from __future__ import annotations

import pandas as pd


INTERPRETATION = "predictive validation only; not a causal conclusion"

REPORT_COLUMNS = [
    "variable",
    "candidate_grade",
    "final_score",
    "review_priority",
    "review_tier",
    "review_reason",
    "conditional_granger_status",
    "conditional_best_lag",
    "conditional_min_p_value",
    "conditional_fdr_q_value",
    "predictive_contribution",
    "risk_level",
    "risk_flags",
    "recommended_use",
    "recommended_action",
    "final_review_decision",
    "final_review_reason",
    "interpretation",
]


def build_causal_review_report(
    ranked_features: pd.DataFrame,
    causal_review_candidates: pd.DataFrame,
    conditional_granger_scores: pd.DataFrame,
    risk_flags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a lightweight v0.4 causal-review summary.

    The report is only a predictive-validation/manual-review summary and does
    not claim final causality.
    """
    if causal_review_candidates.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    report = causal_review_candidates.copy(deep=True)
    _ensure_column(report, "variable")

    ranked_cols = ["variable", "candidate_grade", "final_score", "recommended_use", "recommended_action"]
    report = _left_join_missing(report, ranked_features, ranked_cols)

    conditional_cols = {
        "status": "conditional_granger_status",
        "best_lag": "conditional_best_lag",
        "min_p_value": "conditional_min_p_value",
        "fdr_q_value": "conditional_fdr_q_value",
        "predictive_contribution": "predictive_contribution",
    }
    report = _left_join_missing(report, conditional_granger_scores, ["variable", *conditional_cols], conditional_cols)

    if risk_flags is not None:
        report = _left_join_missing(report, risk_flags, ["variable", "risk_level", "risk_flags"])

    for col in REPORT_COLUMNS:
        if col not in report.columns:
            report[col] = pd.NA

    decisions = report.apply(_decide_review, axis=1, result_type="expand")
    decisions.columns = ["final_review_decision", "final_review_reason"]
    report["final_review_decision"] = decisions["final_review_decision"]
    report["final_review_reason"] = decisions["final_review_reason"]
    report["interpretation"] = INTERPRETATION

    return report[REPORT_COLUMNS].copy()


def _left_join_missing(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str],
    rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    if right.empty or "variable" not in right.columns:
        return left

    rename = rename or {}
    available = [c for c in columns if c in right.columns]
    if "variable" not in available:
        return left

    side = right[available].copy(deep=True).rename(columns=rename)
    value_cols = [c for c in side.columns if c != "variable"]
    if not value_cols:
        return left

    merged = left.merge(side, on="variable", how="left", suffixes=("", "__joined"))
    for col in value_cols:
        joined_col = f"{col}__joined"
        if joined_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[joined_col])
            merged = merged.drop(columns=[joined_col])
    return merged


def _ensure_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        frame[column] = pd.NA


def _decide_review(row: pd.Series) -> tuple[str, str]:
    status = _text(row.get("conditional_granger_status"))
    risk_level = _text(row.get("risk_level")).lower()
    q_value = _number(row.get("conditional_fdr_q_value"))
    contribution = _number(row.get("predictive_contribution"))

    if status == "high_collinearity_risk" and contribution is not None and contribution > 0:
        return (
            "manual_review_only",
            "存在正向预测信号，但条件 Granger 预测验证受高共线性限制，仅建议人工复核，不是因果结论。",
        )
    if _has_risk_limited_signal(row) and not status.startswith("ok"):
        return (
            "risk_limited_review",
            "存在共同负荷驱动或闭环/稳定性风险，预测验证受统计限制，仅限风险提示型人工复核，不是因果结论。",
        )
    if not status.startswith("ok"):
        return (
            "insufficient_evidence",
            "条件 Granger 预测验证未得到可用结果，仅建议人工复核时作为缺失证据处理，不是因果结论。",
        )
    if risk_level in {"high", "strong"}:
        return (
            "manual_review_only",
            "风险等级较高，仅可进入人工复核和工艺解释，不是因果结论。",
        )
    if _has_risk_limited_signal(row):
        return (
            "risk_limited_review",
            "存在共同负荷驱动或稳定性风险，即使预测验证显著也仅限风险提示型人工复核，不是因果结论。",
        )
    if q_value is not None and contribution is not None and q_value <= 0.05 and contribution >= 0.05:
        return (
            "priority_review",
            "条件 Granger 预测验证信号较强，建议优先人工复核，但不是因果结论。",
        )
    if q_value is not None and contribution is not None and q_value <= 0.1 and contribution >= 0.02:
        return (
            "secondary_review",
            "条件 Granger 预测验证信号中等，可作为二级人工复核线索，不是因果结论。",
        )
    return (
        "not_recommended",
        "预测验证证据不足，暂不建议进入优先复核队列，不是因果结论。",
    )


def _has_risk_limited_signal(row: pd.Series) -> bool:
    risk_flags = _text(row.get("risk_flags")).lower()
    recommended_use = _text(row.get("recommended_use")).lower()
    risk_limited_flags = {
        "common_capacity_driver",
        "unstable_candidate",
        "unstable_across_regimes",
        "unstable_over_time",
    }
    risk_limited_uses = {"capacity_driven", "unstable_candidate"}
    return any(flag in risk_flags for flag in risk_limited_flags) or recommended_use in risk_limited_uses


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    if pd.isna(value):
        return ""
    return str(value)


def _number(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)
