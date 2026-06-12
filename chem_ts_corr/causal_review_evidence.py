from __future__ import annotations

import pandas as pd


INTERPRETATION = "integrated review evidence only; not a causal conclusion"

EVIDENCE_COLUMNS = [
    "variable",
    "candidate_grade",
    "final_score",
    "lag",
    "direction",
    "risk_level",
    "risk_flags",
    "risk_count",
    "recommended_use",
    "recommended_action",
    "conditional_granger_status",
    "conditional_best_lag",
    "conditional_min_p_value",
    "conditional_fdr_q_value",
    "predictive_contribution",
    "condition_number",
    "granger_min_p_value",
    "granger_fdr_q_value",
    "granger_status",
    "model_lift",
    "rolling_stability",
    "rolling_sign_consistency",
    "enhanced_validation_status",
    "model_importance_rank",
    "max_importance",
    "best_model_lag",
    "model_explanation_support",
    "evidence_score",
    "evidence_level",
    "evidence_reason",
    "risk_constraint_level",
    "integrated_review_decision",
    "integrated_review_reason",
    "interpretation",
]


RISK_ORDER = {"none": 0, "weak": 1, "medium": 2, "strong": 3}


def build_causal_review_evidence(
    ranked_features: pd.DataFrame,
    conditional_granger_scores: pd.DataFrame,
    risk_flags: pd.DataFrame | None = None,
    enhanced_validation_summary: pd.DataFrame | None = None,
    granger_tests: pd.DataFrame | None = None,
    model_variable_importance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a conservative multi-evidence table for manual review.

    The output integrates predictive-validation signals only and does not change
    screening scores, model calculations, or the existing three-layer review
    decision.
    """
    if ranked_features.empty or "variable" not in ranked_features.columns:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)

    base_cols = [
        "variable",
        "candidate_grade",
        "final_score",
        "lag",
        "direction",
        "risk_level",
        "risk_flags",
        "risk_count",
        "recommended_use",
        "recommended_action",
    ]
    evidence = ranked_features.copy(deep=True)
    evidence = evidence[[c for c in base_cols if c in evidence.columns]].copy(deep=True)
    _ensure_columns(evidence, base_cols)

    evidence = _left_join_missing(
        evidence,
        conditional_granger_scores,
        {
            "status": "conditional_granger_status",
            "best_lag": "conditional_best_lag",
            "min_p_value": "conditional_min_p_value",
            "fdr_q_value": "conditional_fdr_q_value",
            "predictive_contribution": "predictive_contribution",
            "condition_number": "condition_number",
        },
    )
    evidence = _left_join_missing(
        evidence,
        risk_flags,
        {
            "risk_level": "risk_level",
            "risk_flags": "risk_flags",
            "risk_count": "risk_count",
            "recommended_use": "recommended_use",
            "recommended_action": "recommended_action",
        },
    )
    evidence = _left_join_missing(
        evidence,
        enhanced_validation_summary,
        {
            "model_lift": "model_lift",
            "rolling_stability": "rolling_stability",
            "rolling_sign_consistency": "rolling_sign_consistency",
            "status": "enhanced_validation_status",
        },
    )
    evidence = _left_join_missing(
        evidence,
        granger_tests,
        {
            "min_p_value": "granger_min_p_value",
            "fdr_q_value": "granger_fdr_q_value",
            "status": "granger_status",
        },
    )
    evidence = _left_join_missing(
        evidence,
        model_variable_importance,
        {
            "importance_rank": "model_importance_rank",
            "max_importance": "max_importance",
            "best_model_lag": "best_model_lag",
        },
    )

    for col in EVIDENCE_COLUMNS:
        if col not in evidence.columns:
            evidence[col] = pd.NA

    assessed = evidence.apply(_assess_row, axis=1, result_type="expand")
    assessed.columns = [
        "evidence_score",
        "evidence_level",
        "evidence_reason",
        "risk_constraint_level",
        "integrated_review_decision",
        "integrated_review_reason",
        "model_explanation_support",
    ]
    for col in assessed.columns:
        evidence[col] = assessed[col]
    evidence["interpretation"] = INTERPRETATION
    return evidence[EVIDENCE_COLUMNS].copy(deep=True)


def _left_join_missing(left: pd.DataFrame, right: pd.DataFrame | None, rename: dict[str, str]) -> pd.DataFrame:
    if right is None or right.empty or "variable" not in right.columns:
        return left
    source_cols = ["variable", *[c for c in rename if c in right.columns]]
    if len(source_cols) <= 1:
        return left
    side = right.copy(deep=True)[source_cols].rename(columns=rename)
    side = _dedupe_variables(side)
    value_cols = [c for c in side.columns if c != "variable"]
    merged = left.merge(side, on="variable", how="left", suffixes=("", "__joined"))
    for col in value_cols:
        joined_col = f"{col}__joined"
        if joined_col in merged.columns:
            existing = merged[col] if col in merged.columns else pd.Series(pd.NA, index=merged.index)
            missing = existing.isna() | existing.astype(str).str.strip().eq("")
            merged[col] = existing.where(~missing, merged[joined_col])
            merged = merged.drop(columns=[joined_col])
    return merged


def _dedupe_variables(frame: pd.DataFrame) -> pd.DataFrame:
    if "variable" not in frame.columns or frame.empty:
        return frame
    sortable = frame.copy(deep=True)
    sort_cols: list[str] = []
    ascending: list[bool] = []
    for col in ["fdr_q_value", "granger_fdr_q_value", "min_p_value", "granger_min_p_value"]:
        if col in sortable.columns:
            sortable[f"__sort_{col}"] = pd.to_numeric(sortable[col], errors="coerce")
            sort_cols.append(f"__sort_{col}")
            ascending.append(True)
    for col in ["importance_rank", "model_importance_rank"]:
        if col in sortable.columns:
            sortable[f"__sort_{col}"] = pd.to_numeric(sortable[col], errors="coerce")
            sort_cols.append(f"__sort_{col}")
            ascending.append(True)
    if sort_cols:
        sortable = sortable.sort_values(sort_cols, ascending=ascending, na_position="last", kind="mergesort")
    sortable = sortable.drop(columns=[c for c in sortable.columns if c.startswith("__sort_")])
    return sortable.drop_duplicates(subset=["variable"], keep="first")


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col not in frame.columns:
            frame[col] = pd.NA


def _assess_row(row: pd.Series) -> tuple[float, str, str, str, str, str, str]:
    score = 0.0
    reasons: list[str] = []

    grade = _text(row.get("candidate_grade")).upper()
    grade_scores = {"A": 2.0, "B": 1.5, "C": 1.0, "D": 0.5}
    if grade in grade_scores:
        score += grade_scores[grade]
        reasons.append(f"candidate_grade_{grade}")

    conditional_status = _text(row.get("conditional_granger_status")).lower()
    conditional_q = _number(row.get("conditional_fdr_q_value"))
    contribution = _number(row.get("predictive_contribution"))
    if conditional_status.startswith("ok") and conditional_q is not None:
        if conditional_q <= 0.05:
            score += 2.0
            reasons.append("conditional_granger_supported")
        elif conditional_q <= 0.10:
            score += 1.0
            reasons.append("conditional_granger_weak_support")
    if contribution is not None and contribution > 0:
        if conditional_status == "high_collinearity_risk":
            score += 0.3
            reasons.append("high_collinearity_limited_signal")
        else:
            score += 0.5
            reasons.append("predictive_contribution_positive")

    granger_q = _number(row.get("granger_fdr_q_value"))
    if granger_q is not None:
        if granger_q <= 0.05:
            score += 0.75
            reasons.append("granger_auxiliary_support")
        elif granger_q <= 0.10:
            score += 0.4
            reasons.append("granger_auxiliary_weak_support")

    model_lift = _number(row.get("model_lift"))
    if model_lift is not None:
        if model_lift >= 0.05:
            score += 0.75
            reasons.append("model_lift_supported")
        elif model_lift >= 0.01:
            score += 0.4
            reasons.append("model_lift_weak_support")

    rolling_stability = _number(row.get("rolling_stability"))
    if rolling_stability is not None:
        if rolling_stability >= 0.7:
            score += 0.75
            reasons.append("rolling_stability_supported")
        elif rolling_stability >= 0.5:
            score += 0.4
            reasons.append("rolling_stability_weak_support")

    sign_consistency = _number(row.get("rolling_sign_consistency"))
    if sign_consistency is not None and sign_consistency >= 0.8:
        score += 0.3
        reasons.append("rolling_sign_consistency_supported")

    model_support = ""
    importance_rank = _number(row.get("model_importance_rank"))
    if importance_rank is not None:
        if importance_rank <= 5:
            score += 0.75
            model_support = "model_explanation_support"
            reasons.append("model_explanation_support")
        elif importance_rank <= 10:
            score += 0.4
            model_support = "model_explanation_support"
            reasons.append("model_explanation_support")

    risk_level = _risk_constraint_level(row)
    risk_reasons = _risk_reasons(row, risk_level)
    reasons.extend(reason for reason in risk_reasons if reason not in reasons)

    evidence_level = _evidence_level(score, risk_level, conditional_status)
    decision = _integrated_decision(evidence_level, risk_level)
    integrated_reason = ";".join(reasons) if reasons else "no_supporting_predictive_evidence"
    return round(score, 6), evidence_level, integrated_reason, risk_level, decision, integrated_reason, model_support


def _risk_constraint_level(row: pd.Series) -> str:
    risk_text = ";".join(
        _text(row.get(col)).lower()
        for col in ["risk_flags", "recommended_use", "risk_level", "conditional_granger_status"]
    )
    declared_risk_level = _text(row.get("risk_level")).lower()
    level = "none"
    if declared_risk_level in {"strong", "high"}:
        level = _max_risk(level, "strong")
    elif declared_risk_level == "medium":
        level = _max_risk(level, "medium")
    elif declared_risk_level == "weak":
        level = _max_risk(level, "weak")
    if any(flag in risk_text for flag in ["strong_formula_leakage", "poor_data_quality", "target_leads_variable", "closed_loop_suspect"]):
        level = _max_risk(level, "strong")
    if "common_capacity_driver" in risk_text:
        level = _max_risk(level, "medium")
    if any(flag in risk_text for flag in ["lag_boundary", "residual_collinearity", "unstable_over_time", "unstable_across_regimes"]):
        level = _max_risk(level, "weak")
    if _text(row.get("conditional_granger_status")).lower() == "high_collinearity_risk":
        level = _max_risk(level, "medium")
    return level


def _risk_reasons(row: pd.Series, risk_level: str) -> list[str]:
    risk_text = ";".join(_text(row.get(col)).lower() for col in ["risk_flags", "recommended_use", "conditional_granger_status"])
    reasons: list[str] = []
    mapping = [
        ("strong_formula_leakage", "strong_formula_leakage_risk"),
        ("poor_data_quality", "poor_data_quality_risk"),
        ("target_leads_variable", "target_lead_risk"),
        ("closed_loop_suspect", "closed_loop_risk"),
        ("common_capacity_driver", "common_capacity_driver_risk"),
        ("lag_boundary", "lag_boundary_risk"),
        ("residual_collinearity", "residual_collinearity_risk"),
        ("unstable_over_time", "unstable_over_time_risk"),
        ("unstable_across_regimes", "unstable_across_regimes_risk"),
    ]
    for needle, reason in mapping:
        if needle in risk_text:
            reasons.append(reason)
    if risk_level == "none":
        return reasons
    return reasons


def _max_risk(left: str, right: str) -> str:
    return right if RISK_ORDER[right] > RISK_ORDER[left] else left


def _evidence_level(score: float, risk_level: str, conditional_status: str) -> str:
    if risk_level == "strong":
        return "risk_limited_evidence"
    if score >= 4.0 and risk_level in {"none", "weak"}:
        return "strong_predictive_evidence"
    if score >= 2.5:
        return "moderate_predictive_evidence"
    if score >= 1.0:
        return "weak_or_incomplete_evidence"
    if _is_insufficient_status(conditional_status):
        return "insufficient_evidence"
    return "not_supported"


def _integrated_decision(evidence_level: str, risk_level: str) -> str:
    if risk_level == "strong":
        return "manual_review_only"
    if evidence_level == "strong_predictive_evidence" and risk_level in {"none", "weak"}:
        return "priority_review"
    if evidence_level == "moderate_predictive_evidence" and risk_level != "strong":
        return "secondary_review"
    if evidence_level == "risk_limited_evidence":
        return "risk_limited_review"
    if evidence_level == "insufficient_evidence":
        return "insufficient_evidence"
    if evidence_level == "weak_or_incomplete_evidence":
        return "manual_review_only"
    if evidence_level == "not_supported":
        return "not_recommended"
    return "manual_review_only"


def _is_insufficient_status(status: str) -> bool:
    if not status:
        return True
    return any(token in status for token in ["insufficient", "missing", "not_found", "not enough", "skipped", "failed", "error", "empty"])


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _number(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)
