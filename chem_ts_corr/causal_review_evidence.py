from __future__ import annotations

import pandas as pd

from chem_ts_corr.common import as_text, left_join_missing, to_float


INTERPRETATION = "confounder review evidence only; not a causal conclusion"

CONFIDENCE_REVIEW_COLUMNS = [
    "independent_predictive_support",
    "confounder_assessment",
    "control_relation_assessment",
    "statistical_limitation",
    "direction_assessment",
]

EVIDENCE_COLUMNS = [
    "variable",
    "review_priority",
    "review_reason",
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
    "data_priority",
    "evidence_reason",
    "statistical_limit_level",
    "statistical_limit_reason",
    "risk_constraint_level",
    "integrated_review_decision",
    "integrated_review_reason",
    "interpretation",
    *CONFIDENCE_REVIEW_COLUMNS,
]

# ``evidence_matrix`` is a read-only handoff for engineers.  Its fields are
# status/evidence references only; none of them are inputs to screening,
# scoring, ranking, candidate selection, or XGBoost execution.
EVIDENCE_MATRIX_COLUMNS = [
    "variable",
    "initial_rank",
    "final_score",
    "validation_status",
    "evidence_consistency",
    "supporting_methods",
    *CONFIDENCE_REVIEW_COLUMNS,
    "xgb_status",
    "generalization_status",
]

EVIDENCE_MATRIX_STATUS_LABELS = {
    "validation_status": {
        "not_run": "未执行",
        "not_computed": "未计算",
        "missing": "证据缺失",
        "supported": "已有支持证据",
        "limited": "证据有限或存在限制",
        "failed": "执行失败",
    },
    "evidence_consistency": {
        "not_run": "未执行",
        "not_computed": "未计算",
        "missing": "证据缺失",
        "consistent": "证据较一致",
        "partial": "证据部分一致",
    },
    "independent_predictive_support": {
        "supported": "独立预测贡献证据较强",
        "supported_with_limitations": "存在独立预测贡献证据，但存在限制",
        "not_supported": "未形成独立预测贡献证据",
        "not_computed": "未计算",
    },
    "confounder_assessment": {
        "not_assessed": "未审查",
        "no_flagged_confounder": "未发现明显混杂风险",
        "common_driver_risk": "可能存在共同驱动影响",
        "shared_signal_risk": "可能存在共享信号影响",
        "formula_relation_risk": "可能存在公式关系影响",
    },
    "control_relation_assessment": {
        "not_assessed": "未审查",
        "no_control_relation_flagged": "未发现明显控制关系风险",
        "control_reference": "控制参考变量",
        "possible_control_response": "可能属于控制响应信号",
        "shared_capacity_or_control_context": "可能存在负荷或控制背景影响",
    },
    "statistical_limitation": {
        "not_computed": "未计算",
        "no_flagged_statistical_limitation": "未发现明显统计限制",
        "high_collinearity_limitation": "高共线性限制",
        "insufficient_sample_limitation": "样本不足限制",
        "failed_statistical_limitation": "统计计算失败",
    },
    "direction_assessment": {
        "variable_leads_target": "变量领先目标",
        "target_leads_variable": "目标领先变量",
        "zero_lag": "零滞后",
        "not_computed": "未计算",
    },
    "xgb_status": {
        "not_computed": "未计算",
        "missing": "证据缺失",
        "validated_incremental_signal": "时间外预测增量已支持",
        "weak_incremental_value": "时间外预测增量较弱",
        "redundant_with_baseline": "与基线信息重复",
        "unstable_out_of_time": "时间外预测增量不稳定",
        "insufficient_features": "有效特征不足",
    },
    "generalization_status": {
        "not_computed": "未计算",
        "missing": "证据缺失",
        "validated_incremental_signal": "时间外预测增量已支持",
        "weak_incremental_value": "时间外预测增量较弱",
        "redundant_with_baseline": "与基线信息重复",
        "unstable_out_of_time": "时间外预测增量不稳定",
        "insufficient_features": "有效特征不足",
    },
}


RISK_ORDER = {"none": 0, "weak": 1, "medium": 2, "strong": 3}

STATISTICAL_LIMIT_FLAGS = {
    "high_collinearity_risk",
    "residual_collinearity",
    "common_capacity_driver",
    "unstable_over_time",
    "unstable_across_regimes",
    "lag_boundary",
}

HARD_DOWNGRADE_FLAGS = {
    "severe_data_quality",
    "strong_formula_leakage",
    "target_leads_variable",
}

STATISTICAL_LIMIT_LEVELS = {
    "high_collinearity_risk": "medium",
    "common_capacity_driver": "medium",
    "residual_collinearity": "weak",
    "lag_boundary": "weak",
    "unstable_over_time": "weak",
    "unstable_across_regimes": "weak",
}


def build_causal_review_evidence(
    ranked_features: pd.DataFrame,
    conditional_granger_scores: pd.DataFrame,
    risk_flags: pd.DataFrame | None = None,
    enhanced_validation_summary: pd.DataFrame | None = None,
    granger_tests: pd.DataFrame | None = None,
    model_variable_importance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build third-layer confounder-review evidence for manual review.

    The output explains whether predictive value may remain independently
    supported or be limited by confounding, control relations, or statistical
    constraints.  It does not change screening scores, model calculations, or
    first-layer ranking.
    """
    if ranked_features.empty or "variable" not in ranked_features.columns:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)

    base_cols = [
        "variable",
        "review_priority",
        "review_reason",
        "candidate_grade",
        "final_score",
        "lag",
        "direction",
        "risk_level",
        "risk_flags",
        "risk_count",
        "recommended_use",
        "recommended_action",
        # These existing screening fields are retained in the working frame so
        # the third-layer explanations can describe residual/control context.
        # They are not emitted as new score or ranking inputs.
        "residual_corr",
        "residual_status",
        "residual_evidence_status",
        "load_adjusted_relation_status",
        "common_capacity_candidate_flag",
        "control_reference_type",
        "control_reference_source",
        "is_control_reference",
        "is_auto_control_reference",
        "temporal_direction_status",
    ]
    evidence = ranked_features.copy(deep=True)
    evidence = evidence[[c for c in base_cols if c in evidence.columns]].copy(deep=True)
    _ensure_columns(evidence, base_cols)

    evidence = left_join_missing(
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
    evidence = left_join_missing(
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
    evidence = left_join_missing(
        evidence,
        enhanced_validation_summary,
        {
            "model_lift": "model_lift",
            "rolling_stability": "rolling_stability",
            "rolling_sign_consistency": "rolling_sign_consistency",
            "status": "enhanced_validation_status",
        },
    )
    evidence = left_join_missing(
        evidence,
        granger_tests,
        {
            "min_p_value": "granger_min_p_value",
            "fdr_q_value": "granger_fdr_q_value",
            "status": "granger_status",
        },
    )
    evidence = left_join_missing(
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
        "data_priority",
        "evidence_reason",
        "statistical_limit_level",
        "statistical_limit_reason",
        "risk_constraint_level",
        "integrated_review_decision",
        "integrated_review_reason",
        "model_explanation_support",
    ]
    for col in assessed.columns:
        evidence[col] = assessed[col]
    evidence["interpretation"] = INTERPRETATION
    review_assessments = evidence.apply(_confidence_review_assessments, axis=1, result_type="expand")
    review_assessments.columns = CONFIDENCE_REVIEW_COLUMNS
    for col in CONFIDENCE_REVIEW_COLUMNS:
        evidence[col] = review_assessments[col]
    return evidence[EVIDENCE_COLUMNS].copy(deep=True)


def build_evidence_matrix(
    ranked_features: pd.DataFrame | None,
    validation_summary: pd.DataFrame | None,
    causal_review_evidence: pd.DataFrame | None,
    *,
    xgb_candidate_uplift: pd.DataFrame | None = None,
    xgb_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the explanation-only evidence matrix for manual review.

    ``validation_summary`` and the optional XGB tables are read-only inputs.
    ``None`` means that the corresponding stage was not computed; a supplied
    table without a row for a variable is represented as ``missing``.  This
    distinction keeps not-computed, missing, and real numeric zero semantics
    separate while preserving the existing second-/fourth-layer values.
    """
    evidence = (
        causal_review_evidence.copy(deep=True)
        if causal_review_evidence is not None
        else pd.DataFrame()
    )
    if evidence.empty or "variable" not in evidence.columns:
        return pd.DataFrame(columns=EVIDENCE_MATRIX_COLUMNS)

    ranked_by_variable = _rows_by_variable(ranked_features)
    validation_by_variable = _rows_by_variable(validation_summary)
    xgb_by_variable = _rows_by_variable(xgb_candidate_uplift)
    if xgb_summary is not None and not xgb_summary.empty:
        # Some callers may provide a per-variable fourth-layer sidecar rather
        # than the candidate uplift table.  Prefer the explicit sidecar row
        # only when the uplift table has no row for that variable.
        xgb_summary_by_variable = _rows_by_variable(xgb_summary)
    else:
        xgb_summary_by_variable = {}

    validation_missing_state = "not_computed" if validation_summary is None else "missing"
    xgb_missing_state = (
        "not_computed"
        if xgb_candidate_uplift is None and xgb_summary is None
        else "missing"
    )
    rows: list[dict[str, object]] = []
    for _, source in evidence.iterrows():
        variable = _text(source.get("variable"))
        ranked = ranked_by_variable.get(variable, {})
        validation = validation_by_variable.get(variable, {})
        xgb = xgb_by_variable.get(variable) or xgb_summary_by_variable.get(variable, {})

        row = {
            "variable": variable,
            # ``driver_rank`` and ``final_score`` are references to the first
            # layer.  Do not infer a rank from matrix order or recalculate a
            # score when the source field is absent.
            "initial_rank": _first_present(
                source.get("initial_rank"),
                source.get("driver_rank"),
                ranked.get("driver_rank"),
            ),
            "final_score": _first_present(
                source.get("final_score"),
                ranked.get("final_score"),
            ),
            "validation_status": _matrix_state(
                validation,
                "validation_status",
                default=validation_missing_state,
            ),
            "evidence_consistency": _matrix_state(
                validation,
                "evidence_consistency",
                default=validation_missing_state,
            ),
            "supporting_methods": _first_present(
                validation.get("supporting_methods"),
            ),
        }
        for column in CONFIDENCE_REVIEW_COLUMNS:
            row[column] = _matrix_state(source, column, default="not_computed")

        # Only copy fourth-layer fields that already exist.  In particular,
        # do not derive generalization status from an overall JSON status or
        # start XGB here.
        row["xgb_status"] = _matrix_state(
            xgb,
            "xgb_status",
            default=_first_present(xgb.get("validation_status"), xgb_missing_state),
        )
        row["generalization_status"] = _matrix_state(
            xgb,
            "generalization_status",
            default=xgb_missing_state,
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=EVIDENCE_MATRIX_COLUMNS)


def evidence_status_label(field: str, value: object) -> str:
    """Return the shared Chinese display text for a matrix status code."""
    code = _text(value)
    if not code:
        return ""
    return EVIDENCE_MATRIX_STATUS_LABELS.get(field, {}).get(code, code)


def evidence_matrix_status_labels() -> dict[str, dict[str, str]]:
    """Return a copy safe to expose in API/report payloads."""
    return {
        field: dict(labels)
        for field, labels in EVIDENCE_MATRIX_STATUS_LABELS.items()
    }


def _rows_by_variable(frame: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if frame is None or frame.empty or "variable" not in frame.columns:
        return {}
    rows: dict[str, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        variable = _text(row.get("variable"))
        if variable and variable not in rows:
            rows[variable] = row
    return rows


def _first_present(*values: object) -> object:
    for value in values:
        if not _is_missing(value):
            return value
    return pd.NA


def _matrix_state(
    row: dict[str, object] | pd.Series,
    column: str,
    *,
    default: object,
) -> object:
    value = row.get(column) if row is not None else None
    return default if _is_missing(value) else value



def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col not in frame.columns:
            frame[col] = pd.NA


def _confidence_review_assessments(row: pd.Series) -> tuple[str, str, str, str, str]:
    """Classify existing review evidence without changing any calculations."""
    return (
        _independent_predictive_support(row),
        _confounder_assessment(row),
        _control_relation_assessment(row),
        _statistical_limitation(row),
        _direction_assessment(row),
    )


def _independent_predictive_support(row: pd.Series) -> str:
    status = _text(row.get("conditional_granger_status")).lower()
    q_value = _number(row.get("conditional_fdr_q_value"))
    contribution = _number(row.get("predictive_contribution"))
    has_positive_contribution = contribution is not None and contribution > 0
    if not status or status in {"not_computed", "not_run", "not_available"}:
        return "not_computed"
    if status == "high_collinearity_risk":
        return "supported_with_limitations" if has_positive_contribution else "not_supported"
    if "fallback_missing_ranked_lag" in status:
        return (
            "supported_with_limitations"
            if has_positive_contribution and (q_value is None or q_value <= 0.10)
            else "not_supported"
        )
    if _is_insufficient_status(status):
        return "not_computed"
    if status.startswith("ok") and q_value is not None and q_value <= 0.10:
        return "supported" if has_positive_contribution else "not_supported"
    if status.startswith("ok"):
        return "not_supported"
    return "not_computed"


def _confounder_assessment(row: pd.Series) -> str:
    tokens = _risk_flag_tokens(row)
    recommended_use = _text(row.get("recommended_use")).lower()
    relation_status = _text(row.get("load_adjusted_relation_status")).lower()
    residual_evidence_status = _text(row.get("residual_evidence_status")).lower()
    if {
        "formula_like",
        "strong_formula_leakage",
    } & tokens or recommended_use == "formula_coupled_reference":
        return "formula_relation_risk"
    if (
        "common_capacity_driver" in tokens
        or recommended_use == "capacity_driven"
        or relation_status == "raw_only_common_load_risk"
        or _is_true(row.get("common_capacity_candidate_flag"))
    ):
        return "common_driver_risk"
    if (
        "residual_collinearity" in tokens
        or "redundant_proxy" in tokens
        or relation_status == "raw_only_residual_weak"
        or residual_evidence_status == "weak"
    ):
        return "shared_signal_risk"
    if not _has_assessment_input(
        row,
        "risk_flags",
        "risk_level",
        "recommended_use",
        "residual_status",
        "residual_evidence_status",
        "load_adjusted_relation_status",
    ):
        return "not_assessed"
    return "no_flagged_confounder"


def _control_relation_assessment(row: pd.Series) -> str:
    tokens = _risk_flag_tokens(row)
    recommended_use = _text(row.get("recommended_use")).lower()
    control_reference = (
        recommended_use == "control_variable_reference"
        or _is_true(row.get("is_control_reference"))
        or _is_true(row.get("is_auto_control_reference"))
        or _has_text(row.get("control_reference_type"))
        or _has_text(row.get("control_reference_source"))
    )
    if control_reference:
        return "control_reference"
    lag = _number(row.get("lag"))
    temporal_status = _text(row.get("temporal_direction_status")).lower()
    if (
        "target_leads_variable" in tokens
        or temporal_status == "target_leads_supported"
        or (lag is not None and lag < 0)
        or recommended_use == "state_indicator"
    ):
        return "possible_control_response"
    if "common_capacity_driver" in tokens or recommended_use == "capacity_driven":
        return "shared_capacity_or_control_context"
    if not _has_assessment_input(
        row,
        "risk_flags",
        "recommended_use",
        "control_reference_type",
        "control_reference_source",
        "is_control_reference",
        "is_auto_control_reference",
        "temporal_direction_status",
    ):
        return "not_assessed"
    return "no_control_relation_flagged"


def _statistical_limitation(row: pd.Series) -> str:
    status = _text(row.get("conditional_granger_status")).lower()
    tokens = _risk_flag_tokens(row)
    condition_number = _number(row.get("condition_number"))

    if (
        "high_collinearity_risk" in status
        or "high_collinearity_risk" in tokens
        or "residual_collinearity" in tokens
        or (condition_number is not None and condition_number > 1e8)
    ):
        return "high_collinearity_limitation"
    if "insufficient" in status or "not enough" in status:
        return "insufficient_sample_limitation"
    if not status or status in {"not_computed", "not_run", "not_available"}:
        return "not_computed"
    if "failed" in status or "error" in status:
        return "failed_statistical_limitation"
    if "fallback_missing_ranked_lag" in status:
        return "no_flagged_statistical_limitation"
    if _is_insufficient_status(status):
        return "not_computed"
    if status.startswith("ok"):
        return "no_flagged_statistical_limitation"
    # A non-significant result is still a computed result. Preserve its raw
    # q/p values and status, while keeping the explanation separate from
    # sample/collinearity limitations.
    if status.startswith("skipped"):
        return "not_computed"
    return "no_flagged_statistical_limitation"


def _direction_assessment(row: pd.Series) -> str:
    lag = _number(row.get("lag"))
    if lag is None:
        return "not_computed"
    if lag > 0:
        return "variable_leads_target"
    if lag < 0:
        return "target_leads_variable"
    return "zero_lag"


def _has_assessment_input(row: pd.Series, *columns: str) -> bool:
    return any(not _is_missing(row.get(column)) for column in columns)


def _is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).strip().lower() in {"true", "1", "yes", "y"}


def _has_text(value: object) -> bool:
    return _text(value).strip().lower() not in {"", "nan", "none", "missing", "not_computed"}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple, set)) and not value:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _assess_row(row: pd.Series) -> tuple[float, str, str, str, str, str, str, str, str, str]:
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
    is_fallback_signal = "fallback_missing_ranked_lag" in conditional_status
    conditional_supported = (
        conditional_status.startswith("ok")
        and not is_fallback_signal
        and conditional_q is not None
        and conditional_q <= 0.10
    )
    if conditional_status.startswith("ok") and conditional_q is not None:
        if is_fallback_signal and conditional_q <= 0.10:
            score += 0.4
            reasons.append("fallback_predictive_signal")
        elif conditional_q <= 0.05:
            score += 2.0
            reasons.append("independent_predictive_evidence")
        elif conditional_q <= 0.10:
            score += 1.0
            reasons.append("independent_predictive_evidence_limited")
    if contribution is not None and contribution > 0:
        if conditional_status == "high_collinearity_risk":
            score += 0.3
            reasons.append("high_collinearity_limited_signal")
        else:
            score += 0.5
            reasons.append("predictive_contribution_positive")

    granger_q = _number(row.get("granger_fdr_q_value"))
    granger_supported = granger_q is not None and granger_q <= 0.10
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

    score = round(score, 6)
    statistical_limit_level, statistical_limit_reasons = _statistical_limit_assessment(row)
    has_hard_downgrade = _has_hard_downgrade(row)
    risk_level = _risk_constraint_level(row, statistical_limit_level, has_hard_downgrade)
    risk_reasons = _risk_reasons(row, risk_level)
    reasons.extend(reason for reason in risk_reasons if reason not in reasons)

    data_priority = _data_priority(
        grade=grade,
        evidence_score=score,
        importance_rank=importance_rank,
        conditional_supported=conditional_supported,
        contribution=contribution,
        granger_supported=granger_supported,
        sign_consistency=sign_consistency,
        conditional_status=conditional_status,
        conditional_q=conditional_q,
        model_lift=model_lift,
        rolling_stability=rolling_stability,
    )
    evidence_level = _evidence_level(score, risk_level, conditional_status)
    integrated_reasons = list(reasons)
    decision = _integrated_decision(
        evidence_level,
        risk_level,
        data_priority,
        statistical_limit_level,
        has_hard_downgrade,
        row=row,
    )
    if has_hard_downgrade:
        integrated_reasons.append("hard_downgrade_risk")
    if decision in {"priority_review_with_statistical_limit", "secondary_review_with_statistical_limit"}:
        integrated_reasons.extend(["statistical_test_limited"])
        if data_priority == "high":
            integrated_reasons.extend(["strong_data_evidence", "priority_preserved_due_to_strong_data_evidence"])
        integrated_reasons.extend(statistical_limit_reasons)
    integrated_reason = ";".join(dict.fromkeys(integrated_reasons)) if integrated_reasons else "no_supporting_predictive_evidence"
    statistical_limit_reason = ";".join(statistical_limit_reasons)
    evidence_reason = ";".join(dict.fromkeys(reasons)) if reasons else "no_supporting_predictive_evidence"
    return (
        score,
        evidence_level,
        data_priority,
        evidence_reason,
        statistical_limit_level,
        statistical_limit_reason,
        risk_level,
        decision,
        integrated_reason,
        model_support,
    )


def _risk_text(row: pd.Series) -> str:
    return ";".join(
        _text(row.get(col)).lower()
        for col in ["risk_flags", "recommended_use", "risk_level", "conditional_granger_status"]
    )


def _has_hard_downgrade(row: pd.Series) -> bool:
    return bool(_risk_flag_tokens(row) & HARD_DOWNGRADE_FLAGS)


def _risk_flag_tokens(row: pd.Series) -> set[str]:
    return {token.strip().lower() for token in _text(row.get("risk_flags")).split(";") if token.strip()}


def _is_legacy_poor_data_quality(row: pd.Series) -> bool:
    tokens = _risk_flag_tokens(row)
    return (
        "poor_data_quality" in tokens
        and "severe_data_quality" not in tokens
        and not (tokens & (HARD_DOWNGRADE_FLAGS - {"severe_data_quality"}))
        and not any(STATISTICAL_LIMIT_LEVELS.get(token) == "medium" for token in tokens)
    )


def _statistical_limit_assessment(row: pd.Series) -> tuple[str, list[str]]:
    risk_text = _risk_text(row)
    level = "none"
    reasons: list[str] = []
    for flag, flag_level in STATISTICAL_LIMIT_LEVELS.items():
        if flag in risk_text:
            level = _max_risk(level, flag_level)
            reasons.append(_statistical_limit_reason(flag))
    return level, reasons


def _statistical_limit_reason(flag: str) -> str:
    if flag == "high_collinearity_risk":
        return "high_collinearity_limited_signal"
    return flag


def _data_priority(
    *,
    grade: str,
    evidence_score: float,
    importance_rank: float | None,
    conditional_supported: bool,
    contribution: float | None,
    granger_supported: bool,
    sign_consistency: float | None,
    conditional_status: str,
    conditional_q: float | None,
    model_lift: float | None,
    rolling_stability: float | None,
) -> str:
    has_positive_contribution = contribution is not None and contribution > 0
    has_independent_strong_evidence = (
        (importance_rank is not None and importance_rank <= 5)
        or (model_lift is not None and model_lift >= 0.05)
        or (rolling_stability is not None and rolling_stability >= 0.7)
        or (conditional_supported and has_positive_contribution)
        or (granger_supported and sign_consistency is not None and sign_consistency >= 0.8)
    )
    high_collinearity_without_q = conditional_status == "high_collinearity_risk" and conditional_q is None
    if (
        (grade in {"A", "B"} and not high_collinearity_without_q)
        or evidence_score >= 3.5
        or has_independent_strong_evidence
    ):
        return "high"
    if (
        grade in {"C", "D"}
        or evidence_score >= 2.0
        or (importance_rank is not None and importance_rank <= 10)
        or (sign_consistency is not None and sign_consistency >= 0.7)
        or has_positive_contribution
    ):
        return "medium"
    return "low"


def _risk_constraint_level(row: pd.Series, statistical_limit_level: str, has_hard_downgrade: bool) -> str:
    risk_text = _risk_text(row)
    declared_risk_level = _text(row.get("risk_level")).lower()
    if _is_legacy_poor_data_quality(row):
        declared_risk_level = "weak"
    level = "none"
    if has_hard_downgrade:
        level = _max_risk(level, "strong")
    elif declared_risk_level in {"strong", "high"} and not any(flag in risk_text for flag in STATISTICAL_LIMIT_FLAGS):
        level = _max_risk(level, "strong")
    elif declared_risk_level == "medium":
        level = _max_risk(level, "medium")
    elif declared_risk_level == "weak":
        level = _max_risk(level, "weak")
    level = _max_risk(level, statistical_limit_level)
    return level


def _risk_reasons(row: pd.Series, risk_level: str) -> list[str]:
    risk_text = _risk_text(row)
    reasons: list[str] = []
    mapping = [
        ("strong_formula_leakage", "strong_formula_leakage_risk"),
        ("poor_data_quality", "poor_data_quality_warning"),
        ("severe_data_quality", "severe_data_quality_risk"),
        ("target_leads_variable", "target_lead_risk"),
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


def _integrated_decision(
    evidence_level: str,
    risk_level: str,
    data_priority: str,
    statistical_limit_level: str,
    has_hard_downgrade: bool,
    *,
    row: pd.Series,
) -> str:
    if has_hard_downgrade:
        return "manual_review_only"
    if _is_high_collinearity_without_independent_strong_evidence(row):
        if evidence_level == "moderate_predictive_evidence":
            return "secondary_review_with_statistical_limit"
        return "manual_review_only"
    if data_priority == "high" and statistical_limit_level in {"medium", "strong"}:
        return "priority_review_with_statistical_limit"
    if evidence_level == "strong_predictive_evidence" and statistical_limit_level in {"none", "weak"}:
        return "priority_review"
    if evidence_level == "moderate_predictive_evidence":
        if statistical_limit_level in {"medium", "strong"}:
            return "secondary_review_with_statistical_limit"
        return "secondary_review"
    if evidence_level == "weak_or_incomplete_evidence":
        return "manual_review_only"
    if evidence_level == "not_supported":
        return "not_recommended"
    if evidence_level == "insufficient_evidence":
        return "insufficient_evidence"
    if risk_level == "strong":
        return "manual_review_only"
    return "manual_review_only"


def _is_high_collinearity_without_independent_strong_evidence(row: pd.Series) -> bool:
    status = _text(row.get("conditional_granger_status")).lower()
    q = _number(row.get("conditional_fdr_q_value"))
    if status != "high_collinearity_risk" or q is not None:
        return False
    importance_rank = _number(row.get("model_importance_rank"))
    model_lift = _number(row.get("model_lift"))
    rolling_stability = _number(row.get("rolling_stability"))
    sign_consistency = _number(row.get("rolling_sign_consistency"))
    granger_q = _number(row.get("granger_fdr_q_value"))
    return not (
        (importance_rank is not None and importance_rank <= 5)
        or (model_lift is not None and model_lift >= 0.05)
        or (rolling_stability is not None and rolling_stability >= 0.7)
        or (granger_q is not None and granger_q <= 0.10 and sign_consistency is not None and sign_consistency >= 0.8)
    )


def _is_insufficient_status(status: str) -> bool:
    if not status:
        return True
    return any(token in status for token in ["insufficient", "missing", "not_found", "not enough", "skipped", "failed", "error", "empty"])


def _text(value: object) -> str:
    return as_text(value)


def _number(value: object) -> float | None:
    numeric = to_float(value, default=float("nan"))
    return None if pd.isna(numeric) else numeric
