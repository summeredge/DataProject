from __future__ import annotations

from pathlib import Path

import pandas as pd

from chem_ts_corr.common import to_int
from chem_ts_corr.causal_review import build_causal_review_candidates
from chem_ts_corr.model_discovery import build_model_discovered_candidates, build_model_variable_importance
from chem_ts_corr.near_miss import build_near_miss_candidates


DISPLAY_SCORE_COLUMNS = {
    "driver_priority_score",
    "final_score",
    "driver_priority_factor",
    "evidence_completeness",
    "evidence_confidence",
    "data_quality_score",
    "evidence_strength",
    "evidence_score",
    "evidence_score_low",
    "evidence_score_high",
    "证据覆盖度",
    "证据修正系数",
    "数据质量得分",
}


def write_outputs(
    output_dir: Path,
    target: str,
    ranked_features: pd.DataFrame,
    lag_scores: pd.DataFrame,
    granger_tests: pd.DataFrame,
    importance: pd.DataFrame,
    metrics: dict[str, float | str],
    diagnostics: pd.DataFrame | None = None,
    residual_corr_scores: pd.DataFrame | None = None,
    regime_scores: pd.DataFrame | None = None,
    risk_flags: pd.DataFrame | None = None,
    model_lift_scores: pd.DataFrame | None = None,
    lag_peak_quality: pd.DataFrame | None = None,
    rolling_corr_scores: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_variable_importance = build_model_variable_importance(
        importance,
        ranked_features,
        risk_flags=risk_flags,
    )
    model_discovered = build_model_discovered_candidates(
        importance,
        ranked_features,
        risk_flags=risk_flags,
        max_lag=_metric_int(metrics, "max_lag"),
    )
    near_miss = build_near_miss_candidates(
        lag_scores,
        ranked_features,
        residual_corr_scores=residual_corr_scores,
        lag_peak_quality=lag_peak_quality,
        risk_flags=risk_flags,
        screening_top_n=_metric_int(metrics, "top_k") or 50,
    )

    files = {
        "ranked_features.csv": ranked_features,
        "recommended_candidates.csv": build_recommended_candidates(ranked_features),
        "causal_review_candidates.csv": build_causal_review_candidates(ranked_features),
        "lag_scores.csv": lag_scores,
        "granger_tests.csv": granger_tests,
        "shap_or_importance.csv": importance,
        "model_variable_importance.csv": model_variable_importance,
        "model_discovered_candidates.csv": model_discovered,
        "near_miss_candidates.csv": near_miss,
        "diagnostics.csv": diagnostics,
        "residual_corr_scores.csv": residual_corr_scores,
        "regime_scores.csv": regime_scores,
        "risk_flags.csv": risk_flags,
        "model_lift_scores.csv": model_lift_scores,
        "lag_peak_quality.csv": lag_peak_quality,
        "rolling_corr_scores.csv": rolling_corr_scores,
    }
    for name, frame in files.items():
        (frame if frame is not None else pd.DataFrame()).to_csv(
            output_dir / name, index=False, encoding="utf-8-sig"
        )

    summary = build_markdown_summary(
        target,
        ranked_features,
        granger_tests,
        importance,
        metrics,
        risk_flags if risk_flags is not None else pd.DataFrame(),
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")



def _metric_int(metrics: dict[str, float | str], key: str) -> int | None:
    value = metrics.get(key)
    numeric = to_int(value, default=-1)
    return None if numeric == -1 and value not in {-1, "-1"} else numeric


def build_markdown_summary(
    target: str,
    ranked_features: pd.DataFrame,
    granger_tests: pd.DataFrame,
    importance: pd.DataFrame,
    metrics: dict[str, float | str],
    risk_flags: pd.DataFrame,
) -> str:
    risky = risk_flags[risk_flags.get("risk_count", 0) > 0] if not risk_flags.empty else pd.DataFrame()
    common_capacity = _risk_subset(risky, "common_capacity_driver_flag")
    closed_loop = _risk_subset(risky, "closed_loop_suspect_flag")
    strong = _recommended_subset(ranked_features, "strong_screening_candidate")
    predictive = _recommended_subset(ranked_features, "prediction_candidate")
    not_causal = ranked_features[ranked_features.get("recommended_use", pd.Series(dtype=str)).isin(["capacity_driven","formula_coupled_reference","closed_loop_suspect","unstable_candidate","poor_quality_variable"])] if not ranked_features.empty else pd.DataFrame()

    lines = [f"# 四层工业时序筛查摘要：{target}", "", "## 运行信息", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## 强初筛候选", ""])
    lines.extend(_table_lines(_core_columns(strong).head(15)))
    lines.extend(["", "## 相关性线索", ""])
    lines.extend(_table_lines(_core_columns(ranked_features).head(15)))

    lines.extend(["", "## 评分分解 Top 15", ""])
    decomp_cols = [c for c in ["variable","driver_rank","driver_priority_score","final_score","candidate_class","driver_priority_factor","evidence_coverage_status","evidence_missing_items","evidence_score","evidence_score_low","evidence_score_high","evidence_completeness","evidence_confidence","association_score","innovation_score","independent_signal_score","correlation_evidence_score","correlation_evidence_status","regime_stability_final","regime_status","rolling_stability","rolling_status","stability_score","lag_quality","lag_quality_status","model_lift_score","model_lift_status","prediction_score","data_quality_score","score_method","risk_penalty_rate","risk_penalty","risk_score_cap"] if c in ranked_features.columns]
    decomposition = ranked_features[decomp_cols].head(15) if decomp_cols else pd.DataFrame()
    decomposition = decomposition.rename(
        columns={
            "evidence_coverage_status": "证据覆盖状态",
            "evidence_missing_items": "缺失证据",
            "evidence_completeness": "证据覆盖度",
            "data_quality_score": "数据质量得分",
            "evidence_confidence": "证据修正系数",
        }
    )
    lines.extend(_table_lines(decomposition))

    lines.extend(["", "## 预测候选", ""])
    lines.extend(_table_lines(_core_columns(predictive).head(15)))

    lines.extend(["", "## 疑似共同负荷驱动", ""])
    lines.extend(_table_lines(common_capacity.head(15)))

    lines.extend(["", "## 疑似闭环反馈", ""])
    lines.extend(_table_lines(closed_loop.head(15)))

    lines.extend(["", "## 不建议作为因果结论的变量", ""])
    lines.extend(_table_lines(_core_columns(not_causal).head(15)))

    lines.extend(["", "## Granger 二级验证", ""])
    lines.append("Granger 结果仅表示候选变量对目标的预测贡献，不作为因果结论。")
    lines.extend(_table_lines(granger_tests.head(15)))

    lines.extend(["", "## 第三层复核准备说明", ""])
    lines.append("causal_review_candidates.csv 为规则化复核优先级清单（不引入 PCMCI/Transfer Entropy，仅用于三层复核排队）。")
    review = build_causal_review_candidates(ranked_features)
    lines.extend(_table_lines(review.head(15)))

    lines.extend(["", "## 自动诊断建议", ""])
    advice: list[str] = []
    top10 = ranked_features.head(10) if not ranked_features.empty else pd.DataFrame()
    if not top10.empty and "lag_boundary_flag" in top10.columns:
        lag_boundary_ratio = float(top10["lag_boundary_flag"].fillna(False).astype(bool).mean())
        if lag_boundary_ratio >= 0.3:
            advice.append("- Top 10 候选中滞后边界命中占比较高（>=30%），建议适当扩大 max_lag 后复跑。")

    if ranked_features.empty or "candidate_grade" not in ranked_features.columns:
        ab_count = 0
    else:
        ab_count = int(ranked_features["candidate_grade"].astype(str).isin(["A", "B"]).sum())
    if ab_count == 0:
        advice.append("- 当前未筛出 A/B 级强候选，不建议直接进入 APC/软测量建模，建议先补充工况与变量复核。")

    if not risk_flags.empty and "common_capacity_driver_flag" in risk_flags.columns:
        common_capacity_ratio = float(risk_flags["common_capacity_driver_flag"].fillna(False).astype(bool).mean())
        common_capacity_count = int(risk_flags["common_capacity_driver_flag"].fillna(False).astype(bool).sum())
        if common_capacity_ratio >= 0.3 or common_capacity_count >= 3:
            advice.append("- 疑似共同负荷驱动变量较多，建议检查残差控制列设置与负荷变量选择是否合理。")

    if not advice:
        advice.append("- 暂无显著自动诊断告警，建议结合工艺知识继续复核 Top 候选。")
    lines.extend(advice)

    lines.extend(
        [
            "",
            "## 解读提醒",
            "",
            "- final_score 使用工业稳健 V2：在多组合理工程权重下汇总变化量关联、增量预测、稳定性和滞后质量；缺失证据降低证据覆盖度与修正系数，不再放大剩余证据。统计证据风险按 risk_penalty_rate 相对扣减，高风险可触发 risk_score_cap；方向和变量角色通过 driver_priority_score 单独约束工程优先级。",
            "- 证据修正系数由证据覆盖度和数据质量共同计算，仅用于修正综合证据评分，不表示概率、统计置信度或因果置信度。",
            "- residual_corr 是 target 和 candidate 分别剔除 CAPACITY 控制变量后的残差相关。",
            "- regime_stability_final 综合工况覆盖度、方向一致性、强度一致性和滞后一致性。",
            "- lag_scores.csv 中普通 p 值仅供参考；工业时序通常存在自相关、非独立样本和多重比较，应优先查看 corr_q_value 与工程合理性。",
            "- recommended_action 给出下一步处理建议，包括可作为预测候选、疑似共同负荷驱动、疑似闭环反馈、仅作相关性参考、建议人工工艺复核。",
            "- 本报告输出的是筛查线索和预测候选，不直接给出工艺因果结论。",
        ]
    )
    return "\n".join(lines) + "\n"


def _recommended_subset(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    if frame.empty or "recommended_use" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["recommended_use"].astype(str).eq(value)]


def _risk_subset(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    return frame[frame[column].astype(bool)]


def _core_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variable",
        "final_score",
        "lag",
        "direction",
        "raw_corr",
        "residual_corr",
        "risk_flags",
        "recommended_use",
        "recommended_action",
    ]
    return frame[[col for col in columns if col in frame.columns]] if not frame.empty else frame


def _table_lines(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["无可用结果。"]

    display = frame.fillna("")
    columns = [str(col) for col in display.columns]
    rows = [
        [_format_cell(value, columns[index]) for index, value in enumerate(row)]
        for row in display.to_numpy()
    ]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]

    header = "| " + " | ".join(col.ljust(widths[index]) for index, col in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return [header, separator, *body]


def _format_cell(value: object, column: str = "") -> str:
    if isinstance(value, float):
        if column in DISPLAY_SCORE_COLUMNS:
            return f"{value:.3f}"
        return f"{value:.6g}"
    return str(value)


def build_recommended_candidates(ranked_features: pd.DataFrame) -> pd.DataFrame:
    if ranked_features.empty:
        return pd.DataFrame(columns=["variable","candidate_grade","recommended_use","final_score","fallback_reason"])
    ab = ranked_features[ranked_features.get("candidate_grade", pd.Series(dtype=str)).isin(["A","B"])].copy()
    if not ab.empty:
        if "fallback_reason" not in ab.columns:
            ab["fallback_reason"] = ""
        return ab
    top = ranked_features.head(10).copy()
    top["fallback_reason"] = "no_A_or_B_candidates"
    return top
