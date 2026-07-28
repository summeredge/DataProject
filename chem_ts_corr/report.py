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
    strong = _recommended_subset(ranked_features, "strong_screening_candidate")
    not_causal = ranked_features[ranked_features.get("recommended_use", pd.Series(dtype=str)).isin(["capacity_driven", "formula_coupled_reference", "unstable_candidate", "poor_quality_variable"])] if not ranked_features.empty else pd.DataFrame()

    lines = [f"# 初步筛选摘要：{target}", "", "## 运行信息", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## 强初筛候选", ""])
    lines.extend(_table_lines(_core_columns(strong).head(15)))
    lines.extend(["", "## 相关性线索", ""])
    lines.extend(_table_lines(_core_columns(ranked_features).head(15)))

    lines.extend(["", "## 评分分解 Top 15", ""])
    decomp_cols = [c for c in ["variable", "final_score", "association_score", "correlation_evidence_score", "innovation_score", "lag_quality", "data_quality_score", "evidence_score", "risk_flags", "recommended_use"] if c in ranked_features.columns]
    decomposition = ranked_features[decomp_cols].head(15) if decomp_cols else pd.DataFrame()
    decomposition = decomposition.rename(
        columns={
            "data_quality_score": "数据质量得分",
        }
    )
    lines.extend(_table_lines(decomposition))

    lines.extend(["", "## 疑似共同负荷驱动", ""])
    lines.extend(_table_lines(common_capacity.head(15)))

    lines.extend(["", "## 不建议作为因果结论的变量", ""])
    lines.extend(_table_lines(_core_columns(not_causal).head(15)))

    lines.extend(["", "## 当前阶段建议", ""])
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
            "- final_score 是初步统计筛选的稳健综合得分；缺失可选评分组件与真实零值分开处理，可用组件按其实际权重重新归一化。",
            "- 当前摘要只解释初步分析已生成的相关性、滞后、数据质量和基础风险结果。",
            "- 增强筛选、Granger、模型分析和综合复核须在对应页面单独运行后解读。",
            "- lag_scores.csv 中普通 p 值仅供参考；工业时序通常存在自相关、非独立样本和多重比较，应优先查看 corr_q_value 与工程合理性。",
            "- 本报告输出的是初步筛查线索，不直接给出工艺因果或控制源结论。",
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
        "pearson",
        "spearman",
        "method",
        "lag",
        "direction",
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
        return pd.DataFrame(columns=["variable", "final_score", "candidate_grade", "recommended_use", "fallback_reason"])
    ordered = ranked_features.sort_values("final_score", ascending=False, kind="stable") if "final_score" in ranked_features.columns else ranked_features.copy()
    ab = ordered[ordered.get("candidate_grade", pd.Series(index=ordered.index, dtype=str)).isin(["A", "B"])].copy()
    if not ab.empty:
        if "fallback_reason" not in ab.columns:
            ab["fallback_reason"] = ""
        return ab
    top = ordered.head(10).copy()
    top["fallback_reason"] = "no_A_or_B_candidates"
    return top
