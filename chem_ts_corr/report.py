from __future__ import annotations

from pathlib import Path

import pandas as pd


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

    files = {
        "ranked_features.csv": ranked_features,
        "recommended_candidates.csv": build_recommended_candidates(ranked_features),
        "lag_scores.csv": lag_scores,
        "granger_tests.csv": granger_tests,
        "shap_or_importance.csv": importance,
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

    lines.extend(["", "## 强初筛候选", ""]); lines.extend(_table_lines(_core_columns(strong).head(15))); lines.extend(["", "## 相关性线索", ""])
    lines.extend(_table_lines(_core_columns(ranked_features).head(15)))

    lines.extend(["", "## 评分分解 Top 15", ""])
    decomp_cols = [c for c in ["variable","final_score","raw_corr_score","residual_corr_score","regime_stability_final","rolling_stability","lag_quality","model_lift_score","risk_penalty","residual_status","regime_status","rolling_status","model_lift_status","lag_quality_status"] if c in ranked_features.columns]
    lines.extend(_table_lines(ranked_features[decomp_cols].head(15) if decomp_cols else pd.DataFrame()))

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

    lines.extend(
        [
            "",
            "## 解读提醒",
            "",
            "- final_score 使用动态权重：未计算项不参与评分，剩余已计算项按原始权重重归一；risk_penalty 按强/弱风险扣减。",
            "- residual_corr 是 target 和 candidate 分别剔除 CAPACITY 控制变量后的残差相关。",
            "- regime_stability 综合相关强度、符号一致性和滞后一致性。",
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
    rows = [[_format_cell(value) for value in row] for row in display.to_numpy()]
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


def _format_cell(value: object) -> str:
    if isinstance(value, float):
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
