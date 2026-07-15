from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "reports" / "web_runs"

PACKAGE_KEYS = [
    "meta",
    "overview",
    "highly_correlated_variables",
    "attention_variables",
    "predictive_causal_evidence",
    "control_candidate_variables",
    "risk_and_limitations",
    "variable_role_hints",
    "xgb_out_of_time_validation",
]

RISK_EXPLANATIONS = {
    "common_capacity_driver": "可能共同受负荷/产能变化驱动，相关性可能来自共同工况而非可干预链路。",
    "closed_loop_suspect": "可能受现有闭环控制影响，变量和目标之间的时序关系会被控制器动作重塑。",
    "high_collinearity_risk": "与其它变量高度共线，单变量证据或重要性可能被替代变量分摊或放大。",
    "target_leads_variable": "目标变量领先候选变量，不应解释为候选变量先影响目标。",
    "lag_boundary": "最佳滞后靠近搜索边界，需要扩大窗口或做工程复核。",
    "ranked lag outside maxlag": "排序滞后超出复核最大滞后，条件检验可能无法覆盖原始峰值。",
    "fallback_missing_ranked_lag": "缺少主筛查滞后，只能 fallback 扫描，证据可信度需降级。",
    "strong_formula_leakage": "变量可能与目标存在公式/计算泄漏，不能作为独立解释或控制依据。",
}


def build_llm_analysis_package(run_dir: str | Path | None = None, *, run_id: str | None = None, top_n: int = 20) -> dict[str, Any]:
    """Build a compact, JSON-serializable analysis package for offline LLM prompting."""
    path = Path(run_dir) if run_dir is not None else DEFAULT_RUNS_DIR / str(run_id or "")
    top_n = max(1, int(top_n or 20))
    ranked = _read_csv(path / "ranked_features.csv")
    risk = _read_csv(path / "risk_flags.csv")
    conditional = _read_csv(path / "conditional_granger_scores.csv")
    evidence = _read_csv(path / "causal_review_evidence.csv")
    final = _read_csv(path / "final_review_summary.csv")
    importance = _read_csv(path / "model_variable_importance.csv")
    enhanced = _read_csv(path / "enhanced_validation_summary.csv")
    summary = _read_summary(path / "summary.md")
    xgb_validation = _xgb_out_of_time_validation(path, top_n)

    risk_by_var = _index_by_variable(risk)
    evidence_by_var = _index_by_variable(evidence)
    conditional_by_var = _index_by_variable(conditional)
    final_by_var = _index_by_variable(final)

    highly = [_compact(row, ["variable", "final_score", "candidate_grade", "lag", "direction", "risk_flags", "risk_level", "recommended_use"]) for row in _rows(ranked, top_n)]
    attention = [_compact(row, ["final_rank", "variable", "integrated_review_decision", "priority_label", "key_reason", "conflict_type", "conflict_reason", "lag_boundary_hint"]) for row in _rows(final, top_n)]

    names = list(dict.fromkeys(
        [str(r.get("variable")) for r in _rows(conditional, top_n) if r.get("variable")]
        + [str(r.get("variable")) for r in _rows(evidence, top_n) if r.get("variable")]
        + [str(r.get("variable")) for r in _rows(ranked, top_n) if r.get("variable")]
    ))[:top_n]
    predictive = []
    for name in names:
        merged = {"variable": name}
        merged.update(_compact(conditional_by_var.get(name, {}), ["status", "best_lag", "tested_lags", "fdr_q_value", "predictive_contribution", "interpretation"]))
        merged.update(_compact(evidence_by_var.get(name, {}), ["evidence_score", "evidence_level", "data_priority", "risk_constraint_level", "statistical_limit_level", "statistical_limit_reason", "integrated_review_decision", "integrated_review_reason"]))
        merged["evidence_scope"] = "预测验证/复核证据，不是确定性因果结论"
        predictive.append(_clean(merged))

    control = []
    ranked_by_var = _index_by_variable(ranked)
    control_names = list(dict.fromkeys([str(r.get("variable")) for r in _rows(ranked, top_n) + _rows(final, top_n) if r.get("variable")]))
    all_variable_names = set(control_names)
    for df in (ranked, final, evidence, risk, conditional):
        if not df.empty and "variable" in df.columns:
            all_variable_names.update(str(v) for v in df["variable"].dropna().tolist())
    for name in control_names:
        fields = {}
        for source in (ranked_by_var.get(name, {}), final_by_var.get(name, {}), evidence_by_var.get(name, {}), risk_by_var.get(name, {})):
            fields.update({k: v for k, v in source.items() if k not in fields or fields[k] in ("", None)})
        classified = _classify_control(name, fields, all_variable_names)
        role = classified.pop("suggested_control_role")
        comment = classified.pop("control_comment")
        control.append(_clean({"variable": name, "suggested_control_role": role, **classified, "control_comment": comment, "lag": fields.get("lag") or fields.get("best_lag"), "risk_flags": fields.get("risk_flags") or fields.get("conflict_type"), "review_decision": fields.get("integrated_review_decision")}))

    risks = []
    for name in list(dict.fromkeys([str(r.get("variable")) for r in _rows(risk, top_n * 3) + _rows(final, top_n * 3) + _rows(evidence, top_n * 3) if r.get("variable")])):
        merged = {"variable": name}
        for source in (risk_by_var.get(name, {}), final_by_var.get(name, {}), evidence_by_var.get(name, {})):
            merged.update(source)
        text = _joined_risks(merged)
        if text:
            risks.append(_clean({"variable": name, "risk_types": text, "risk_level": merged.get("risk_level") or merged.get("risk_constraint_level") or merged.get("statistical_limit_level"), "recommended_handling": _risk_handling(text), "conflict_reason": merged.get("conflict_reason"), "statistical_limit_reason": merged.get("statistical_limit_reason")}))
        if len(risks) >= top_n:
            break

    role_hints = [_role_hint(name) for name in list(dict.fromkeys([str(r.get("variable")) for r in _rows(ranked, top_n) if r.get("variable")]))]

    package = {
        "meta": {"run_dir": str(path), **summary},
        "overview": {
            "top_n": top_n,
            "available_files": sorted(p.name for p in path.glob("*.csv")) if path.exists() else [],
            "model_variable_importance_rows": int(len(importance)),
            "enhanced_validation_rows": int(len(enhanced)),
            "xgb_validation_status": xgb_validation["status"],
            "xgb_validation_available": xgb_validation["available"],
            "xgb_model_summary_rows": len(xgb_validation["model_comparison"]),
            "xgb_candidate_uplift_rows": len(xgb_validation["candidate_uplift"]),
            "xgb_available_files": _xgb_available_files(xgb_validation),
        },
        "highly_correlated_variables": highly,
        "attention_variables": attention,
        "predictive_causal_evidence": predictive,
        "control_candidate_variables": control,
        "risk_and_limitations": risks,
        "variable_role_hints": role_hints,
        "xgb_out_of_time_validation": xgb_validation,
    }
    json.dumps(package, ensure_ascii=False)
    return package


def build_llm_prompt(package: dict[str, Any], report_type: str = "general") -> str:
    data = json.dumps(package, ensure_ascii=False, indent=2)
    return f"""# 化工/APC/DCS 工程分析 Prompt（{report_type}）

你将面向化工/APC/DCS 工程人员，基于下方压缩分析包撰写中文报告。

## 硬约束
- 不得声称发现确定性因果关系。
- 不得使用未经限定的“X 导致 Y”、“X 是 Y 的原因”、“确定因果关系”、“可以直接作为控制变量”、“应直接投用控制”。
- 必须区分：高相关变量、最需要关注变量、预测验证/因果复核证据靠前变量、可能 MV 候选、可能 DV / 前馈候选（DV / FF = 扰动变量 / 前馈变量）、可能 CV（被控变量 / 约束变量）候选、监控变量候选、不建议直接用于控制的变量。
- APC 术语必须严格：DV / FF = 扰动变量 / 前馈变量候选；CV = 被控变量 / 约束变量候选；不得把 DV 写成被控变量，也不得把前馈扰动候选误写为受控目标。
- predictive_causal_evidence 只能解释为预测验证/复核证据，不是确定性因果。
- 结论必须引用变量名、证据来源、滞后、风险标签或复核决策；如果证据不足，必须明确说“证据不足”，不要编造原因。
- 核心工程原则：用 PV 做分析，用回路做 MV 候选，用 SV/MV/APC 写入点做实际操纵点确认。
- .PV 可以作为过程变量或控制回路历史数据代表；即使属于 FIC/TIC/PIC/AIC 等 PID 回路，相关性、滞后、Granger、模型解释等分析仍可使用 .PV。
- FIC.PV 作为流量控制回路历史数据代表进入 loop_mv_candidate 是合理的；但 AIC/AI/TI/PI/PV 类变量默认更偏分析测量、过程状态、监控或 DV / FF 候选。
- 只有明确存在同回路 SV/MV、远程设定值或 APC 可写入点时，AIC/AI/TI/PI/PV 类变量才可列为 loop_mv_candidate；不得仅凭 AIC/AI/TI/PI 前缀就把它们列为 MV 候选。
- loop_mv_candidate 表示“当前 PV 数据代表的控制回路可进入 MV 候选复核”，不是说 .PV 点本身就是最终写入 MV。
- 实际操纵点必须由工程人员核对 .SV、.MV、远程设定值或 APC 可写入点；有同回路 SV/MV 时应说明 related_sv/related_mv，没有时也应提示核对 DCS 中是否存在写入点。
- 控制回路历史数据通常属于闭环数据，必须提示 PID 控制动作、SV/MV 变化、MV 饱和、自动/手动状态对相关性和预测性的影响。
- 不得机械排除 .PV 并写成“.PV 不能做 MV，所以只能作为监控变量”；也不得写成“.PV 点本身就是最终写入 MV”。
- 如果 meta/overview 显示 model_status=skipped、skip_model_lift=True 或 skip_rolling_corr=True，不得把 model_explanation_support、model_lift_support、rolling_stability_supported 作为主要证据；只能提示“输入证据字段中出现，但需核对该模块是否实际启用”。
- XGBoost 时间外增量验证仅代表时间外预测增量证据，不代表确定性因果，也不改变前三层排名；不得用它回写或重排前三层候选，单独证明变量可操纵、是根本原因或作为 APC 投用依据。
- XGB 模型定义：M0 仅使用目标变量最短期历史作为简单基线；M1 使用目标变量历史滞后及配置的控制变量历史作为基线；M2 在 M1 基础上加入本次选中的全部候选变量滞后特征；逐候选增量验证比较“M1 + 单个候选变量”与 M1。不得将 M2_vs_M1 总体改善归因给某个单一候选，也不得将单候选改善解释为全体候选共同贡献。
- 改善率 = (基线误差 - 候选模型误差) / 基线误差 × 100%：大于 0 表示加入候选后误差下降，等于或接近 0 表示相对基线增量有限，小于 0 表示误差上升。
- positive_rmse_fold_ratio 只能解释为 RMSE 改善大于 0 的时间折占比，不得称为稳定性分数或因果置信度；数值越高只表示正向改善出现在更多时间折，仍须结合改善幅度、最差折结果和 validation_status 判断跨时间一致性。
- validation_status 的边界：validated_incremental_signal 表示多个时间折支持正向预测增量但仍不是因果结论；weak_incremental_value 表示存在一定预测增量但强度或跨折一致性不足；redundant_with_baseline 表示候选没有明显超过目标历史和控制变量基线，不等于工艺上无关；unstable_out_of_time 表示时间折方向不一致，应检查工况变化、漂移或数据分布变化但不得断言具体原因；insufficient_features 表示当前滞后或数据条件下特征不足，不能据此否定变量。
- 当 xgb_out_of_time_validation.available=false 时，必须按其 status 明确说明 XGB 未运行、运行失败、摘要无效或输出不完整；未运行时不得编造结果，不得编造 RMSE、MAE、R² 或候选验证结论。

## 风险解释必须覆盖
common_capacity_driver、closed_loop_suspect、high_collinearity_risk、target_leads_variable、lag_boundary、ranked lag outside maxlag、fallback_missing_ranked_lag、strong_formula_leakage。

## 输出结构
1. 总体结论
2. 与目标变量高度相关的过程变量
3. 最需要关注的变量
4. 相关性与因果复核证据靠前的变量
5. XGBoost 时间外增量验证：报告是否实际运行成功，解释 M0、M1、M2 总体比较和逐候选增量结果，区分正向增量、弱增量、基线冗余、跨时间不稳定和特征不足；与前三层证据交叉核对但不改变前三层排序，也不写成工艺因果结论。
6. 可能适合作为控制变量 / APC建模变量的候选：分别列出可能 MV 候选、可能 DV / 前馈候选（DV / FF = 扰动变量 / 前馈变量）、可能 CV（被控变量 / 约束变量）候选、监控变量候选、不建议直接用于控制的变量
7. 主要风险与解释限制
8. 下一步工程验证动作

## 压缩分析包
```json
{data}
```
"""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_required_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None


def _xgb_out_of_time_validation(path: Path, top_n: int) -> dict[str, Any]:
    empty = {
        "available": False,
        "summary": {},
        "model_comparison": [],
        "candidate_uplift": [],
        "evidence_scope": "时间外预测增量证据，不是工艺因果结论，也不改变前三层排名",
    }
    xgb_dir = path / "xgb_validation"
    summary_path = xgb_dir / "xgb_validation_summary.json"
    if not summary_path.exists():
        return {"status": "not_run", **empty}

    summary = _read_json(summary_path)
    if summary is None:
        return {"status": "invalid_summary", **empty}

    compact_summary = _compact(
        summary,
        [
            "status", "target", "row_count", "candidate_count", "candidate_pool_count",
            "fold_count", "m0_feature_count", "m1_feature_count", "m2_feature_count",
            "max_used_lag", "resolved_max_lag", "top_n", "created_at",
        ],
    )
    status = str(summary.get("status") or "invalid_summary")
    if status != "success":
        return {"status": status, **empty, "summary": compact_summary}

    model_summary = _read_required_csv(xgb_dir / "xgb_model_summary.csv")
    candidate_uplift = _read_required_csv(xgb_dir / "xgb_candidate_uplift.csv")
    if model_summary is None or candidate_uplift is None:
        return {"status": "incomplete_outputs", **empty, "summary": compact_summary}

    return {
        "status": "success",
        "available": True,
        "summary": compact_summary,
        "model_comparison": [
            _compact(
                row,
                [
                    "model_name", "mean_rmse", "median_rmse", "mean_mae", "median_mae",
                    "mean_r2", "fold_count", "M2_vs_M1_rmse_improvement_pct",
                    "M2_vs_M1_mae_improvement_pct",
                ],
            )
            for row in _rows(model_summary, len(model_summary))
        ],
        "candidate_uplift": [
            _compact(
                row,
                [
                    "variable", "fold_count", "positive_rmse_fold_count",
                    "positive_mae_fold_count", "positive_rmse_fold_ratio",
                    "median_rmse_improvement_pct", "median_mae_improvement_pct",
                    "mean_rmse_improvement_pct", "mean_mae_improvement_pct",
                    "worst_fold_rmse_improvement_pct", "validation_status",
                ],
            )
            for row in _rows(candidate_uplift, top_n)
        ],
        "evidence_scope": empty["evidence_scope"],
    }


def _xgb_available_files(xgb_validation: dict[str, Any]) -> list[str]:
    if xgb_validation["status"] == "invalid_summary" or xgb_validation["status"] == "not_run":
        return []
    files = ["xgb_validation/xgb_validation_summary.json"]
    if xgb_validation["available"]:
        files.extend(
            [
                "xgb_validation/xgb_model_summary.csv",
                "xgb_validation/xgb_candidate_uplift.csv",
            ]
        )
    return files


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    meta = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\s*-\s*([^:：]+)[:：]\s*(.*)\s*$", line)
        if m:
            meta[m.group(1).strip()] = _clean_value(m.group(2).strip())
    return meta


def _rows(df: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    return [] if df.empty else df.head(n).to_dict("records")


def _index_by_variable(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if df.empty or "variable" not in df.columns:
        return {}
    return {str(r.get("variable")): r for r in df.to_dict("records")}


def _compact(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return _clean({k: row.get(k) for k in keys if k in row and pd.notna(row.get(k)) and row.get(k) != ""})


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items() if _clean(v) not in (None, "", [], {})}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return _clean_value(obj)


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, float):
        return round(value, 4)
    return value.item() if hasattr(value, "item") else value


def _joined_risks(row: dict[str, Any]) -> str:
    parts = [str(row.get(k, "")) for k in ["risk_flags", "conflict_type", "lag_boundary_hint", "statistical_limit_reason", "integrated_review_reason"]]
    return ";".join(p for p in parts if p and p.lower() != "nan")


def _risk_handling(text: str) -> str:
    matches = [explain for key, explain in RISK_EXPLANATIONS.items() if key in text]
    return "；".join(matches) if matches else "需结合工艺机理、趋势图和现场可操纵性做人工复核。"


def _classify_control(name: str, fields: dict[str, Any], all_variable_names: set[str] | None = None) -> dict[str, Any]:
    all_variable_names = all_variable_names or set()
    text = (name + ";" + ";".join(str(v) for v in fields.values())).lower()
    loop_tag, suffix = _split_loop_point(name)
    related_sv = f"{loop_tag}.SV" if loop_tag and f"{loop_tag}.SV" in all_variable_names else None
    related_mv = f"{loop_tag}.MV" if loop_tag and f"{loop_tag}.MV" in all_variable_names else None
    role_hint = _control_role_hint(name, loop_tag, suffix)

    base = {
        "suggested_control_role": "manual_review_only",
        "loop_tag": loop_tag,
        "related_sv": related_sv,
        "related_mv": related_mv,
        "has_related_sv": bool(related_sv),
        "has_related_mv": bool(related_mv),
        "role_hint": role_hint,
        "control_comment": "需结合工艺机理、趋势图和现场可操纵性做人工复核。",
    }

    if any(x in text for x in ["target_leads_variable", "strong_formula_leakage", "weak_or_incomplete_evidence"]):
        base.update(
            suggested_control_role="not_recommended_for_control",
            control_comment="存在目标领先、公式泄漏或证据不足风险，不建议直接用于控制。",
        )
        return base

    downgrade = [x for x in ["closed_loop_suspect", "common_capacity_driver", "high_collinearity_risk", "fallback_missing_ranked_lag"] if x in text]
    lower = name.lower()
    is_loop_pv = suffix == "PV" and _looks_like_control_loop(loop_tag or "")
    is_measurement_heavy_loop_pv = suffix == "PV" and _is_measurement_heavy_loop(loop_tag or "")
    has_explicit_write_point = bool(related_sv or related_mv or any(x in text for x in ["apc", "write", "writable", "remote set", "remote_set", "可写入", "写入点", "远程设定值"]))

    if suffix == "MV":
        role = "mv_candidate"
        comment = f"{name} 名称指向 MV，可作为实际操纵点候选复核；仍需确认 DCS/APC 写入权限、约束和安全边界。"
    elif suffix == "SV":
        role = "mv_candidate"
        comment = f"{name} 是设定值候选，可作为实际操纵点候选复核；需确认 APC 是否允许调整该设定值及其上下限/联锁约束。"
    elif is_loop_pv and (not is_measurement_heavy_loop_pv or has_explicit_write_point):
        role = "loop_mv_candidate"
        write_points = "/".join(p for p in [related_sv, related_mv, "远程设定值", "APC 可写入点"] if p)
        if not (related_sv or related_mv):
            write_points = "SV/MV/远程设定值/APC 可写入点"
        comment = (
            f"{name} 可作为 {loop_tag} 控制回路的历史数据代表；{loop_tag} 控制回路可作为 MV 候选复核，"
            f"实际操纵点需核对或确认 {write_points}。由于该变量来自闭环回路，需检查控制模式、SV/MV 变化、"
            "MV 饱和、自动/手动状态和 lag_boundary 风险；不得把 PV 点本身直接视为最终写入 MV。"
        )
    elif any(x in lower for x in ["load", "feed", "flow", "负荷", "流量", "进料"]):
        role = "dv_feedforward_candidate"
        comment = "名称提示可能为负荷/进料/流量扰动或前馈候选；需验证可测性、领先滞后和现场可用性。"
    elif suffix == "PV" or any(x in lower for x in ["aic", "ai", "tic", "ti", "pic", "pi"]):
        role = "monitor_candidate"
        comment = "名称提示偏分析测量、过程状态、监控变量或 DV / FF 候选，可用于分析和监控；只有明确存在 SV/MV、远程设定值或 APC 可写入点时，才应列为 loop_mv_candidate。"
    else:
        role = "manual_review_only"
        comment = "变量角色不明确，需人工确认工艺含义、可操纵性和与目标的工程链路。"

    if downgrade:
        if role in {"mv_candidate", "loop_mv_candidate", "dv_feedforward_candidate"}:
            role = "manual_review_only" if role != "mv_candidate" else "monitor_candidate"
        comment += " 因 " + ",".join(downgrade) + " 风险，需保守降级或人工复核。"
    base.update(suggested_control_role=role, control_comment=comment)
    return base


def _split_loop_point(name: str) -> tuple[str | None, str | None]:
    match = re.match(r"^(.+)\.([A-Za-z]+)$", str(name).strip())
    if not match:
        return None, None
    return match.group(1), match.group(2).upper()


def _looks_like_control_loop(loop_tag: str) -> bool:
    return bool(re.match(r"^(FIC|TIC|PIC|AIC|LIC|FRC|TRC|PRC|ARC|LRC|FC|TC|PC|AC|LC)\w*", loop_tag.upper()))


def _is_measurement_heavy_loop(loop_tag: str) -> bool:
    return bool(re.match(r"^(AIC|AI|TIC|TI|PIC|PI)\w*", loop_tag.upper()))


def _control_role_hint(name: str, loop_tag: str | None, suffix: str | None) -> str:
    if suffix == "PV" and _looks_like_control_loop(loop_tag or ""):
        if _is_measurement_heavy_loop(loop_tag or ""):
            return "PV 偏分析测量/过程状态/监控或 DV / FF 候选；只有明确存在 SV/MV、远程设定值或 APC 可写入点时，回路才进入 MV 候选复核。"
        return "PV 可作为控制回路历史数据代表；回路可进入 MV 候选复核，实际写入点需另行确认。"
    if suffix == "SV":
        return "设定值候选；需确认 APC 是否允许调整。"
    if suffix == "MV":
        return "MV 写入点候选；需确认权限、约束和安全边界。"
    return _role_hint(name)["role_hint"]


def _role_hint(name: str) -> dict[str, str]:
    lower = name.lower()
    if any(x in lower for x in [".mv", ".sv", "fic"]):
        hint = "名称提示可能为可操纵量或设定值；仍需现场确认。"
    elif any(x in lower for x in [".pv", "ai", "ti", "pi"]):
        hint = "名称提示多为过程测量/监控变量，不能凭名称确认可操纵性。"
    elif any(x in lower for x in ["load", "feed", "flow", "负荷", "流量", "进料"]):
        hint = "名称提示可能为负荷/扰动/前馈候选，需验证可测性和滞后。"
    else:
        hint = "角色不明确，需人工确认工艺含义和可操纵性。"
    return {"variable": name, "role_hint": hint}
