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
    for name in list(dict.fromkeys([str(r.get("variable")) for r in _rows(ranked, top_n) + _rows(final, top_n) if r.get("variable")])):
        fields = {}
        for source in (ranked_by_var.get(name, {}), final_by_var.get(name, {}), evidence_by_var.get(name, {}), risk_by_var.get(name, {})):
            fields.update({k: v for k, v in source.items() if k not in fields or fields[k] in ("", None)})
        role, comment = _classify_control(name, fields)
        control.append(_clean({"variable": name, "suggested_control_role": role, "control_comment": comment, "lag": fields.get("lag") or fields.get("best_lag"), "risk_flags": fields.get("risk_flags") or fields.get("conflict_type"), "review_decision": fields.get("integrated_review_decision")}))

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
        "overview": {"top_n": top_n, "available_files": sorted(p.name for p in path.glob("*.csv")) if path.exists() else [], "model_variable_importance_rows": int(len(importance)), "enhanced_validation_rows": int(len(enhanced))},
        "highly_correlated_variables": highly,
        "attention_variables": attention,
        "predictive_causal_evidence": predictive,
        "control_candidate_variables": control,
        "risk_and_limitations": risks,
        "variable_role_hints": role_hints,
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
- 必须区分：高相关变量、最需要关注变量、预测验证/因果复核证据靠前变量、可能 MV 候选、可能 DV / 前馈候选、监控变量候选、不建议直接用于控制的变量。
- predictive_causal_evidence 只能解释为预测验证/复核证据，不是确定性因果。
- 结论必须引用变量名、证据来源、滞后、风险标签或复核决策；如果证据不足，必须明确说“证据不足”，不要编造原因。

## 风险解释必须覆盖
common_capacity_driver、closed_loop_suspect、high_collinearity_risk、target_leads_variable、lag_boundary、ranked lag outside maxlag、fallback_missing_ranked_lag、strong_formula_leakage。

## 输出结构
1. 总体结论
2. 与目标变量高度相关的过程变量
3. 最需要关注的变量
4. 相关性与因果复核证据靠前的变量
5. 可能适合作为控制变量 / APC建模变量的候选：分别列出可能 MV 候选、可能 DV / 前馈候选、监控变量候选、不建议直接用于控制的变量
6. 主要风险与解释限制
7. 下一步工程验证动作

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
    if isinstance(value, float):
        return round(value, 4)
    return value.item() if hasattr(value, "item") else value


def _joined_risks(row: dict[str, Any]) -> str:
    parts = [str(row.get(k, "")) for k in ["risk_flags", "conflict_type", "lag_boundary_hint", "statistical_limit_reason", "integrated_review_reason"]]
    return ";".join(p for p in parts if p and p.lower() != "nan")


def _risk_handling(text: str) -> str:
    matches = [explain for key, explain in RISK_EXPLANATIONS.items() if key in text]
    return "；".join(matches) if matches else "需结合工艺机理、趋势图和现场可操纵性做人工复核。"


def _classify_control(name: str, fields: dict[str, Any]) -> tuple[str, str]:
    text = (name + ";" + ";".join(str(v) for v in fields.values())).lower()
    if any(x in text for x in ["target_leads_variable", "strong_formula_leakage", "weak_or_incomplete_evidence"]):
        return "not_recommended_for_control", "存在目标领先、公式泄漏或证据不足风险，不建议直接用于控制。"
    downgrade = [x for x in ["closed_loop_suspect", "common_capacity_driver", "high_collinearity_risk", "fallback_missing_ranked_lag"] if x in text]
    base_comment = "需确认是否可操纵；不得仅凭变量名直接认定为 MV。"
    if any(x in name.lower() for x in [".mv", ".sv", "fic"]):
        role = "mv_candidate"
    elif any(x in name.lower() for x in ["load", "feed", "flow", "负荷", "流量", "进料"]):
        role = "dv_feedforward_candidate"
    elif any(x in name.lower() for x in [".pv", "ai", "ti", "pi"]):
        role = "monitor_candidate"
    else:
        role = "manual_review_only"
    if downgrade:
        role = "monitor_candidate" if role == "mv_candidate" else "manual_review_only"
        base_comment += " 因 " + ",".join(downgrade) + " 已保守降级。"
    return role, base_comment


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
