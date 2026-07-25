from __future__ import annotations

import re
from typing import Any

import pandas as pd


STATUS_LABELS = {
    "supported": "支持",
    "partially_supported": "部分支持",
    "not_supported": "不支持",
    "not_available": "未获得",
    "insufficient_data": "数据不足",
    "conflicting": "存在冲突",
}

_LAYER_FIELDS = (
    ("Layer 1 关联", "layer1_association_status"),
    ("Layer 2 时间", "layer2_temporal_status"),
    ("Layer 3 独立性", "layer3_independence_status"),
    ("Layer 4 模型", "layer4_model_status"),
    ("稳定性", "stability_status"),
    ("数据质量", "data_quality_status"),
)


def add_evidence_explanations(frame: pd.DataFrame) -> pd.DataFrame:
    """Append deterministic, non-scoring evidence explanations to result rows."""
    if frame.empty:
        for column in [
            "evidence_support_items",
            "evidence_against_items",
            "four_layer_missing_items",
            "four_layer_coverage_status",
            "evidence_conflict_items",
            "candidate_summary",
        ]:
            frame[column] = pd.Series(dtype=str)
        return frame

    out = frame.copy()
    values = out.apply(_explain_row, axis=1, result_type="expand")
    for column in values.columns:
        out[column] = values[column]
    return out


def _explain_row(row: pd.Series) -> dict[str, str]:
    support: list[str] = []
    against: list[str] = []
    missing: list[str] = []
    existing_conflict = _text(row.get("evidence_conflict_items"))
    conflict: list[str] = _tokens(existing_conflict)
    generated_conflict: list[str] = []
    for label, field in _LAYER_FIELDS:
        status = _text(row.get(field))
        if status in {"supported", "partially_supported"}:
            support.append(f"{label}{STATUS_LABELS[status]}")
        elif status == "not_supported":
            against.append(f"{label}不支持")
        elif status in {"not_available", "insufficient_data"}:
            missing.append(f"{label}{STATUS_LABELS[status]}")
        elif status == "conflicting":
            generated_conflict.append(f"{label}存在冲突")

    conflict = _unique([*conflict, *generated_conflict])
    conflict_items = existing_conflict
    for item in _unique(generated_conflict):
        if item not in _tokens(conflict_items):
            conflict_items = "；".join(filter(None, [conflict_items, item]))
    summary = _summary(row, support, against, missing, conflict)
    return {
        "evidence_support_items": "；".join(_unique(support)),
        "evidence_against_items": "；".join(_unique(against)),
        "four_layer_missing_items": "；".join(_unique(missing)),
        "four_layer_coverage_status": _coverage_status(missing),
        "evidence_conflict_items": conflict_items,
        "candidate_summary": summary,
    }


def _summary(
    row: pd.Series,
    support: list[str],
    against: list[str],
    missing: list[str],
    conflict: list[str],
) -> str:
    flags = set(_tokens(row.get("risk_flags")))
    temporal = _text(row.get("layer2_temporal_status"))
    lag = _number(row.get("lag"))
    if "target_leads_variable" in flags or (
        temporal == "conflicting" and lag is not None and lag < 0
    ):
        return "下游响应可能：目标变量在时间上领先该变量，不建议直接解释为上游驱动。"
    if (
        conflict
        or _text(row.get("layer3_independence_status")) == "not_supported"
        or _text(row.get("stability_status")) == "conflicting"
        or flags
        & {
            "common_capacity_driver",
            "redundant_proxy",
            "residual_collinearity",
            "strong_formula_leakage",
            "poor_data_quality",
        }
    ):
        statistical_statuses = [
            _text(row.get("layer1_association_status")),
            _text(row.get("layer2_temporal_status")),
            _text(row.get("layer3_independence_status")),
            _text(row.get("layer4_model_status")),
            _text(row.get("stability_status")),
        ]
        has_statistical_support = any(
            status in {"supported", "partially_supported"}
            for status in statistical_statuses
        )
        if has_statistical_support:
            return "需要工程复核：存在部分统计证据支持，但同时存在独立性、稳定性、数据质量或时间冲突提示。"
        return "当前缺少明确支持证据，并存在独立性、稳定性、数据质量或时间冲突提示。"
    if temporal == "partially_supported" and lag == 0:
        return "同步关联候选：相关性明显，但未观察到稳定领先关系。"
    core_statuses = [
        _text(row.get("layer1_association_status")),
        _text(row.get("layer2_temporal_status")),
        _text(row.get("layer3_independence_status")),
        _text(row.get("layer4_model_status")),
        _text(row.get("stability_status")),
    ]
    if all(status == "supported" for status in core_statuses):
        return "潜在驱动因素候选：关联、时间、独立性、模型与稳定性证据均支持，建议工程复核。"
    if missing:
        return "潜在驱动因素候选：部分统计证据未获得或数据不足，建议补充证据后工程复核。"
    if against:
        return "潜在驱动因素候选：统计证据支持有限，建议结合工程信息复核。"
    return "潜在驱动因素候选：请结合统计证据支持与工程复核判断。"


def _tokens(value: Any) -> list[str]:
    return [item.strip() for item in re.split(r"[;；,，|]", _text(value)) if item.strip()]


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _coverage_status(missing: list[str]) -> str:
    if not missing:
        return "完整"
    return "部分完整" if len(missing) == 1 else "证据不足"


def _text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _number(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
