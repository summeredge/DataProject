from __future__ import annotations

import argparse
import json
import math
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_CLASSES = {"reasonable_driver", "implausible_driver", "neutral"}
RISK_FLAG_COLUMNS = {
    "target_leads_variable_flag": "target_leads_variable",
    "common_capacity_driver_flag": "common_capacity_driver",
    "strong_formula_leakage_flag": "strong_formula_leakage",
    "poor_data_quality_flag": "poor_data_quality",
    "lag_boundary_flag": "lag_boundary",
}
OUTPUT_COLUMNS = [
    "variable",
    "current_rank",
    "final_score",
    "candidate_grade",
    "recommended_use",
    "recommended_action",
    "lag",
    "direction",
    "risk_flags",
    "target_leads_variable_flag",
    "common_capacity_driver_flag",
    "strong_formula_leakage_flag",
    "poor_data_quality_flag",
    "lag_boundary_flag",
    "expected_class",
    "expectation_reason",
    "expectation_provided",
]


def evaluate_ranking_baseline(
    ranked_features: pd.DataFrame,
    risk_flags: pd.DataFrame | None = None,
    expectations: pd.DataFrame | None = None,
    cutoffs: tuple[int, ...] = (10, 20),
) -> tuple[pd.DataFrame, dict[str, object]]:
    ranked = ranked_features.copy(deep=True).reset_index(drop=True)
    if "variable" not in ranked.columns:
        ranked["variable"] = ""

    detail = ranked.copy(deep=True)
    detail.insert(0, "current_rank", range(1, len(detail) + 1))
    detail["variable"] = detail["variable"].fillna("").astype(str)

    risk_source = risk_flags.copy(deep=True) if risk_flags is not None else pd.DataFrame()
    detail = _merge_existing_risks(detail, ranked, risk_source)
    detail = _merge_expectations(detail, _validate_expectations(expectations))
    detail = _ensure_output_columns(detail)

    metrics = _build_metrics(detail, _validate_expectations(expectations), cutoffs)
    return detail[OUTPUT_COLUMNS], metrics


def evaluate_run_directory(
    run_dir: Path,
    expectations_path: Path | None = None,
    output_dir: Path | None = None,
    cutoffs: tuple[int, ...] = (10, 20),
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    ranked_path = run_dir / "ranked_features.csv"
    if not ranked_path.exists():
        raise FileNotFoundError("未找到 ranked_features.csv，请先完成主筛查")

    ranked = pd.read_csv(ranked_path, encoding="utf-8-sig")
    risk_path = run_dir / "risk_flags.csv"
    risk = pd.read_csv(risk_path, encoding="utf-8-sig") if risk_path.exists() else None
    expectations = (
        pd.read_csv(expectations_path, encoding="utf-8-sig")
        if expectations_path is not None
        else None
    )
    detail, metrics = evaluate_ranking_baseline(ranked, risk, expectations, cutoffs=cutoffs)

    target_dir = Path(output_dir) if output_dir is not None else run_dir / "ranking_baseline"
    target_dir.mkdir(parents=True, exist_ok=True)

    variables_path = target_dir / "ranking_baseline_variables.csv"
    metrics_path = target_dir / "ranking_baseline_metrics.json"
    markdown_path = target_dir / "ranking_baseline.md"
    detail.to_csv(variables_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(
        json.dumps(_json_ready(metrics), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(build_baseline_markdown(detail, metrics), encoding="utf-8")
    return {
        "variables": variables_path,
        "metrics": metrics_path,
        "markdown": markdown_path,
    }


def build_baseline_markdown(detail: pd.DataFrame, metrics: dict[str, object]) -> str:
    lines = [
        "# 筛选评分评价基线",
        "",
        "## 运行概况",
        "",
        f"- ranked_row_count: {metrics.get('ranked_row_count', 0)}",
        f"- duplicate_ranked_variable_count: {metrics.get('duplicate_ranked_variable_count', 0)}",
        f"- duplicate_ranked_variables: {', '.join(metrics.get('duplicate_ranked_variables', [])) or '无'}",
        "",
        "## Top10 / Top20 指标",
        "",
    ]
    cutoff_rows = []
    for key, value in (metrics.get("cutoffs") or {}).items():
        row = {"cutoff": key}
        row.update(value)
        cutoff_rows.append(row)
    lines.extend(_markdown_table(pd.DataFrame(cutoff_rows)))

    lines.extend(["", "## 已知合理变量当前排名", ""])
    reasonable = detail[detail["expected_class"].eq("reasonable_driver")]
    lines.extend(_markdown_table(_expectation_rank_table(reasonable)))

    lines.extend(["", "## 已知不合理变量当前排名", ""])
    implausible = detail[detail["expected_class"].eq("implausible_driver")]
    lines.extend(_markdown_table(_expectation_rank_table(implausible)))

    lines.extend(["", "## 评价清单中未找到的变量", ""])
    missing = (metrics.get("expectations") or {}).get("missing_variables", [])
    lines.extend(_markdown_table(pd.DataFrame({"variable": missing})))

    lines.extend(["", "## Top20 风险变量", ""])
    top20_detail = detail.head(20)
    risk_mask = (
        top20_detail[list(RISK_FLAG_COLUMNS)].any(axis=1)
        if not top20_detail.empty
        else pd.Series(dtype=bool)
    )
    risk_cols = ["variable", "current_rank", "final_score", "candidate_grade", "risk_flags", "expected_class"]
    lines.extend(_markdown_table(top20_detail.loc[risk_mask, risk_cols]))

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "本报告只评价现有筛选结果，不修改排名，也不代表因果结论。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评价已有 ranked_features.csv 的筛选评分基线")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expectations", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cutoffs", default="10,20")
    args = parser.parse_args(argv)
    cutoffs = tuple(int(item.strip()) for item in args.cutoffs.split(",") if item.strip())
    evaluate_run_directory(
        args.run_dir,
        expectations_path=args.expectations,
        output_dir=args.output_dir,
        cutoffs=cutoffs,
    )
    return 0


def _merge_existing_risks(
    detail: pd.DataFrame,
    ranked: pd.DataFrame,
    risk_source: pd.DataFrame,
) -> pd.DataFrame:
    result = detail.copy(deep=True)
    if "risk_flags" not in result.columns:
        result["risk_flags"] = ""
    result["risk_flags"] = result["risk_flags"].fillna("").astype(str)

    if not risk_source.empty and "variable" in risk_source.columns:
        source = risk_source.copy(deep=True)
        source["variable"] = source["variable"].fillna("").astype(str)
        occurrence_column = "__risk_occurrence"
        result[occurrence_column] = result.groupby("variable", sort=False).cumcount()
        source[occurrence_column] = source.groupby("variable", sort=False).cumcount()
        merge_cols = ["variable"]
        for column in ["risk_flags", *RISK_FLAG_COLUMNS.keys()]:
            if column in source.columns:
                merge_cols.append(column)
        merge_cols.append(occurrence_column)
        source = source[merge_cols].rename(
            columns={column: f"{column}__risk_source" for column in merge_cols if column != "variable"}
        )
        source = source.rename(columns={f"{occurrence_column}__risk_source": occurrence_column})
        merged = result.merge(
            source,
            on=["variable", occurrence_column],
            how="left",
            sort=False,
        )
        if "risk_flags__risk_source" in merged.columns:
            missing_text = merged["risk_flags"].astype(str).str.strip().eq("")
            merged["risk_flags"] = merged["risk_flags"].where(
                ~missing_text,
                merged["risk_flags__risk_source"].fillna("").astype(str),
            )
            merged = merged.drop(columns=["risk_flags__risk_source"])
        result = merged.drop(columns=[occurrence_column])

    for column, token in RISK_FLAG_COLUMNS.items():
        ranked_explicit = (
            result[column].map(_optional_bool).astype("boolean")
            if column in ranked.columns
            else pd.Series(pd.NA, index=result.index, dtype="boolean")
        )
        joined = f"{column}__risk_source"
        risk_file_explicit = (
            result[joined].map(_optional_bool).astype("boolean")
            if joined in result.columns
            else pd.Series(pd.NA, index=result.index, dtype="boolean")
        )
        text_fallback = result["risk_flags"].map(
            lambda value, flag=token: _has_risk_token(value, flag)
        )
        result[column] = (
            ranked_explicit.combine_first(risk_file_explicit)
            .combine_first(text_fallback.astype("boolean"))
            .fillna(False)
            .astype(bool)
        )
        if joined in result.columns:
            result = result.drop(columns=[joined])
    return result


def _validate_expectations(expectations: pd.DataFrame | None) -> pd.DataFrame:
    if expectations is None:
        return pd.DataFrame(columns=["variable", "expected_class", "reason"])
    frame = expectations.copy(deep=True)
    for column in ["variable", "expected_class"]:
        if column not in frame.columns:
            raise ValueError(f"评价清单缺少必填字段: {column}")
        blank = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(f"评价清单存在空值字段: {column}")
    frame["variable"] = frame["variable"].astype(str)
    frame["expected_class"] = frame["expected_class"].astype(str)
    if "reason" not in frame.columns:
        frame["reason"] = ""
    frame["reason"] = frame["reason"].fillna("").astype(str)

    duplicated = frame.loc[frame["variable"].duplicated(), "variable"].tolist()
    if duplicated:
        raise ValueError(f"评价清单变量重复: {duplicated[0]}")
    illegal = sorted(set(frame["expected_class"]) - EXPECTED_CLASSES)
    if illegal:
        raise ValueError(f"非法 expected_class: {illegal[0]}")
    return frame[["variable", "expected_class", "reason"]]


def _merge_expectations(detail: pd.DataFrame, expectations: pd.DataFrame) -> pd.DataFrame:
    result = detail.copy(deep=True)
    if expectations.empty:
        result["expected_class"] = ""
        result["expectation_reason"] = ""
        result["expectation_provided"] = False
        return result
    side = expectations.rename(columns={"reason": "expectation_reason"})
    result = result.merge(side, on="variable", how="left", sort=False)
    result["expected_class"] = result["expected_class"].fillna("")
    result["expectation_reason"] = result["expectation_reason"].fillna("")
    result["expectation_provided"] = result["expected_class"].astype(str).str.strip().ne("")
    return result


def _ensure_output_columns(detail: pd.DataFrame) -> pd.DataFrame:
    result = detail.copy(deep=True)
    text_columns = [
        "variable",
        "candidate_grade",
        "recommended_use",
        "recommended_action",
        "direction",
        "risk_flags",
        "expected_class",
        "expectation_reason",
    ]
    numeric_columns = ["final_score", "lag"]
    for column in text_columns:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str)
    for column in numeric_columns:
        if column not in result.columns:
            result[column] = pd.NA
    for column in RISK_FLAG_COLUMNS:
        if column not in result.columns:
            result[column] = False
        result[column] = result[column].map(_to_bool).fillna(False).astype(bool)
    if "expectation_provided" not in result.columns:
        result["expectation_provided"] = False
    result["expectation_provided"] = result["expectation_provided"].map(_to_bool).fillna(False).astype(bool)
    return result


def _build_metrics(
    detail: pd.DataFrame,
    expectations: pd.DataFrame,
    cutoffs: tuple[int, ...],
) -> dict[str, object]:
    ranked_variables = detail["variable"].astype(str)
    duplicate_variables = sorted(ranked_variables[ranked_variables.duplicated()].unique().tolist())
    expected_vars = set(expectations["variable"].astype(str)) if not expectations.empty else set()
    found_vars = expected_vars & set(ranked_variables)
    missing_vars = sorted(expected_vars - found_vars)
    reasonable_vars = set(expectations.loc[expectations["expected_class"].eq("reasonable_driver"), "variable"])
    implausible_vars = set(expectations.loc[expectations["expected_class"].eq("implausible_driver"), "variable"])
    found_reasonable_vars = reasonable_vars & found_vars

    metrics: dict[str, object] = {
        "schema_version": 1,
        "ranked_row_count": int(len(detail)),
        "duplicate_ranked_variable_count": int(len(duplicate_variables)),
        "duplicate_ranked_variables": duplicate_variables,
        "expectations": {
            "provided": bool(not expectations.empty),
            "total": int(len(expectations)),
            "reasonable_total": int(len(reasonable_vars)),
            "implausible_total": int(len(implausible_vars)),
            "neutral_total": int(expectations["expected_class"].eq("neutral").sum()) if not expectations.empty else 0,
            "found_total": int(len(found_vars)),
            "missing_total": int(len(missing_vars)),
            "missing_variables": missing_vars,
        },
        "cutoffs": {},
    }

    for cutoff in cutoffs:
        available = min(int(cutoff), len(detail))
        top = detail.head(available)
        top_vars = set(top["variable"].astype(str))
        reasonable_hits = len(top_vars & reasonable_vars)
        implausible_hits = len(top_vars & implausible_vars)
        entry = {
            "requested_cutoff": int(cutoff),
            "available_rows": int(available),
            "known_reasonable_hits": int(reasonable_hits),
            "known_reasonable_recall_of_expected": _ratio(reasonable_hits, len(reasonable_vars)),
            "known_reasonable_recall_of_found": _ratio(reasonable_hits, len(found_reasonable_vars)),
            "known_implausible_hits": int(implausible_hits),
            "known_implausible_share_of_topn": _ratio(implausible_hits, available),
        }
        for column, key in [
            ("target_leads_variable_flag", "target_leads_count"),
            ("common_capacity_driver_flag", "common_capacity_count"),
            ("strong_formula_leakage_flag", "strong_formula_leakage_count"),
            ("poor_data_quality_flag", "poor_data_quality_count"),
            ("lag_boundary_flag", "lag_boundary_count"),
        ]:
            entry[key] = int(top[column].sum())
        ab = top["candidate_grade"].astype(str).str.upper().isin({"A", "B"})
        for column, key in [
            ("target_leads_variable_flag", "ab_target_leads_count"),
            ("common_capacity_driver_flag", "ab_common_capacity_count"),
            ("strong_formula_leakage_flag", "ab_strong_formula_leakage_count"),
            ("poor_data_quality_flag", "ab_poor_data_quality_count"),
        ]:
            entry[key] = int((top[column] & ab).sum())
        metrics["cutoffs"][str(cutoff)] = entry
    return metrics


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _has_risk_token(value: object, token: str) -> bool:
    try:
        if value is None or pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    parts = {item.strip().lower() for item in str(value).split(";") if item.strip()}
    return token.lower() in parts


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Number):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _to_bool(value: object) -> bool:
    parsed = _optional_bool(value)
    return parsed if parsed is not None else False


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if value is pd.NA:
        return None
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _expectation_rank_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["variable", "current_rank", "final_score", "candidate_grade", "expectation_reason"]
    return frame[columns] if not frame.empty else pd.DataFrame(columns=columns)


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["无"]
    display = frame.fillna("")
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(_markdown_cell(row[column]) for column in display.columns) + " |")
    return lines


def _markdown_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
