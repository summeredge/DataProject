from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import causal_review_evidence, final_review_summary, screening
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import run_analysis
from chem_ts_corr.screening import final_ranked_features
from chem_ts_corr.service import analyze_numeric_frame


REGISTRY_PATH = Path("docs/four_layer_evidence_registry.json")
RANKING_FUNCTIONS = {
    screening: {
        "risk_flags", "_risk_adjustment", "classify_candidate", "final_ranked_features",
        "_finalize_driver_ranking", "_grade_candidate", "_recommend_use", "_recommended_action",
    },
    causal_review_evidence: {
        "build_causal_review_evidence", "_assess_row", "_risk_text", "_has_hard_downgrade",
        "_statistical_limit_assessment", "_risk_constraint_level", "_risk_reasons",
        "_evidence_level", "_integrated_decision", "_is_high_collinearity_without_independent_strong_evidence",
    },
    final_review_summary: {
        "build_final_review_summary", "_key_reason", "_lag_boundary_hint", "_conflicts",
    },
}


def _registry() -> list[dict[str, object]]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _field_accesses(module, function_names: set[str]) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    assert {node.name for node in functions} == function_names
    fields: set[str] = set()
    for function in functions:
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                fields.add(node.args[0].value)
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                fields.add(node.slice.value)
    return fields


def _production_ranking_fields() -> set[str]:
    return set().union(*(
        _field_accesses(module, names) for module, names in RANKING_FUNCTIONS.items()
    ))


def _ranking_inputs() -> tuple[pd.DataFrame, ...]:
    ranked = pd.DataFrame([
        {"variable": "upstream", "score": 0.81, "innovation_score": 0.64, "lag": 2},
        {"variable": "downstream", "score": 0.86, "innovation_score": 0.72, "lag": -1},
        {"variable": "capacity", "score": 0.78, "innovation_score": 0.70, "lag": 3},
    ])
    residual = pd.DataFrame([
        {"variable": "upstream", "residual_corr": 0.75},
        {"variable": "downstream", "residual_corr": 0.80},
        {"variable": "capacity", "residual_corr": 0.20},
    ])
    stability = pd.DataFrame([
        {"variable": "upstream", "regime_stability_final": 0.80},
        {"variable": "downstream", "regime_stability_final": 0.85},
        {"variable": "capacity", "regime_stability_final": 0.72},
    ])
    lift = pd.DataFrame([
        {"variable": "upstream", "model_lift_score": 0.66, "status": "ok"},
        {"variable": "downstream", "model_lift_score": 0.73, "status": "ok"},
        {"variable": "capacity", "model_lift_score": 0.55, "status": "ok"},
    ])
    risks = pd.DataFrame([
        {"variable": "upstream", "risk_flags": "", "data_quality_score": 0.96},
        {"variable": "downstream", "risk_flags": "target_leads_variable", "data_quality_score": 0.93},
        {"variable": "capacity", "risk_flags": "common_capacity_driver", "data_quality_score": 0.90},
    ])
    lag_quality = pd.DataFrame([
        {"variable": "upstream", "lag_quality": 0.82, "lag_boundary_flag": False},
        {"variable": "downstream", "lag_quality": 0.76, "lag_boundary_flag": False},
        {"variable": "capacity", "lag_quality": 0.68, "lag_boundary_flag": False},
    ])
    rolling = pd.DataFrame([
        {"variable": "upstream", "rolling_stability": 0.84},
        {"variable": "downstream", "rolling_stability": 0.81},
        {"variable": "capacity", "rolling_stability": 0.71},
    ])
    return ranked, residual, stability, lift, risks, lag_quality, rolling


def test_final_ranking_baseline_is_frozen():
    result = final_ranked_features(*_ranking_inputs())
    actual = result[[
        "variable", "final_score", "driver_priority_factor", "driver_priority_score",
        "driver_rank", "candidate_grade", "recommended_use",
    ]].to_dict("records")
    expected = pd.DataFrame([
        {"variable": "upstream", "final_score": 0.7271088721789382, "driver_priority_factor": 1.0,
         "driver_priority_score": 0.7271088721789382, "driver_rank": 1, "candidate_grade": "B",
         "recommended_use": "prediction_candidate"},
        {"variable": "capacity", "final_score": 0.5451660268468049, "driver_priority_factor": 0.75,
         "driver_priority_score": 0.4088745201351037, "driver_rank": 2, "candidate_grade": "C",
         "recommended_use": "capacity_driven"},
        {"variable": "downstream", "final_score": 0.7233067357159925, "driver_priority_factor": 0.45,
         "driver_priority_score": 0.3254880310721966, "driver_rank": 3, "candidate_grade": "C",
         "recommended_use": "state_indicator"},
    ])
    pd.testing.assert_frame_equal(pd.DataFrame(actual), expected, check_exact=False, rtol=1e-12)


def _config(tmp_path: Path, name: str) -> AnalysisConfig:
    return AnalysisConfig(
        input_path=tmp_path / "input.csv", time_column="time", target="target",
        output_dir=tmp_path / name, max_lag=3, top_k=3, skip_model_lift=True,
        skip_rolling_corr=True,
    )


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    size = 96
    upstream = rng.normal(size=size)
    target = np.roll(upstream, 2) + rng.normal(scale=0.25, size=size)
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=size, freq="h"),
        "target": target, "upstream": upstream, "other": rng.normal(size=size),
    })


def test_service_and_pipeline_entries_write_the_same_frozen_ranked_features(tmp_path: Path):
    frame = _frame()
    config = _config(tmp_path, "service")
    service_ranked = analyze_numeric_frame(frame.set_index("time"), config).ranked_features

    frame.to_csv(config.input_path, index=False, encoding="utf-8-sig")
    pipeline_config = _config(tmp_path, "pipeline")
    run_analysis(pipeline_config)
    exported = pd.read_csv(pipeline_config.output_dir / "ranked_features.csv", encoding="utf-8-sig")

    fields = [
        "variable", "final_score", "driver_priority_factor", "driver_priority_score",
        "driver_rank", "candidate_grade", "recommended_use",
    ]
    pd.testing.assert_frame_equal(
        service_ranked[fields].reset_index(drop=True), exported[fields].reset_index(drop=True),
        check_dtype=False,
    )
    expected = pd.DataFrame([
        {"variable": "upstream", "final_score": 0.8967827122876123,
         "driver_priority_factor": 1.0, "driver_priority_score": 0.8967827122876123,
         "driver_rank": 1, "candidate_grade": "A", "recommended_use": "strong_screening_candidate"},
        {"variable": "other", "final_score": 0.4126856616471688,
         "driver_priority_factor": 0.45, "driver_priority_score": 0.18570854774122597,
         "driver_rank": 2, "candidate_grade": "D", "recommended_use": "state_indicator"},
    ])
    pd.testing.assert_frame_equal(exported[fields], expected, check_dtype=False, check_exact=False, rtol=1e-12)


def test_registry_covers_all_production_ranking_fields_and_direct_roles_are_ast_accesses():
    registry = _registry()
    registered = [str(entry["field"]) for entry in registry]
    actual = _production_ranking_fields()
    assert not (actual - set(registered))
    assert len(registered) == len(set(registered))
    direct = {str(entry["field"]) for entry in registry if "direct" in str(entry["score_role"])}
    assert direct <= actual


def test_engineering_context_is_not_accessed_by_any_scoring_or_review_sort_function():
    registry = _registry()
    accessed = set().union(*(
        _field_accesses(module, names) for module, names in RANKING_FUNCTIONS.items()
    ))
    for entry in registry:
        if entry["score_role"] == "not_in_scoring":
            assert str(entry["field"]) not in accessed


def test_four_layer_explanation_fields_are_not_scoring_inputs():
    scoring_functions = {
        "_available_weight_profile_scores",
        "_combine_correlation_evidence",
        "_risk_adjustment",
        "classify_candidate",
        "_finalize_driver_ranking",
        "_grade_candidate",
        "_recommend_use",
        "_recommended_action",
    }
    accessed = _field_accesses(screening, scoring_functions)
    assert not accessed.intersection({
        "evidence_support_items",
        "evidence_against_items",
        "four_layer_missing_items",
        "evidence_conflict_items",
        "four_layer_coverage_status",
        "candidate_summary",
    })


def test_audit_document_and_registry_use_current_coverage_semantics():
    audit = Path("docs/four_layer_evidence_audit.md").read_text(encoding="utf-8")
    registry = _registry()
    assert "evidence_confidence = data_quality_score" in audit
    assert "evidence_score = evidence_strength × evidence_confidence" in audit
    for obsolete in [
        "sqrt(data_quality_score * evidence_" + "completeness)",
        "quality × " + "coverage",
        "evidence_completeness 是直接评分" + "输入",
    ]:
        assert obsolete not in audit
    entries = {entry["field"]: entry for entry in registry}
    for field in [
        "evidence_completeness",
        "evidence_confidence",
        "evidence_coverage_status",
        "evidence_missing_items",
        "four_layer_missing_items",
        "four_layer_coverage_status",
    ]:
        assert field in entries
    assert entries["evidence_missing_items"]["layer"] == "scoring_component_coverage"
    assert entries["four_layer_missing_items"]["layer"] == "four_layer_explanation_coverage"


def _single_evidence_result(*, model_status: str = "not_computed", model_score: float = np.nan, complete: bool = False):
    ranked = pd.DataFrame([{"variable": "x", "score": 0.8, "lag": 1}])
    empty = pd.DataFrame(columns=["variable"])
    model = pd.DataFrame([
        {"variable": "x", "model_lift_score": model_score, "status": model_status}
    ])
    stability = pd.DataFrame([{"variable": "x", "regime_stability_final": 0.8}]) if complete else empty
    lag_quality = pd.DataFrame([{"variable": "x", "lag_quality": 0.8}]) if complete else empty
    rolling = pd.DataFrame([{"variable": "x", "rolling_stability": 0.8}]) if complete else empty
    return final_ranked_features(
        ranked, empty, stability, model,
        pd.DataFrame([{"variable": "x", "risk_flags": "", "data_quality_score": 1.0}]),
        lag_quality, rolling,
    ).iloc[0]


def test_missing_optional_evidence_does_not_reduce_ranking_score():
    association_only = _single_evidence_result()
    complete = _single_evidence_result(model_status="ok", model_score=0.8, complete=True)

    assert association_only["evidence_strength"] == pytest.approx(complete["evidence_strength"])
    assert association_only["evidence_score"] == pytest.approx(complete["evidence_score"])
    assert association_only["final_score"] == pytest.approx(complete["final_score"])
    assert association_only["evidence_completeness"] != complete["evidence_completeness"]
    assert association_only["evidence_coverage_status"] != complete["evidence_coverage_status"]
    assert association_only["evidence_missing_items"] != complete["evidence_missing_items"]


def test_unavailable_model_evidence_is_not_zero_model_evidence():
    not_computed = _single_evidence_result(model_status="not_computed")
    insufficient = _single_evidence_result(model_status="skipped: insufficient rows")
    zero_lift = _single_evidence_result(model_status="ok", model_score=0.0)

    assert pd.isna(not_computed["prediction_score"])
    assert pd.isna(insufficient["prediction_score"])
    assert zero_lift["prediction_score"] == 0.0
    assert not_computed["evidence_score"] == pytest.approx(insufficient["evidence_score"])
    assert zero_lift["evidence_score"] < not_computed["evidence_score"]


def _optional_component_result(component: str, value: float | None) -> pd.Series:
    ranked = pd.DataFrame([{"variable": "x", "score": 0.8, "lag": 1}])
    ranked["innovation_score"] = 0.8 if component != "innovation" else value
    empty = pd.DataFrame(columns=["variable"])
    model = pd.DataFrame([{
        "variable": "x",
        "model_lift_score": 0.8 if component != "prediction" else value,
        "status": "ok" if component != "prediction" or value is not None else "skipped: insufficient rows",
    }])
    stability = pd.DataFrame([{"variable": "x", "regime_stability_final": 0.8 if component != "stability" else value}])
    rolling = pd.DataFrame([{"variable": "x", "rolling_stability": 0.8 if component != "stability" else value}])
    lag_quality = pd.DataFrame([{"variable": "x", "lag_quality": 0.8 if component != "lag_quality" else value}])
    if value is None and component == "stability":
        stability = empty
        rolling = empty
    if value is None and component == "lag_quality":
        lag_quality = empty
    return final_ranked_features(
        ranked, empty, stability, model,
        pd.DataFrame([{"variable": "x", "risk_flags": "", "data_quality_score": 1.0}]),
        lag_quality, rolling,
    ).iloc[0]


@pytest.mark.parametrize(
    ("component", "missing_label"),
    [
        ("prediction", "模型提升"),
        ("stability", "稳定性验证"),
        ("lag_quality", "滞后质量"),
        ("innovation", "变化量验证"),
    ],
)
def test_missing_optional_component_is_not_explicit_zero(component: str, missing_label: str):
    missing = _optional_component_result(component, None)
    zero = _optional_component_result(component, 0.0)

    assert missing["evidence_strength"] > zero["evidence_strength"]
    assert missing["evidence_score"] > zero["evidence_score"]
    assert missing["final_score"] > zero["final_score"]
    assert missing_label in str(missing["evidence_missing_items"])
    assert missing_label not in str(zero["evidence_missing_items"])
    if component == "prediction":
        assert pd.isna(missing["prediction_score"])
        assert zero["prediction_score"] == 0.0
        assert missing["layer4_model_status"] in {"not_available", "insufficient_data"}
        assert zero["layer4_model_status"] == "not_supported"


def test_pr_8c_followup_source_contracts_exclude_coverage_from_scores():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")
    production_source = "\n".join(
        path.read_text(encoding="utf-8") for path in Path("chem_ts_corr").rglob("*.py")
    )

    assert "order = {column: index for index, column in enumerate(frame.columns)}" not in production_source
    assert "anchor, proxy" not in production_source
    assert "** (1 / 3)" not in inspect.getsource(screening._data_quality_score)
    assert 'final["data_quality_score"] * final["evidence_completeness"]' not in source
