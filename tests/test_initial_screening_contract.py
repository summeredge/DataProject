from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

from chem_ts_corr import screening
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.report import build_recommended_candidates, build_markdown_summary
from chem_ts_corr.screening import final_ranked_features
from chem_ts_corr.service import analyze_numeric_frame
from chem_ts_corr.web import INDEX_HTML, _build_result_payload


FORBIDDEN_INITIAL_FIELDS = {
    "layer1_association_status",
    "layer2_temporal_status",
    "layer3_independence_status",
    "layer4_model_status",
    "four_layer_coverage_status",
    "four_layer_missing_items",
    "evidence_support_items",
    "evidence_against_items",
    "evidence_conflict_items",
    "candidate_summary",
}


def _ranked(rows: list[tuple[str, float, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "variable": variable,
                "score": score,
                "lag": lag,
                "direction": "变量领先目标" if lag > 0 else "变量滞后目标" if lag < 0 else "同步变化",
            }
            for variable, score, lag in rows
        ]
    )
    frame["innovation_score"] = frame["score"]
    return frame


def _run_ranked(
    rows: list[tuple[str, float, int]],
    *,
    top_k: int | None = None,
    risk_flags: dict[str, str] | None = None,
) -> pd.DataFrame:
    ranked = _ranked(rows)
    empty = pd.DataFrame(columns=["variable"])
    variables = ranked[["variable", "score"]]
    risks = pd.DataFrame(
        [{"variable": variable, "risk_flags": flags} for variable, flags in (risk_flags or {}).items()]
    )
    return final_ranked_features(
        ranked,
        empty,
        empty,
        variables.rename(columns={"score": "model_lift_score"}).assign(status="ok"),
        risks,
        variables.rename(columns={"score": "lag_quality"}),
        variables.rename(columns={"score": "rolling_stability"}),
        top_k=top_k,
    )


def _inject_opposed_compatibility_priorities(monkeypatch):
    original = screening._finalize_driver_ranking

    def finalize_without_compatibility_ranking(*args, **kwargs):
        kwargs["top_k"] = None
        result = original(*args, **kwargs)
        result["driver_priority_score"] = 1 - result["final_score"]
        result["driver_rank"] = result["final_score"].rank(method="first").astype(int)
        return result

    monkeypatch.setattr(screening, "_finalize_driver_ranking", finalize_without_compatibility_ranking)


def test_initial_screening_orders_candidates_by_final_score_descending(monkeypatch):
    _inject_opposed_compatibility_priorities(monkeypatch)
    result = _run_ranked([("high_final", 0.90, -1), ("low_final", 0.80, 1)])

    assert result.set_index("variable").loc["high_final", "final_score"] > result.set_index("variable").loc["low_final", "final_score"]
    assert result.set_index("variable").loc["high_final", "driver_priority_score"] < result.set_index("variable").loc["low_final", "driver_priority_score"]
    assert result.set_index("variable").loc["high_final", "driver_rank"] > result.set_index("variable").loc["low_final", "driver_rank"]
    assert result["variable"].tolist() == ["high_final", "low_final"]


def test_initial_screening_top_k_uses_final_score_not_driver_priority_score(monkeypatch):
    _inject_opposed_compatibility_priorities(monkeypatch)
    result = _run_ranked([("high_final", 0.90, -1), ("low_final", 0.80, 1)], top_k=1)

    assert result.loc[0, "driver_priority_score"] < 0.5
    assert result.loc[0, "driver_rank"] == 2
    assert result["variable"].tolist() == ["high_final"]


def test_initial_recommendations_preserve_final_score_order(monkeypatch):
    _inject_opposed_compatibility_priorities(monkeypatch)
    ranked = _run_ranked([("high_final", 0.90, -1), ("low_final", 0.80, 1)])
    recommendations = build_recommended_candidates(ranked)

    assert recommendations["variable"].tolist() == ["high_final", "low_final"]
    assert recommendations["final_score"].tolist() == [0.90, 0.80]
    assert recommendations["driver_priority_score"].tolist() == [0.10, 0.20]


def test_closed_loop_legacy_inputs_do_not_change_initial_results():
    clean = _run_ranked([("a", 0.90, 1), ("b", 0.80, 1)])
    legacy = _run_ranked(
        [("a", 0.90, 1), ("b", 0.80, 1)],
        risk_flags={"a": "closed_loop_suspect;closed_loop_confirmed;closed_loop_conflict"},
    )

    fields = ["variable", "final_score", "candidate_grade", "recommended_use"]
    pd.testing.assert_frame_equal(clean[fields], legacy[fields], check_dtype=False)
    config = AnalysisConfig(
        input_path=Path("input.csv"),
        time_column="time",
        target="target",
        output_dir=Path("out"),
        manual_closed_loop_variables=["a"],
        manual_non_closed_loop_variables=["b"],
    )
    assert config.manual_closed_loop_variables == ["a"]


def test_initial_api_filters_four_layer_fields(tmp_path: Path):
    ranked = pd.DataFrame(
        [
            {
                "variable": "x",
                "final_score": 0.8,
                "driver_rank": 1,
                "driver_priority_score": 0.8,
                **{field: "supported" for field in FORBIDDEN_INITIAL_FIELDS if field.endswith("status")},
                "four_layer_missing_items": "",
                "evidence_support_items": "Layer 1",
                "evidence_against_items": "",
                "evidence_conflict_items": "",
                "candidate_summary": "四层解释",
            }
        ]
    )
    ranked.to_csv(tmp_path / "ranked_features.csv", index=False)
    (tmp_path / "summary.md").write_text("# 初步筛选摘要\n", encoding="utf-8")
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path)

    payload = _build_result_payload("run", tmp_path, config)
    assert not (FORBIDDEN_INITIAL_FIELDS & set(payload["rankedFeatures"][0]))
    assert not (FORBIDDEN_INITIAL_FIELDS & set(payload["overview"]["top10"][0]))


def test_initial_web_contract_excludes_four_layer_fields():
    initial_blocks = [
        INDEX_HTML.split("function coreCandidateColumns()", 1)[1].split("function renderCompactDetailTable", 1)[0],
        INDEX_HTML.split("function renderScreeningScoreDetails", 1)[1].split("function timeRelationshipExplanation", 1)[0],
        INDEX_HTML.split("overviewTop:", 1)[1].split("],", 1)[0],
    ]

    for block in initial_blocks:
        assert not any(field in block for field in FORBIDDEN_INITIAL_FIELDS)
    assert '"final_score"' in initial_blocks[0]
    assert '"pearson"' in initial_blocks[0]
    assert '"spearman"' in initial_blocks[0]


def test_initial_analysis_does_not_expose_unexecuted_followup_results(tmp_path: Path):
    ranked = pd.DataFrame([{"variable": "x", "final_score": 0.8, "recommended_use": "manual_review_required"}])
    ranked.to_csv(tmp_path / "ranked_features.csv", index=False)
    (tmp_path / "summary.md").write_text("# 初步筛选摘要\n", encoding="utf-8")
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path)
    payload = _build_result_payload("run", tmp_path, config)

    assert payload["enhancedValidationSummary"] == []
    assert payload["grangerTests"] == []
    assert payload["importance"] == []
    assert not any(
        value in {"supported", "not_supported", "conflicting", "insufficient_data"}
        for row in payload["rankedFeatures"]
        for value in row.values()
    )


def test_initial_service_does_not_execute_followup_analyses():
    source = inspect.getsource(analyze_numeric_frame)

    for call in [
        "model_lift_scores(",
        "rolling_corr_scores(",
        "regime_scores(",
        "fit_explainable_model(",
        "run_granger_tests(",
    ]:
        assert call not in source


def test_initial_screening_source_has_no_four_layer_explanation_call():
    source = inspect.getsource(final_ranked_features)

    assert "add_evidence_explanations" not in source
    assert not any(field in source for field in FORBIDDEN_INITIAL_FIELDS)


def test_summary_restores_initial_screening_positioning():
    ranked = pd.DataFrame([{"variable": "x", "final_score": 0.8, "recommended_use": "manual_review_required"}])
    summary = build_markdown_summary("target", ranked, pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())

    assert "# 初步筛选摘要：target" in summary
    assert "四层工业时序筛查摘要" not in summary
    assert "四层证据解释" not in summary
    assert "驱动因素候选排序" not in summary
