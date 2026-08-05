from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import screening
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.report import build_recommended_candidates, build_markdown_summary, write_outputs
from chem_ts_corr.screening import final_ranked_features
from chem_ts_corr.service import analyze_numeric_frame
from chem_ts_corr.web import (
    INDEX_HTML,
    _build_result_payload,
    _order_recommended_candidates,
    _secondary_variables_from_ranked,
)


FORBIDDEN_INITIAL_FIELDS = {
    "evidence_strength",
    "evidence_score",
    "evidence_completeness",
    "evidence_available_count",
    "evidence_coverage_status",
    "evidence_missing_items",
    "evidence_score_low",
    "evidence_score_high",
    "score_method",
    "prediction_score",
    "model_lift_score",
    "model_lift_status",
    "stability_score",
    "stability_status",
    "rolling_stability",
    "rolling_status",
    "regime_stability_final",
    "regime_status",
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

INITIAL_SCREENING_FIELDS = {
    "variable", "driver_rank", "final_score", "pearson", "spearman", "method", "dominant_corr",
    "correlation_direction", "lag", "direction", "lag_quality", "lag_quality_status",
    "lag_boundary_flag", "n", "data_quality_score", "risk_flags", "risk_level",
    "human_reason", "recommended_use", "recommended_action", "force_included",
    "variable_role", "is_residual_control", "is_capacity_reference", "is_segment_reference",
    "innovation_score", "innovation_lag", "innovation_direction", "innovation_sign",
    "innovation_status", "pearson_p", "spearman_p", "pearson_q", "spearman_q",
    "corr_q_value", "pearson_r2", "spearman_r2",
    "association_score", "near_peak_lag_min", "near_peak_lag_max", "near_peak_lag_count",
    "temporal_direction_status", "temporal_penalty_rate", "temporal_score_cap",
    "is_auto_control_reference", "is_control_reference", "control_reference_type",
    "control_reference_source",
}

FOLLOWUP_OUTPUTS = {
    "granger_tests.csv", "shap_or_importance.csv", "residual_corr_scores.csv",
    "regime_scores.csv", "model_lift_scores.csv", "rolling_corr_scores.csv",
    "enhanced_validation_summary.csv", "conditional_granger_scores.csv",
    "causal_review_report.csv", "causal_review_evidence.csv", "final_review_summary.csv",
}

PR3_CANDIDATE_FIELDS = {
    "candidate_source", "selected_by_raw", "selected_by_residual", "raw_candidate_rank",
    "residual_candidate_rank", "candidate_pool_rank", "common_capacity_candidate_flag",
}
PR4_CANDIDATE_FIELDS = {
    "residual_signal_score", "residual_evidence_status", "load_adjusted_relation_status",
    "candidate_priority_tier", "candidate_priority_score", "candidate_priority_rank",
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
    assert result.set_index("variable").loc["high_final", "driver_rank"] < result.set_index("variable").loc["low_final", "driver_rank"]
    assert result["variable"].tolist() == ["high_final", "low_final"]


def test_initial_screening_top_k_does_not_truncate_complete_final_score_order(monkeypatch):
    _inject_opposed_compatibility_priorities(monkeypatch)
    result = _run_ranked([("high_final", 0.90, -1), ("low_final", 0.80, 1)], top_k=1)

    assert result.loc[0, "driver_priority_score"] < 0.5
    assert result.loc[0, "driver_rank"] == 1
    assert result["variable"].tolist() == ["high_final", "low_final"]


def test_initial_recommendations_preserve_final_score_order(monkeypatch):
    _inject_opposed_compatibility_priorities(monkeypatch)
    ranked = _run_ranked([("high_final", 0.90, -1), ("low_final", 0.80, 1)])
    recommendations = build_recommended_candidates(ranked)

    assert recommendations["variable"].tolist() == ["high_final", "low_final"]
    assert recommendations["final_score"].tolist() == [0.90, 0.80]
    assert recommendations["driver_priority_score"].tolist() == pytest.approx([0.10, 0.20])


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
                **{field: "forbidden" for field in FORBIDDEN_INITIAL_FIELDS},
                "pearson": 0.7,
                "lag": 2,
            }
        ]
    )
    ranked.to_csv(tmp_path / "ranked_features.csv", index=False)
    (tmp_path / "summary.md").write_text("# 初步筛选摘要\n", encoding="utf-8")
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path)

    payload = _build_result_payload("run", tmp_path, config)
    assert not (FORBIDDEN_INITIAL_FIELDS & set(payload["rankedFeatures"][0]))
    assert not (FORBIDDEN_INITIAL_FIELDS & set(payload["overview"]["top10"][0]))
    assert set(payload["rankedFeatures"][0]) <= INITIAL_SCREENING_FIELDS
    assert set(payload["overview"]["top10"][0]) <= INITIAL_SCREENING_FIELDS


def _run_complete_initial_output(
    tmp_path: Path,
    force_include: list[str] | None = None,
    residual_controls: bool = True,
):
    rows = 120
    time = np.arange(rows, dtype=float)
    controls = [f"control_{index}" for index in range(8)]
    candidates = [f"candidate_{index}" for index in range(5)]
    frame = pd.DataFrame(
        {
            "target": np.sin(time / 7),
            **{name: np.sin((time + index + 1) / 7) for index, name in enumerate(candidates)},
            **{name: np.cos((time + index + 1) / 7) for index, name in enumerate(controls)},
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="min"),
    )
    config = AnalysisConfig(
        input_path=tmp_path / "input.csv",
        time_column="time",
        target="target",
        output_dir=tmp_path,
        max_lag=3,
        top_k=15,
        residual_control_columns=controls if residual_controls else [],
        force_include_variables=force_include or [],
        enable_model=False,
        skip_model_lift=True,
        skip_rolling_corr=True,
    )
    tables = analyze_numeric_frame(frame, config)
    write_outputs(
        tmp_path,
        config.target,
        tables.ranked_features,
        tables.lag_scores,
        tables.granger_tests,
        tables.importance,
        tables.metrics,
        diagnostics=tables.diagnostics,
        risk_flags=tables.risk_flags,
        lag_peak_quality=tables.lag_peak_quality,
        residual_corr_scores=tables.residual_corr_scores,
        recommended_candidates=tables.recommended_candidates,
    )
    return controls, candidates, _build_result_payload("run", tmp_path, config)


def test_complete_initial_output_separates_candidates_through_payload(tmp_path: Path):
    controls, candidates, payload = _run_complete_initial_output(tmp_path)
    ranked = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")
    recommended = pd.read_csv(tmp_path / "recommended_candidates.csv", encoding="utf-8-sig")
    causal = pd.read_csv(tmp_path / "causal_review_candidates.csv", encoding="utf-8-sig")

    assert len(ranked) == len(payload["rankedFeatures"]) == 13
    assert len(recommended) == len(causal) == len(payload["recommendedCandidates"]) == 5
    assert ranked["variable"].is_unique
    assert recommended["variable"].is_unique
    assert causal["variable"].is_unique
    assert ranked["driver_rank"].tolist() == list(range(1, 14))
    assert [row["variable"] for row in payload["rankedFeatures"]] == ranked["variable"].tolist()
    assert set(controls) <= set(ranked["variable"])
    assert not set(controls) & set(recommended["variable"])
    assert not set(controls) & set(causal["variable"])
    assert set(candidates) == set(recommended["variable"])
    assert payload["overview"]["effective_variables"] == 13
    assert payload["overview"]["recommended_candidate_count"] == 5
    assert payload["overview"]["control_reference_count"] == 8
    assert [row["variable"] for row in payload["overview"]["top10"]] == ranked.head(10)["variable"].tolist()
    assert set(ranked.set_index("variable").loc[controls, "variable_role"]) == {"residual_control"}
    assert not (PR3_CANDIDATE_FIELDS & set(ranked.columns))
    assert not (PR4_CANDIDATE_FIELDS & set(ranked.columns))
    assert all(not (PR3_CANDIDATE_FIELDS & set(row)) for row in payload["rankedFeatures"])
    assert all(not (PR4_CANDIDATE_FIELDS & set(row)) for row in payload["rankedFeatures"])
    assert all(not (PR3_CANDIDATE_FIELDS & set(row)) for row in payload["overview"]["top10"])
    assert all(not (PR4_CANDIDATE_FIELDS & set(row)) for row in payload["overview"]["top10"])
    assert PR3_CANDIDATE_FIELDS <= set(recommended.columns)
    assert PR4_CANDIDATE_FIELDS <= set(recommended.columns)
    assert all(PR3_CANDIDATE_FIELDS <= set(row) for row in payload["recommendedCandidates"])
    assert all(PR4_CANDIDATE_FIELDS <= set(row) for row in payload["recommendedCandidates"])
    assert sorted(recommended["candidate_pool_rank"].tolist()) == list(range(1, len(recommended) + 1))
    assert recommended["candidate_priority_rank"].tolist() == list(range(1, len(recommended) + 1))
    assert [row["variable"] for row in payload["recommendedCandidates"]] == recommended["variable"].tolist()
    assert _secondary_variables_from_ranked(ranked, config=AnalysisConfig(
        tmp_path / "input.csv", "time", "target", tmp_path, top_k=15
    )) == ranked["variable"].tolist()
    source_values = {"raw_only", "residual_only", "raw_and_residual", "force_included", "control_reference"}
    assert set(recommended["candidate_source"]) <= source_values
    assert {row["candidate_source"] for row in payload["recommendedCandidates"]} <= source_values
    residual_statuses = {"strong", "weak", "insufficient", "missing", "control_reference"}
    relation_statuses = {
        "dual_channel_supported", "residual_only_supported", "raw_only_supported",
        "raw_only_common_load_risk", "raw_only_residual_weak", "raw_only_residual_missing",
        "force_included_only", "control_reference",
    }
    assert set(recommended["residual_evidence_status"]) <= residual_statuses
    assert set(recommended["load_adjusted_relation_status"]) <= relation_statuses
    assert {row["residual_evidence_status"] for row in payload["recommendedCandidates"]} <= residual_statuses
    assert {row["load_adjusted_relation_status"] for row in payload["recommendedCandidates"]} <= relation_statuses
    assert PR3_CANDIDATE_FIELDS <= set(causal.columns)
    assert PR4_CANDIDATE_FIELDS <= set(causal.columns)
    candidate_fields = sorted(PR3_CANDIDATE_FIELDS | PR4_CANDIDATE_FIELDS)
    recommended_sources = recommended.set_index("variable")[candidate_fields]
    causal_sources = causal.set_index("variable").loc[recommended_sources.index, candidate_fields]
    pd.testing.assert_frame_equal(recommended_sources, causal_sources, check_dtype=False)

    _run_complete_initial_output(tmp_path / "without_roles", residual_controls=False)
    unmarked = pd.read_csv(tmp_path / "without_roles" / "ranked_features.csv", encoding="utf-8-sig")
    pd.testing.assert_series_equal(
        ranked.set_index("variable").loc[controls, "final_score"],
        unmarked.set_index("variable").loc[controls, "final_score"],
        check_names=False,
    )


def test_forced_control_flows_to_causal_review_without_losing_role(tmp_path: Path):
    controls, _, payload = _run_complete_initial_output(tmp_path, ["control_0"])
    recommended = pd.read_csv(tmp_path / "recommended_candidates.csv", encoding="utf-8-sig")
    causal = pd.read_csv(tmp_path / "causal_review_candidates.csv", encoding="utf-8-sig")
    forced = recommended.set_index("variable").loc["control_0"]

    assert len(payload["rankedFeatures"]) == 13
    assert len(recommended) == len(causal) == len(payload["recommendedCandidates"]) == 6
    assert forced["variable_role"] == "residual_control"
    assert bool(forced["force_included"])
    assert "control_0" in set(causal["variable"])
    assert not set(controls[1:]) & set(recommended["variable"])
    assert not set(controls[1:]) & set(causal["variable"])


def test_result_payload_does_not_truncate_complete_results_at_fifty(tmp_path: Path):
    ranked = pd.DataFrame(
        [
            {"variable": f"x{index}", "final_score": 1 - index / 100, "variable_role": "candidate"}
            for index in range(60)
        ]
    )
    ranked.to_csv(tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig")
    ranked.iloc[:3].to_csv(tmp_path / "recommended_candidates.csv", index=False, encoding="utf-8-sig")
    (tmp_path / "summary.md").write_text("# 初步筛选摘要\n", encoding="utf-8")
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path)

    payload = _build_result_payload("run", tmp_path, config)

    assert len(payload["rankedFeatures"]) == 60
    assert [row["variable"] for row in payload["overview"]["top10"]] == ranked.head(10)["variable"].tolist()


def test_historical_recommended_csv_without_pool_rank_preserves_file_order(tmp_path: Path):
    ranked = pd.DataFrame([
        {"variable": "a", "final_score": .9},
        {"variable": "b", "final_score": .8},
        {"variable": "c", "final_score": .7},
    ])
    recommended = ranked.iloc[[2, 0, 1]].assign(candidate_source="raw_only")
    ranked.to_csv(tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig")
    recommended.to_csv(tmp_path / "recommended_candidates.csv", index=False, encoding="utf-8-sig")
    (tmp_path / "summary.md").write_text("# 初步筛选摘要\n", encoding="utf-8")
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path)

    payload = _build_result_payload("run", tmp_path, config)

    assert [row["variable"] for row in payload["recommendedCandidates"]] == ["c", "a", "b"]


def test_historical_recommended_csv_uses_pool_rank_without_priority_rank(tmp_path: Path):
    ranked = pd.DataFrame([{"variable": value, "final_score": score} for value, score in [("a", .9), ("b", .8), ("c", .7)]])
    recommended = ranked.assign(candidate_pool_rank=[3, 1, 2], candidate_source="raw_only")
    ranked.to_csv(tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig")
    recommended.to_csv(tmp_path / "recommended_candidates.csv", index=False, encoding="utf-8-sig")
    (tmp_path / "summary.md").write_text("# 初步筛选摘要\n", encoding="utf-8")
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path)

    payload = _build_result_payload("run", tmp_path, config)

    assert [row["variable"] for row in payload["recommendedCandidates"]] == ["b", "c", "a"]


def test_valid_residual_output_does_not_change_complete_initial_results(tmp_path: Path, monkeypatch):
    rows = 120
    time = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "target": np.sin(time / 7),
            "candidate": np.sin((time + 2) / 7),
            "control": np.cos(time / 9),
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="min"),
    )
    config = AnalysisConfig(
        tmp_path / "input.csv", "time", "target", tmp_path,
        max_lag=3, top_k=10, residual_control_columns=["control"],
        enable_model=False, skip_model_lift=True, skip_rolling_corr=True,
    )
    with_residual = analyze_numeric_frame(frame, config).ranked_features
    monkeypatch.setattr(screening, "residual_corr_scores", lambda *args, **kwargs: pd.DataFrame(columns=["variable"]))
    without_residual = analyze_numeric_frame(frame, config).ranked_features

    fields = ["variable", "final_score", "driver_rank", "candidate_grade", "recommended_use", "risk_flags"]
    pd.testing.assert_frame_equal(with_residual[fields], without_residual[fields])


def test_initial_web_contract_excludes_four_layer_fields():
    initial_blocks = [
        INDEX_HTML.split("function coreCandidateColumns()", 1)[1].split("}\n", 1)[0],
        INDEX_HTML.split("const INITIAL_SCREENING_DETAIL_COLUMNS", 1)[1].split("];", 1)[0],
        INDEX_HTML.split("overviewTop:", 1)[1].split("],", 1)[0],
    ]

    for block in initial_blocks:
        assert not any(field in block for field in FORBIDDEN_INITIAL_FIELDS)
    assert '"final_score"' in initial_blocks[0]
    assert '"pearson"' in initial_blocks[0]
    assert '"spearman"' in initial_blocks[0]


def test_initial_web_shows_variable_role_and_complete_result_copy():
    payload_source = inspect.getsource(_build_result_payload)

    core_columns = INDEX_HTML.split("function coreCandidateColumns()", 1)[1].split("}", 1)[0]
    overview_columns = INDEX_HTML.split("overviewTop:", 1)[1].split("],", 1)[0]
    detail_columns = INDEX_HTML.split("const INITIAL_SCREENING_DETAIL_COLUMNS", 1)[1].split("];", 1)[0]
    assert core_columns.index('"variable_role"') > core_columns.index('"variable"')
    assert '"variable_role"' in overview_columns
    for field in ["is_residual_control", "is_capacity_reference", "is_segment_reference"]:
        assert f'"{field}"' in detail_columns
    assert 'variable_role: "变量角色"' in INDEX_HTML
    for value in ["普通候选", "残差控制参考", "负荷参考", "工况分段参考"]:
        assert value in INDEX_HTML
    assert "完整初步分析结果" in INDEX_HTML
    assert "默认只展示候选排序结果的核心列和前 50 行" not in INDEX_HTML
    assert "display_ranked.head(50)" not in payload_source
    assert "_initial_screening_frame(recommended).head(50)" not in payload_source


def test_web_residual_candidate_display_has_no_outer_card_or_legacy_label():
    candidate_area = INDEX_HTML.split('<div id="candidatesTab">', 1)[1].split(
        '<div id="trendTab"', 1
    )[0]

    assert '<section id="candidatesTab"' not in INDEX_HTML
    assert "去负荷(残差)验证候选" in candidate_area
    assert "基于原始关联和去负荷后的残差信号筛选得到，用于后续验证排序，不代表因果关系或独立驱动结论。" in candidate_area
    assert "重点候选池" not in INDEX_HTML
    assert "初步分析 Top 10" in INDEX_HTML
    assert "完整初步分析结果" in candidate_area
    assert "结果质量提示" in candidate_area


def test_web_control_reference_table_uses_complete_ranked_rows_and_existing_details():
    candidate_area = INDEX_HTML.split('<div id="candidatesTab">', 1)[1].split(
        '<div id="trendTab"', 1
    )[0]
    render_source = INDEX_HTML.split("function renderAnalysisResult(data)", 1)[1].split(
        "function sleep", 1
    )[0]
    table_source = INDEX_HTML.split("function renderControlReferenceTable(rows)", 1)[1].split(
        "function coreCandidateColumns", 1
    )[0]

    assert candidate_area.index("去负荷(残差)验证候选") < candidate_area.index(
        "控制/负荷参考变量"
    ) < candidate_area.index("完整初步分析结果")
    assert "由已配置的残差控制列、负荷列、分段列，以及位号末尾的.SV/.SP/.MV自动识别。" in candidate_area
    assert "以.PV结尾的变量，不会仅凭前缀自动识别" in candidate_area
    assert '<section id="controlReferenceTable"' not in INDEX_HTML
    assert 'lastRows.filter((row) => row.is_control_reference === true)' in render_source
    assert "renderCompactDetailTable" in table_source
    assert "candidateDetailColumns" in table_source
    assert "当前未识别到控制或负荷参考变量。" in table_source
    for field in [
        "driver_rank", "variable", "control_reference_type", "control_reference_source",
        "final_score", "temporal_direction_status", "data_quality_score", "risk_flags",
    ]:
        assert f'"{field}"' in INDEX_HTML.split("function controlReferenceColumns()", 1)[1].split("}", 1)[0]
    for suffix in [".SV", ".SP", ".MV"]:
        assert suffix in candidate_area


def test_web_candidate_payload_preserves_variable_names(tmp_path: Path):
    candidates = pd.DataFrame(
        [
            {"variable": "AI400014.PV", "final_score": 0.9, "candidate_priority_rank": 1},
            {"variable": "AIC450005.PV", "final_score": 0.8, "candidate_priority_rank": 2},
        ]
    )
    candidates.to_csv(tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(tmp_path / "recommended_candidates.csv", index=False, encoding="utf-8-sig")
    (tmp_path / "summary.md").write_text("# 初步筛选摘要\n", encoding="utf-8")
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path)

    payload = _build_result_payload("run", tmp_path, config)

    assert [row["variable"] for row in payload["recommendedCandidates"]] == [
        "AI400014.PV",
        "AIC450005.PV",
    ]


def test_recommended_web_contract_uses_pool_rank_and_maps_all_candidate_sources():
    payload_source = inspect.getsource(_build_result_payload)
    order_source = inspect.getsource(_order_recommended_candidates)
    display_source = INDEX_HTML.split("function displayCellValue", 1)[1].split("function openTrendForCandidate", 1)[0]

    assert "_order_recommended_candidates(" in payload_source
    assert "order_initial_candidates" not in order_source
    assert "candidate_pool_rank" in order_source
    assert 'candidate_source: "候选来源"' in INDEX_HTML
    assert "return labels[value] || value" in display_source
    for source in ["raw_only", "residual_only", "raw_and_residual", "force_included", "control_reference"]:
        assert source in display_source
    for label in ["全量数据", "去负荷数据", "全量数据和去负荷数据", "人工强制包含", "控制/负荷参考"]:
        assert label in display_source


def test_pr4_web_contract_displays_priority_fields_and_statuses_only_in_recommendations():
    display_source = INDEX_HTML.split("function displayCellValue", 1)[1].split("function openTrendForCandidate", 1)[0]
    recommended_columns = INDEX_HTML.split("function recommendedCandidateColumns()", 1)[1].split("}", 1)[0]
    initial_columns = INDEX_HTML.split("function coreCandidateColumns()", 1)[1].split("}", 1)[0]

    for field in [
        "candidate_priority_rank", "candidate_source", "load_adjusted_relation_status",
        "candidate_priority_score", "residual_signal_score", "residual_evidence_status",
        "common_capacity_candidate_flag", "final_score",
    ]:
        assert f'"{field}"' in recommended_columns
    for field in PR4_CANDIDATE_FIELDS:
        assert field not in initial_columns
    for label in [
        "候选优先级", "候选综合优先分", "去负荷后独立关联强度", "去负荷验证证据",
        "去负荷验证", "共同负荷风险",
    ]:
        assert label in INDEX_HTML
    for enum_value in [
        "strong", "weak", "insufficient", "missing", "control_reference",
        "dual_channel_supported", "residual_only_supported", "raw_only_supported",
        "raw_only_common_load_risk", "raw_only_residual_weak", "raw_only_residual_missing",
        "force_included_only",
    ]:
        assert enum_value in display_source
    for label in [
        "去负荷后独立关联明显", "去负荷后独立关联较弱", "去负荷验证不可计算", "未提供去负荷验证结果",
        "全量数据和去负荷后关联均有支持", "仅在去负荷后发现独立关联", "全量数据关联明显",
        "全量数据关联明显，去负荷后明显减弱", "全量数据关联明显，去负荷后独立关联较弱",
        "全量数据关联明显，本次分析未执行去负荷验证", "人工强制包含",
    ]:
        assert label in display_source
    assert display_source.count("return labels[value] || value") >= 3
    assert "基于原始关联和去负荷后的残差信号筛选得到，用于后续验证排序，不代表因果关系或独立驱动结论。" in INDEX_HTML


def _javascript_function(name: str) -> str:
    source = INDEX_HTML.split(f"function {name}", 1)[1]
    return f"function {name}" + source.split("\n}\n", 1)[0] + "\n}"


def test_initial_tables_preserve_api_order_until_the_user_sorts():
    render_source = INDEX_HTML.split("function renderAnalysisResult(data)", 1)[1].split("function sleep", 1)[0]
    compact_source = INDEX_HTML.split("function renderCompactDetailTable", 1)[1].split("function selectCompactDetailRow", 1)[0]
    for target_id in ['"table"', '"overviewTop"']:
        assert f"delete tableSortStates[{target_id}]" in render_source
        assert f"tableSortStates[{target_id}] = {{ column: \"final_score\"" not in render_source
    assert "delete tableSortStates[controlReferenceTable]" in render_source
    assert 'targetId === candidateTable || targetId === recommendedCandidateTable || targetId === controlReferenceTable || targetId === "overviewTop"' in compact_source
    assert "ensureTableSortState(targetId, preserveInputOrder ? null : columns[0])" in compact_source

    rows = [
        {"variable": "b", "final_score": 0.7},
        {"variable": "c", "final_score": 0.9},
        {"variable": "a", "final_score": 0.8},
    ]
    script = "\n".join([
        "const tableSortStates = { table: undefined };",
        _javascript_function("sortedRowsForTable"),
        _javascript_function("compareValues"),
        f"const rows = {json.dumps(rows)};",
        "const original = rows.map((row) => row.variable);",
        "const none = sortedRowsForTable('table', rows).map((row) => row.variable);",
        "tableSortStates.table = { column: 'final_score', direction: 'asc' };",
        "const asc = sortedRowsForTable('table', rows).map((row) => row.variable);",
        "tableSortStates.table = { column: 'final_score', direction: 'desc' };",
        "const desc = sortedRowsForTable('table', rows).map((row) => row.variable);",
        "console.log(JSON.stringify({ original, none, asc, desc }));",
    ])
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    actual = json.loads(result.stdout)

    assert actual == {
        "original": ["b", "c", "a"],
        "none": ["b", "c", "a"],
        "asc": ["b", "a", "c"],
        "desc": ["c", "a", "b"],
    }


def test_initial_score_details_only_reference_whitelisted_fields():
    source = INDEX_HTML.split("function renderScreeningScoreDetails(row)", 1)[1].split("function lagProfileCacheKey", 1)[0]
    detail_modal_source = INDEX_HTML.split("function renderGenericDetailModalBody", 1)[1].split("function timeRelationshipExplanation", 1)[0]

    for forbidden in ["evidence_strength", "evidence_score", "证据强度", "证据覆盖度", "证据完整度"]:
        assert forbidden not in source
    for allowed in ["final_score", "data_quality_score", "risk_flags", "recommended_use", "recommended_action"]:
        assert allowed in source
    assert "final_score 是初步分析中实际可用统计证据经过风险处理后的综合筛选得分。" not in source
    assert "初步筛选得分" in source
    assert "evidence_strength" not in detail_modal_source
    assert "evidence_score" not in detail_modal_source


def test_web_final_score_explanations_match_the_initial_scoring_contract():
    expected = "final_score 是基础关联强度经过数据质量、明确风险和时间方向约束后的初步筛选得分。"
    old = "final_score 是当前初步分析可用统计证据、滞后质量、数据质量经过风险处理后的综合筛选得分。"
    explanations = [
        line.strip()
        for line in INDEX_HTML.splitlines()
        if "final_score 是" in line
    ]

    assert explanations
    assert all(expected in explanation for explanation in explanations)
    assert old not in INDEX_HTML
    for forbidden in ["lag_quality", "滞后质量", "创新得分", "残差证据", "稳定性", "模型结果"]:
        assert all(forbidden not in explanation for explanation in explanations)


def test_initial_detail_columns_are_whitelisted():
    detail_source = INDEX_HTML.split("function candidateDetailColumns(row)", 1)[1].split("function renderCandidateTable", 1)[0]

    assert "Object.keys(row" not in detail_source
    assert "INITIAL_SCREENING_DETAIL_COLUMNS" in detail_source
    assert "driver_priority_score" not in detail_source


def test_initial_write_outputs_does_not_create_followup_placeholders(tmp_path: Path):
    ranked = pd.DataFrame([{"variable": "x", "final_score": 0.8, "candidate_grade": "A", "recommended_use": "prediction_candidate"}])
    write_outputs(
        tmp_path, "target", ranked, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {},
    )

    assert not {path.name for path in tmp_path.iterdir()} & FOLLOWUP_OUTPUTS
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path)
    payload = _build_result_payload("run", tmp_path, config)
    assert not ({item["name"] for item in payload["downloads"]} & FOLLOWUP_OUTPUTS)
    assert payload["enhancedValidationSummary"] == []
    assert payload["grangerTests"] == []
    assert payload["importance"] == []


def test_initial_tie_break_order_is_independent_of_input_order(monkeypatch):
    original = screening._finalize_driver_ranking

    def force_tied_final_scores(frame, *args, **kwargs):
        result = original(frame, *args, **kwargs)
        result["final_score"] = 0.8
        result["association_score"] = result["variable"].map({"a": 0.7, "b": 0.8, "c": 0.8})
        result["lag_quality"] = result["variable"].map({"a": 0.9, "b": 0.7, "c": 0.7})
        result["driver_priority_score"] = result["variable"].map({"a": 0.9, "b": 0.1, "c": 0.2})
        return result

    monkeypatch.setattr(screening, "_finalize_driver_ranking", force_tied_final_scores)
    result = _run_ranked([("c", 0.8, 1), ("a", 0.8, 1), ("b", 0.8, 1)])

    assert result["variable"].tolist() == ["b", "c", "a"]
    assert result["driver_rank"].tolist() == [1, 2, 3]


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
    service_source = Path("chem_ts_corr/service.py").read_text(encoding="utf-8")
    report_source = Path("chem_ts_corr/report.py").read_text(encoding="utf-8")
    payload_source = inspect.getsource(_build_result_payload)
    assert "def _candidate_list(" not in service_source
    assert "top_filtered = [v for v in top if v not in excluded]" not in service_source
    assert "build_causal_review_candidates(ranked_features)" not in report_source
    assert "display_ranked.head(50)" not in payload_source
    assert "_initial_screening_frame(recommended).head(50)" not in payload_source


def test_initial_screening_source_has_no_four_layer_explanation_call():
    source = inspect.getsource(final_ranked_features)

    assert "add_evidence_explanations" not in source
    assert not any(field in source for field in {
        "layer1_association_status", "layer2_temporal_status", "layer3_independence_status",
        "layer4_model_status", "four_layer_coverage_status", "four_layer_missing_items",
        "evidence_support_items", "evidence_against_items", "evidence_conflict_items", "candidate_summary",
    })


def test_summary_restores_initial_screening_positioning():
    ranked = pd.DataFrame([{"variable": "x", "final_score": 0.8, "recommended_use": "manual_review_required"}])
    summary = build_markdown_summary("target", ranked, pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())

    assert "# 初步筛选摘要：target" in summary
    assert "四层工业时序筛查摘要" not in summary
    assert "四层证据解释" not in summary
    assert "驱动因素候选排序" not in summary
