from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.screening import (
    CLASS_PRIORITY_ADJUSTMENT,
    EVIDENCE_COMPONENT_WEIGHTS,
    PRIMARY_RANK_COLUMN,
    PRIMARY_SCORE_COLUMN,
    final_ranked_features,
    risk_flags,
)


def _ranked(rows: list[tuple[str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": variable, "score": score, "lag": lag, "direction": ""}
            for variable, score, lag in rows
        ]
    )


def _risks(flags: dict[str, str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [{"variable": variable, "risk_flags": value} for variable, value in (flags or {}).items()],
        columns=["variable", "risk_flags"],
    )


def _run_ranked(
    rows: list[tuple[str, float, int]],
    flags: dict[str, str] | None = None,
    *,
    top_k: int | None = None,
    force_include_variables: list[str] | None = None,
    control_columns: list[str] | None = None,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["variable"])
    return final_ranked_features(
        ranked=_ranked(rows),
        residual=empty,
        stability=empty,
        model_lift=empty,
        risks=_risks(flags),
        lag_peak_quality=empty,
        rolling_corr_scores=empty,
        top_k=top_k,
        force_include_variables=force_include_variables,
        control_columns=control_columns,
    )


def _reversal(top_k: int | None = None) -> pd.DataFrame:
    return _run_ranked(
        [("a", 0.90, -1), ("b", 0.58, 1)],
        {"a": "target_leads_variable", "b": ""},
        top_k=top_k,
    )


def test_primary_rank_constants_are_fixed():
    assert PRIMARY_RANK_COLUMN == "driver_rank"
    assert PRIMARY_SCORE_COLUMN == "driver_priority_score"


def test_main_order_is_driver_rank_not_final_score():
    result = _reversal()
    indexed = result.set_index("variable")

    assert indexed.loc["a", "final_score"] > indexed.loc["b", "final_score"]
    assert indexed.loc["b", "driver_priority_score"] > indexed.loc["a", "driver_priority_score"]
    assert indexed.loc["b", "driver_rank"] < indexed.loc["a", "driver_rank"]
    assert result["variable"].tolist() == ["b", "a"]


def test_topk_selects_smallest_driver_rank():
    result = _reversal(top_k=1)

    assert result["variable"].tolist() == ["b"]


def test_association_and_driver_ranks_keep_distinct_semantics():
    result = _reversal().set_index("variable")

    assert result.loc["a", "association_rank"] < result.loc["b", "association_rank"]
    assert result.loc["b", "driver_rank"] < result.loc["a", "driver_rank"]


def test_final_score_and_candidate_grade_are_unchanged_by_sort_switch():
    row = _reversal().set_index("variable").loc["a"]

    assert row["evidence_score"] == pytest.approx(0.90)
    assert row["risk_penalty"] == pytest.approx(0.10)
    assert row["risk_score_cap"] == pytest.approx(0.59)
    assert row["final_score"] == pytest.approx(0.59)
    assert row["driver_priority_score"] == pytest.approx(0.29)
    assert row["candidate_grade"] == "C"


def test_equal_driver_scores_keep_input_order():
    result = _run_ranked([("x", 0.8, 1), ("y", 0.8, 1)])

    assert result["variable"].tolist() == ["x", "y"]
    assert result["driver_rank"].tolist() == [1, 2]


def test_without_topk_driver_rank_is_monotonic_and_final_score_need_not_be():
    result = _run_ranked(
        [("down", 0.9, -1), ("up", 0.58, 1), ("sync", 0.7, 0), ("capacity", 0.8, 1)],
        {"down": "target_leads_variable", "capacity": "common_capacity_driver"},
    )

    assert result["driver_rank"].is_monotonic_increasing
    assert not result["final_score"].is_monotonic_decreasing


def test_force_include_keeps_global_rank_and_final_order():
    result = _run_ranked(
        [("a", 0.9, 1), ("b", 0.8, 1), ("c", 0.7, 1)],
        top_k=1,
        force_include_variables=["c"],
    )

    assert result["variable"].tolist() == ["a", "c"]
    assert result["driver_rank"].tolist() == [1, 3]
    assert bool(result.set_index("variable").loc["c", "force_included"]) is True


def test_force_include_duplicate_is_deduplicated():
    result = _run_ranked(
        [("a", 0.9, 1), ("b", 0.8, 1)], top_k=1, force_include_variables=["a"]
    )

    assert result["variable"].tolist() == ["a"]
    assert bool(result.loc[0, "force_included"]) is True


def test_topk_zero_returns_only_forced_global_rank():
    result = _run_ranked(
        [("a", 0.9, 1), ("b", 0.8, 1), ("c", 0.7, 1)],
        top_k=0,
        force_include_variables=["c"],
    )

    assert result["variable"].tolist() == ["c"]
    assert result["driver_rank"].tolist() == [3]


def test_control_variable_is_excluded_from_automatic_topk():
    result = _run_ranked(
        [("ctrl", 0.9, 1), ("a", 0.8, 1)], top_k=1, control_columns=["ctrl"]
    )

    assert result["variable"].tolist() == ["a"]
    assert result.loc[0, "driver_rank"] == 2


def test_forced_control_variable_is_kept_and_sorted_by_global_rank():
    result = _run_ranked(
        [("ctrl", 0.9, 1), ("a", 0.8, 1)],
        top_k=1,
        force_include_variables=["ctrl"],
        control_columns=["ctrl"],
    )

    assert result["variable"].tolist() == ["ctrl", "a"]
    assert result["driver_rank"].tolist() == [1, 2]
    assert result.set_index("variable").loc["ctrl", "recommended_use"] == "control_variable_reference"


def test_topk_does_not_compress_global_driver_rank():
    rows = [(f"x{index}", 1.0 - index * 0.1, 1) for index in range(5)]
    result = _run_ranked(rows, top_k=2, force_include_variables=["x4"])

    assert result["driver_rank"].tolist() == [1, 2, 5]


def test_risk_penalty_affects_formal_driver_rank():
    result = _run_ranked(
        [("safe", 0.8, 1), ("risky", 0.8, -1)],
        {"safe": "", "risky": "target_leads_variable"},
    ).set_index("variable")

    assert result.loc["safe", "final_score"] > result.loc["risky", "final_score"]
    assert result.loc["safe", "driver_priority_score"] > result.loc["risky", "driver_priority_score"]
    assert result.loc["safe", "driver_rank"] < result.loc["risky", "driver_rank"]


def test_class_adjustments_determine_formal_order_at_equal_final_score():
    result = _run_ranked(
        [("up", 0.59, 1), ("sync", 0.59, 0), ("capacity", 0.71, 1), ("down", 0.90, -1)],
        {"capacity": "common_capacity_driver", "down": "target_leads_variable"},
    )

    assert result["variable"].tolist() == ["up", "sync", "capacity", "down"]
    assert result["final_score"].tolist() == pytest.approx([0.59] * 4)
    for _, row in result.iterrows():
        assert row["driver_priority_score"] == pytest.approx(
            row["final_score"] + CLASS_PRIORITY_ADJUSTMENT[row["candidate_class"]]
        )


def test_direction_semantics_survive_production_risk_pipeline():
    ranked = _ranked([("up", 0.8, 1), ("down", 0.8, -1), ("sync", 0.8, 0)])
    empty = pd.DataFrame()
    generated = risk_flags(ranked, empty, empty, empty, {name: "PV" for name in ranked["variable"]}, [])
    variable_only = pd.DataFrame(columns=["variable"])
    result = final_ranked_features(
        ranked, variable_only, variable_only, variable_only, generated, variable_only, variable_only
    ).set_index("variable")

    assert result.loc["up", "candidate_class"] == "upstream_driver_candidate"
    assert result.loc["down", "candidate_class"] == "downstream_response"
    assert result.loc["sync", "candidate_class"] == "synchronous_association"


def test_pr5_and_pr6_scoring_guards_remain_active():
    source = inspect.getsource(final_ranked_features)
    parts = source.split("parts = {", 1)[1].split("}", 1)[0]
    insufficient = pd.DataFrame(
        [{"variable": "x", "regime_stability_final": np.nan, "regime_evidence_status": "insufficient_regimes"}]
    )
    variable_only = pd.DataFrame(columns=["variable"])
    result = final_ranked_features(
        _ranked([("x", 0.8, 1)]), variable_only, insufficient, variable_only,
        variable_only, variable_only, variable_only,
    ).iloc[0]

    assert parts.count('"correlation":') == 1
    assert '"raw":' not in parts and '"residual":' not in parts
    assert EVIDENCE_COMPONENT_WEIGHTS["regime"] == pytest.approx(0.15)
    assert pd.isna(result["regime_stability_final"])
    assert result["regime_status"] == "insufficient_regimes"


def test_output_fields_remain_compatible():
    result = _run_ranked([("x", 0.8, 1)])
    required = {
        "variable", "raw_corr", "correlation_evidence_score", "regime_stability_final",
        "evidence_score", "risk_penalty", "risk_score_cap", "final_score", "association_rank",
        "candidate_class", "driver_priority_score", "driver_rank", "candidate_grade",
        "recommended_use", "force_included",
    }

    assert required.issubset(result.columns)


def test_inputs_are_not_modified():
    frames = [_ranked([("x", 0.8, 1)])] + [pd.DataFrame(columns=["variable"]) for _ in range(6)]
    before = [frame.copy(deep=True) for frame in frames]

    final_ranked_features(*frames)

    for actual, expected in zip(frames, before):
        pd.testing.assert_frame_equal(actual, expected)


def test_old_final_score_primary_sort_is_absent():
    source = inspect.getsource(final_ranked_features)

    assert 'sort_values("final_score"' not in source
    assert "sort_values('final_score'" not in source
    assert source.count("PRIMARY_RANK_COLUMN") >= 2
    assert "ascending=True" in source
    assert ".head(top_k)" in source


def _web_source() -> str:
    return Path("chem_ts_corr/web.py").read_text(encoding="utf-8")


def _javascript_function(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def test_web_primary_table_default_sort_is_driver_rank_ascending():
    source = _web_source()

    assert 'table: { column: "driver_rank", direction: "asc" }' in source


def test_web_old_final_score_default_sort_is_absent():
    source = _web_source()

    assert 'table: { column: "final_score", direction: "desc" }' not in source
    assert 'tableSortStates["table"] = { column: "final_score", direction: "desc" }' not in source


def test_new_analysis_resets_primary_sort_before_rendering_rows():
    body = _javascript_function(_web_source(), "renderAnalysisResult(data) {", "sleep")
    reset_sort = 'tableSortStates["table"] = { column: "driver_rank", direction: "asc" };'

    assert reset_sort in body
    assert body.index(reset_sort) < body.index("renderTable(lastRows);")


def test_web_reset_restores_driver_rank_primary_sort():
    body = _web_source().split("function reset() {", 1)[1].split("\n}", 1)[0]

    assert 'table: { column: "driver_rank", direction: "asc" }' in body
    assert 'table: { column: "final_score", direction: "desc" }' not in body


def test_final_review_summary_default_sort_is_unchanged():
    source = _web_source()

    assert 'finalReviewSummaryTable: { column: "final_rank", direction: "asc" }' in source


def test_web_sortable_header_interactions_are_preserved():
    source = _web_source()

    for expected in [
        'class="sortable"',
        'header.addEventListener("click", sort)',
        "updateTableSortState(targetId, header.dataset.column)",
        'state.direction = state.direction === "asc" ? "desc" : "asc"',
        "tableSortStates[targetId]",
    ]:
        assert expected in source
