from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.screening import (
    PRIMARY_RANK_COLUMN,
    PRIMARY_SCORE_COLUMN,
    build_recommended_candidates,
    final_ranked_features,
    order_initial_candidates,
    risk_flags,
)
from chem_ts_corr.web import _overview_payload


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
    ranked = _ranked(rows)
    ranked["innovation_score"] = ranked["score"]
    complete = ranked[["variable", "score"]]
    return final_ranked_features(
        ranked=ranked,
        residual=empty,
        stability=empty,
        model_lift=complete.rename(columns={"score": "model_lift_score"}).assign(status="ok"),
        risks=_risks(flags),
        lag_peak_quality=complete.rename(columns={"score": "lag_quality"}),
        rolling_corr_scores=complete.rename(columns={"score": "rolling_stability"}),
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


def test_primary_rank_constants_are_final_score():
    assert PRIMARY_RANK_COLUMN == "final_score"
    assert PRIMARY_SCORE_COLUMN == "final_score"


def test_main_order_is_final_score_descending():
    result = _reversal()
    indexed = result.set_index("variable")

    assert indexed.loc["a", "final_score"] > indexed.loc["b", "final_score"]
    assert result["variable"].tolist() == ["a", "b"]
    assert result["driver_rank"].tolist() == [1, 2]


def test_overview_payload_orders_by_final_score():
    ranked = pd.DataFrame(
        [
            {"variable": "risky", "driver_rank": 2, "driver_priority_score": 0.40, "final_score": 0.90},
            {"variable": "safe", "driver_rank": 1, "driver_priority_score": 0.70, "final_score": 0.70},
        ]
    )

    overview = _overview_payload(ranked, pd.DataFrame(), SimpleNamespace(target="target"), {})

    assert [row["variable"] for row in overview["top10"]] == ["risky", "safe"]


def test_topk_does_not_truncate_complete_ranking():
    result = _reversal(top_k=1)

    assert result["variable"].tolist() == ["a", "b"]
    assert result["driver_rank"].tolist() == [1, 2]


def test_association_and_primary_ranks_follow_same_score_order():
    result = _reversal().set_index("variable")

    assert result.loc["a", "association_rank"] < result.loc["b", "association_rank"]
    assert result.loc["a", "driver_rank"] < result.loc["b", "driver_rank"]


def test_engineering_direction_does_not_change_initial_score_order():
    row = _reversal().set_index("variable").loc["a"]

    assert row["evidence_score"] == pytest.approx(0.90)
    assert row["final_score"] == pytest.approx(0.90)
    assert row["driver_priority_factor"] == pytest.approx(1.0)
    assert row["driver_priority_score"] == pytest.approx(row["final_score"])


def test_equal_scores_keep_input_order():
    result = _run_ranked([("x", 0.8, 1), ("y", 0.8, 1)])

    assert result["variable"].tolist() == ["x", "y"]
    assert result["driver_rank"].tolist() == [1, 2]


def test_complete_ranking_is_stable_when_input_column_order_changes():
    first = _run_ranked([("b", 0.8, 1), ("a", 0.8, 1)])
    second = _run_ranked([("a", 0.8, 1), ("b", 0.8, 1)])

    assert first["variable"].tolist() == second["variable"].tolist() == ["a", "b"]


def test_final_score_is_monotonic_without_topk():
    result = _run_ranked(
        [("down", 0.9, -1), ("up", 0.58, 1), ("sync", 0.7, 0), ("capacity", 0.8, 1)],
        {"down": "target_leads_variable", "capacity": "common_capacity_driver"},
    )

    assert result["final_score"].is_monotonic_decreasing
    assert result["driver_rank"].is_monotonic_increasing


def test_force_include_marks_complete_ranking_without_changing_it():
    result = _run_ranked(
        [("a", 0.9, 1), ("b", 0.8, 1), ("c", 0.7, 1)],
        top_k=1,
        force_include_variables=["c"],
    )

    assert result["variable"].tolist() == ["a", "b", "c"]
    assert result["driver_rank"].tolist() == [1, 2, 3]
    assert bool(result.set_index("variable").loc["c", "force_included"])


def test_force_include_does_not_remove_other_complete_results():
    result = _run_ranked(
        [("a", 0.9, 1), ("b", 0.8, 1)], top_k=1, force_include_variables=["a"]
    )

    assert result["variable"].tolist() == ["a", "b"]
    assert bool(result.loc[0, "force_included"])


def test_topk_zero_does_not_truncate_complete_ranking():
    result = _run_ranked(
        [("a", 0.9, 1), ("b", 0.8, 1), ("c", 0.7, 1)],
        top_k=0,
        force_include_variables=["c"],
    )

    assert result["variable"].tolist() == ["a", "b", "c"]
    assert result["driver_rank"].tolist() == [1, 2, 3]


def test_control_variable_remains_in_complete_ranking_with_role():
    result = _run_ranked(
        [("ctrl", 0.9, 1), ("a", 0.8, 1)], top_k=1, control_columns=["ctrl"]
    )

    assert result["variable"].tolist() == ["ctrl", "a"]
    assert result["driver_rank"].tolist() == [1, 2]
    control = result.set_index("variable").loc["ctrl"]
    assert bool(control["is_residual_control"])
    assert control["variable_role"] == "residual_control"


def test_forced_control_variable_is_kept_and_sorted_by_final_score():
    result = _run_ranked(
        [("ctrl", 0.9, 1), ("a", 0.8, 1)],
        top_k=1,
        force_include_variables=["ctrl"],
        control_columns=["ctrl"],
    )

    assert result["variable"].tolist() == ["ctrl", "a"]
    assert result["driver_rank"].tolist() == [1, 2]
    assert result.set_index("variable").loc["ctrl", "variable_role"] == "residual_control"


def test_topk_does_not_compress_global_rank():
    rows = [(f"x{index}", 1.0 - index * 0.1, 1) for index in range(5)]
    result = _run_ranked(rows, top_k=2, force_include_variables=["x4"])

    assert result["driver_rank"].tolist() == [1, 2, 3, 4, 5]


def test_recommended_candidates_exclude_references_and_keep_forced_reference():
    result = _run_ranked(
        [("ctrl", 0.9, 1), ("a", 0.8, 1), ("b", 0.7, 1)],
        control_columns=["ctrl"],
    )

    candidates = build_recommended_candidates(result, 1)
    forced = build_recommended_candidates(result, 1, ["ctrl"])

    assert candidates["variable"].tolist() == ["a"]
    assert forced["variable"].tolist() == ["a", "ctrl"]
    assert forced.set_index("variable").loc["ctrl", "variable_role"] == "residual_control"
    candidates.loc[:, "variable"] = "changed"
    assert result["variable"].tolist() == ["ctrl", "a", "b"]


def test_complete_ranking_and_candidate_pool_have_independent_fixed_sizes():
    controls = [f"control_{index}" for index in range(8)]
    ordinary = [f"candidate_{index}" for index in range(5)]
    rows = [(name, 0.95 - index * 0.01, 1) for index, name in enumerate(controls + ordinary)]

    complete = _run_ranked(rows, top_k=15, control_columns=controls)
    candidates = build_recommended_candidates(complete, 15)
    baseline = _run_ranked(rows, top_k=15)

    assert len(complete) == 13
    assert len(candidates) == 5
    assert complete["driver_rank"].tolist() == list(range(1, 14))
    assert set(complete.loc[complete["is_residual_control"], "variable"]) == set(controls)
    assert set(candidates["variable"]) == set(ordinary)
    assert complete.set_index("variable").loc[controls, "final_score"].tolist() == pytest.approx(
        baseline.set_index("variable").loc[controls, "final_score"].tolist()
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


def test_missing_regime_remains_missing_without_reviving_weight_normalization():
    source = inspect.getsource(final_ranked_features)
    insufficient = pd.DataFrame(
        [{"variable": "x", "regime_stability_final": np.nan, "regime_evidence_status": "insufficient_regimes"}]
    )
    variable_only = pd.DataFrame(columns=["variable"])
    result = final_ranked_features(
        _ranked([("x", 0.8, 1)]), variable_only, insufficient, variable_only,
        variable_only, variable_only, variable_only,
    ).iloc[0]

    assert "EVIDENCE_COMPONENT_WEIGHTS" not in source
    assert "den.replace(0" not in source
    assert "regime_stability_final" not in result
    assert "regime_status" not in result


def test_output_fields_remain_compatible_without_four_layer_fields():
    result = _run_ranked([("x", 0.8, 1)])
    required = {
        "variable", "raw_corr", "correlation_evidence_score",
        "evidence_score", "evidence_completeness", "evidence_confidence",
        "data_quality_score",
        "risk_penalty", "risk_score_cap", "final_score", "association_rank",
        "candidate_class", "driver_priority_factor", "driver_priority_score", "driver_rank",
        "candidate_grade", "recommended_use", "force_included",
    }

    assert required.issubset(result.columns)
    assert not any(column.startswith("layer") or column.startswith("four_layer") for column in result.columns)
    assert "candidate_summary" not in result.columns


def test_inputs_are_not_modified():
    frames = [_ranked([("x", 0.8, 1)])] + [pd.DataFrame(columns=["variable"]) for _ in range(6)]
    before = [frame.copy(deep=True) for frame in frames]

    final_ranked_features(*frames)

    for actual, expected in zip(frames, before):
        pd.testing.assert_frame_equal(actual, expected)


def test_final_score_primary_sort_is_explicit():
    source = inspect.getsource(order_initial_candidates)

    assert '"_initial_final_score"' in source
    assert '"_initial_association_score"' in source
    assert '"_initial_lag_quality"' in source
    assert 'sort_values("driver_rank"' not in source
    assert 'sort_values("driver_priority_score"' not in source
    assert ".head(top_k)" not in inspect.getsource(final_ranked_features)


def _web_source() -> str:
    return Path("chem_ts_corr/web.py").read_text(encoding="utf-8")


def _javascript_function(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def test_web_primary_table_default_sort_is_final_score_descending():
    source = _web_source()

    assert 'table: { column: "final_score", direction: "desc" }' in source
    assert 'table: { column: "driver_rank", direction: "asc" }' not in source
    assert 'sort_values("driver_rank"' not in source


def test_new_analysis_clears_primary_table_sort_before_rendering_rows():
    body = _javascript_function(_web_source(), "renderAnalysisResult(data) {", "sleep")
    reset_sort = 'delete tableSortStates["table"];'

    assert reset_sort in body
    assert body.index(reset_sort) < body.index("renderTable(lastRows);")


def test_web_reset_restores_final_score_primary_sort():
    body = _web_source().split("function reset() {", 1)[1].split("\n}", 1)[0]

    assert 'table: { column: "final_score", direction: "desc" }' in body


def test_final_review_summary_default_sort_is_unchanged():
    assert 'finalReviewSummaryTable: { column: "final_rank", direction: "asc" }' in _web_source()


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
