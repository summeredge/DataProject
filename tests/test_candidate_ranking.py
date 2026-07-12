from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr.screening import (
    CLASS_PRIORITY_ADJUSTMENT,
    classify_candidate,
    final_ranked_features,
    risk_flags,
)


def _evaluate(
    rows: list[dict[str, object]],
    risk_flags: dict[str, object] | None = None,
    *,
    top_k: int | None = None,
) -> pd.DataFrame:
    ranked = pd.DataFrame(rows)
    risks = pd.DataFrame(
        [
            {"variable": variable, "risk_flags": flags}
            for variable, flags in (risk_flags or {}).items()
        ],
        columns=["variable", "risk_flags"],
    )
    empty = pd.DataFrame(columns=["variable"])
    return final_ranked_features(
        ranked=ranked,
        residual=empty,
        stability=empty,
        model_lift=empty,
        risks=risks,
        lag_peak_quality=empty,
        rolling_corr_scores=empty,
        top_k=top_k,
    )


def _row(variable: str, score: float, lag: int = 0) -> dict[str, object]:
    return {
        "variable": variable,
        "score": score,
        "lag": lag,
        "direction": "同步变化",
    }


def test_association_rank_ignores_risk_adjustment():
    result = _evaluate(
        [_row("a", 0.9, -1), _row("b", 0.8)],
        {"a": "target_leads_variable", "b": ""},
    ).set_index("variable")

    assert result.loc["a", "association_rank"] == 1
    assert result.loc["b", "association_rank"] == 2


def test_driver_rank_can_reverse_association_order():
    result = _evaluate(
        [_row("a", 0.9), _row("b", 0.8)],
        {"a": "target_leads_variable", "b": ""},
    ).set_index("variable")

    assert result.loc["b", "driver_rank"] == 1
    assert result.loc["a", "driver_rank"] == 2


@pytest.mark.parametrize(
    ("risk", "expected"),
    [
        ("strong_formula_leakage", "formula_or_derived"),
        ("poor_data_quality", "poor_quality"),
        ("target_leads_variable", "downstream_response"),
        ("closed_loop_suspect", "closed_loop_related"),
        ("common_capacity_driver", "capacity_driven"),
    ],
)
def test_candidate_class_risk_priority(risk: str, expected: str):
    row = pd.Series({"lag": -20, "risk_flags": risk})

    assert classify_candidate(row) == expected


def test_candidate_class_uses_declared_risk_priority():
    row = pd.Series(
        {
            "lag": -20,
            "risk_flags": "common_capacity_driver;target_leads_variable;poor_data_quality;strong_formula_leakage",
        }
    )

    assert classify_candidate(row) == "formula_or_derived"


def test_positive_lag_without_risk_is_upstream_driver_candidate():
    assert classify_candidate(pd.Series({"best_lag": 20, "risk_flags": ""})) == "upstream_driver_candidate"


def test_zero_lag_without_risk_is_synchronous_association():
    assert classify_candidate(pd.Series({"best_lag": 0, "risk_flags": ""})) == "synchronous_association"


def test_other_or_missing_lag_is_uncertain_candidate():
    assert classify_candidate(pd.Series({"best_lag": -2, "risk_flags": ""})) == "uncertain_candidate"
    assert classify_candidate(pd.Series({"risk_flags": ""})) == "uncertain_candidate"


def test_upstream_class_is_reachable_through_production_risk_pipeline():
    ranked = pd.DataFrame(
        [{"variable": "x", "score": 0.8, "lag": 1, "direction": "变量领先目标"}]
    )
    empty = pd.DataFrame()
    generated_risks = risk_flags(
        ranked=ranked,
        residual=empty,
        stability=empty,
        diag=empty,
        roles={"x": "PV"},
        control_columns=[],
    )
    variable_only = pd.DataFrame(columns=["variable"])

    result = final_ranked_features(
        ranked=ranked,
        residual=variable_only,
        stability=variable_only,
        model_lift=variable_only,
        risks=generated_risks,
        lag_peak_quality=variable_only,
        rolling_corr_scores=variable_only,
    )

    assert generated_risks.loc[0, "risk_flags"] == ""
    assert result.loc[0, "candidate_class"] == "upstream_driver_candidate"


@pytest.mark.parametrize("candidate_class", CLASS_PRIORITY_ADJUSTMENT)
def test_driver_priority_score_uses_fixed_class_adjustment(candidate_class: str):
    risk_by_class = {
        "downstream_response": "target_leads_variable",
        "closed_loop_related": "closed_loop_suspect",
        "capacity_driven": "common_capacity_driver",
        "formula_or_derived": "strong_formula_leakage",
        "poor_quality": "poor_data_quality",
    }
    lag = 1 if candidate_class == "upstream_driver_candidate" else 0 if candidate_class == "synchronous_association" else -1
    result = _evaluate(
        [_row("x", 0.9, lag)],
        {"x": risk_by_class.get(candidate_class, "")},
    ).iloc[0]

    assert result["candidate_class"] == candidate_class
    expected = min(1.0, max(0.0, result["final_score"] + CLASS_PRIORITY_ADJUSTMENT[candidate_class]))
    assert result["driver_priority_score"] == pytest.approx(expected)


def test_shadow_fields_do_not_change_final_score_or_legacy_behavior():
    result = _evaluate(
        [_row("a", 0.9, -1), _row("b", 0.8)],
        {"a": "target_leads_variable", "b": ""},
    ).set_index("variable")

    assert result.loc["a", "final_score"] == pytest.approx(0.59)
    assert result.loc["b", "final_score"] == pytest.approx(0.8)
    assert result.loc["a", "candidate_grade"] == "C"
    assert result.loc["a", "recommended_use"] == "state_indicator"


def test_output_contains_legacy_and_shadow_fields_in_order():
    result = _evaluate([_row("x", 0.8)])

    for column in ["variable", "final_score", "candidate_grade"]:
        assert column in result.columns
    shadow = ["association_rank", "candidate_class", "driver_priority_score", "driver_rank"]
    assert all(column in result.columns for column in shadow)
    final_index = result.columns.get_loc("final_score")
    assert result.columns[final_index + 1 : final_index + 5].tolist() == shadow


def test_main_output_order_and_topk_use_driver_rank():
    result = _evaluate(
        [_row("a", 0.9), _row("b", 0.8)],
        {"a": "target_leads_variable", "b": ""},
        top_k=1,
    )

    assert result["variable"].tolist() == ["b"]
    assert result.loc[0, "driver_rank"] == 1


def test_equal_scores_keep_original_order_for_both_shadow_ranks():
    result = _evaluate([_row("first", 0.8), _row("second", 0.8)])
    indexed = result.set_index("variable")

    assert indexed.loc["first", "association_rank"] == 1
    assert indexed.loc["second", "association_rank"] == 2
    assert indexed.loc["first", "driver_rank"] == 1
    assert indexed.loc["second", "driver_rank"] == 2


def test_all_inputs_are_not_modified():
    ranked = pd.DataFrame([_row("x", 0.8)])
    risks = pd.DataFrame([{"variable": "x", "risk_flags": ""}])
    frames = [ranked, risks] + [pd.DataFrame(columns=["variable"]) for _ in range(5)]
    before = [frame.copy(deep=True) for frame in frames]

    final_ranked_features(
        ranked=frames[0],
        residual=frames[2],
        stability=frames[3],
        model_lift=frames[4],
        risks=frames[1],
        lag_peak_quality=frames[5],
        rolling_corr_scores=frames[6],
    )

    for actual, expected in zip(frames, before):
        pd.testing.assert_frame_equal(actual, expected)


def test_shadow_output_static_guardrails():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")

    assert "final_score =\ndriver_priority_score" not in source
    assert 'sort_values("driver_rank")' not in source
    assert 'sort_values("driver_priority_score")' not in source
    assert "PRIMARY_RANK_COLUMN" in source
    assert "ascending=True" in source
