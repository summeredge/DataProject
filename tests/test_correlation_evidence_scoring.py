from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.screening import (
    INDUSTRIAL_SCORE_WEIGHT_PROFILES,
    final_ranked_features,
    risk_flags,
)


def _frame(values: dict[str, object] | None = None) -> pd.DataFrame:
    if values is None:
        return pd.DataFrame(columns=["variable"])
    return pd.DataFrame([values])


def _evaluate(
    ranked_rows: list[dict[str, object]],
    *,
    residual: pd.DataFrame | None = None,
    stability: pd.DataFrame | None = None,
    model_lift: pd.DataFrame | None = None,
    risks: pd.DataFrame | None = None,
    lag_peak: pd.DataFrame | None = None,
    rolling: pd.DataFrame | None = None,
    top_k: int | None = None,
    force_include: list[str] | None = None,
) -> pd.DataFrame:
    return final_ranked_features(
        ranked=pd.DataFrame(ranked_rows),
        residual=_frame() if residual is None else residual,
        stability=_frame() if stability is None else stability,
        model_lift=_frame() if model_lift is None else model_lift,
        risks=_frame() if risks is None else risks,
        lag_peak_quality=_frame() if lag_peak is None else lag_peak,
        rolling_corr_scores=_frame() if rolling is None else rolling,
        top_k=top_k,
        force_include_variables=force_include,
    )


def _ranked(variable: str, score: float, lag: int = 0) -> dict[str, object]:
    return {"variable": variable, "score": score, "innovation_score": score, "lag": lag, "direction": "同步变化"}


def test_v1_fixed_weights_are_replaced_by_robust_profiles():
    assert len(INDUSTRIAL_SCORE_WEIGHT_PROFILES) > 10
    assert all(sum(profile.values()) == pytest.approx(1.0) for profile in INDUSTRIAL_SCORE_WEIGHT_PROFILES)


def test_verified_residual_uses_fixed_family_formula():
    row = _evaluate(
        [_ranked("x", 0.9)],
        residual=_frame({"variable": "x", "residual_corr": 0.3}),
    ).iloc[0]
    expected = np.sqrt(0.9 * 0.3)

    assert row["association_score"] == pytest.approx(0.9)
    assert row["independent_signal_score"] == pytest.approx(0.3)
    assert row["correlation_evidence_score"] == pytest.approx(expected)
    assert row["correlation_evidence_status"] == "innovation_and_independent_verified"


def test_missing_residual_uses_association_only():
    row = _evaluate([_ranked("x", 0.85)]).iloc[0]

    assert row["association_score"] == pytest.approx(0.85)
    assert pd.isna(row["independent_signal_score"])
    assert row["correlation_evidence_score"] == pytest.approx(0.85)
    assert row["correlation_evidence_status"] == "innovation_verified"


def test_zero_residual_is_valid_independent_signal():
    row = _evaluate(
        [_ranked("x", 0.8)],
        residual=_frame({"variable": "x", "residual_corr": 0.0}),
    ).iloc[0]

    assert row["independent_signal_score"] == 0.0
    assert row["correlation_evidence_status"] == "innovation_and_independent_verified"
    assert row["correlation_evidence_score"] == 0.0


@pytest.mark.parametrize(
    ("raw", "residual", "expected"),
    [(0.7, 0.7, 0.7), (0.4, 0.8, np.sqrt(0.4 * 0.8))],
)
def test_family_formula_handles_equal_or_higher_residual(raw: float, residual: float, expected: float):
    row = _evaluate(
        [_ranked("x", raw)],
        residual=_frame({"variable": "x", "residual_corr": residual}),
    ).iloc[0]

    assert row["correlation_evidence_score"] == pytest.approx(expected)


def test_independent_signal_can_reverse_raw_correlation_order():
    result = _evaluate(
        [_ranked("a", 0.95), _ranked("b", 0.80)],
        residual=pd.DataFrame(
            [{"variable": "a", "residual_corr": 0.20}, {"variable": "b", "residual_corr": 0.75}]
        ),
    )
    indexed = result.set_index("variable")

    assert indexed.loc["a", "raw_corr"] > indexed.loc["b", "raw_corr"]
    assert indexed.loc["b", "correlation_evidence_score"] > indexed.loc["a", "correlation_evidence_score"]
    assert indexed.loc["b", "evidence_score"] > indexed.loc["a", "evidence_score"]
    assert indexed.loc["b", "final_score"] > indexed.loc["a", "final_score"]
    assert result["variable"].tolist() == ["b", "a"]


def test_all_evidence_uses_fixed_component_budget():
    row = _evaluate(
        [_ranked("x", 0.9)],
        residual=_frame({"variable": "x", "residual_corr": 0.3}),
        stability=_frame({"variable": "x", "regime_stability_final": 0.7}),
        rolling=_frame({"variable": "x", "rolling_stability": 0.6}),
        lag_peak=_frame({"variable": "x", "lag_quality": 0.5}),
        model_lift=_frame({"variable": "x", "model_lift": 0.4}),
    ).iloc[0]
    correlation = np.sqrt(0.9 * 0.3)
    stability_score = np.sqrt(0.7 * 0.6)
    expected = np.median([
        profile["association"] * correlation
        + profile["prediction"] * 0.4
        + profile["stability"] * stability_score
        + profile["lag_quality"] * 0.5
        for profile in INDUSTRIAL_SCORE_WEIGHT_PROFILES
    ])

    assert row["evidence_score"] == pytest.approx(expected)


def test_missing_evidence_is_not_renormalized():
    row = _evaluate(
        [_ranked("x", 0.8)],
        residual=_frame({"variable": "x", "residual_corr": 0.4}),
        lag_peak=_frame({"variable": "x", "lag_quality": 0.6}),
    ).iloc[0]
    assert row["evidence_completeness"] == pytest.approx(0.5)
    assert row["evidence_score"] < row["correlation_evidence_score"]


def test_pr3_risk_penalty_and_cap_are_unchanged():
    row = _evaluate(
        [_ranked("x", 0.9, -1)],
        risks=_frame({"variable": "x", "risk_flags": "target_leads_variable"}),
    ).iloc[0]

    assert row["risk_penalty"] == 0.0
    assert row["risk_score_cap"] == 1.0
    assert row["final_score"] == pytest.approx(row["evidence_score"])


def test_pr4_direction_classes_survive_production_risk_pipeline():
    ranked = pd.DataFrame([_ranked("upstream", 0.8, 1), _ranked("downstream", 0.8, -1)])
    empty = pd.DataFrame()
    generated = risk_flags(
        ranked, empty, empty, empty, {"upstream": "PV", "downstream": "PV"}, []
    )
    variable_only = pd.DataFrame(columns=["variable"])

    result = final_ranked_features(
        ranked, variable_only, variable_only, variable_only, generated, variable_only, variable_only
    ).set_index("variable")

    assert result.loc["upstream", "candidate_class"] == "upstream_driver_candidate"
    assert "target_leads_variable" in generated.set_index("variable").loc["downstream", "risk_flags"]
    assert result.loc["downstream", "candidate_class"] == "downstream_response"


def test_shadow_ranking_fields_remain_positive_integers():
    result = _evaluate([_ranked("a", 0.7), _ranked("b", 0.8)])

    for column in ["association_rank", "driver_rank"]:
        assert result[column].tolist() == [1, 2]
        assert pd.api.types.is_integer_dtype(result[column])
    assert {"candidate_class", "driver_priority_score"}.issubset(result.columns)


def test_main_sort_and_topk_use_driver_rank():
    rows = [_ranked("a", 0.95), _ranked("b", 0.80), _ranked("c", 0.70)]
    residual = pd.DataFrame(
        [
            {"variable": "a", "residual_corr": 0.10},
            {"variable": "b", "residual_corr": 0.75},
            {"variable": "c", "residual_corr": 0.70},
        ]
    )
    full = _evaluate(rows, residual=residual)
    top = _evaluate(rows, residual=residual, top_k=1)

    assert full["driver_rank"].is_monotonic_increasing
    assert top.loc[0, "variable"] == full.loc[0, "variable"]


def test_force_include_keeps_adjusted_low_rank_variable():
    rows = [_ranked("safe", 0.8), _ranked("forced", 0.9)]
    risks = pd.DataFrame(
        [{"variable": "safe", "risk_flags": ""}, {"variable": "forced", "risk_flags": "target_leads_variable"}]
    )
    result = _evaluate(rows, risks=risks, top_k=1, force_include=["forced"]).set_index("variable")

    assert "forced" in result.index
    assert bool(result.loc["forced", "force_included"]) is True
    assert result.loc["forced", "correlation_evidence_score"] == pytest.approx(0.9)
    assert result.loc["forced", "risk_penalty"] == 0.0
    assert result.loc["forced", "risk_score_cap"] == 1.0


def test_output_uses_formal_correlation_evidence_fields():
    result = _evaluate([_ranked("x", 0.8)])
    required = {
        "variable", "raw_corr", "association_score", "residual_corr",
        "independent_signal_score", "residual_status",
        "correlation_evidence_score", "correlation_evidence_status", "evidence_score",
        "risk_penalty", "risk_score_cap", "final_score", "candidate_grade",
        "recommended_use", "candidate_class", "association_rank", "driver_rank",
    }

    assert required.issubset(result.columns)


def test_all_inputs_are_not_modified():
    frames = [
        pd.DataFrame([_ranked("x", 0.8)]),
        _frame({"variable": "x", "residual_corr": 0.4}),
        _frame({"variable": "x", "regime_stability_final": 0.7}),
        _frame({"variable": "x", "model_lift": 0.3}),
        _frame({"variable": "x", "risk_flags": ""}),
        _frame({"variable": "x", "lag_quality": 0.6}),
        _frame({"variable": "x", "rolling_stability": 0.5}),
    ]
    before = [frame.copy(deep=True) for frame in frames]

    final_ranked_features(*frames)

    for actual, expected in zip(frames, before):
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "residual",
    [pd.DataFrame(), pd.DataFrame(columns=["variable"]), pd.DataFrame({"variable": ["x"], "residual_corr": [np.nan]})],
)
def test_empty_or_nan_residual_is_association_only(residual: pd.DataFrame):
    row = _evaluate([_ranked("x", 0.8)], residual=residual).iloc[0]

    assert pd.isna(row["independent_signal_score"])
    assert row["correlation_evidence_status"] == "innovation_verified"
    assert row["correlation_evidence_score"] == pytest.approx(0.8)


def test_all_scores_stay_in_range():
    result = _evaluate(
        [_ranked("high", 2.0), _ranked("low", -1.0)],
        residual=pd.DataFrame(
            [{"variable": "high", "residual_corr": 2.0}, {"variable": "low", "residual_corr": -1.0}]
        ),
    )

    assert result["association_score"].between(0, 1).all()
    assert result["independent_signal_score"].dropna().between(0, 1).all()
    assert result["correlation_evidence_score"].between(0, 1).all()
    assert result["evidence_score"].between(0, 1).all()
    assert result["final_score"].between(0, 1).all()


def test_old_double_counting_source_pattern_is_absent():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")

    assert '"raw_corr_score"' not in source
    assert '"residual": (residual_score, 0.25)' not in source
    assert "residual_score =" not in source
    assert "EVIDENCE_COMPONENT_WEIGHTS" not in source
    assert "CORRELATION_EVIDENCE_WEIGHTS" not in source
