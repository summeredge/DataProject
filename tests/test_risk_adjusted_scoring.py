from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr.screening import (
    EVIDENCE_SCORE_CAPS,
    RISK_RELATIVE_PENALTY_WEIGHTS,
    final_ranked_features,
)


def _ranked(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": variable, "lag": 0, "direction": "同步变化", "score": score}
            for variable, score in rows
        ]
    )


def _risks(rows: list[tuple[str, object, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": variable,
                "risk_flags": flags,
                "strong_risk_count": strong,
                "weak_risk_count": weak,
            }
            for variable, flags, strong, weak in rows
        ]
    )


def _score(
    rows: list[tuple[str, float]],
    risks: pd.DataFrame | None = None,
    *,
    top_k: int | None = None,
    force_include_variables: list[str] | None = None,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["variable"])
    complete = pd.DataFrame(
        [{"variable": variable, "value": score} for variable, score in rows]
    )
    ranked = _ranked(rows)
    ranked["innovation_score"] = ranked["score"]
    return final_ranked_features(
        ranked=ranked,
        residual=empty,
        stability=empty,
        model_lift=complete.rename(columns={"value": "model_lift_score"}).assign(status="ok"),
        risks=pd.DataFrame() if risks is None else risks,
        lag_peak_quality=complete.rename(columns={"value": "lag_quality"}),
        rolling_corr_scores=complete.rename(columns={"value": "rolling_stability"}),
        top_k=top_k,
        force_include_variables=force_include_variables,
    )


def _one(flags: object, evidence: float = 1.0) -> pd.Series:
    return _score([("x", evidence)], _risks([("x", flags, 0, 0)])).iloc[0]


def test_no_risk_preserves_evidence_score():
    row = _one("", 0.9)

    assert row["evidence_score"] == pytest.approx(0.9)
    assert row["risk_penalty"] == 0
    assert row["risk_score_cap"] == 1
    assert row["risk_cap_reason"] == ""
    assert row["final_score"] == pytest.approx(row["evidence_score"])


def test_target_leads_changes_driver_class_without_reducing_prediction_evidence():
    row = _one("target_leads_variable", 0.9)

    assert row["risk_penalty_rate"] == 0.0
    assert row["risk_penalty"] == 0.0
    assert row["risk_score_cap"] == 1.0
    assert row["final_score"] == pytest.approx(0.9)
    assert row["candidate_class"] == "downstream_response"


@pytest.mark.parametrize(
    ("token", "cap", "grade"),
    [
        ("common_capacity_driver", 1.0, "A"),
        ("poor_data_quality", 0.44, "D"),
    ],
)
def test_engineering_risks_do_not_duplicate_component_penalties(token: str, cap: float, grade: str):
    row = _one(token, 0.95)

    assert row["risk_penalty"] == 0.0
    assert row["risk_score_cap"] == pytest.approx(cap)
    assert row["final_score"] == pytest.approx(min(0.95, cap))
    assert row["candidate_grade"] == grade


def test_formula_like_does_not_duplicate_strong_formula_penalty():
    row = _one("formula_like;strong_formula_leakage", 1.0)

    assert row["risk_penalty_rate"] == pytest.approx(0.50)
    assert row["risk_penalty"] == pytest.approx(0.50)
    assert row["risk_score_cap"] == pytest.approx(0.25)
    assert row["risk_cap_reason"] == "strong_formula_leakage"
    assert row["final_score"] == pytest.approx(0.25)
    assert row["candidate_grade"] == "E"


def test_engineering_risks_are_handled_by_driver_priority_not_evidence_subtraction():
    row = _one("common_capacity_driver;target_leads_variable;lag_boundary", 0.95)

    assert row["risk_penalty"] == 0.0
    assert row["risk_score_cap"] == 1.0
    assert row["risk_cap_reason"] == ""
    assert row["final_score"] == pytest.approx(0.95)


def test_multiple_caps_choose_lowest_value():
    row = _one("strong_formula_leakage;poor_data_quality;target_leads_variable")

    assert row["risk_score_cap"] == pytest.approx(0.25)
    assert row["risk_cap_reason"] == "strong_formula_leakage"


def test_relative_penalty_rate_is_bounded():
    row = _one(";".join(RISK_RELATIVE_PENALTY_WEIGHTS))

    assert row["risk_penalty_rate"] == pytest.approx(0.60)
    assert row["risk_penalty"] == pytest.approx(0.60)


def test_unknown_risk_does_not_adjust_score():
    row = _one("unknown_risk", 0.9)

    assert row["risk_penalty"] == 0
    assert row["risk_score_cap"] == 1
    assert row["final_score"] == pytest.approx(row["evidence_score"])


@pytest.mark.parametrize(
    "flags",
    [
        "not_target_leads_variable",
        "target_leads_variable_extra",
        "common_capacity_driver_backup",
    ],
)
def test_risk_tokens_are_matched_exactly(flags: str):
    row = _one(flags, 0.9)

    assert row["risk_penalty"] == 0
    assert row["risk_score_cap"] == 1


@pytest.mark.parametrize(
    "risks",
    [
        pd.DataFrame(),
        pd.DataFrame({"variable": ["x"]}),
        pd.DataFrame({"variable": ["x"], "risk_flags": [pd.NA]}),
    ],
)
def test_missing_risk_data_is_safe(risks: pd.DataFrame):
    row = _score([("x", 0.9)], risks).iloc[0]

    assert row["risk_penalty"] == 0
    assert row["risk_score_cap"] == 1
    assert row["risk_cap_reason"] == ""
    assert row["final_score"] == pytest.approx(row["evidence_score"])


def test_force_included_variable_still_receives_risk_adjustment():
    risks = _risks([("safe", "", 0, 0), ("risky", "strong_formula_leakage", 1, 0)])

    result = _score(
        [("safe", 0.9), ("risky", 1.0)],
        risks,
        top_k=1,
        force_include_variables=["risky"],
    )
    risky = result.set_index("variable").loc["risky"]

    assert bool(risky["force_included"]) is True
    assert risky["risk_penalty"] == pytest.approx(0.50)
    assert risky["risk_score_cap"] == pytest.approx(0.25)
    assert risky["candidate_grade"] == "E"


def test_risk_adjustment_changes_ranking():
    risks = _risks([("a", "target_leads_variable", 0, 0), ("b", "", 0, 0)])

    result = _score([("a", 0.95), ("b", 0.85)], risks)

    assert result["variable"].tolist() == ["a", "b"]


def test_risk_counts_do_not_control_penalty():
    risks = _risks(
        [
            ("a", "target_leads_variable", 0, 0),
            ("b", "target_leads_variable", 10, 20),
        ]
    )

    result = _score([("a", 0.9), ("b", 0.9)], risks).set_index("variable")

    assert result.loc["a", "risk_penalty"] == result.loc["b", "risk_penalty"]
    assert result.loc["a", "final_score"] == result.loc["b", "final_score"]


def test_adjusted_output_values_stay_in_range():
    risks = _risks(
        [
            ("none", "", 0, 0),
            ("all", ";".join(RISK_RELATIVE_PENALTY_WEIGHTS), 99, 99),
            ("unknown", "unknown", 0, 0),
        ]
    )

    result = _score([("none", 1.5), ("all", 1.0), ("unknown", -0.5)], risks)

    assert result["evidence_score"].between(0, 1).all()
    assert result["risk_penalty_rate"].between(0, 0.80).all()
    assert result["risk_penalty"].between(0, 1).all()
    assert result["risk_score_cap"].between(0, 1).all()
    assert result["final_score"].between(0, 1).all()


def test_inputs_are_not_modified():
    ranked = _ranked([("x", 0.9)])
    risks = _risks([("x", "target_leads_variable", 1, 0)])
    empty = pd.DataFrame(columns=["variable"])
    frames = [ranked, risks, empty.copy(), empty.copy(), empty.copy(), empty.copy(), empty.copy()]
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


def test_recommended_use_uses_exact_risk_tokens():
    row = _one("poor_data_quality_extra", 0.9)

    assert row["recommended_use"] != "poor_quality_variable"


def test_old_count_and_secondary_scaling_formulas_are_absent():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")

    for forbidden in [
        "0.12 * strong",
        "0.03 * weak",
        '0.10 * final["risk_penalty"]',
        "0.10 * final['risk_penalty']",
    ]:
        assert forbidden not in source
    assert '"raw": (final["raw_corr_score"], 0.25)' not in source
    assert '"residual": (residual_score, 0.25)' not in source
    assert "EVIDENCE_COMPONENT_WEIGHTS" not in source
    assert "CORRELATION_EVIDENCE_WEIGHTS" not in source
    assert "den.replace(0" not in source


def test_v2_risk_constants_separate_evidence_and_engineering_priority():
    assert RISK_RELATIVE_PENALTY_WEIGHTS == {
        "formula_like": 0.00,
        "strong_formula_leakage": 0.50,
        "common_capacity_driver": 0.00,
        "target_leads_variable": 0.00,
        "unstable_across_regimes": 0.00,
        "unstable_over_time": 0.00,
        "lag_boundary": 0.00,
        "low_model_lift": 0.00,
        "poor_data_quality": 0.00,
        "residual_collinearity": 0.10,
        "redundant_proxy": 0.00,
    }
    assert EVIDENCE_SCORE_CAPS == {
        "strong_formula_leakage": 0.25,
        "poor_data_quality": 0.44,
    }
