from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.screening import _data_quality_score, final_ranked_features, risk_flags


@pytest.mark.parametrize("missing_rate", [0.0, 0.1, 0.2, 0.3])
def test_missing_rate_quality_score_decays_smoothly(missing_rate: float):
    score = _data_quality_score(
        {
            "missing_rate": missing_rate,
            "saturation_ratio": 0.0,
            "abnormal_jump_ratio": 0.0,
        }
    )

    expected = np.exp(-np.log(2) * missing_rate / 0.20 / 4)
    assert score == pytest.approx(expected)


def test_missing_rate_scores_are_monotonic_and_do_not_drop_to_zero_at_threshold():
    scores = [
        _data_quality_score(
            {
                "missing_rate": missing_rate,
                "saturation_ratio": 0.0,
                "abnormal_jump_ratio": 0.0,
            }
        )
        for missing_rate in [0.0, 0.1, 0.2, 0.3]
    ]

    assert all(left > right for left, right in zip(scores, scores[1:]))
    assert scores[2] > 0
    assert scores[3] > 0
    assert scores[2] ** 4 == pytest.approx(0.5)


def test_four_reference_thresholds_have_geometric_mean_half_score():
    score = _data_quality_score(
        {
            "missing_rate": 0.20,
            "saturation_ratio": 0.20,
            "abnormal_jump_ratio": 0.01,
            "robust_outlier_ratio": 0.01,
        }
    )

    assert score == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("field", "reference_rate"),
    [
        ("missing_rate", 0.20),
        ("saturation_ratio", 0.20),
        ("abnormal_jump_ratio", 0.01),
    ],
)
def test_each_quality_component_remains_positive_above_reference_threshold(
    field: str, reference_rate: float
):
    diagnostic = {
        "missing_rate": 0.0,
        "saturation_ratio": 0.0,
        "abnormal_jump_ratio": 0.0,
    }
    diagnostic[field] = reference_rate * 1.5

    assert _data_quality_score(diagnostic) > 0


@pytest.mark.parametrize(
    "diagnostic",
    [
        {},
        {"missing_rate": -0.1, "saturation_ratio": -0.1, "abnormal_jump_ratio": -0.1},
        {"missing_rate": 1.0, "saturation_ratio": 1.0, "abnormal_jump_ratio": 1.0},
    ],
)
def test_data_quality_score_stays_in_unit_interval(diagnostic: dict[str, float]):
    assert 0.0 <= _data_quality_score(diagnostic) <= 1.0


def test_poor_data_quality_risk_threshold_is_preserved():
    result = risk_flags(
        pd.DataFrame([{"variable": "x", "score": 0.8, "lag": 1}]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(
            [
                {
                    "variable": "x",
                    "missing_rate": 0.21,
                    "saturation_ratio": 0.0,
                    "abnormal_jump_ratio": 0.0,
                }
            ]
        ),
        {"x": "PV"},
        [],
    ).iloc[0]

    assert bool(result["poor_data_quality_flag"]) is True
    assert "poor_data_quality" in result["risk_flags"].split(";")


def _quality_row(diagnostic: dict[str, float]) -> pd.Series:
    return risk_flags(
        pd.DataFrame([{"variable": "x", "score": 0.8, "lag": 1}]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"variable": "x", **diagnostic}]),
        {"x": "PV"},
        [],
    ).iloc[0]


@pytest.mark.parametrize(
    "diagnostic",
    [
        {"missing_rate": 0.21},
        {"saturation_ratio": 0.21},
        {"abnormal_jump_ratio": 0.011},
        {"robust_outlier_ratio": 0.011},
    ],
)
def test_mild_quality_overflow_is_poor_but_not_severe(diagnostic: dict[str, float]):
    row = _quality_row(diagnostic)
    tokens = row["risk_flags"].split(";")

    assert "poor_data_quality" in tokens
    assert "severe_data_quality" not in tokens
    assert row["data_quality_score"] < 1.0
    assert row["strong_risk_count"] == 0
    assert row["risk_level"] == "weak"
    assert "数据质量需关注，建议核查缺失、单值集中和异常点" in row["human_reason"]


@pytest.mark.parametrize(
    "diagnostic",
    [
        {"missing_rate": 0.51},
        {"saturation_ratio": 0.81},
        {"abnormal_jump_ratio": 0.051},
        {"robust_outlier_ratio": 0.051},
    ],
)
def test_extreme_quality_overflow_marks_severe(diagnostic: dict[str, float]):
    row = _quality_row(diagnostic)
    tokens = row["risk_flags"].split(";")

    assert "poor_data_quality" not in tokens
    assert "severe_data_quality" in tokens
    assert row["data_quality_score"] < 1.0
    assert row["strong_risk_count"] == 1
    assert row["risk_level"] == "medium"
    assert row["human_reason"] == "数据质量严重不足"


@pytest.mark.parametrize(
    "diagnostic",
    [
        {"missing_rate": 0.50},
        {"saturation_ratio": 0.80},
        {"abnormal_jump_ratio": 0.05},
        {"robust_outlier_ratio": 0.05},
    ],
)
def test_severe_quality_thresholds_require_strictly_greater(diagnostic: dict[str, float]):
    row = _quality_row(diagnostic)
    tokens = row["risk_flags"].split(";")

    assert "severe_data_quality" not in tokens


def test_zero_quality_rates_keep_full_score_and_no_risk():
    row = _quality_row(
        {
            "missing_rate": 0.0,
            "saturation_ratio": 0.0,
            "abnormal_jump_ratio": 0.0,
            "robust_outlier_ratio": 0.0,
        }
    )

    assert row["data_quality_score"] == pytest.approx(1.0)
    assert "poor_data_quality" not in row["risk_flags"].split(";")
    assert "severe_data_quality" not in row["risk_flags"].split(";")


def _chain_row(
    diagnostic: dict[str, float], *, temporal_status: str
) -> tuple[pd.Series, pd.Series]:
    ranked = pd.DataFrame(
        [{"variable": "x", "lag": 1, "direction": "变量领先目标", "score": 0.95}]
    )
    risks = risk_flags(
        ranked.copy(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"variable": "x", **diagnostic}]),
        {"x": "PV"},
        [],
    ).iloc[0]
    complete = pd.DataFrame([{"variable": "x", "value": 0.95}])
    final = final_ranked_features(
        ranked=ranked,
        residual=pd.DataFrame(columns=["variable"]),
        stability=pd.DataFrame(columns=["variable"]),
        model_lift=complete.rename(columns={"value": "model_lift_score"}).assign(status="ok"),
        risks=pd.DataFrame(
            [
                {
                    "variable": "x",
                    "risk_flags": risks["risk_flags"],
                    "human_reason": risks["human_reason"],
                    "data_quality_score": risks["data_quality_score"],
                }
            ]
        ),
        lag_peak_quality=complete.rename(columns={"value": "lag_quality"}).assign(
            temporal_direction_status=temporal_status
        ),
        rolling_corr_scores=complete.rename(columns={"value": "rolling_stability"}),
    ).iloc[0]
    return risks, final


def test_full_chain_mild_quality_keeps_normal_use_without_cap():
    risks, row = _chain_row(
        {
            "missing_rate": 0.21,
            "saturation_ratio": 0.0,
            "abnormal_jump_ratio": 0.0,
            "robust_outlier_ratio": 0.0,
        },
        temporal_status="variable_leads_supported",
    )

    assert "poor_data_quality" in risks["risk_flags"].split(";")
    assert "severe_data_quality" not in risks["risk_flags"].split(";")
    assert row["risk_score_cap"] == 1.0
    assert row["risk_cap_reason"] == ""
    assert row["final_score"] > 0.44
    assert row["candidate_class"] == "upstream_driver_candidate"
    assert row["recommended_use"] == "strong_screening_candidate"


def test_full_chain_severe_quality_keeps_continuous_quality_effect_without_cap():
    risks, row = _chain_row(
        {
            "missing_rate": 0.51,
            "saturation_ratio": 0.0,
            "abnormal_jump_ratio": 0.0,
            "robust_outlier_ratio": 0.0,
        },
        temporal_status="variable_leads_supported",
    )

    assert risks["risk_flags"].split(";") == ["severe_data_quality"]
    assert row["risk_score_cap"] == pytest.approx(1.0)
    assert row["risk_cap_reason"] == ""
    assert row["final_score"] == pytest.approx(0.95 * row["data_quality_score"])
    assert row["candidate_class"] == "poor_quality"
    assert row["recommended_use"] == "poor_quality_variable"
    assert row["recommended_action"] == "数据质量严重不足，建议清洗数据或剔除该变量后重新分析"


def test_full_chain_severe_scores_are_mutually_exclusive_for_each_metric():
    diagnostics = [
        {"saturation_ratio": 0.81},
        {"abnormal_jump_ratio": 0.051},
        {"robust_outlier_ratio": 0.051},
    ]
    for diagnostic in diagnostics:
        risks, row = _chain_row(diagnostic, temporal_status="variable_leads_supported")
        assert risks["risk_flags"].split(";") == ["severe_data_quality"]
        assert "poor_data_quality" not in risks["risk_flags"].split(";")
        assert row["risk_score_cap"] == pytest.approx(1.0)
        assert row["risk_cap_reason"] == ""
        assert row["final_score"] == pytest.approx(0.95 * row["data_quality_score"])
        assert row["candidate_class"] == "poor_quality"
        assert row["recommended_use"] == "poor_quality_variable"


def test_full_chain_keeps_final_score_primary_ranking_and_schema():
    mild = _chain_row(
        {"missing_rate": 0.21}, temporal_status="variable_leads_supported"
    )[1]
    severe = _chain_row(
        {"missing_rate": 0.51}, temporal_status="variable_leads_supported"
    )[1]
    combined = pd.DataFrame([mild, severe])

    assert combined["final_score"].is_monotonic_decreasing
    for column in [
        "variable",
        "final_score",
        "data_quality_score",
        "risk_flags",
        "risk_score_cap",
        "risk_cap_reason",
        "candidate_class",
        "recommended_use",
        "recommended_action",
    ]:
        assert column in combined.columns


def test_data_quality_source_uses_exponential_components_and_not_minimum():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")
    function = source.split("def _data_quality_score", 1)[1].split(
        "def risk_flags", 1
    )[0]

    assert "np.exp" in function
    assert "np.log(2)" in function
    assert "min(" not in function
