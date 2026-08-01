from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.screening import _data_quality_score, risk_flags


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
    assert "数据质量需关注" in row["human_reason"]


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

    assert "poor_data_quality" in tokens
    assert "severe_data_quality" in tokens
    assert row["data_quality_score"] < 1.0
    assert "数据质量严重不足" in row["human_reason"]


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


def test_data_quality_source_uses_exponential_components_and_not_minimum():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")
    function = source.split("def _data_quality_score", 1)[1].split(
        "def risk_flags", 1
    )[0]

    assert "np.exp" in function
    assert "np.log(2)" in function
    assert "min(" not in function
