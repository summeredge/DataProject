from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.report import write_outputs
from chem_ts_corr.screening import (
    _combine_correlation_evidence,
    _summarize_regime_robustness,
    final_ranked_features,
)


def _combined(
    association: float,
    innovation: float = np.nan,
    independent: float = np.nan,
) -> float:
    score, _ = _combine_correlation_evidence(
        pd.Series([association]),
        pd.Series([independent]),
        pd.Series([innovation]),
    )
    return float(score.iloc[0])


def _regime_summary(signed_correlations: tuple[float, float, float]) -> pd.Series:
    scores = pd.DataFrame(
        {
            "variable": ["x", "x", "x"],
            "regime": ["low", "mid", "high"],
            "score": [0.8, 0.8, 0.8],
            "signed_corr": signed_correlations,
            "lag": [1, 1, 1],
        }
    )
    return _summarize_regime_robustness(scores, max_lag=10).iloc[0]


def test_three_correlation_evidence_scores_use_equal_weight_geometric_mean():
    actual = _combined(association=0.8, innovation=0.6, independent=0.7)
    equal_weight = (0.8 * 0.6 * 0.7) ** (1 / 3)
    old_nested_square_root = np.sqrt(np.sqrt(0.8 * 0.6) * 0.7)

    assert actual == pytest.approx(equal_weight)
    assert actual != pytest.approx(old_nested_square_root)


@pytest.mark.parametrize(
    ("innovation", "independent", "expected"),
    [
        (np.nan, np.nan, 0.8),
        (0.6, np.nan, np.sqrt(0.8 * 0.6)),
        (np.nan, 0.7, np.sqrt(0.8 * 0.7)),
    ],
)
def test_missing_correlation_evidence_is_not_treated_as_zero(
    innovation: float,
    independent: float,
    expected: float,
):
    assert _combined(0.8, innovation, independent) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("signed_correlations", "expected_consistency", "expected_reversal"),
    [
        ((0.8, 0.7, -0.3), 2 / 3, True),
        ((0.8, 0.7, 0.6), 1.0, False),
        ((0.1, 0.1, -0.9), 7 / 11, True),
    ],
)
def test_regime_sign_consistency_uses_signed_strength_and_reports_reversal(
    signed_correlations: tuple[float, float, float],
    expected_consistency: float,
    expected_reversal: bool,
):
    row = _regime_summary(signed_correlations)

    assert row["regime_sign_consistency"] == pytest.approx(expected_consistency)
    assert bool(row["regime_sign_reversal_flag"]) is expected_reversal
    assert row["regime_stability_final"] == pytest.approx(expected_consistency)


def test_regime_reversal_flag_is_diagnostic_only_and_not_in_final_ranking_schema():
    stability = _summarize_regime_robustness(
        pd.DataFrame(
            {
                "variable": ["x", "x", "x"],
                "regime": ["low", "mid", "high"],
                "score": [0.8, 0.8, 0.8],
                "signed_corr": [0.8, 0.7, -0.3],
                "lag": [1, 1, 1],
            }
        ),
        max_lag=10,
    )
    variable_only = pd.DataFrame(columns=["variable"])
    ranked = pd.DataFrame([{"variable": "x", "score": 0.8, "lag": 1}])

    final = final_ranked_features(
        ranked,
        variable_only,
        stability,
        variable_only,
        variable_only,
        pd.DataFrame(
            [{"variable": "x", "lag_quality": 0.8, "lag_boundary_flag": False}]
        ),
        variable_only,
    )

    assert "regime_sign_reversal_flag" in stability.columns
    assert "regime_sign_reversal_flag" not in final.columns


def test_regime_reversal_flag_is_only_written_to_regime_detail_csv(tmp_path: Path):
    stability = _summarize_regime_robustness(
        pd.DataFrame(
            {
                "variable": ["x", "x", "x"],
                "regime": ["low", "mid", "high"],
                "score": [0.8, 0.8, 0.8],
                "signed_corr": [0.8, 0.7, -0.3],
                "lag": [1, 1, 1],
            }
        ),
        max_lag=10,
    )
    variable_only = pd.DataFrame(columns=["variable"])
    final = final_ranked_features(
        pd.DataFrame([{"variable": "x", "score": 0.8, "lag": 1}]),
        variable_only,
        stability,
        variable_only,
        variable_only,
        pd.DataFrame(
            [{"variable": "x", "lag_quality": 0.8, "lag_boundary_flag": False}]
        ),
        variable_only,
    )

    write_outputs(
        tmp_path,
        "target",
        final,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        regime_scores=stability,
    )

    for csv_path in tmp_path.glob("*.csv"):
        header = csv_path.read_text(encoding="utf-8-sig").splitlines()
        if csv_path.name == "regime_scores.csv":
            assert header and "regime_sign_reversal_flag" in header[0]
        else:
            assert not header or "regime_sign_reversal_flag" not in header[0]
