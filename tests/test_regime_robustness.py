from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.screening import (
    EVIDENCE_COMPONENT_WEIGHTS,
    MIN_REGIMES_FOR_STABILITY,
    REGIME_CONSISTENCY_WEIGHTS,
    REGIME_NAMES,
    REGIME_STABILITY_COLUMNS,
    REGIME_UNSTABLE_THRESHOLD,
    _summarize_regime_robustness,
    final_ranked_features,
    risk_flags,
)


def _scores(
    variable: str = "x",
    regimes: tuple[str, ...] = REGIME_NAMES,
    scores: tuple[float, ...] = (0.8, 0.8, 0.8),
    signs: tuple[float, ...] = (0.8, 0.8, 0.8),
    lags: tuple[float, ...] = (10, 10, 10),
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": variable, "regime": regime, "score": score, "signed_corr": sign, "lag": lag}
            for regime, score, sign, lag in zip(regimes, scores, signs, lags)
        ]
    )


def _summary(scores: pd.DataFrame, max_lag: int = 20) -> pd.Series:
    return _summarize_regime_robustness(scores, max_lag).iloc[0]


def _final(stability: pd.DataFrame, *, lag_quality: float | None = None) -> pd.Series:
    ranked = pd.DataFrame([{"variable": "x", "score": 0.8, "lag": 1, "direction": "变量领先目标"}])
    variable_only = pd.DataFrame(columns=["variable"])
    lag_peak = (
        pd.DataFrame([{"variable": "x", "lag_quality": lag_quality}])
        if lag_quality is not None
        else variable_only
    )
    return final_ranked_features(
        ranked, variable_only, stability, variable_only, variable_only, lag_peak, variable_only
    ).iloc[0]


def _risks(stability: pd.DataFrame) -> pd.Series:
    ranked = pd.DataFrame([{"variable": "x", "score": 0.8, "lag": 1}])
    return risk_flags(
        ranked,
        pd.DataFrame(),
        stability,
        pd.DataFrame(),
        {"x": "PV"},
        [],
    ).iloc[0]


def test_regime_constants_are_fixed():
    assert REGIME_NAMES == ("low", "mid", "high")
    assert MIN_REGIMES_FOR_STABILITY == 2
    assert REGIME_UNSTABLE_THRESHOLD == 0.50
    assert REGIME_CONSISTENCY_WEIGHTS == {"strength": 0.60, "lag": 0.40}
    assert sum(REGIME_CONSISTENCY_WEIGHTS.values()) == pytest.approx(1.0)


def test_empty_input_has_fixed_schema():
    result = _summarize_regime_robustness(pd.DataFrame(), max_lag=20)

    assert result.empty
    assert result.columns.tolist() == REGIME_STABILITY_COLUMNS


def test_three_identical_regimes_have_full_robustness():
    row = _summary(_scores())

    assert row["regime_count"] == 3
    assert row["regime_coverage"] == 1
    assert row["regime_strength_consistency"] == 1
    assert row["regime_sign_consistency"] == 1
    assert row["regime_lag_consistency"] == 1
    assert row["regime_consistency_score"] == 1
    assert row["regime_stability_final"] == 1
    assert row["regime_evidence_status"] == "full_coverage"


def test_two_identical_regimes_are_partial_coverage():
    row = _summary(_scores(regimes=("low", "high"), scores=(0.8, 0.8), signs=(0.8, 0.8), lags=(10, 10)))

    assert row["regime_count"] == 2
    assert row["regime_coverage"] == pytest.approx(2 / 3)
    assert row["regime_consistency_score"] == 1
    assert row["regime_stability_final"] == pytest.approx(2 / 3)
    assert row["regime_evidence_status"] == "partial_coverage"


def test_single_regime_is_insufficient_not_neutral_or_stable():
    row = _summary(_scores(regimes=("low",), scores=(0.8,), signs=(0.8,), lags=(10,)))

    assert row["regime_count"] == 1
    assert row["regime_coverage"] == pytest.approx(1 / 3)
    assert row["regime_evidence_status"] == "insufficient_regimes"
    assert pd.isna(row["regime_stability_final"])
    assert pd.isna(row["regime_consistency_score"])


def test_opposite_signs_gate_robustness_to_zero():
    row = _summary(_scores(regimes=("low", "high"), scores=(0.8, 0.8), signs=(0.8, -0.8), lags=(10, 10)))

    assert row["regime_sign_consistency"] == 0
    assert row["regime_consistency_score"] == 0
    assert row["regime_stability_final"] == 0
    assert row["regime_evidence_status"] == "partial_coverage"


def test_one_of_three_signs_reversed_is_strongly_reduced():
    row = _summary(_scores(signs=(0.8, 0.8, -0.8)))

    assert row["regime_sign_consistency"] == pytest.approx(1 / 3)
    assert row["regime_stability_final"] < 0.5


def test_strength_consistency_uses_min_max_ratio():
    row = _summary(_scores(scores=(0.8, 0.4, 0.8)))

    assert row["regime_strength_consistency"] == pytest.approx(0.5)


def test_strength_consistency_is_scale_invariant():
    high = _summary(_scores(regimes=("low", "high"), scores=(0.8, 0.4), signs=(1, 1), lags=(0, 0)))
    low = _summary(_scores(regimes=("low", "high"), scores=(0.4, 0.2), signs=(1, 1), lags=(0, 0)))

    assert high["regime_strength_consistency"] == pytest.approx(low["regime_strength_consistency"])


def test_all_zero_strength_has_zero_consistency():
    row = _summary(_scores(regimes=("low", "high"), scores=(0.0, 0.0), signs=(1, 1), lags=(0, 0)))

    assert row["regime_strength_consistency"] == 0


@pytest.mark.parametrize(
    ("lags", "max_lag", "expected"),
    [((10, 10), 20, 1.0), ((-20, 20), 20, 0.0), ((0, 0), 0, 1.0)],
)
def test_lag_consistency_formula(lags: tuple[int, int], max_lag: int, expected: float):
    row = _summary(
        _scores(regimes=("low", "high"), scores=(0.8, 0.8), signs=(1, 1), lags=lags),
        max_lag=max_lag,
    )

    assert row["regime_lag_consistency"] == pytest.approx(expected)


def test_missing_metric_marks_insufficient_metrics():
    frame = _scores(regimes=("low", "high"), scores=(0.8, 0.8), signs=(0.8, np.nan), lags=(1, 1))
    row = _summary(frame)

    assert row["regime_evidence_status"] == "insufficient_metrics"
    assert pd.isna(row["regime_stability_final"])
    assert pd.isna(row["regime_sign_consistency"])


def test_duplicate_regime_does_not_inflate_coverage():
    frame = pd.concat([_scores(regimes=("low",), scores=(0.8,), signs=(1,), lags=(1,))] * 2, ignore_index=True)
    row = _summary(frame)

    assert row["regime_count"] == 1
    assert row["regime_coverage"] == pytest.approx(1 / 3)


def test_unknown_regimes_are_ignored():
    frame = pd.concat([
        _scores(regimes=("low", "high"), scores=(0.8, 0.8), signs=(1, 1), lags=(1, 1)),
        _scores(regimes=("startup", "shutdown"), scores=(1, 1), signs=(1, 1), lags=(1, 1)),
    ], ignore_index=True)
    row = _summary(frame)

    assert row["regime_count"] == 2
    assert row["regime_coverage"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "summary",
    [
        _summarize_regime_robustness(_scores(regimes=("low",), scores=(0.8,), signs=(1,), lags=(1,)), 20),
        _summarize_regime_robustness(_scores(regimes=("low", "high"), scores=(0.8, 0.8), signs=(1, np.nan), lags=(1, 1)), 20),
    ],
)
def test_insufficient_regime_evidence_does_not_trigger_risk(summary: pd.DataFrame):
    row = _risks(summary)

    assert "unstable_across_regimes" not in row["risk_flags"]


def test_sign_reversal_triggers_unstable_regime_risk():
    summary = _summarize_regime_robustness(
        _scores(regimes=("low", "high"), scores=(0.8, 0.8), signs=(1, -1), lags=(1, 1)), 20
    )
    row = _risks(summary)

    assert "unstable_across_regimes" in row["risk_flags"]


@pytest.mark.parametrize(
    ("stability", "expected_status"),
    [
        (pd.DataFrame(), "not_computed"),
        (_summarize_regime_robustness(_scores(regimes=("low",), scores=(0.8,), signs=(1,), lags=(1,)), 20), "insufficient_regimes"),
        (_summarize_regime_robustness(_scores(regimes=("low", "high"), scores=(0.8, 0.8), signs=(1, np.nan), lags=(1, 1)), 20), "insufficient_metrics"),
    ],
)
def test_final_output_does_not_fill_missing_regime_with_half(stability: pd.DataFrame, expected_status: str):
    row = _final(stability)

    assert pd.isna(row["regime_stability_final"])
    assert row["regime_status"] == expected_status


def test_missing_regime_is_excluded_from_dynamic_weight_denominator():
    row = _final(pd.DataFrame(), lag_quality=0.6)
    expected = (0.50 * 0.8 + 0.10 * 0.6) / 0.60

    assert row["evidence_score"] == pytest.approx(expected)


def test_partial_regime_evidence_enters_total_score():
    stability = _summarize_regime_robustness(
        _scores(regimes=("low", "high"), scores=(0.8, 0.8), signs=(1, 1), lags=(1, 1)), 20
    )
    row = _final(stability)
    expected = (0.50 * 0.8 + 0.15 * (2 / 3)) / 0.65

    assert row["regime_status"] == "partial_coverage"
    assert row["evidence_score"] == pytest.approx(expected)


def test_regime_component_weight_remains_fifteen_percent():
    assert EVIDENCE_COMPONENT_WEIGHTS["regime"] == pytest.approx(0.15)


def test_pr3_risk_penalty_and_cap_remain_active():
    variable_only = pd.DataFrame(columns=["variable"])
    ranked = pd.DataFrame([{"variable": "x", "score": 0.9, "lag": -1, "direction": "变量滞后目标"}])
    risks = pd.DataFrame([{"variable": "x", "risk_flags": "target_leads_variable"}])
    row = final_ranked_features(
        ranked, variable_only, variable_only, variable_only, risks, variable_only, variable_only,
        force_include_variables=["x"], top_k=0,
    ).iloc[0]

    assert row["risk_penalty"] == pytest.approx(0.10)
    assert row["risk_score_cap"] == pytest.approx(0.59)
    assert bool(row["force_included"]) is True


def test_pr4_direction_and_driver_rank_sort_remain_active():
    variable_only = pd.DataFrame(columns=["variable"])
    ranked = pd.DataFrame([
        {"variable": "up", "score": 0.7, "lag": 1},
        {"variable": "down", "score": 0.9, "lag": -1},
    ])
    generated = risk_flags(ranked, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"up": "PV", "down": "PV"}, [])
    result = final_ranked_features(
        ranked, variable_only, variable_only, variable_only, generated, variable_only, variable_only
    )
    indexed = result.set_index("variable")

    assert indexed.loc["up", "candidate_class"] == "upstream_driver_candidate"
    assert indexed.loc["down", "candidate_class"] == "downstream_response"
    assert result["driver_rank"].is_monotonic_increasing


def test_pr5_parts_keep_single_correlation_entry():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")
    parts = source.split("parts = {", 1)[1].split("}", 1)[0]

    assert parts.count('"correlation":') == 1
    assert '"raw":' not in parts
    assert '"residual":' not in parts


def test_summarizer_and_final_inputs_are_not_modified():
    scores = _scores()
    scores_before = scores.copy(deep=True)
    stability = _summarize_regime_robustness(scores, 20)
    ranked = pd.DataFrame([{"variable": "x", "score": 0.8, "lag": 1}])
    inputs = [ranked, pd.DataFrame(columns=["variable"]), stability]
    inputs += [pd.DataFrame(columns=["variable"]) for _ in range(4)]
    before = [frame.copy(deep=True) for frame in inputs]

    final_ranked_features(*inputs)

    pd.testing.assert_frame_equal(scores, scores_before)
    for actual, expected in zip(inputs, before):
        pd.testing.assert_frame_equal(actual, expected)


def test_all_computed_regime_metrics_stay_in_range():
    result = _summarize_regime_robustness(
        pd.concat([_scores("a"), _scores("b", scores=(0.8, 0.4, 0.6), signs=(1, 1, -1), lags=(-20, 0, 20))]),
        20,
    )

    for column in [
        "regime_coverage", "regime_strength_consistency", "regime_sign_consistency",
        "regime_lag_consistency", "regime_consistency_score", "regime_stability_final",
    ]:
        assert result[column].dropna().between(0, 1).all()


def test_old_regime_formulas_are_absent():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")

    for forbidden in [
        'strength_stability = (1 - stability["regime_score_cv"])',
        "0.5 * strength_stability",
        "value_counts(normalize=True).iloc[0]",
        "display_regime = regime_raw.fillna(0.5)",
    ]:
        assert forbidden not in source
