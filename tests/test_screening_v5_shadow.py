from __future__ import annotations

import inspect

import pandas as pd
import pytest

from chem_ts_corr.screening import (
    V5_SHADOW_COMPARISON_COLUMNS,
    V5_SHADOW_SUMMARY_COLUMNS,
    build_v5_shadow_comparison,
    build_v5_shadow_summary,
    compute_v5_shadow_components,
    write_v5_shadow_outputs,
)


def test_v5_base_score_is_association_times_data_quality():
    result = compute_v5_shadow_components(
        association_score=0.8,
        data_quality_score=0.5,
    )

    assert result["base_score_v5"] == pytest.approx(0.4)
    assert result["evidence_score_v5"] == pytest.approx(0.4)
    assert result["shadow_final_score_v5"] == pytest.approx(0.4)


def test_v5_residual_bonus_requires_ok_and_preserves_zero_semantics():
    supported = compute_v5_shadow_components(
        association_score=0.8,
        data_quality_score=1.0,
        residual_corr=0.6,
        residual_status="ok",
    )
    measured_zero = compute_v5_shadow_components(
        association_score=0.8,
        data_quality_score=1.0,
        residual_corr=0.0,
        residual_status="ok",
    )
    unavailable_zero = compute_v5_shadow_components(
        association_score=0.8,
        data_quality_score=1.0,
        residual_corr=0.0,
        residual_status="not_computed",
    )

    assert supported["residual_support"] == pytest.approx(0.6)
    assert supported["residual_bonus_rate"] == pytest.approx(0.06)
    assert measured_zero["residual_support"] == pytest.approx(0.0)
    assert measured_zero["residual_bonus_rate"] == pytest.approx(0.0)
    assert pd.isna(unavailable_zero["residual_support"])
    assert unavailable_zero["residual_bonus_rate"] == pytest.approx(0.0)
    assert unavailable_zero["shadow_final_score_v5"] == pytest.approx(0.8)


def test_v5_residual_is_support_only_and_cannot_reduce_base_score():
    low = compute_v5_shadow_components(
        association_score=0.75,
        data_quality_score=0.8,
        residual_corr=0.01,
        residual_status="ok",
    )
    high = compute_v5_shadow_components(
        association_score=0.75,
        data_quality_score=0.8,
        residual_corr=0.99,
        residual_status="ok",
    )

    assert low["shadow_final_score_v5"] >= low["base_score_v5"]
    assert high["shadow_final_score_v5"] > low["shadow_final_score_v5"]


def test_v5_stability_uses_pure_consistency_and_geometric_merge():
    result = compute_v5_shadow_components(
        association_score=0.7,
        data_quality_score=1.0,
        rolling_sign_consistency=0.8,
        rolling_corr_iqr=0.2,
        regime_coverage=0.9,
        regime_strength_consistency=0.5,
        regime_sign_consistency=0.8,
        regime_lag_consistency=1.0,
        regime_evidence_status="full_coverage",
    )

    rolling = 0.8 * (1.0 - 0.2)
    regime = 0.9 * 0.8 * (0.60 * 0.5 + 0.40 * 1.0)
    stability = (rolling * regime) ** 0.5
    assert result["rolling_support"] == pytest.approx(rolling)
    assert result["regime_support"] == pytest.approx(regime)
    assert result["stability_support"] == pytest.approx(stability)
    assert result["stability_bonus_rate"] == pytest.approx(0.1 * stability)


def test_v5_missing_stability_does_not_penalize_or_fake_zero():
    result = compute_v5_shadow_components(
        association_score=0.7,
        data_quality_score=1.0,
    )

    assert pd.isna(result["rolling_support"])
    assert pd.isna(result["regime_support"])
    assert pd.isna(result["stability_support"])
    assert result["stability_bonus_rate"] == pytest.approx(0.0)
    assert result["shadow_final_score_v5"] == pytest.approx(result["base_score_v5"])


def test_v5_total_support_bonus_is_capped_at_twenty_percent():
    result = compute_v5_shadow_components(
        association_score=1.0,
        data_quality_score=1.0,
        residual_corr=1.0,
        residual_status="ok",
        rolling_sign_consistency=1.0,
        rolling_corr_iqr=0.0,
        regime_coverage=1.0,
        regime_strength_consistency=1.0,
        regime_sign_consistency=1.0,
        regime_lag_consistency=1.0,
        regime_evidence_status="full_coverage",
    )

    assert result["support_bonus_rate"] == pytest.approx(0.20)
    assert result["evidence_score_v5"] == pytest.approx(1.0)


def test_v5_only_target_leads_status_applies_existing_penalty_and_cap():
    target_leads = compute_v5_shadow_components(
        association_score=0.9,
        data_quality_score=1.0,
        residual_corr=1.0,
        residual_status="ok",
        temporal_direction_status="target_leads_supported",
    )
    variable_leads = compute_v5_shadow_components(
        association_score=0.9,
        data_quality_score=1.0,
        temporal_direction_status="variable_leads_supported",
    )

    assert target_leads["evidence_score_v5"] == pytest.approx(0.99)
    assert target_leads["shadow_final_score_v5"] == pytest.approx(0.25)
    assert variable_leads["shadow_final_score_v5"] == pytest.approx(0.9)


def test_v5_comparison_does_not_use_risk_flags_or_mutate_formal_frame():
    ranked = pd.DataFrame(
        [
            {
                "variable": "x",
                "final_score": 0.44,
                "driver_rank": 1,
                "association_score": 0.8,
                "data_quality_score": 1.0,
                "risk_flags": "strong_formula_leakage;severe_data_quality",
                "temporal_direction_status": "variable_leads_supported",
            }
        ]
    )
    before = ranked.copy(deep=True)

    result = build_v5_shadow_comparison(ranked)

    pd.testing.assert_frame_equal(ranked, before)
    assert result.loc[0, "base_score_v5"] == pytest.approx(0.8)
    assert result.loc[0, "shadow_final_score_v5"] == pytest.approx(0.8)
    assert result.loc[0, "risk_flags"] == "strong_formula_leakage;severe_data_quality"


def test_v5_comparison_uses_support_sources_and_keeps_rolling_abs_median_out():
    ranked = pd.DataFrame(
        [
            {
                "variable": "x",
                "final_score": 0.7,
                "driver_rank": 1,
                "association_score": 0.7,
                "data_quality_score": 1.0,
            }
        ]
    )
    rolling = pd.DataFrame(
        [
            {
                "variable": "x",
                "rolling_abs_corr_median": 999.0,
                "rolling_sign_consistency": 0.8,
                "rolling_corr_iqr": 0.2,
            }
        ]
    )
    regime = pd.DataFrame(
        [
            {
                "variable": "x",
                "regime_coverage": 0.9,
                "regime_strength_consistency": 0.5,
                "regime_sign_consistency": 0.8,
                "regime_lag_consistency": 1.0,
                "regime_evidence_status": "full_coverage",
            }
        ]
    )

    result = build_v5_shadow_comparison(
        ranked,
        rolling_corr_scores=rolling,
        regime_stability=regime,
    ).iloc[0]
    expected_rolling = 0.8 * 0.8
    expected_regime = 0.9 * 0.8 * (0.60 * 0.5 + 0.40 * 1.0)
    expected_stability = (expected_rolling * expected_regime) ** 0.5

    assert result["rolling_support"] == pytest.approx(expected_rolling)
    assert result["regime_support"] == pytest.approx(expected_regime)
    assert result["stability_support"] == pytest.approx(expected_stability)


def test_v5_comparison_rank_delta_and_top_k_summary_are_objective():
    ranked = pd.DataFrame(
        [
            {"variable": "a", "final_score": 0.90, "driver_rank": 1, "association_score": 0.90, "data_quality_score": 1.0},
            {"variable": "b", "final_score": 0.89, "driver_rank": 2, "association_score": 0.89, "data_quality_score": 1.0},
            {"variable": "c", "final_score": 0.70, "driver_rank": 3, "association_score": 0.70, "data_quality_score": 1.0},
        ]
    )
    residual = pd.DataFrame(
        [{"variable": "b", "residual_corr": 1.0, "residual_status": "ok"}]
    )

    comparison = build_v5_shadow_comparison(ranked, residual_corr_scores=residual)
    summary = build_v5_shadow_summary(comparison, top_ks=(1, 2, 10))
    by_variable = comparison.set_index("variable")

    assert by_variable.loc["b", "rank_delta"] == 1
    assert by_variable.loc["a", "rank_delta"] == -1
    assert summary["k"].tolist() == [1, 2, 10]
    assert summary.loc[summary["k"] == 1, "top_k_entrants"].item() == "b"
    assert summary.loc[summary["k"] == 1, "top_k_dropouts"].item() == "a"
    assert summary.loc[summary["k"] == 10, "effective_k"].item() == 3
    assert build_v5_shadow_summary(comparison)["k"].tolist() == [10, 20, 30]


def test_v5_writer_creates_only_shadow_artifacts_and_preserves_formal_file(tmp_path):
    ranked = pd.DataFrame(
        [
            {"variable": "x", "final_score": 0.5, "driver_rank": 1, "association_score": 0.5, "data_quality_score": 1.0}
        ]
    )
    comparison = build_v5_shadow_comparison(ranked)
    formal_path = tmp_path / "ranked_features.csv"
    formal_path.write_bytes(b"formal-output")

    paths = write_v5_shadow_outputs(tmp_path, comparison)

    assert paths["comparison"].name == "screening_v5_shadow_comparison.csv"
    assert paths["summary"].name == "screening_v5_shadow_summary.csv"
    assert paths["comparison"].exists()
    assert paths["summary"].exists()
    assert formal_path.read_bytes() == b"formal-output"
    assert pd.read_csv(paths["comparison"], encoding="utf-8-sig").columns.tolist() == V5_SHADOW_COMPARISON_COLUMNS
    assert pd.read_csv(paths["summary"], encoding="utf-8-sig").columns.tolist() == V5_SHADOW_SUMMARY_COLUMNS


def test_v5_formula_source_excludes_existing_stability_strength_field():
    source = inspect.getsource(compute_v5_shadow_components)

    assert "rolling_stability" not in source
    assert "regime_stability_final" not in source
    assert "rolling_abs_corr_median" not in source
