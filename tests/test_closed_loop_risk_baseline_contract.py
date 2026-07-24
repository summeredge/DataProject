from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.screening import (
    CLASS_PRIORITY_FACTORS,
    EVIDENCE_SCORE_CAPS,
    RISK_RELATIVE_PENALTY_WEIGHTS,
    final_ranked_features,
    risk_flags,
)
from chem_ts_corr.web import _build_result_payload


def _golden_inputs() -> tuple[pd.DataFrame, ...]:
    ranked = pd.DataFrame(
        [
            {
                "variable": "safe_upstream",
                "score": 0.90,
                "lag": 2,
                "direction": "variable_leads_target",
                "innovation_score": 0.90,
            },
            {
                "variable": "mv_negative_lag",
                "score": 0.90,
                "lag": -2,
                "direction": "target_leads_variable",
                "innovation_score": 0.90,
            },
            {
                "variable": "pv_negative_lag",
                "score": 0.90,
                "lag": -2,
                "direction": "target_leads_variable",
                "innovation_score": 0.90,
            },
            {
                "variable": "total_formula",
                "score": 0.99,
                "lag": 0,
                "direction": "synchronous",
                "innovation_score": 0.99,
            },
            {
                "variable": "common_load",
                "score": 0.85,
                "lag": 1,
                "direction": "variable_leads_target",
                "innovation_score": 0.85,
            },
            {
                "variable": "poor_signal",
                "score": 0.82,
                "lag": 1,
                "direction": "variable_leads_target",
                "innovation_score": 0.82,
            },
            {
                "variable": "boundary_signal",
                "score": 0.80,
                "lag": 12,
                "direction": "variable_leads_target",
                "innovation_score": 0.80,
            },
            {
                "variable": "unstable_signal",
                "score": 0.78,
                "lag": 3,
                "direction": "variable_leads_target",
                "innovation_score": 0.78,
            },
        ]
    )
    ranked["pearson"] = [0.86, 0.86, 0.86, 0.98, 0.80, 0.78, 0.76, 0.74]
    ranked["spearman"] = [0.84, 0.84, 0.84, 0.97, 0.79, 0.77, 0.75, 0.73]
    ranked["method"] = "pearson"
    ranked["pearson_p"] = 0.001
    ranked["spearman_p"] = 0.002
    ranked["pearson_q"] = 0.008
    ranked["spearman_q"] = 0.016
    ranked["corr_q_value"] = 0.008
    ranked["pearson_r2"] = ranked["pearson"] ** 2
    ranked["spearman_r2"] = ranked["spearman"] ** 2
    ranked["n"] = 120
    variables = ranked["variable"].tolist()
    residual = pd.DataFrame(
        [{"variable": variable, "residual_corr": 0.90} for variable in variables]
        + [{"variable": "common_load", "residual_corr": 0.20}]
    ).drop_duplicates("variable", keep="last")
    stability = pd.DataFrame(
        [
            {
                "variable": variable,
                "regime_stability_final": 0.90,
                "regime_evidence_status": "full_coverage",
            }
            for variable in variables
        ]
        + [
            {
                "variable": "unstable_signal",
                "regime_stability_final": 0.40,
                "regime_evidence_status": "full_coverage",
            }
        ]
    ).drop_duplicates("variable", keep="last")
    model_lift = pd.DataFrame(
        [
            {"variable": variable, "model_lift": 0.20, "model_lift_score": 0.80, "status": "ok"}
            for variable in variables
        ]
    )
    lag_peak_quality = pd.DataFrame(
        [
            {
                "variable": variable,
                "lag_quality": 0.80,
                "lag_boundary_flag": variable == "boundary_signal",
            }
            for variable in variables
        ]
    )
    rolling = pd.DataFrame(
        [{"variable": variable, "rolling_stability": 0.90} for variable in variables]
        + [{"variable": "unstable_signal", "rolling_stability": 0.20}]
    ).drop_duplicates("variable", keep="last")
    diag = pd.DataFrame(
        [
            {
                "variable": variable,
                "missing_rate": 0.30 if variable == "poor_signal" else 0.0,
                "saturation_ratio": 0.0,
                "abnormal_jump_ratio": 0.0,
            }
            for variable in variables
        ]
    )
    risks = risk_flags(
        ranked,
        residual,
        stability,
        diag,
        {variable: "MV" if variable == "mv_negative_lag" else "PV" for variable in variables},
        control_columns=["capacity"],
        lag_peak_quality=lag_peak_quality,
        rolling_corr_scores=rolling,
        model_lift_scores=model_lift,
    )
    return ranked, residual, stability, model_lift, risks, lag_peak_quality, rolling


def _golden_result(*, top_k: int | None = None) -> pd.DataFrame:
    return final_ranked_features(*_golden_inputs(), top_k=top_k, control_columns=["capacity"])


def test_current_closed_loop_auto_detection_contract():
    ranked = pd.DataFrame(
        [
            {"variable": "mv_negative_lag", "score": 0.8, "lag": -1},
            {"variable": "pv_negative_lag", "score": 0.8, "lag": -1},
            {"variable": "mv_zero_lag", "score": 0.8, "lag": 0},
            {"variable": "mv_positive_lag", "score": 0.8, "lag": 1},
        ]
    )
    risks = risk_flags(
        ranked,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {
            "mv_negative_lag": "MV",
            "pv_negative_lag": "PV",
            "mv_zero_lag": "MV",
            "mv_positive_lag": "MV",
        },
        [],
    ).set_index("variable")

    assert bool(risks.loc["mv_negative_lag", "closed_loop_suspect_flag"])
    assert risks.loc["mv_negative_lag", "risk_flags"] == "target_leads_variable"
    assert risks.loc["pv_negative_lag", "risk_flags"] == "target_leads_variable"
    assert not bool(risks.loc["mv_zero_lag", "closed_loop_suspect_flag"])
    assert not bool(risks.loc["mv_zero_lag", "target_leads_variable_flag"])
    assert risks.loc["mv_zero_lag", "risk_flags"] == ""
    assert risks.loc["mv_positive_lag", "risk_flags"] == ""


def test_current_risk_constants_are_frozen():
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
    }
    assert EVIDENCE_SCORE_CAPS == {"strong_formula_leakage": 0.25, "poor_data_quality": 0.44}
    assert CLASS_PRIORITY_FACTORS == {
        "upstream_driver_candidate": 1.00,
        "synchronous_association": 0.90,
        "downstream_response": 0.45,
        "capacity_driven": 0.75,
        "formula_or_derived": 0.25,
        "poor_quality": 0.35,
        "uncertain_candidate": 0.80,
    }


def test_golden_case_freezes_risk_scoring_and_recommendation_baseline():
    result = _golden_result().set_index("variable")
    expected = {
        "safe_upstream": (
            "",
            0,
            0,
            0,
            "none",
            0.850000000000000,
            0.0,
            0.0,
            1.0,
            0.850000000000000,
            "upstream_driver_candidate",
            1.0,
            0.850000000000000,
            1,
            "A",
            "strong_screening_candidate",
            "优先进入机理复核",
        ),
        "boundary_signal": (
            "lag_boundary",
            1,
            0,
            1,
            "weak",
            0.833008382305190,
            0.0,
            0.0,
            1.0,
            0.833008382305190,
            "upstream_driver_candidate",
            1.0,
            0.833008382305190,
            2,
            "A",
            "strong_screening_candidate",
            "优先进入机理复核",
        ),
        "unstable_signal": (
            "unstable_across_regimes;unstable_over_time",
            2,
            0,
            2,
            "weak",
            0.675237646144890,
            0.0,
            0.0,
            1.0,
            0.675237646144890,
            "upstream_driver_candidate",
            1.0,
            0.675237646144890,
            3,
            "B",
            "unstable_candidate",
            "跨工况/时间不稳定，建议复核",
        ),
        "common_load": (
            "common_capacity_driver",
            1,
            1,
            0,
            "medium",
            0.756188557787204,
            0.0,
            0.0,
            1.0,
            0.756188557787204,
            "capacity_driven",
            0.75,
            0.567141418340403,
            4,
            "A",
            "capacity_driven",
            "疑似共同负荷驱动",
        ),
        "mv_negative_lag": (
            "target_leads_variable",
            1,
            0,
            1,
            "weak",
            0.850000000000000,
            0.0,
            0.0,
            1.0,
            0.850000000000000,
            "downstream_response",
            0.45,
            0.382500000000000,
            5,
            "A",
            "state_indicator",
            "更可能是状态指示量",
        ),
        "pv_negative_lag": (
            "target_leads_variable",
            1,
            0,
            1,
            "weak",
            0.850000000000000,
            0.0,
            0.0,
            1.0,
            0.850000000000000,
            "downstream_response",
            0.45,
            0.382500000000000,
            6,
            "A",
            "state_indicator",
            "更可能是状态指示量",
        ),
        "poor_signal": (
            "poor_data_quality",
            1,
            1,
            0,
            "medium",
            0.703376973864192,
            0.0,
            0.0,
            0.44,
            0.44,
            "poor_quality",
            0.35,
            0.154000000000000,
            7,
            "D",
            "poor_quality_variable",
            "数据质量风险，建议剔除",
        ),
        "total_formula": (
            "formula_like;strong_formula_leakage",
            2,
            1,
            1,
            "medium",
            0.864760503272487,
            0.5,
            0.432380251636244,
            0.25,
            0.25,
            "formula_or_derived",
            0.25,
            0.0625,
            8,
            "E",
            "formula_coupled_reference",
            "疑似公式耦合，仅参考",
        ),
    }
    fields = ["risk_flags", "risk_count", "strong_risk_count", "weak_risk_count", "risk_level"]
    numeric = [
        "evidence_score",
        "risk_penalty_rate",
        "risk_penalty",
        "risk_score_cap",
        "final_score",
        "driver_priority_factor",
        "driver_priority_score",
    ]
    for variable, values in expected.items():
        row = result.loc[variable]
        assert tuple(row[field] for field in fields) == values[:5]
        for field, value in zip(numeric, (*values[5:10], values[11], values[12])):
            assert row[field] == pytest.approx(value)
        assert row["candidate_class"] == values[10]
        assert row["driver_rank"] == values[13]
        assert row["candidate_grade"] == values[14]
        assert row["recommended_use"] == values[15]
        assert row["recommended_action"] == values[16]


def test_risk_flags_csv_schema_is_frozen(tmp_path: Path):
    risk_path = tmp_path / "risk_flags.csv"
    _golden_inputs()[4].to_csv(risk_path, index=False, encoding="utf-8-sig")
    frame = pd.read_csv(risk_path, encoding="utf-8-sig")

    assert frame.columns.tolist() == [
        "variable",
        "formula_like_flag",
        "strong_formula_leakage_flag",
        "common_capacity_driver_flag",
        "closed_loop_suspect_flag",
        "target_leads_variable_flag",
        "unstable_across_regimes_flag",
        "unstable_over_time_flag",
        "lag_boundary_flag",
        "low_model_lift_flag",
        "poor_data_quality_flag",
        "residual_collinearity_flag",
        "data_quality_score",
        "risk_flags",
        "risk_count",
        "strong_risk_count",
        "weak_risk_count",
        "risk_level",
        "human_reason",
    ]
    assert not {
        "manual_closed_loop_variables",
        "manual_non_closed_loop_variables",
        "manual_closed_loop_status",
        "closed_loop_evidence_level",
        "closed_loop_evidence_source",
        "closed_loop_conflict",
        "auto_closed_loop_score",
        "original_driver_rank",
    }.intersection(frame.columns)


def test_no_manual_input_keeps_golden_result_deterministic_and_statistics_separate():
    first = _golden_result()
    second = _golden_result()
    pd.testing.assert_frame_equal(first, second)
    indexed = first.set_index("variable")

    assert indexed.loc["mv_negative_lag", "evidence_score"] == pytest.approx(0.85)
    assert indexed.loc["pv_negative_lag", "evidence_score"] == pytest.approx(0.85)
    assert indexed.loc["mv_negative_lag", "driver_priority_score"] == pytest.approx(0.3825)
    assert indexed.loc["pv_negative_lag", "driver_priority_score"] == pytest.approx(0.3825)
    assert indexed.loc["safe_upstream", "raw_corr"] == pytest.approx(0.90)
    assert indexed.loc["mv_negative_lag", "pearson"] == pytest.approx(0.86)
    assert indexed.loc["mv_negative_lag", "spearman"] == pytest.approx(0.84)
    assert indexed.loc["mv_negative_lag", "method"] == "pearson"
    assert indexed.loc["mv_negative_lag", "lag"] == -2
    assert indexed.loc["mv_negative_lag", "direction"] == "target_leads_variable"
    assert indexed.loc["mv_negative_lag", "association_score"] == pytest.approx(0.90)
    assert indexed.loc["safe_upstream", "innovation_score"] == pytest.approx(0.90)
    assert indexed.loc["safe_upstream", "independent_signal_score"] == pytest.approx(0.90)


def test_top_k_and_global_rank_baseline_are_frozen():
    full = _golden_result()
    top_three = _golden_result(top_k=3)

    assert full["variable"].tolist() == [
        "safe_upstream",
        "boundary_signal",
        "unstable_signal",
        "common_load",
        "mv_negative_lag",
        "pv_negative_lag",
        "poor_signal",
        "total_formula",
    ]
    assert full["driver_rank"].tolist() == list(range(1, 9))
    assert top_three["variable"].tolist() == ["safe_upstream", "boundary_signal", "unstable_signal"]
    assert top_three["driver_rank"].tolist() == [1, 2, 3]


def test_force_include_and_control_exclusion_keep_global_rank_semantics():
    result = final_ranked_features(
        *_golden_inputs(),
        top_k=1,
        force_include_variables=["total_formula"],
        control_columns=["safe_upstream"],
    )

    assert result["variable"].tolist() == ["boundary_signal", "total_formula"]
    assert result["driver_rank"].tolist() == [2, 8]
    assert bool(result.set_index("variable").loc["total_formula", "force_included"])


def test_output_schema_and_future_closed_loop_fields_are_frozen():
    result = _golden_result()
    assert result.columns.tolist() == [
        "variable",
        "lag",
        "direction",
        "pearson",
        "spearman",
        "method",
        "pearson_p",
        "spearman_p",
        "pearson_q",
        "spearman_q",
        "corr_q_value",
        "pearson_r2",
        "spearman_r2",
        "n",
        "raw_corr",
        "association_score",
        "correlation_strength",
        "correlation_direction",
        "statistical_significance",
        "innovation_score",
        "innovation_lag",
        "innovation_direction",
        "innovation_sign",
        "innovation_status",
        "residual_corr",
        "independent_signal_score",
        "independent_score",
        "residual_status",
        "correlation_evidence_score",
        "correlation_evidence_status",
        "regime_stability_final",
        "regime_consistency_score",
        "regime_coverage",
        "regime_strength_consistency",
        "regime_sign_consistency",
        "regime_lag_consistency",
        "regime_count",
        "regime_status",
        "rolling_stability",
        "rolling_status",
        "stability_score",
        "lag_quality",
        "lag_quality_status",
        "lag_boundary_flag",
        "temporal_score",
        "temporal_consistency",
        "model_lift_score",
        "model_lift_status",
        "prediction_score",
        "predictive_score",
        "data_quality_score",
        "evidence_strength",
        "evidence_available_count",
        "evidence_completeness",
        "evidence_confidence",
        "evidence_coverage_status",
        "evidence_missing_items",
        "evidence_score_low",
        "evidence_score_high",
        "score_method",
        "risk_count",
        "strong_risk_count",
        "weak_risk_count",
        "risk_level",
        "human_reason",
        "risk_flags",
        "evidence_score",
        "risk_penalty_rate",
        "risk_penalty",
        "risk_score_cap",
        "risk_cap_reason",
        "final_score",
        "candidate_driver_score",
        "driver_evidence_summary",
        "association_rank",
        "candidate_class",
        "driver_priority_factor",
        "driver_priority_score",
        "driver_rank",
        "candidate_grade",
        "recommended_use",
        "recommended_action",
        "force_included",
        "closed_loop_context",
        "closed_loop_status",
        "closed_loop_reason",
    ]
    future_fields = {
        "manual_closed_loop_status",
        "closed_loop_evidence_level",
        "closed_loop_evidence_source",
        "closed_loop_conflict",
        "auto_closed_loop_score",
        "original_driver_rank",
    }
    assert not future_fields.intersection(result.columns)
    assert {"closed_loop_context", "closed_loop_status", "closed_loop_reason"}.issubset(result.columns)
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")
    for token in [
        "manual_closed_loop_variables",
        "manual_non_closed_loop_variables",
        "confirmed_closed_loop",
        "confirmed_not_closed_loop",
        "auto_closed_loop_score",
        "update_risk_annotations",
        "original_driver_rank",
    ]:
        assert token not in source


def test_csv_and_web_payload_preserve_existing_result_without_manual_fields(tmp_path: Path):
    result = _golden_result()
    ranked_path = tmp_path / "ranked_features.csv"
    result.to_csv(ranked_path, index=False, encoding="utf-8-sig")
    before = ranked_path.read_bytes()
    (tmp_path / "summary.md").write_text("- rows_after_preprocess: 8\n", encoding="utf-8")
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path)

    payload = _build_result_payload("a" * 32, tmp_path, config)

    assert ranked_path.read_bytes() == before
    assert (
        pd.read_csv(ranked_path, encoding="utf-8-sig").columns.tolist() == result.columns.tolist()
    )
    assert [row["variable"] for row in payload["rankedFeatures"]] == result["variable"].tolist()
    assert payload["rankedFeatures"][0]["driver_rank"] == 1
    assert payload["rankedFeatures"][0]["final_score"] == pytest.approx(0.85)
    assert not any(key.startswith("manual_closed_loop") for key in payload["rankedFeatures"][0])
