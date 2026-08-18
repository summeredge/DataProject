from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import pipeline, service
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import (
    FORMAL_SCREENING_FILES,
    MODEL_DOWNSTREAM_FORMAL_INPUT_FILES,
    CAUSAL_REVIEW_FORMAL_INPUT_FILES,
    DOWNSTREAM_FORMAL_INPUT_FILES,
    REQUIRED_FORMAL_SCREENING_FILES,
    XGB_FORMAL_INPUT_FILES,
    confirm_initial_screening_branch,
    run_initial_screening_branch,
    run_initial_screening_workflow,
)
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


def _write_shadow_input(
    root: Path,
    *,
    with_regime: bool,
    skip_rolling_corr: bool = False,
) -> AnalysisConfig:
    root.mkdir(parents=True, exist_ok=True)
    rows = 240
    index = pd.date_range("2026-01-01", periods=rows, freq="min")
    t = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "target": np.sin(t / 9.0),
            "candidate": np.sin((t - 2.0) / 9.0),
            "other": np.cos(t / 13.0),
        },
        index=index,
    )
    if with_regime:
        frame["capacity"] = np.linspace(-1.0, 1.0, rows)
    input_path = root / "input.csv"
    table = frame.copy()
    table.insert(0, "time", index)
    table.to_csv(input_path, index=False, encoding="utf-8-sig")
    return AnalysisConfig(
        input_path=input_path,
        time_column="time",
        target="target",
        output_dir=root,
        max_lag=3,
        top_k=3,
        preprocess_mode="raw",
        segment_column="capacity" if with_regime else None,
        capacity_columns=["capacity"] if with_regime else [],
        residual_control_columns=["capacity"] if with_regime else [],
        skip_model_lift=True,
        skip_rolling_corr=skip_rolling_corr,
    )


def _shadow_paths(root: Path) -> tuple[Path, Path, Path]:
    branch = root / "screening_branches" / "raw"
    return (
        branch / "ranked_features.csv",
        branch / "screening_v5_shadow_comparison.csv",
        branch / "screening_v5_shadow_summary.csv",
    )


def test_initial_branch_writes_real_shadow_evidence_after_formal_v4(tmp_path):
    config = _write_shadow_input(tmp_path, with_regime=True)

    run_initial_screening_branch(config, branch="raw")

    ranked_path, comparison_path, summary_path = _shadow_paths(tmp_path)
    assert ranked_path.exists()
    assert comparison_path.exists()
    assert summary_path.exists()
    assert not (tmp_path / "screening_v5_shadow_comparison.csv").exists()

    ranked = pd.read_csv(ranked_path, encoding="utf-8-sig")
    comparison = pd.read_csv(comparison_path, encoding="utf-8-sig")
    residual = pd.read_csv(
        tmp_path / "screening_branches" / "raw" / "residual_corr_scores.csv",
        encoding="utf-8-sig",
    )
    row = comparison.loc[comparison["variable"].eq("candidate")].iloc[0]
    formal = ranked.loc[ranked["variable"].eq("candidate")].iloc[0]
    residual_row = residual.loc[residual["variable"].eq("candidate")].iloc[0]

    assert row["final_score_v4"] == pytest.approx(formal["final_score"])
    assert row["rank_v4"] == pytest.approx(formal["driver_rank"])
    assert pd.notna(row["rolling_support"])
    assert pd.notna(row["regime_support"])
    assert row["residual_status"] == residual_row["residual_status"]
    assert row["residual_corr"] == pytest.approx(residual_row["residual_corr"])
    if residual_row["residual_status"] == "ok":
        assert row["residual_support"] == pytest.approx(
            min(max(float(residual_row["residual_corr"]), 0.0), 1.0)
        )
    assert row["base_score_v5"] == pytest.approx(
        row["association_score"] * row["data_quality_score"]
    )


def test_shadow_without_regime_basis_keeps_regime_support_missing(tmp_path):
    config = _write_shadow_input(tmp_path, with_regime=False)

    run_initial_screening_branch(config, branch="raw")

    comparison = pd.read_csv(_shadow_paths(tmp_path)[1], encoding="utf-8-sig")
    candidate = comparison.loc[comparison["variable"].eq("candidate")].iloc[0]
    assert pd.notna(candidate["rolling_support"])
    assert pd.isna(candidate["regime_support"])
    assert pd.notna(candidate["stability_support"])


def test_shadow_stability_does_not_change_formal_v4_or_candidate_outputs(tmp_path):
    enabled_root = tmp_path / "enabled"
    disabled_root = tmp_path / "disabled"
    enabled = _write_shadow_input(enabled_root, with_regime=True, skip_rolling_corr=False)
    disabled = _write_shadow_input(disabled_root, with_regime=True, skip_rolling_corr=True)

    run_initial_screening_branch(enabled, branch="raw")
    run_initial_screening_branch(disabled, branch="raw")

    for filename in ["ranked_features.csv", "recommended_candidates.csv"]:
        left = pd.read_csv(
            enabled_root / "screening_branches" / "raw" / filename,
            encoding="utf-8-sig",
        )
        right = pd.read_csv(
            disabled_root / "screening_branches" / "raw" / filename,
            encoding="utf-8-sig",
        )
        pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_shadow_failure_clears_old_sidecars_but_keeps_formal_outputs(
    monkeypatch,
    tmp_path,
):
    config = _write_shadow_input(tmp_path, with_regime=False, skip_rolling_corr=True)
    branch_dir = tmp_path / "screening_branches" / "raw"
    branch_dir.mkdir(parents=True)
    for name in [
        "screening_v5_shadow_comparison.csv",
        "screening_v5_shadow_summary.csv",
    ]:
        (branch_dir / name).write_text("old-shadow", encoding="utf-8")

    def fail_shadow(*args, **kwargs):
        raise RuntimeError("synthetic shadow failure")

    monkeypatch.setattr(pipeline, "build_v5_shadow_comparison", fail_shadow)
    with pytest.raises(RuntimeError, match="synthetic shadow failure"):
        run_initial_screening_branch(config, branch="raw")

    assert not (branch_dir / "screening_v5_shadow_comparison.csv").exists()
    assert not (branch_dir / "screening_v5_shadow_summary.csv").exists()
    assert (branch_dir / "ranked_features.csv").exists()


def test_shadow_evidence_failure_fails_closed_without_blocking_formal_output(
    monkeypatch,
    tmp_path,
):
    config = _write_shadow_input(tmp_path, with_regime=True, skip_rolling_corr=False)

    def fail_evidence(*args, **kwargs):
        raise RuntimeError("synthetic evidence failure")

    monkeypatch.setattr(service, "_v5_shadow_stability_evidence", fail_evidence)
    run_initial_screening_branch(config, branch="raw")

    ranked_path, comparison_path, summary_path = _shadow_paths(tmp_path)
    assert ranked_path.exists()
    assert comparison_path.exists()
    assert summary_path.exists()
    comparison = pd.read_csv(comparison_path, encoding="utf-8-sig")
    assert comparison["rolling_support"].isna().all()
    assert comparison["regime_support"].isna().all()


def test_shadow_sidecars_are_not_formal_or_downstream_inputs():
    shadow_names = {
        "screening_v5_shadow_comparison.csv",
        "screening_v5_shadow_summary.csv",
    }
    for file_set in [
        REQUIRED_FORMAL_SCREENING_FILES,
        FORMAL_SCREENING_FILES,
        DOWNSTREAM_FORMAL_INPUT_FILES,
        MODEL_DOWNSTREAM_FORMAL_INPUT_FILES,
        CAUSAL_REVIEW_FORMAL_INPUT_FILES,
        XGB_FORMAL_INPUT_FILES,
    ]:
        assert not shadow_names.intersection(file_set)


def test_shadow_sidecars_stay_in_raw_and_processed_branches_after_confirmation(tmp_path):
    config = _write_shadow_input(tmp_path, with_regime=True, skip_rolling_corr=True)
    processed_config = AnalysisConfig(
        **{
            **config.__dict__,
            "preprocess_mode": "lowpass",
        }
    )

    run_initial_screening_workflow(processed_config)
    raw_dir = tmp_path / "screening_branches" / "raw"
    processed_dir = tmp_path / "screening_branches" / "processed"
    for branch_dir in [raw_dir, processed_dir]:
        assert (branch_dir / "screening_v5_shadow_comparison.csv").exists()
        assert (branch_dir / "screening_v5_shadow_summary.csv").exists()
    assert not (tmp_path / "screening_v5_shadow_comparison.csv").exists()
    assert not (tmp_path / "screening_v5_shadow_summary.csv").exists()

    confirm_initial_screening_branch(tmp_path, branch="processed")
    assert not (tmp_path / "screening_v5_shadow_comparison.csv").exists()
    assert not (tmp_path / "screening_v5_shadow_summary.csv").exists()


def test_raw_only_promotion_does_not_copy_shadow_sidecars_to_root(tmp_path):
    config = _write_shadow_input(tmp_path, with_regime=False, skip_rolling_corr=True)

    run_initial_screening_workflow(config)

    raw_dir = tmp_path / "screening_branches" / "raw"
    assert (raw_dir / "screening_v5_shadow_comparison.csv").exists()
    assert (raw_dir / "screening_v5_shadow_summary.csv").exists()
    assert not (tmp_path / "screening_v5_shadow_comparison.csv").exists()
    assert not (tmp_path / "screening_v5_shadow_summary.csv").exists()
