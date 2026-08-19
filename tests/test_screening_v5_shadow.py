from __future__ import annotations

import json
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import pipeline, screening, service
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
    compute_v5_score_components,
    write_v5_shadow_outputs,
)


def test_v5_base_score_is_association_times_data_quality():
    result = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=0.5,
    )

    assert result["base_score_v5"] == pytest.approx(0.4)
    assert result["evidence_score_v5"] == pytest.approx(0.4)
    assert result["final_score_v5"] == pytest.approx(0.4)


def test_v5_residual_bonus_requires_ok_and_preserves_zero_semantics():
    supported = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=1.0,
        residual_corr=0.6,
        residual_status="ok",
    )
    measured_zero = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=1.0,
        residual_corr=0.0,
        residual_status="ok",
    )
    unavailable_zero = compute_v5_score_components(
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
    assert unavailable_zero["final_score_v5"] == pytest.approx(0.8)


def test_v5_residual_is_support_only_and_cannot_reduce_base_score():
    low = compute_v5_score_components(
        association_score=0.75,
        data_quality_score=0.8,
        residual_corr=0.01,
        residual_status="ok",
    )
    high = compute_v5_score_components(
        association_score=0.75,
        data_quality_score=0.8,
        residual_corr=0.99,
        residual_status="ok",
    )

    assert low["final_score_v5"] >= low["base_score_v5"]
    assert high["final_score_v5"] > low["final_score_v5"]


def test_v5_stability_uses_pure_consistency_and_geometric_merge():
    result = compute_v5_score_components(
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
    result = compute_v5_score_components(
        association_score=0.7,
        data_quality_score=1.0,
    )

    assert pd.isna(result["rolling_support"])
    assert pd.isna(result["regime_support"])
    assert pd.isna(result["stability_support"])
    assert result["stability_bonus_rate"] == pytest.approx(0.0)
    assert result["final_score_v5"] == pytest.approx(result["base_score_v5"])


def test_v5_support_statuses_distinguish_skipped_failures_insufficient_and_zero():
    rolling_zero = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=1.0,
        rolling_sign_consistency=0.0,
        rolling_corr_iqr=0.0,
        rolling_support_status="ok",
    )
    rolling_skipped = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=1.0,
        rolling_sign_consistency=1.0,
        rolling_corr_iqr=0.0,
        rolling_support_status="not_computed",
    )
    rolling_failed = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=1.0,
        rolling_support_status="calculation_failed",
    )
    rolling_insufficient = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=1.0,
        rolling_support_status="insufficient_data",
    )
    regime_zero = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=1.0,
        regime_coverage=1.0,
        regime_strength_consistency=1.0,
        regime_sign_consistency=0.0,
        regime_lag_consistency=1.0,
        regime_support_status="ok",
    )
    regime_no_basis = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=1.0,
        regime_support_status="no_regime_basis",
    )
    regime_insufficient = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=1.0,
        regime_support_status="insufficient_regimes",
    )
    regime_metrics = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=1.0,
        regime_support_status="insufficient_metrics",
    )
    regime_failed = compute_v5_score_components(
        association_score=0.8,
        data_quality_score=1.0,
        regime_support_status="calculation_failed",
    )

    assert rolling_zero["rolling_support"] == pytest.approx(0.0)
    assert rolling_zero["rolling_support_status"] == "ok"
    for result, status in [
        (rolling_skipped, "not_computed"),
        (rolling_failed, "calculation_failed"),
        (rolling_insufficient, "insufficient_data"),
    ]:
        assert pd.isna(result["rolling_support"])
        assert result["rolling_support_status"] == status
    assert regime_zero["regime_support"] == pytest.approx(0.0)
    assert regime_zero["regime_support_status"] == "ok"
    for result, status in [
        (regime_no_basis, "no_regime_basis"),
        (regime_insufficient, "insufficient_regimes"),
        (regime_metrics, "insufficient_metrics"),
        (regime_failed, "calculation_failed"),
    ]:
        assert pd.isna(result["regime_support"])
        assert result["regime_support_status"] == status


def test_v5_total_support_bonus_is_capped_at_twenty_percent():
    result = compute_v5_score_components(
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
    target_leads = compute_v5_score_components(
        association_score=0.9,
        data_quality_score=1.0,
        residual_corr=1.0,
        residual_status="ok",
        temporal_direction_status="target_leads_supported",
    )
    variable_leads = compute_v5_score_components(
        association_score=0.9,
        data_quality_score=1.0,
        temporal_direction_status="variable_leads_supported",
    )

    assert target_leads["evidence_score_v5"] == pytest.approx(0.99)
    assert target_leads["final_score_v5"] == pytest.approx(0.25)
    assert variable_leads["final_score_v5"] == pytest.approx(0.9)


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
    source = inspect.getsource(compute_v5_score_components)

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


def test_initial_branch_writes_formal_v5_without_automatic_shadow_sidecars(tmp_path):
    config = _write_shadow_input(tmp_path, with_regime=True)

    run_initial_screening_branch(config, branch="raw")

    ranked_path, comparison_path, summary_path = _shadow_paths(tmp_path)
    ranked = pd.read_csv(ranked_path, encoding="utf-8-sig")
    assert ranked["score_method"].eq("initial_association_temporal_v5").all()
    assert ranked["final_score"].is_monotonic_decreasing
    assert ranked["driver_rank"].tolist() == list(range(1, len(ranked) + 1))
    assert not comparison_path.exists()
    assert not summary_path.exists()


def test_formal_v5_without_regime_basis_keeps_regime_support_missing(tmp_path):
    config = _write_shadow_input(tmp_path, with_regime=False)
    raw = pipeline.load_analysis_source_frame(config)

    tables = service.analyze_initial_screening_branch_frame(raw, config)

    regime = tables.shadow_regime_stability.set_index("variable").loc["candidate"]
    rolling = tables.shadow_rolling_corr_scores.set_index("variable").loc["candidate"]
    assert regime["regime_support_status"] == "no_regime_basis"
    assert rolling["rolling_support_status"] == "ok"
    assert tables.ranked_features["score_method"].eq("initial_association_temporal_v5").all()


def test_formal_v5_support_never_reduces_base_score(tmp_path):
    config = _write_shadow_input(tmp_path, with_regime=True)
    raw = pipeline.load_analysis_source_frame(config)

    tables = service.analyze_initial_screening_branch_frame(raw, config)
    ranked = tables.ranked_features
    base = ranked["association_score"] * ranked["data_quality_score"]
    non_target_leads = ~ranked["temporal_direction_status"].eq("target_leads_supported")

    assert (ranked.loc[non_target_leads, "final_score"] >= base.loc[non_target_leads] - 1e-12).all()


def test_formal_branch_clears_obsolete_shadow_sidecars(tmp_path):
    config = _write_shadow_input(tmp_path, with_regime=False, skip_rolling_corr=True)
    branch_dir = tmp_path / "screening_branches" / "raw"
    branch_dir.mkdir(parents=True)
    for name in ["screening_v5_shadow_comparison.csv", "screening_v5_shadow_summary.csv"]:
        (branch_dir / name).write_text("old-shadow", encoding="utf-8")

    timings = run_initial_screening_branch(config, branch="raw")

    assert set(timings) == {
        "read_data_seconds",
        "analysis_core_seconds",
        "write_outputs_seconds",
        "pipeline_total_seconds",
    }
    assert not (branch_dir / "screening_v5_shadow_comparison.csv").exists()
    assert not (branch_dir / "screening_v5_shadow_summary.csv").exists()
    assert (branch_dir / "ranked_features.csv").exists()


def test_raw_workflow_promotes_formal_v5(tmp_path):
    config = _write_shadow_input(tmp_path, with_regime=False, skip_rolling_corr=True)

    result = run_initial_screening_workflow(config)

    assert result["branch"] == "raw"
    ranked = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")
    assert ranked["score_method"].eq("initial_association_temporal_v5").all()
    context = json.loads((tmp_path / "preprocessing_context.json").read_text(encoding="utf-8"))
    assert context["branch_selection_status"] == "not_required"


def test_processed_workflow_keeps_both_formal_v5_branches(tmp_path):
    base = _write_shadow_input(tmp_path, with_regime=True, skip_rolling_corr=True)
    config = replace(base, preprocess_mode="lowpass")

    run_initial_screening_workflow(config)

    branches_root = tmp_path / "screening_branches"
    for branch in ["raw", "processed"]:
        ranked = pd.read_csv(branches_root / branch / "ranked_features.csv", encoding="utf-8-sig")
        assert ranked["score_method"].eq("initial_association_temporal_v5").all()
    assert (tmp_path / "preprocessing_comparison.csv").exists()
    context = json.loads((tmp_path / "preprocessing_context.json").read_text(encoding="utf-8"))
    assert context["branch_selection_status"] == "awaiting_confirmation"


def test_v5_support_evidence_failure_fails_closed_to_zero_bonus(monkeypatch, tmp_path):
    config = _write_shadow_input(tmp_path, with_regime=True, skip_rolling_corr=False)

    def fail_evidence(*args, **kwargs):
        raise RuntimeError("synthetic evidence failure")

    monkeypatch.setattr(service, "_v5_shadow_stability_evidence", fail_evidence)
    run_initial_screening_branch(config, branch="raw")

    ranked = pd.read_csv(_shadow_paths(tmp_path)[0], encoding="utf-8-sig")
    residual = pd.read_csv(
        tmp_path / "screening_branches" / "raw" / "residual_corr_scores.csv",
        encoding="utf-8-sig",
    )
    base = ranked["association_score"] * ranked["data_quality_score"]
    no_temporal_penalty = ~ranked["temporal_direction_status"].eq("target_leads_supported")
    unavailable_names = set(
        residual.loc[~residual["residual_status"].eq("ok"), "variable"].astype(str)
    )
    residual_unavailable = ranked["variable"].astype(str).isin(unavailable_names)
    selected = no_temporal_penalty & residual_unavailable
    assert ranked["score_method"].eq("initial_association_temporal_v5").all()
    assert np.allclose(ranked.loc[selected, "final_score"], base.loc[selected])


def test_v5_rolling_and_regime_failures_are_isolated(monkeypatch, tmp_path):
    original_rolling_scores = screening.rolling_corr_scores
    rolling_config = _write_shadow_input(tmp_path / "rolling", with_regime=True)
    raw = pipeline.load_analysis_source_frame(rolling_config)

    def fail_rolling(*args, **kwargs):
        raise RuntimeError("synthetic rolling failure")

    monkeypatch.setattr(screening, "rolling_corr_scores", fail_rolling)
    rolling_tables = service.analyze_initial_screening_branch_frame(raw, rolling_config)
    rolling_row = rolling_tables.shadow_rolling_corr_scores.set_index("variable").loc["candidate"]
    regime_row = rolling_tables.shadow_regime_stability.set_index("variable").loc["candidate"]
    assert rolling_row["rolling_support_status"] == "calculation_failed"
    assert regime_row["regime_support_status"] == "ok"

    monkeypatch.setattr(screening, "rolling_corr_scores", original_rolling_scores)

    def fail_regime(*args, **kwargs):
        raise RuntimeError("synthetic regime failure")

    monkeypatch.setattr(screening, "regime_scores", fail_regime)
    regime_tables = service.analyze_initial_screening_branch_frame(raw, rolling_config)
    rolling_row = regime_tables.shadow_rolling_corr_scores.set_index("variable").loc["candidate"]
    regime_row = regime_tables.shadow_regime_stability.set_index("variable").loc["candidate"]
    assert rolling_row["rolling_support_status"] == "ok"
    assert regime_row["regime_support_status"] == "calculation_failed"


@pytest.mark.parametrize("mode", ["lowpass_detrend", "lowpass_diff"])
def test_processed_shadow_regime_uses_level_basis_and_processed_correlations(
    monkeypatch,
    tmp_path,
    mode,
):
    config = replace(
        _write_shadow_input(tmp_path / mode, with_regime=True),
        preprocess_mode=mode,
    )
    raw = pipeline.load_analysis_source_frame(config)
    captured: dict[str, pd.Series | None] = {}
    original_regime_scores = screening.regime_scores

    def capture_regime_inputs(*args, **kwargs):
        processed = args[0]
        capacity_column = args[2]
        basis = kwargs.get("regime_basis")
        captured["processed_capacity"] = processed[capacity_column].copy()
        captured["processed_target"] = processed[config.target].copy()
        captured["basis"] = (
            None if basis is None else basis[capacity_column].copy()
        )
        return original_regime_scores(*args, **kwargs)

    monkeypatch.setattr(screening, "regime_scores", capture_regime_inputs)
    service.analyze_initial_screening_branch_frame(raw, config)

    basis = captured["basis"]
    processed_target = captured["processed_target"]
    processed_capacity = captured["processed_capacity"]
    assert basis is not None
    assert processed_target is not None
    assert processed_capacity is not None
    assert basis.index.equals(processed_capacity.index)
    expected_level = pd.Series(
        np.linspace(-1.0, 1.0, len(raw)),
        index=raw.index,
        name="capacity",
    ).reindex(basis.index)
    pd.testing.assert_series_equal(basis, expected_level)
    raw_target = raw[config.target].reindex(processed_target.index)
    assert not np.allclose(
        processed_target.to_numpy(dtype=float),
        raw_target.to_numpy(dtype=float),
    )


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


def test_automatic_shadow_sidecars_are_absent_from_both_branches_and_promotion(tmp_path):
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
        assert not (branch_dir / "screening_v5_shadow_comparison.csv").exists()
        assert not (branch_dir / "screening_v5_shadow_summary.csv").exists()
    assert not (tmp_path / "screening_v5_shadow_comparison.csv").exists()
    assert not (tmp_path / "screening_v5_shadow_summary.csv").exists()

    confirm_initial_screening_branch(tmp_path, branch="processed")
    assert not (tmp_path / "screening_v5_shadow_comparison.csv").exists()
    assert not (tmp_path / "screening_v5_shadow_summary.csv").exists()
    ranked = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")
    assert ranked["score_method"].eq("initial_association_temporal_v5").all()


def test_raw_only_promotion_does_not_create_shadow_sidecars(tmp_path):
    config = _write_shadow_input(tmp_path, with_regime=False, skip_rolling_corr=True)

    run_initial_screening_workflow(config)

    raw_dir = tmp_path / "screening_branches" / "raw"
    assert not (raw_dir / "screening_v5_shadow_comparison.csv").exists()
    assert not (raw_dir / "screening_v5_shadow_summary.csv").exists()
    assert not (tmp_path / "screening_v5_shadow_comparison.csv").exists()
    assert not (tmp_path / "screening_v5_shadow_summary.csv").exists()
