from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import service
from chem_ts_corr.config import NOT_WIRED_ANALYSIS_PREPROCESS_MODES, AnalysisConfig
from chem_ts_corr.pipeline import run_analysis, run_initial_screening_branch
from chem_ts_corr.preprocess import (
    difference_by_physical_interval,
    lowpass_filter_frame,
)
from chem_ts_corr.service import (
    analyze_initial_screening_branch_frame,
    analyze_numeric_frame,
)


ROOT_LEVEL_SCREENING_FILES = [
    "ranked_features.csv",
    "recommended_candidates.csv",
    "causal_review_candidates.csv",
    "lag_scores.csv",
    "near_miss_candidates.csv",
    "diagnostics.csv",
    "risk_flags.csv",
    "lag_peak_quality.csv",
    "summary.md",
]


def _raw_frame() -> pd.DataFrame:
    rows = 120
    time = np.arange(rows, dtype=float)
    controls = [f"control_{index}" for index in range(8)]
    candidates = [f"candidate_{index}" for index in range(5)]
    return pd.DataFrame(
        {
            "target": np.sin(time / 7),
            **{name: np.sin((time + index + 1) / 7) for index, name in enumerate(candidates)},
            **{name: np.cos((time + index + 1) / 7) for index, name in enumerate(controls)},
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="min"),
    )


def _raw_config(tmp_path: Path, **overrides) -> AnalysisConfig:
    controls = [f"control_{index}" for index in range(8)]
    kwargs = {
        "input_path": tmp_path / "input.csv",
        "time_column": "time",
        "target": "target",
        "output_dir": tmp_path,
        "max_lag": 3,
        "top_k": 15,
        "residual_control_columns": controls,
        "force_include_variables": [],
        "enable_model": False,
        "skip_model_lift": True,
        "skip_rolling_corr": True,
    }
    kwargs.update(overrides)
    return AnalysisConfig(**kwargs)


def _write_input(config: AnalysisConfig, frame: pd.DataFrame) -> None:
    config.input_path.parent.mkdir(parents=True, exist_ok=True)
    table = frame.copy()
    table[config.time_column] = frame.index
    table[[config.time_column, *frame.columns]].to_csv(
        config.input_path, index=False, encoding=config.encoding
    )


def _assert_no_root_publish(run_dir: Path) -> None:
    for name in ROOT_LEVEL_SCREENING_FILES:
        assert not (run_dir / name).exists(), f"root-level {name} must not be published"
    assert not (run_dir / "preprocessing_comparison.csv").exists()
    assert not (run_dir / "preprocessing_context.json").exists()


def test_invalid_branch_raises_value_error(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="raw")

    with pytest.raises(ValueError, match="Unknown screening branch"):
        run_initial_screening_branch(config, branch="abc")


@pytest.mark.parametrize(
    "mode", ["lowpass", "lowpass_detrend", "lowpass_diff", "detrend", "diff", "detrend_diff"]
)
def test_raw_branch_rejects_non_raw_modes(tmp_path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode)

    with pytest.raises(ValueError, match="Raw branch requires preprocess_mode='raw'"):
        run_initial_screening_branch(config, branch="raw")


@pytest.mark.parametrize("mode", ["raw", "detrend", "diff", "detrend_diff"])
def test_processed_branch_rejects_raw_and_legacy_modes(tmp_path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode)

    with pytest.raises(ValueError, match="Processed branch requires preprocess_mode"):
        run_initial_screening_branch(config, branch="processed")


def test_raw_branch_writes_only_to_screening_branches_raw(tmp_path):
    config = _raw_config(tmp_path)
    _write_input(config, _raw_frame())

    timings = run_initial_screening_branch(config, branch="raw")

    assert set(timings) == {
        "read_data_seconds",
        "analysis_core_seconds",
        "write_outputs_seconds",
        "pipeline_total_seconds",
    }
    raw_dir = tmp_path / "screening_branches" / "raw"
    assert (raw_dir / "ranked_features.csv").exists()
    assert (raw_dir / "recommended_candidates.csv").exists()
    assert (raw_dir / "causal_review_candidates.csv").exists()
    assert not (tmp_path / "screening_branches" / "processed").exists()
    _assert_no_root_publish(tmp_path)


@pytest.mark.parametrize("mode", ["lowpass", "lowpass_detrend", "lowpass_diff"])
def test_processed_branch_writes_only_to_screening_branches_processed(
    tmp_path, mode: str
):
    config = _raw_config(tmp_path, preprocess_mode=mode)
    _write_input(config, _raw_frame())

    run_initial_screening_branch(config, branch="processed")

    processed_dir = tmp_path / "screening_branches" / "processed"
    assert (processed_dir / "ranked_features.csv").exists()
    assert (processed_dir / "recommended_candidates.csv").exists()
    assert not (tmp_path / "screening_branches" / "raw").exists()
    _assert_no_root_publish(tmp_path)


def test_raw_branch_matches_formal_raw_screening(tmp_path):
    config = _raw_config(tmp_path)
    _write_input(config, _raw_frame())

    run_analysis(config)
    run_initial_screening_branch(config, branch="raw")

    root_dir = tmp_path
    branch_dir = tmp_path / "screening_branches" / "raw"
    formal_ranked = pd.read_csv(root_dir / "ranked_features.csv", encoding="utf-8-sig")
    branch_ranked = pd.read_csv(
        branch_dir / "ranked_features.csv", encoding="utf-8-sig"
    )
    pd.testing.assert_frame_equal(formal_ranked, branch_ranked, check_exact=False)
    for column in [
        "variable",
        "final_score",
        "driver_rank",
        "lag",
        "direction",
        "candidate_grade",
        "risk_flags",
    ]:
        pd.testing.assert_series_equal(
            branch_ranked[column], formal_ranked[column], check_dtype=False
        )
    assert (
        branch_ranked["variable"].head(config.top_k).tolist()
        == formal_ranked["variable"].head(config.top_k).tolist()
    )

    formal_recommended = pd.read_csv(
        root_dir / "recommended_candidates.csv", encoding="utf-8-sig"
    )
    branch_recommended = pd.read_csv(
        branch_dir / "recommended_candidates.csv", encoding="utf-8-sig"
    )
    pd.testing.assert_frame_equal(
        formal_recommended, branch_recommended, check_exact=False
    )
    assert branch_recommended["variable"].tolist() == formal_recommended["variable"].tolist()
    for column in ["candidate_source", "candidate_pool_rank"]:
        pd.testing.assert_series_equal(
            branch_recommended[column],
            formal_recommended[column],
            check_dtype=False,
        )


def test_branch_core_passes_configured_lowpass_parameters(monkeypatch, tmp_path):
    config = _raw_config(
        tmp_path,
        preprocess_mode="lowpass_diff",
        lowpass_tau_minutes=7.5,
        diff_interval_minutes=5.0,
    )
    _write_input(config, _raw_frame())
    calls: list[tuple[str, dict[str, object]]] = []
    original = service.transform_frame

    def spy(frame, mode, detrend_window, **kwargs):
        calls.append((mode, dict(kwargs)))
        return original(frame, mode, detrend_window, **kwargs)

    monkeypatch.setattr(service, "transform_frame", spy)

    run_initial_screening_branch(config, branch="processed")

    assert len(calls) == 1
    mode, kwargs = calls[0]
    assert mode == "lowpass_diff"
    assert kwargs["lowpass_tau_minutes"] == 7.5
    assert kwargs["diff_interval_minutes"] == 5.0


def test_lowpass_diff_branch_uses_five_point_difference(monkeypatch, tmp_path):
    config = _raw_config(
        tmp_path,
        preprocess_mode="lowpass_diff",
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=5.0,
    )
    frame = _raw_frame()
    _write_input(config, frame)
    captured: list[tuple[pd.DataFrame, dict[str, object]]] = []
    original = service.transform_frame

    def spy(frame_input, mode, detrend_window, **kwargs):
        captured.append((frame_input.copy(deep=True), dict(kwargs)))
        return original(frame_input, mode, detrend_window, **kwargs)

    monkeypatch.setattr(service, "transform_frame", spy)

    run_initial_screening_branch(config, branch="processed")

    cleaned, kwargs = captured[0]
    assert kwargs["diff_interval_minutes"] == 5.0
    expected = difference_by_physical_interval(
        lowpass_filter_frame(cleaned, tau_minutes=5.0),
        diff_interval_minutes=5.0,
    ).dropna()
    transformed = original(cleaned, "lowpass_diff", config.detrend_window, **kwargs)
    pd.testing.assert_frame_equal(transformed, expected)
    assert transformed.index[0] == cleaned.index[5]


@pytest.mark.parametrize("mode", sorted(NOT_WIRED_ANALYSIS_PREPROCESS_MODES))
def test_analyze_numeric_frame_still_rejects_lowpass_modes(tmp_path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode)

    with pytest.raises(ValueError, match="analysis/screening flow"):
        analyze_numeric_frame(_raw_frame(), config)


@pytest.mark.parametrize("mode", sorted(NOT_WIRED_ANALYSIS_PREPROCESS_MODES))
def test_run_analysis_still_rejects_lowpass_modes(tmp_path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode)
    _write_input(config, _raw_frame())

    with pytest.raises(ValueError, match="analysis/screening flow"):
        run_analysis(config)


def test_raw_and_processed_branch_outputs_do_not_overwrite_each_other(tmp_path):
    raw_config = _raw_config(tmp_path, preprocess_mode="raw")
    processed_config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(raw_config, _raw_frame())

    run_initial_screening_branch(raw_config, branch="raw")
    raw_ranked_bytes = (tmp_path / "screening_branches/raw/ranked_features.csv").read_bytes()

    run_initial_screening_branch(processed_config, branch="processed")
    assert (tmp_path / "screening_branches/raw/ranked_features.csv").read_bytes() == raw_ranked_bytes
    processed_ranked_bytes = (
        tmp_path / "screening_branches/processed/ranked_features.csv"
    ).read_bytes()

    run_initial_screening_branch(raw_config, branch="raw")
    assert (tmp_path / "screening_branches/raw/ranked_features.csv").exists()
    assert (
        tmp_path / "screening_branches/processed/ranked_features.csv"
    ).read_bytes() == processed_ranked_bytes


def test_raw_branch_rerun_clears_stale_optional_file(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="raw")
    _write_input(config, _raw_frame())
    run_initial_screening_branch(config, branch="raw")
    residual = tmp_path / "screening_branches" / "raw" / "residual_corr_scores.csv"
    assert residual.exists()

    rerun_config = _raw_config(
        tmp_path, preprocess_mode="raw", residual_control_columns=[]
    )
    run_initial_screening_branch(rerun_config, branch="raw")

    assert not residual.exists()
    assert (tmp_path / "screening_branches" / "raw" / "ranked_features.csv").exists()


@pytest.mark.parametrize("mode", ["lowpass", "lowpass_detrend", "lowpass_diff"])
def test_processed_branch_rerun_clears_stale_optional_file(tmp_path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode)
    _write_input(config, _raw_frame())
    run_initial_screening_branch(config, branch="processed")
    residual = (
        tmp_path / "screening_branches" / "processed" / "residual_corr_scores.csv"
    )
    assert residual.exists()

    rerun_config = _raw_config(
        tmp_path, preprocess_mode=mode, residual_control_columns=[]
    )
    run_initial_screening_branch(rerun_config, branch="processed")

    assert not residual.exists()
    assert (
        tmp_path / "screening_branches" / "processed" / "ranked_features.csv"
    ).exists()


@pytest.mark.parametrize(
    ("rerun_branch", "mode"), [("raw", "raw"), ("processed", "lowpass")]
)
def test_rerun_one_branch_does_not_touch_the_other(
    tmp_path, rerun_branch: str, mode: str
):
    raw_config = _raw_config(tmp_path, preprocess_mode="raw")
    processed_config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(raw_config, _raw_frame())
    run_initial_screening_branch(raw_config, branch="raw")
    run_initial_screening_branch(processed_config, branch="processed")

    other_branch = "processed" if rerun_branch == "raw" else "raw"
    other_dir = tmp_path / "screening_branches" / other_branch
    other_before = {
        path.name: path.read_bytes() for path in other_dir.iterdir() if path.is_file()
    }
    assert other_before, f"{other_branch} branch must already have outputs"

    rerun_config = _raw_config(tmp_path, preprocess_mode=mode)
    run_initial_screening_branch(rerun_config, branch=rerun_branch)

    assert (
        tmp_path / "screening_branches" / rerun_branch / "ranked_features.csv"
    ).exists()
    assert other_dir.exists()
    assert {
        path.name: path.read_bytes() for path in other_dir.iterdir() if path.is_file()
    } == other_before


def test_branch_runner_creates_no_comparison_or_context_files(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())

    run_initial_screening_branch(config, branch="processed")

    assert not (tmp_path / "preprocessing_comparison.csv").exists()
    assert not (tmp_path / "preprocessing_context.json").exists()
    processed_dir = tmp_path / "screening_branches" / "processed"
    assert not (processed_dir / "preprocessing_comparison.csv").exists()
    assert not (processed_dir / "preprocessing_context.json").exists()


def _assert_already_differenced_innovation(ranked: pd.DataFrame) -> None:
    assert "innovation_status" in ranked.columns
    assert ranked["innovation_status"].eq("innovation_verified").all()
    pd.testing.assert_series_equal(
        ranked["innovation_lag"].astype(int),
        ranked["lag"].astype(int),
        check_names=False,
    )
    merged = ranked[["innovation_score", "raw_corr"]].dropna()
    assert merged["innovation_score"].to_numpy() == pytest.approx(
        merged["raw_corr"].to_numpy(), rel=0, abs=1e-9
    )


def test_lowpass_diff_innovation_evidence_skips_second_difference(monkeypatch, tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass_diff")
    calls: list[pd.DataFrame] = []
    original = service.difference_by_contiguous_segment

    def spy(frame):
        calls.append(frame)
        return original(frame)

    monkeypatch.setattr(service, "difference_by_contiguous_segment", spy)
    tables = analyze_initial_screening_branch_frame(_raw_frame(), config)

    assert calls == []
    _assert_already_differenced_innovation(tables.ranked_features)


@pytest.mark.parametrize("mode", ["diff", "detrend_diff"])
def test_legacy_diff_modes_remain_already_differenced(monkeypatch, tmp_path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode)
    calls: list[pd.DataFrame] = []
    original = service.difference_by_contiguous_segment

    def spy(frame):
        calls.append(frame)
        return original(frame)

    monkeypatch.setattr(service, "difference_by_contiguous_segment", spy)
    tables = analyze_numeric_frame(_raw_frame(), config)

    assert calls == []
    _assert_already_differenced_innovation(tables.ranked_features)


@pytest.mark.parametrize("mode", ["raw", "lowpass", "lowpass_detrend"])
def test_non_differenced_modes_still_use_innovation_difference(
    monkeypatch, tmp_path, mode: str
):
    config = _raw_config(tmp_path, preprocess_mode=mode)
    calls: list[pd.DataFrame] = []
    original = service.difference_by_contiguous_segment

    def spy(frame):
        calls.append(frame)
        return original(frame)

    monkeypatch.setattr(service, "difference_by_contiguous_segment", spy)
    tables = analyze_initial_screening_branch_frame(_raw_frame(), config)

    assert calls, "non-differenced modes must still use the innovation difference path"
    ranked = tables.ranked_features
    assert "innovation_status" in ranked.columns
    assert ranked["innovation_status"].isin(
        [
            "innovation_verified",
            "innovation_lag_conflict",
            "innovation_sign_conflict",
            "innovation_sign_unknown",
        ]
    ).all()
