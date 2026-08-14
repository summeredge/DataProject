from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from chem_ts_corr import web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import (
    _build_preprocessing_context,
    _downstream_config_from_context,
    _write_context,
    load_analysis_source_frame,
    prepare_downstream_analysis_context,
    run_initial_screening_comparison,
    run_initial_screening_workflow,
)
from chem_ts_corr.preprocess import preprocess_frame, transform_frame
from chem_ts_corr.screening import rolling_corr_scores


def _config(tmp_path: Path, **overrides) -> AnalysisConfig:
    values = {
        "input_path": tmp_path / "input.csv",
        "time_column": "time",
        "target": "target",
        "output_dir": tmp_path / "run",
        "max_lag": 3,
        "top_k": 3,
        "preprocess_mode": "raw",
    }
    values.update(overrides)
    return AnalysisConfig(**values)


def _write_input(config: AnalysisConfig, periods: int = 80) -> pd.DatetimeIndex:
    times = pd.date_range("2026-08-14 08:00", periods=periods, freq="1min")
    pd.DataFrame(
        {
            "time": times,
            "target": range(periods),
            "predictor": [value * 2 for value in range(periods)],
        }
    ).to_csv(config.input_path, index=False, encoding="utf-8-sig")
    return times


def test_analysis_source_applies_exclusions_before_preprocessing_and_keeps_input_file(tmp_path):
    config = _config(
        tmp_path,
        exclude_windows=[{"start": "2026-08-14T08:20:00", "end": "2026-08-14T08:29:00"}],
    )
    times = _write_input(config)
    original = config.input_path.read_bytes()

    source = load_analysis_source_frame(config)
    cleaned = preprocess_frame(source, "target", "1min", 0.7)

    assert set(times[20:30]).isdisjoint(source.index)
    assert set(times[20:30]).isdisjoint(cleaned.index)
    assert config.input_path.read_bytes() == original


def test_exclusion_gap_resets_lowpass_and_prevents_cross_gap_difference(tmp_path):
    config = _config(
        tmp_path,
        exclude_windows=[{"start": "2026-08-14T08:04:00", "end": "2026-08-14T08:05:00"}],
    )
    times = _write_input(config, periods=12)
    source = load_analysis_source_frame(config)
    cleaned = preprocess_frame(source, "target", None, 0.7)

    lowpass = transform_frame(cleaned, "lowpass", detrend_window=3, lowpass_tau_minutes=5.0)
    diff = transform_frame(cleaned, "lowpass_diff", detrend_window=3, diff_interval_minutes=1.0)

    assert lowpass.loc[times[6], "target"] == source.loc[times[6], "target"]
    assert times[6] not in diff.index


def test_resample_keeps_legal_boundary_bins_without_merging_segments(tmp_path):
    config = _config(
        tmp_path,
        exclude_windows=[{"start": "2026-08-14T08:14:00", "end": "2026-08-14T08:15:00"}],
    )
    times = _write_input(config)

    source = load_analysis_source_frame(config)
    resampled = preprocess_frame(source, "target", "5min", 0.7)

    assert set(times[14:16]).isdisjoint(source.index)
    assert resampled.loc[pd.Timestamp("2026-08-14 08:10"), "target"] == 11.5
    assert resampled.loc[pd.Timestamp("2026-08-14 08:16"), "target"] == 18.0
    assert pd.Timestamp("2026-08-14 08:15") not in resampled.index


def test_rolling_correlation_does_not_cross_exclusion_gap(tmp_path):
    config = _config(
        tmp_path,
        exclude_windows=[{"start": "2026-08-14T08:20:00", "end": "2026-08-14T08:27:00"}],
    )
    _write_input(config, periods=48)
    source = load_analysis_source_frame(config)

    result = rolling_corr_scores(
        source, "target", ["predictor"], max_lag=1, window=12, min_periods=6
    )

    assert int(result.iloc[0]["valid_window_count"]) == 30


def test_rolling_correlation_keeps_existing_contiguous_behavior(tmp_path):
    config = _config(tmp_path)
    _write_input(config, periods=40)
    source = load_analysis_source_frame(config)

    result = rolling_corr_scores(
        source, "target", ["predictor"], max_lag=1, window=12, min_periods=6
    )

    assert int(result.iloc[0]["valid_window_count"]) == 35


def test_raw_and_processed_branches_receive_the_same_excluded_timestamps(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        preprocess_mode="lowpass",
        resample_rule="5min",
        exclude_windows=[{"start": "2026-08-14T08:20:00", "end": "2026-08-14T08:29:00"}],
    )
    _write_input(config)
    import chem_ts_corr.pipeline as pipeline

    captured: dict[str, pd.DatetimeIndex] = {}
    original = pipeline.analyze_initial_screening_branch_frame

    def capture(frame, config_arg, progress_callback=None):
        captured[config_arg.preprocess_mode] = frame.index.copy()
        return original(frame, config_arg, progress_callback=progress_callback)

    monkeypatch.setattr(pipeline, "analyze_initial_screening_branch_frame", capture)
    run_initial_screening_comparison(config)

    assert captured["raw"].equals(captured["lowpass"])
    assert pd.Timestamp("2026-08-14 08:20") not in captured["raw"]


def test_run_context_freezes_exclusion_configuration_for_downstream(tmp_path):
    config = _config(
        tmp_path,
        exclude_windows=[{"start": "2026-08-14T08:10:00", "end": "2026-08-14T08:15:00"}],
    )
    _write_input(config)
    context = _build_preprocessing_context(
        config,
        active_screening_branch="raw",
        active_preprocessing_mode="raw",
        branch_selection_status="not_required",
    )
    changed_current_config = replace(config, exclude_windows=[])

    downstream_config = _downstream_config_from_context(changed_current_config, context)

    assert context["excluded_rows"] == 6
    assert context["remaining_rows"] == 74
    assert downstream_config.exclude_windows == config.exclude_windows


def test_public_downstream_context_uses_frozen_exclusion_windows(tmp_path):
    config = _config(
        tmp_path,
        exclude_windows=[{"start": "2026-08-14T08:10:00", "end": "2026-08-14T08:15:00"}],
    )
    _write_input(config)
    context = _build_preprocessing_context(
        config,
        active_screening_branch="raw",
        active_preprocessing_mode="raw",
        branch_selection_status="not_required",
    )
    _write_context(config.output_dir, context)
    (config.output_dir / "ranked_features.csv").write_text("variable\n", encoding="utf-8-sig")
    (config.output_dir / "recommended_candidates.csv").write_text("variable\n", encoding="utf-8-sig")

    _, downstream_config = prepare_downstream_analysis_context(
        config.output_dir,
        base_config=replace(config, exclude_windows=[]),
        required_formal_files=[],
    )

    assert downstream_config.exclude_windows == config.exclude_windows
    assert pd.Timestamp("2026-08-14 08:10") not in web._numeric_frame(downstream_config).index


def test_empty_exclusions_keep_analysis_source_baseline(tmp_path):
    config = _config(tmp_path)
    _write_input(config)

    source = load_analysis_source_frame(config)
    baseline = pd.read_csv(config.input_path, encoding="utf-8-sig")
    baseline["time"] = pd.to_datetime(baseline["time"])
    baseline = baseline.set_index("time")

    pd.testing.assert_frame_equal(source, baseline)


def test_empty_exclusions_keep_formal_raw_screening_baseline(tmp_path):
    default_config = _config(tmp_path, output_dir=tmp_path / "default")
    _write_input(default_config)
    explicit_empty_config = replace(
        default_config,
        output_dir=tmp_path / "explicit_empty",
        exclude_windows=[],
    )

    run_initial_screening_workflow(default_config)
    run_initial_screening_workflow(explicit_empty_config)

    for filename, columns in {
        "ranked_features.csv": ["variable", "final_score", "driver_rank", "lag"],
        "recommended_candidates.csv": ["variable", "candidate_pool_rank"],
    }.items():
        default = pd.read_csv(default_config.output_dir / filename, encoding="utf-8-sig")
        explicit_empty = pd.read_csv(
            explicit_empty_config.output_dir / filename, encoding="utf-8-sig"
        )
        pd.testing.assert_frame_equal(default[columns], explicit_empty[columns])
