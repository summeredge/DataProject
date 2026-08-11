from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chem_ts_corr.pipeline as pipeline
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import (
    COMPARISON_COLUMNS,
    build_preprocessing_comparison,
    run_initial_screening_comparison,
    _validate_comparison_mode,
)


PROCESSED_MODES = ["lowpass", "lowpass_detrend", "lowpass_diff"]


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


def _ranked_frame(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    columns = [
        "variable",
        "final_score",
        "driver_rank",
        "lag",
        "pearson",
        "spearman",
        "risk_flags",
    ]
    return pd.DataFrame(rows, columns=columns)


def _candidate_frame(variables: list[str]) -> pd.DataFrame:
    if not variables:
        return pd.DataFrame(columns=["variable"])
    return pd.DataFrame({"variable": variables})


@pytest.mark.parametrize("mode", PROCESSED_MODES)
def test_processed_modes_run_dual_branches(tmp_path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode)
    _write_input(config, _raw_frame())

    result = run_initial_screening_comparison(config)

    raw_dir = tmp_path / "screening_branches" / "raw"
    processed_dir = tmp_path / "screening_branches" / "processed"
    assert (raw_dir / "ranked_features.csv").exists()
    assert (processed_dir / "ranked_features.csv").exists()
    assert (raw_dir / "recommended_candidates.csv").exists()
    assert (processed_dir / "recommended_candidates.csv").exists()
    assert set(result) == {"raw", "processed", "comparison_path"}
    assert result["comparison_path"] == tmp_path / "preprocessing_comparison.csv"
    assert result["comparison_path"].exists()
    assert not (tmp_path / "preprocessing_context.json").exists()

    comparison = pd.read_csv(
        tmp_path / "preprocessing_comparison.csv", encoding="utf-8-sig"
    )
    assert comparison.columns.tolist() == COMPARISON_COLUMNS
    assert comparison["processed_mode"].eq(mode).all()


def test_comparison_runner_rejects_raw_mode(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="raw")
    _write_input(config, _raw_frame())

    with pytest.raises(ValueError, match="requires preprocess_mode"):
        run_initial_screening_comparison(config)

    assert not (tmp_path / "preprocessing_comparison.csv").exists()


@pytest.mark.parametrize("mode", ["detrend", "diff", "detrend_diff"])
def test_comparison_runner_rejects_legacy_modes(tmp_path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode)
    _write_input(config, _raw_frame())

    with pytest.raises(ValueError, match="requires preprocess_mode"):
        run_initial_screening_comparison(config)

    assert not (tmp_path / "preprocessing_comparison.csv").exists()


def test_original_config_is_not_modified(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass_diff")
    _write_input(config, _raw_frame())

    run_initial_screening_comparison(config)

    assert config.preprocess_mode == "lowpass_diff"


def test_dual_branches_run_independently(monkeypatch, tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    calls: list[tuple[str, str]] = []
    original = pipeline.run_initial_screening_branch

    def spy(config_arg, *, branch, progress_callback=None):
        calls.append((branch, config_arg.preprocess_mode))
        return original(config_arg, branch=branch, progress_callback=progress_callback)

    monkeypatch.setattr(pipeline, "run_initial_screening_branch", spy)

    run_initial_screening_comparison(config)

    assert calls == [("raw", "raw"), ("processed", "lowpass")]


def test_comparison_columns_exact_order():
    raw = _ranked_frame([{"variable": "A", "final_score": 0.5, "driver_rank": 1, "lag": 1}])
    processed = _ranked_frame(
        [{"variable": "A", "final_score": 0.6, "driver_rank": 1, "lag": -1}]
    )

    comparison = build_preprocessing_comparison(
        raw,
        processed,
        _candidate_frame(["A"]),
        _candidate_frame(["A"]),
        processed_mode="lowpass",
        top_k=5,
    )

    assert comparison.columns.tolist() == COMPARISON_COLUMNS


def test_final_score_delta_is_processed_minus_raw():
    raw = _ranked_frame(
        [{"variable": "A", "final_score": 0.40, "driver_rank": 1, "lag": 1}]
    )
    processed = _ranked_frame(
        [{"variable": "A", "final_score": 0.55, "driver_rank": 1, "lag": 1}]
    )

    comparison = build_preprocessing_comparison(
        raw,
        processed,
        _candidate_frame(["A"]),
        _candidate_frame(["A"]),
        processed_mode="lowpass",
        top_k=5,
    )

    row = comparison.iloc[0]
    assert row["raw_final_score"] == pytest.approx(0.40)
    assert row["processed_final_score"] == pytest.approx(0.55)
    assert row["final_score_delta"] == pytest.approx(0.15)


def test_rank_delta_is_raw_minus_processed():
    raw = _ranked_frame(
        [{"variable": "A", "final_score": 0.5, "driver_rank": 8, "lag": 1}]
    )
    processed = _ranked_frame(
        [{"variable": "A", "final_score": 0.6, "driver_rank": 3, "lag": 1}]
    )

    comparison = build_preprocessing_comparison(
        raw,
        processed,
        _candidate_frame(["A"]),
        _candidate_frame(["A"]),
        processed_mode="lowpass",
        top_k=5,
    )

    row = comparison.iloc[0]
    assert row["raw_rank"] == 8
    assert row["processed_rank"] == 3
    # Positive rank_delta means the processed branch improved.
    assert row["rank_delta"] == 5


@pytest.mark.parametrize(
    ("raw_lag", "processed_lag", "expected"),
    [
        (-3, 2, True),
        (5, 1, False),
        (0, 1, True),
        (0, 0, False),
        (None, 1, pd.NA),
        (-2, None, pd.NA),
    ],
)
def test_lag_direction_changed_semantics(raw_lag, processed_lag, expected):
    raw = _ranked_frame(
        [{"variable": "A", "final_score": 0.5, "driver_rank": 1, "lag": raw_lag}]
    )
    processed = _ranked_frame(
        [{"variable": "A", "final_score": 0.6, "driver_rank": 1, "lag": processed_lag}]
    )

    comparison = build_preprocessing_comparison(
        raw,
        processed,
        _candidate_frame(["A"]),
        _candidate_frame(["A"]),
        processed_mode="lowpass",
        top_k=5,
    )

    value = comparison["lag_direction_changed"].iloc[0]
    if pd.isna(expected):
        assert pd.isna(value)
    else:
        assert bool(value) is expected
        assert comparison["raw_best_lag"].iloc[0] == raw_lag
        assert comparison["processed_best_lag"].iloc[0] == processed_lag


def test_union_align_and_missing_side_semantics():
    raw = _ranked_frame(
        [
            {"variable": "A", "final_score": 0.7, "driver_rank": 1, "lag": 2,
             "pearson": 0.8, "spearman": 0.7, "risk_flags": "lag_boundary"},
            {"variable": "B", "final_score": 0.5, "driver_rank": 2, "lag": 1,
             "pearson": 0.6, "spearman": 0.5, "risk_flags": ""},
            {"variable": "C", "final_score": 0.4, "driver_rank": 3, "lag": -1,
             "pearson": 0.5, "spearman": 0.4, "risk_flags": ""},
        ]
    )
    processed = _ranked_frame(
        [
            {"variable": "B", "final_score": 0.6, "driver_rank": 1, "lag": 2,
             "pearson": 0.7, "spearman": 0.6, "risk_flags": ""},
            {"variable": "C", "final_score": 0.5, "driver_rank": 2, "lag": -2,
             "pearson": 0.6, "spearman": 0.5, "risk_flags": "formula_like"},
            {"variable": "D", "final_score": 0.8, "driver_rank": 3, "lag": 1,
             "pearson": 0.9, "spearman": 0.8, "risk_flags": ""},
        ]
    )
    raw_snapshot = raw.copy(deep=True)
    processed_snapshot = processed.copy(deep=True)

    comparison = build_preprocessing_comparison(
        raw,
        processed,
        _candidate_frame(["B", "C"]),
        _candidate_frame(["C", "D"]),
        processed_mode="lowpass_diff",
        top_k=5,
    )

    assert comparison["variable"].tolist() == ["A", "B", "C", "D"]
    pd.testing.assert_frame_equal(raw, raw_snapshot)
    pd.testing.assert_frame_equal(processed, processed_snapshot)

    row_a = comparison.set_index("variable").loc["A"]
    assert bool(row_a["raw_available"])
    assert not bool(row_a["processed_available"])
    assert pd.isna(row_a["processed_final_score"])
    assert pd.isna(row_a["processed_pearson"])
    assert pd.isna(row_a["processed_spearman"])
    assert pd.isna(row_a["processed_best_lag"])
    assert pd.isna(row_a["processed_rank"])
    assert pd.isna(row_a["rank_delta"])
    assert pd.isna(row_a["final_score_delta"])
    assert pd.isna(row_a["lag_direction_changed"])
    assert pd.isna(row_a["processed_risk_tags"])

    row_d = comparison.set_index("variable").loc["D"]
    assert not bool(row_d["raw_available"])
    assert bool(row_d["processed_available"])
    assert pd.isna(row_d["raw_final_score"])
    assert pd.isna(row_d["raw_pearson"])
    assert pd.isna(row_d["raw_spearman"])
    assert pd.isna(row_d["raw_best_lag"])
    assert pd.isna(row_d["raw_rank"])
    assert pd.isna(row_d["raw_risk_tags"])

    row_b = comparison.set_index("variable").loc["B"]
    assert row_b["raw_rank"] == 2
    assert row_b["processed_rank"] == 1
    assert row_b["rank_delta"] == 1
    assert row_b["processed_mode"] == "lowpass_diff"


def test_top_k_is_independent_per_branch():
    raw = _ranked_frame(
        [
            {"variable": "A", "final_score": 0.5, "driver_rank": 2, "lag": 1},
            {"variable": "B", "final_score": 0.4, "driver_rank": 3, "lag": 1},
        ]
    )
    processed = _ranked_frame(
        [
            {"variable": "A", "final_score": 0.6, "driver_rank": 8, "lag": 1},
            {"variable": "C", "final_score": 0.7, "driver_rank": 1, "lag": 1},
        ]
    )

    comparison = build_preprocessing_comparison(
        raw,
        processed,
        _candidate_frame([]),
        _candidate_frame([]),
        processed_mode="lowpass",
        top_k=5,
    ).set_index("variable")

    assert bool(comparison.loc["A", "raw_in_top_k"])
    assert not bool(comparison.loc["A", "processed_in_top_k"])
    assert bool(comparison.loc["C", "processed_in_top_k"])
    assert not bool(comparison.loc["C", "raw_in_top_k"])


def test_candidate_membership_is_independent_per_branch():
    raw = _ranked_frame(
        [
            {"variable": "A", "final_score": 0.5, "driver_rank": 1, "lag": 1},
            {"variable": "B", "final_score": 0.4, "driver_rank": 2, "lag": 1},
            {"variable": "C", "final_score": 0.3, "driver_rank": 3, "lag": 1},
            {"variable": "D", "final_score": 0.2, "driver_rank": 4, "lag": 1},
        ]
    )
    processed = raw.copy(deep=True)

    comparison = build_preprocessing_comparison(
        raw,
        processed,
        _candidate_frame(["A", "C"]),
        _candidate_frame(["B", "C"]),
        processed_mode="lowpass",
        top_k=5,
    ).set_index("variable")

    assert bool(comparison.loc["A", "raw_candidate"])
    assert not bool(comparison.loc["A", "processed_candidate"])
    assert not bool(comparison.loc["B", "raw_candidate"])
    assert bool(comparison.loc["B", "processed_candidate"])
    assert bool(comparison.loc["C", "raw_candidate"])
    assert bool(comparison.loc["C", "processed_candidate"])
    assert not bool(comparison.loc["D", "raw_candidate"])
    assert not bool(comparison.loc["D", "processed_candidate"])


def test_risk_tags_distinguish_absent_from_no_risk():
    raw = _ranked_frame(
        [
            {"variable": "X", "final_score": 0.5, "driver_rank": 1, "lag": 1,
             "risk_flags": ""},
            {"variable": "Y", "final_score": 0.4, "driver_rank": 2, "lag": 1,
             "risk_flags": "formula_like"},
        ]
    )
    processed = _ranked_frame(
        [
            {"variable": "X", "final_score": 0.6, "driver_rank": 1, "lag": 1,
             "risk_flags": "lag_boundary"},
            {"variable": "Z", "final_score": 0.3, "driver_rank": 2, "lag": 1,
             "risk_flags": ""},
        ]
    )

    comparison = build_preprocessing_comparison(
        raw,
        processed,
        _candidate_frame([]),
        _candidate_frame([]),
        processed_mode="lowpass",
        top_k=5,
    ).set_index("variable")

    assert comparison.loc["X", "raw_risk_tags"] == ""
    assert comparison.loc["X", "processed_risk_tags"] == "lag_boundary"
    assert comparison.loc["Y", "raw_risk_tags"] == "formula_like"
    assert pd.isna(comparison.loc["Y", "processed_risk_tags"])
    assert pd.isna(comparison.loc["Z", "raw_risk_tags"])
    assert comparison.loc["Z", "processed_risk_tags"] == ""


def test_risk_tags_semantics_survive_csv_roundtrip(tmp_path):
    """The CSV itself must keep 'exists with no risk' distinct from 'absent'."""
    raw = _ranked_frame(
        [
            {"variable": "X", "final_score": 0.5, "driver_rank": 1, "lag": 1,
             "risk_flags": ""},
            {"variable": "Y", "final_score": 0.4, "driver_rank": 2, "lag": 1,
             "risk_flags": "formula_like"},
        ]
    )
    processed = _ranked_frame(
        [
            {"variable": "X", "final_score": 0.6, "driver_rank": 1, "lag": 1,
             "risk_flags": "lag_boundary"},
            {"variable": "Z", "final_score": 0.3, "driver_rank": 2, "lag": 1,
             "risk_flags": ""},
        ]
    )
    comparison = build_preprocessing_comparison(
        raw,
        processed,
        _candidate_frame([]),
        _candidate_frame([]),
        processed_mode="lowpass",
        top_k=5,
    )
    comparison_path = tmp_path / "preprocessing_comparison.csv"
    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN",
    )

    roundtrip = pd.read_csv(
        comparison_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    ).set_index("variable")

    # X exists on both sides: raw has no risk (empty string), processed has one.
    assert roundtrip.loc["X", "raw_risk_tags"] == ""
    assert roundtrip.loc["X", "processed_risk_tags"] == "lag_boundary"
    # Y exists only in raw; the processed side is absent, not 'no risk'.
    assert roundtrip.loc["Y", "raw_risk_tags"] == "formula_like"
    assert roundtrip.loc["Y", "processed_risk_tags"] == "NaN"
    # Z exists only in processed with no risk; the raw side is absent.
    assert roundtrip.loc["Z", "raw_risk_tags"] == "NaN"
    assert roundtrip.loc["Z", "processed_risk_tags"] == ""


def test_no_root_formal_results_published(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())

    run_initial_screening_comparison(config)

    assert (tmp_path / "preprocessing_comparison.csv").exists()
    for name in [
        "ranked_features.csv",
        "recommended_candidates.csv",
        "causal_review_candidates.csv",
    ]:
        assert not (tmp_path / name).exists()


def test_no_context_file_generated(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())

    run_initial_screening_comparison(config)

    assert not (tmp_path / "preprocessing_context.json").exists()
    assert not (
        tmp_path / "screening_branches" / "raw" / "preprocessing_context.json"
    ).exists()
    assert not (
        tmp_path / "screening_branches" / "processed" / "preprocessing_context.json"
    ).exists()


def test_no_recommendation_columns_in_comparison():
    raw = _ranked_frame([{"variable": "A", "final_score": 0.5, "driver_rank": 1, "lag": 1}])
    processed = _ranked_frame(
        [{"variable": "A", "final_score": 0.6, "driver_rank": 1, "lag": 1}]
    )

    comparison = build_preprocessing_comparison(
        raw,
        processed,
        _candidate_frame([]),
        _candidate_frame([]),
        processed_mode="lowpass",
        top_k=5,
    )

    assert not {
        "recommended_branch",
        "best_branch",
        "selected_branch",
        "comparison_score",
    } & set(comparison.columns)


@pytest.mark.parametrize("failing_branch", ["raw", "processed"])
def test_branch_failure_skips_comparison_write(monkeypatch, tmp_path, failing_branch):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    original = pipeline.run_initial_screening_branch

    def spy(config_arg, *, branch, progress_callback=None):
        if branch == failing_branch:
            raise RuntimeError("synthetic branch failure")
        return original(config_arg, branch=branch, progress_callback=progress_callback)

    monkeypatch.setattr(pipeline, "run_initial_screening_branch", spy)

    with pytest.raises(RuntimeError, match="synthetic branch failure"):
        run_initial_screening_comparison(config)

    assert not (tmp_path / "preprocessing_comparison.csv").exists()


@pytest.mark.parametrize("failing_branch", ["raw", "processed"])
def test_failed_rerun_clears_stale_comparison(monkeypatch, tmp_path, failing_branch):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    comparison_path = tmp_path / "preprocessing_comparison.csv"
    comparison_path.write_text("old-comparison\n", encoding="utf-8")
    original = pipeline.run_initial_screening_branch

    def spy(config_arg, *, branch, progress_callback=None):
        if branch == failing_branch:
            raise RuntimeError("synthetic branch failure")
        return original(config_arg, branch=branch, progress_callback=progress_callback)

    monkeypatch.setattr(pipeline, "run_initial_screening_branch", spy)

    with pytest.raises(RuntimeError, match="synthetic branch failure"):
        run_initial_screening_comparison(config)

    assert not comparison_path.exists()


def test_successful_rerun_replaces_stale_comparison(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass_diff")
    _write_input(config, _raw_frame())
    comparison_path = tmp_path / "preprocessing_comparison.csv"
    comparison_path.write_text("old-comparison\n", encoding="utf-8")

    run_initial_screening_comparison(config)

    assert comparison_path.exists()
    assert "old-comparison" not in comparison_path.read_text(encoding="utf-8-sig")
    comparison = pd.read_csv(
        comparison_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    assert comparison["processed_mode"].eq("lowpass_diff").all()


def test_invalid_mode_does_not_delete_existing_comparison(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="raw")
    comparison_path = tmp_path / "preprocessing_comparison.csv"
    comparison_path.write_text("old-comparison\n", encoding="utf-8")

    with pytest.raises(ValueError):
        run_initial_screening_comparison(config)

    assert comparison_path.exists()
    assert comparison_path.read_text(encoding="utf-8") == "old-comparison\n"


def test_validate_comparison_mode_rejects_raw_without_side_effects(tmp_path):
    comparison_path = tmp_path / "preprocessing_comparison.csv"
    comparison_path.write_text("old-comparison\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _validate_comparison_mode("raw")

    assert comparison_path.exists()
