from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chem_ts_corr.pipeline as pipeline
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import (
    CONTEXT_FIELDS,
    FORMAL_SCREENING_FILES,
    REQUIRED_FORMAL_SCREENING_FILES,
    begin_downstream_stage,
    confirm_initial_screening_branch,
    run_initial_screening_workflow,
)


PROCESSED_MODES = ["lowpass", "lowpass_detrend", "lowpass_diff"]
LEGACY_MODES = ["detrend", "diff", "detrend_diff"]


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


def _read_context(run_dir: Path) -> dict[str, object]:
    return json.loads(
        (Path(run_dir) / "preprocessing_context.json").read_text(encoding="utf-8")
    )


def _assert_root_has_no_formal_files(run_dir: Path) -> None:
    for name in FORMAL_SCREENING_FILES:
        assert not (run_dir / name).exists(), f"root-level {name} must not be published"


def _assert_root_equals_branch(run_dir: Path, branch: str) -> None:
    branch_dir = run_dir / "screening_branches" / branch
    for name in REQUIRED_FORMAL_SCREENING_FILES:
        assert (run_dir / name).exists(), f"root-level {name} missing"
        assert (run_dir / name).read_bytes() == (branch_dir / name).read_bytes(), (
            f"root {name} must be byte-identical to branch {branch}"
        )


def _assert_no_promotion_temp_dirs(run_dir: Path) -> None:
    leftovers = [
        path.name
        for path in Path(run_dir).iterdir()
        if path.name.startswith(".screening_promote_")
        or path.name.startswith(".screening_backup_")
        or path.name.startswith(".preprocessing_context_")
    ]
    assert leftovers == []


def _fail_if_called(*args, **kwargs):
    raise AssertionError("confirmation must not re-run screening analysis")


# --- Test 1: Raw workflow -----------------------------------------------


def test_raw_workflow_runs_only_raw_branch_and_promotes(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="raw")
    _write_input(config, _raw_frame())

    result = run_initial_screening_workflow(config)

    assert set(result) == {"branch", "timings", "context_path"}
    assert result["branch"] == "raw"
    raw_dir = tmp_path / "screening_branches" / "raw"
    assert (raw_dir / "ranked_features.csv").exists()
    assert (raw_dir / "recommended_candidates.csv").exists()
    assert not (tmp_path / "screening_branches" / "processed").exists()
    assert not (tmp_path / "preprocessing_comparison.csv").exists()
    assert (tmp_path / "ranked_features.csv").exists()

    context = _read_context(tmp_path)
    assert context["branch_selection_status"] == "not_required"
    assert context["selected_preprocessing_mode"] == "raw"
    assert context["active_screening_branch"] == "raw"
    assert context["active_preprocessing_mode"] == "raw"
    _assert_root_equals_branch(tmp_path, "raw")
    _assert_no_promotion_temp_dirs(tmp_path)


# --- Test 2: non-Raw workflow -------------------------------------------


@pytest.mark.parametrize("mode", PROCESSED_MODES)
def test_non_raw_workflow_runs_dual_branches_and_waits(tmp_path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode)
    _write_input(config, _raw_frame())

    result = run_initial_screening_workflow(config)

    raw_dir = tmp_path / "screening_branches" / "raw"
    processed_dir = tmp_path / "screening_branches" / "processed"
    assert (raw_dir / "ranked_features.csv").exists()
    assert (processed_dir / "ranked_features.csv").exists()
    assert (raw_dir / "recommended_candidates.csv").exists()
    assert (processed_dir / "recommended_candidates.csv").exists()
    assert (tmp_path / "preprocessing_comparison.csv").exists()
    assert set(result) == {
        "raw",
        "processed",
        "comparison_path",
        "context_path",
    }
    assert result["context_path"] == tmp_path / "preprocessing_context.json"

    context = _read_context(tmp_path)
    assert context["branch_selection_status"] == "awaiting_confirmation"
    assert context["selected_preprocessing_mode"] == mode
    assert context["active_screening_branch"] is None
    assert context["active_preprocessing_mode"] is None
    _assert_root_has_no_formal_files(tmp_path)
    _assert_no_promotion_temp_dirs(tmp_path)


# --- Test 3: missing-value semantics ------------------------------------


def test_awaiting_context_uses_real_json_null(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    context = _read_context(tmp_path)
    assert context["active_screening_branch"] is None
    assert context["active_preprocessing_mode"] is None

    raw_text = (tmp_path / "preprocessing_context.json").read_text(encoding="utf-8")
    assert '"active_screening_branch": null' in raw_text
    assert '"active_preprocessing_mode": null' in raw_text
    assert '"active_screening_branch": ""' not in raw_text
    assert '"active_screening_branch": false' not in raw_text
    assert '"active_screening_branch": 0' not in raw_text


def test_context_fields_fixed_set(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass_diff")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    context = _read_context(tmp_path)
    assert set(context) == set(CONTEXT_FIELDS)
    assert not {
        "final_score",
        "driver_rank",
        "recommended_branch",
        "best_branch",
        "selected_branch",
    } & set(context)


# --- Test 4: lowpass context parameters ---------------------------------


@pytest.mark.parametrize("mode", ["lowpass", "lowpass_detrend"])
def test_lowpass_context_records_tau_and_null_diff_fields(tmp_path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode, lowpass_tau_minutes=7.5)
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    context = _read_context(tmp_path)
    assert context["lowpass_tau_minutes"] == 7.5
    assert context["requested_diff_interval_minutes"] is None
    assert context["effective_diff_points"] is None
    assert context["effective_diff_interval_minutes"] is None


def test_context_records_real_resample_rule(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass", resample_rule="2min")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    context = _read_context(tmp_path)
    assert context["resample_rule"] == "2min"


# --- Tests 5/6: lowpass_diff effective interval -------------------------


def test_lowpass_diff_context_records_five_minute_interval(tmp_path):
    config = _raw_config(
        tmp_path,
        preprocess_mode="lowpass_diff",
        lowpass_tau_minutes=5.0,
        diff_interval_minutes=5.0,
    )
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    context = _read_context(tmp_path)
    assert context["lowpass_tau_minutes"] == 5.0
    assert context["requested_diff_interval_minutes"] == 5.0
    assert context["effective_diff_points"] == 5
    assert context["effective_diff_interval_minutes"] == 5.0


def test_lowpass_diff_context_records_auto_interval(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass_diff")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    context = _read_context(tmp_path)
    assert context["requested_diff_interval_minutes"] is None
    assert context["effective_diff_points"] == 1
    assert context["effective_diff_interval_minutes"] == 1.0


# --- Tests 7/8: confirmation --------------------------------------------


def test_confirm_raw_promotes_raw_branch(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    confirm_initial_screening_branch(tmp_path, branch="raw")

    context = _read_context(tmp_path)
    assert context["branch_selection_status"] == "confirmed"
    assert context["selected_preprocessing_mode"] == "lowpass"
    assert context["active_screening_branch"] == "raw"
    assert context["active_preprocessing_mode"] == "raw"
    _assert_root_equals_branch(tmp_path, "raw")
    _assert_no_promotion_temp_dirs(tmp_path)


def test_confirm_processed_promotes_processed_branch(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass_diff")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    confirm_initial_screening_branch(tmp_path, branch="processed")

    context = _read_context(tmp_path)
    assert context["branch_selection_status"] == "confirmed"
    assert context["selected_preprocessing_mode"] == "lowpass_diff"
    assert context["active_screening_branch"] == "processed"
    assert context["active_preprocessing_mode"] == "lowpass_diff"
    _assert_root_equals_branch(tmp_path, "processed")


# --- Test 9: confirmation never re-runs screening -----------------------


def test_confirmation_does_not_rerun_screening(monkeypatch, tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    for name in [
        "run_initial_screening_branch",
        "run_initial_screening_comparison",
        "analyze_initial_screening_branch_frame",
    ]:
        monkeypatch.setattr(pipeline, name, _fail_if_called)

    confirm_initial_screening_branch(tmp_path, branch="processed")

    context = _read_context(tmp_path)
    assert context["branch_selection_status"] == "confirmed"


# --- Test 10: repeated confirmation is idempotent -----------------------


def test_repeat_confirm_same_branch_is_idempotent(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)
    confirm_initial_screening_branch(tmp_path, branch="processed")

    context_path = tmp_path / "preprocessing_context.json"
    context_before = context_path.read_bytes()
    root_before = {
        name: (tmp_path / name).read_bytes()
        for name in FORMAL_SCREENING_FILES
        if (tmp_path / name).exists()
    }
    file_count_before = len(list(tmp_path.iterdir()))

    confirm_initial_screening_branch(tmp_path, branch="processed")

    assert context_path.read_bytes() == context_before
    for name, content in root_before.items():
        assert (tmp_path / name).read_bytes() == content
    assert len(list(tmp_path.iterdir())) == file_count_before


# --- Test 11: switch branch before downstream ---------------------------


def test_switch_branch_before_downstream(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    confirm_initial_screening_branch(tmp_path, branch="raw")
    assert _read_context(tmp_path)["active_screening_branch"] == "raw"

    confirm_initial_screening_branch(tmp_path, branch="processed")

    context = _read_context(tmp_path)
    assert context["branch_selection_status"] == "confirmed"
    assert context["selected_preprocessing_mode"] == "lowpass"
    assert context["active_screening_branch"] == "processed"
    assert context["active_preprocessing_mode"] == "lowpass"
    _assert_root_equals_branch(tmp_path, "processed")


# --- Test 12: incomplete branch output ----------------------------------


def test_branch_incomplete_rejected_without_touching_root(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)
    (tmp_path / "screening_branches" / "processed" / "ranked_features.csv").unlink()

    with pytest.raises(ValueError, match="initial_screening_branch_output_incomplete"):
        confirm_initial_screening_branch(tmp_path, branch="processed")

    context = _read_context(tmp_path)
    assert context["branch_selection_status"] == "awaiting_confirmation"
    _assert_root_has_no_formal_files(tmp_path)
    _assert_no_promotion_temp_dirs(tmp_path)


# --- Test 13: promotion rollback ----------------------------------------


def test_promotion_rollback_restores_previous_root(monkeypatch, tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)
    confirm_initial_screening_branch(tmp_path, branch="raw")

    context_path = tmp_path / "preprocessing_context.json"
    context_before = context_path.read_bytes()
    root_before = {
        name: (tmp_path / name).read_bytes()
        for name in FORMAL_SCREENING_FILES
        if (tmp_path / name).exists()
    }

    calls = {"count": 0}
    original_replace = os.replace

    def flaky_replace(source, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("synthetic os.replace failure")
        return original_replace(source, target)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(RuntimeError, match="synthetic os.replace failure"):
        confirm_initial_screening_branch(tmp_path, branch="processed")

    assert calls["count"] >= 2
    assert context_path.read_bytes() == context_before
    for name, content in root_before.items():
        assert (tmp_path / name).read_bytes() == content, f"root {name} not restored"
    _assert_root_equals_branch(tmp_path, "raw")
    assert _read_context(tmp_path)["active_screening_branch"] == "raw"
    _assert_no_promotion_temp_dirs(tmp_path)


# --- Test 14: optional-file residue removal -----------------------------


def test_optional_file_residue_removed_on_branch_switch(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)
    confirm_initial_screening_branch(tmp_path, branch="raw")
    assert (tmp_path / "residual_corr_scores.csv").exists()

    processed_dir = tmp_path / "screening_branches" / "processed"
    assert (processed_dir / "residual_corr_scores.csv").exists()
    (processed_dir / "residual_corr_scores.csv").unlink()

    confirm_initial_screening_branch(tmp_path, branch="processed")

    assert not (tmp_path / "residual_corr_scores.csv").exists()
    _assert_root_equals_branch(tmp_path, "processed")


# --- Tests 15-17: downstream gate ---------------------------------------


def test_awaiting_gate_rejects_without_lock(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    with pytest.raises(ValueError, match="initial_screening_branch_not_confirmed"):
        begin_downstream_stage(tmp_path)

    assert not (tmp_path / "screening_downstream.lock").exists()


def test_confirmed_gate_creates_lock(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)
    confirm_initial_screening_branch(tmp_path, branch="processed")

    begin_downstream_stage(tmp_path)

    assert (tmp_path / "screening_downstream.lock").exists()


def test_raw_gate_passes(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="raw")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    begin_downstream_stage(tmp_path)

    assert (tmp_path / "screening_downstream.lock").exists()


# --- Tests 18/19: lock semantics ----------------------------------------


def test_lock_blocks_branch_switch(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)
    confirm_initial_screening_branch(tmp_path, branch="raw")
    root_before = {
        name: (tmp_path / name).read_bytes()
        for name in FORMAL_SCREENING_FILES
        if (tmp_path / name).exists()
    }
    context_before = (tmp_path / "preprocessing_context.json").read_bytes()
    begin_downstream_stage(tmp_path)

    with pytest.raises(ValueError, match="initial_screening_branch_locked"):
        confirm_initial_screening_branch(tmp_path, branch="processed")

    assert (tmp_path / "preprocessing_context.json").read_bytes() == context_before
    for name, content in root_before.items():
        assert (tmp_path / name).read_bytes() == content


def test_lock_same_branch_confirm_is_noop(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)
    confirm_initial_screening_branch(tmp_path, branch="raw")
    begin_downstream_stage(tmp_path)
    context_before = (tmp_path / "preprocessing_context.json").read_bytes()
    root_before = {
        name: (tmp_path / name).read_bytes()
        for name in FORMAL_SCREENING_FILES
        if (tmp_path / name).exists()
    }

    confirm_initial_screening_branch(tmp_path, branch="raw")

    assert (tmp_path / "preprocessing_context.json").read_bytes() == context_before
    for name, content in root_before.items():
        assert (tmp_path / name).read_bytes() == content


# --- Test 20: locked run rejects a new workflow -------------------------


def test_locked_run_rejects_new_workflow(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="raw")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)
    begin_downstream_stage(tmp_path)
    root_before = {
        name: (tmp_path / name).read_bytes()
        for name in FORMAL_SCREENING_FILES
        if (tmp_path / name).exists()
    }
    context_before = (tmp_path / "preprocessing_context.json").read_bytes()

    with pytest.raises(ValueError, match="initial_screening_run_locked"):
        run_initial_screening_workflow(config)

    assert (tmp_path / "screening_downstream.lock").exists()
    assert (tmp_path / "preprocessing_context.json").read_bytes() == context_before
    for name, content in root_before.items():
        assert (tmp_path / name).read_bytes() == content


# --- Test 21: re-run clears previous formal root ------------------------


def test_rerun_non_raw_clears_previous_formal_root(tmp_path):
    raw_config = _raw_config(tmp_path, preprocess_mode="raw")
    _write_input(raw_config, _raw_frame())
    run_initial_screening_workflow(raw_config)
    assert (tmp_path / "ranked_features.csv").exists()
    assert (tmp_path / "preprocessing_context.json").exists()

    lowpass_config = _raw_config(tmp_path, preprocess_mode="lowpass")
    run_initial_screening_workflow(lowpass_config)

    context = _read_context(tmp_path)
    assert context["branch_selection_status"] == "awaiting_confirmation"
    assert context["selected_preprocessing_mode"] == "lowpass"
    assert (tmp_path / "preprocessing_comparison.csv").exists()
    _assert_root_has_no_formal_files(tmp_path)


# --- Test 22: invalid mode has no side effects --------------------------


@pytest.mark.parametrize("mode", LEGACY_MODES)
def test_invalid_mode_has_no_side_effects(tmp_path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)
    confirm_initial_screening_branch(tmp_path, branch="raw")
    root_before = {
        name: (tmp_path / name).read_bytes()
        for name in FORMAL_SCREENING_FILES
        if (tmp_path / name).exists()
    }
    context_before = (tmp_path / "preprocessing_context.json").read_bytes()
    comparison_before = (tmp_path / "preprocessing_comparison.csv").read_bytes()

    invalid_config = _raw_config(tmp_path, preprocess_mode=mode)
    with pytest.raises(ValueError, match="run_initial_screening_workflow"):
        run_initial_screening_workflow(invalid_config)

    assert (tmp_path / "preprocessing_context.json").read_bytes() == context_before
    assert (tmp_path / "preprocessing_comparison.csv").read_bytes() == comparison_before
    for name, content in root_before.items():
        assert (tmp_path / name).read_bytes() == content


# --- Additional gate/context failure semantics --------------------------


def test_gate_missing_context_fails(tmp_path):
    with pytest.raises(ValueError, match="initial_screening_context_missing"):
        begin_downstream_stage(tmp_path)
    assert not (tmp_path / "screening_downstream.lock").exists()


def test_gate_invalid_context_fails(tmp_path):
    (tmp_path / "preprocessing_context.json").write_text(
        "{not-valid-json", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="initial_screening_context_invalid"):
        begin_downstream_stage(tmp_path)
    assert not (tmp_path / "screening_downstream.lock").exists()


def test_confirm_missing_context_fails(tmp_path):
    with pytest.raises(ValueError, match="initial_screening_context_missing"):
        confirm_initial_screening_branch(tmp_path, branch="raw")


def test_confirm_invalid_branch_name_fails(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)

    with pytest.raises(ValueError, match="Unknown screening branch"):
        confirm_initial_screening_branch(tmp_path, branch="both")
