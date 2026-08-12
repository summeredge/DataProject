from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import (
    DOWNSTREAM_LOCK_FILENAME,
    confirm_initial_screening_branch,
    run_enhanced_screening_for_active_branch,
    run_granger_for_active_branch,
    run_initial_screening_workflow,
)


FORMAL_ROOT_FILES = [
    "ranked_features.csv",
    "recommended_candidates.csv",
    "causal_review_candidates.csv",
]
ENHANCED_OUTPUT_FILES = [
    "model_lift_scores.csv",
    "rolling_corr_scores.csv",
    "enhanced_validation_summary.csv",
]
GRANGER_OUTPUT_FILES = ["granger_tests.csv"]


def _raw_frame() -> pd.DataFrame:
    rows = 120
    time = np.arange(rows, dtype=float)
    controls = [f"control_{index}" for index in range(8)]
    candidates = [f"candidate_{index}" for index in range(5)]
    return pd.DataFrame(
        {
            "target": np.sin(time / 7),
            **{
                name: np.sin((time + index + 1) / 7)
                for index, name in enumerate(candidates)
            },
            **{
                name: np.cos((time + index + 1) / 7)
                for index, name in enumerate(controls)
            },
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="min"),
    )


def _lagged_granger_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 160
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.4 * y[t - 1] + 0.9 * x[t - 2] + rng.normal(scale=0.1)
    return pd.DataFrame(
        {
            "target": y,
            "candidate_0": x,
            "candidate_1": rng.normal(size=n),
            "candidate_2": rng.normal(size=n),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="min"),
    )


def _config(tmp_path: Path, **overrides) -> AnalysisConfig:
    kwargs = {
        "input_path": tmp_path / "input.csv",
        "time_column": "time",
        "target": "target",
        "output_dir": tmp_path,
        "max_lag": 3,
        "top_k": 15,
        "residual_control_columns": [f"control_{index}" for index in range(8)],
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


def _run_raw_workflow(tmp_path: Path) -> AnalysisConfig:
    config = _config(tmp_path, preprocess_mode="raw")
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)
    return config


def _run_lowpass_workflow(tmp_path: Path, **overrides) -> AnalysisConfig:
    kwargs = {
        "preprocess_mode": "lowpass_diff",
        "lowpass_tau_minutes": 7.5,
        "diff_interval_minutes": 5.0,
        "resample_rule": "2min",
    }
    kwargs.update(overrides)
    config = _config(tmp_path, **kwargs)
    _write_input(config, _raw_frame())
    run_initial_screening_workflow(config)
    return config


def _assert_no_lock(run_dir: Path) -> None:
    assert not (Path(run_dir) / DOWNSTREAM_LOCK_FILENAME).exists()


def _assert_no_downstream_outputs(run_dir: Path) -> None:
    for name in ENHANCED_OUTPUT_FILES + GRANGER_OUTPUT_FILES:
        assert not (Path(run_dir) / name).exists(), name


def _formal_root_bytes(run_dir: Path) -> dict[str, bytes]:
    return {
        name: (Path(run_dir) / name).read_bytes() for name in FORMAL_ROOT_FILES
    }


def _full_secondary_frame() -> pd.DataFrame:
    columns = [f"candidate_{index}" for index in range(5)]
    columns += [f"control_{index}" for index in range(8)]
    return pd.DataFrame(
        {
            "target": np.arange(60, dtype=float),
            **{
                name: np.arange(60, dtype=float) + index
                for index, name in enumerate(columns, start=1)
            },
        }
    )


def _spy_scaled_config(
    monkeypatch, variables: list[str] | None = None
) -> dict[str, object]:
    from chem_ts_corr import web

    captured: dict[str, object] = {}
    variables = variables or ["v1"]
    frame = pd.DataFrame(
        {
            "target": np.arange(60, dtype=float),
            **{
                name: np.arange(60, dtype=float) + index
                for index, name in enumerate(variables, start=1)
            },
        }
    )

    def capture_scaled(config, protected_columns=None):
        captured["config"] = config
        return frame

    monkeypatch.setattr(
        web,
        "_secondary_variables_from_ranked",
        lambda ranked, config, extra_variables=None: list(variables),
    )
    monkeypatch.setattr(
        web, "_save_secondary_candidate_context", lambda output_dir, variables: None
    )
    monkeypatch.setattr(web, "_scaled_frame_for_secondary", capture_scaled)
    monkeypatch.setattr(web, "_target_segment_mask", lambda frame: None)
    return captured


def _patch_enhanced_core(monkeypatch) -> dict[str, object]:
    from chem_ts_corr import screening, web

    captured: dict[str, object] = {}

    def capture_evidence(frame, target, variables, max_lag, **kwargs):
        captured["variables"] = list(variables)
        return {}, {}

    monkeypatch.setattr(screening, "prepare_best_lag_evidence", capture_evidence)
    monkeypatch.setattr(
        screening,
        "model_lift_scores",
        lambda *args, **kwargs: pd.DataFrame({"variable": []}),
    )
    monkeypatch.setattr(
        screening,
        "rolling_corr_scores",
        lambda *args, **kwargs: pd.DataFrame({"variable": []}),
    )
    monkeypatch.setattr(
        web,
        "_enhanced_validation_summary",
        lambda *args, **kwargs: pd.DataFrame({"variable": []}),
    )
    return captured


def _patch_granger_core(monkeypatch) -> dict[str, object]:
    from chem_ts_corr import causality

    captured: dict[str, object] = {}

    def capture_granger(frame, target, variables, maxlag, **kwargs):
        captured["variables"] = list(variables)
        return pd.DataFrame({"variable": variables})

    monkeypatch.setattr(causality, "run_granger_tests", capture_granger)
    return captured


def _inject_sentinel_into_processed_branch(run_dir: Path) -> None:
    processed_path = Path(run_dir) / "screening_branches" / "processed"
    ranked = pd.read_csv(processed_path / "ranked_features.csv", encoding="utf-8-sig")
    sentinel = ranked.iloc[[0]].copy()
    sentinel["variable"] = "sentinel_only_in_processed"
    ranked = pd.concat([ranked, sentinel], ignore_index=True)
    ranked.to_csv(
        processed_path / "ranked_features.csv", index=False, encoding="utf-8-sig"
    )


# --- Test 1: awaiting blocks enhanced screening --------------------------


def test_awaiting_confirmation_blocks_enhanced_screening(tmp_path):
    _run_lowpass_workflow(tmp_path)
    assert _read_context(tmp_path)["branch_selection_status"] == "awaiting_confirmation"

    with pytest.raises(ValueError, match="initial_screening_branch_not_confirmed"):
        run_enhanced_screening_for_active_branch(tmp_path)

    _assert_no_lock(tmp_path)
    _assert_no_downstream_outputs(tmp_path)


# --- Test 2: awaiting blocks Granger -------------------------------------


def test_awaiting_confirmation_blocks_granger(tmp_path):
    _run_lowpass_workflow(tmp_path)

    with pytest.raises(ValueError, match="initial_screening_branch_not_confirmed"):
        run_granger_for_active_branch(tmp_path)

    _assert_no_lock(tmp_path)
    _assert_no_downstream_outputs(tmp_path)


# --- Test 3: selected processed but confirmed raw -> enhanced raw --------


def test_enhanced_screening_uses_confirmed_raw_over_selected_mode(tmp_path, monkeypatch):
    config = _run_lowpass_workflow(tmp_path)
    confirm_initial_screening_branch(tmp_path, branch="raw")
    context = _read_context(tmp_path)
    assert context["active_screening_branch"] == "raw"
    assert context["active_preprocessing_mode"] == "raw"

    captured = _spy_scaled_config(monkeypatch)
    _patch_enhanced_core(monkeypatch)

    run_enhanced_screening_for_active_branch(tmp_path, base_config=config)

    assert captured["config"].preprocess_mode == "raw"
    assert captured["config"].preprocess_mode != "lowpass_diff"


# --- Test 4: selected processed but confirmed raw -> Granger raw ---------


def test_granger_uses_confirmed_raw_over_selected_mode(tmp_path, monkeypatch):
    config = _run_lowpass_workflow(tmp_path)
    confirm_initial_screening_branch(tmp_path, branch="raw")

    captured = _spy_scaled_config(monkeypatch)
    _patch_granger_core(monkeypatch)

    run_granger_for_active_branch(tmp_path, base_config=config)

    assert captured["config"].preprocess_mode == "raw"
    assert captured["config"].preprocess_mode != "lowpass_diff"


# --- Test 5: confirmed processed -> enhanced context params --------------


def test_enhanced_screening_uses_processed_context_params(tmp_path, monkeypatch):
    config = _run_lowpass_workflow(tmp_path)
    confirm_initial_screening_branch(tmp_path, branch="processed")
    context = _read_context(tmp_path)
    assert context["active_preprocessing_mode"] == "lowpass_diff"

    captured = _spy_scaled_config(monkeypatch)
    _patch_enhanced_core(monkeypatch)

    run_enhanced_screening_for_active_branch(tmp_path, base_config=config)

    downstream = captured["config"]
    assert downstream.preprocess_mode == "lowpass_diff"
    assert downstream.lowpass_tau_minutes == 7.5
    assert downstream.diff_interval_minutes == 5.0
    assert downstream.resample_rule == "2min"


# --- Test 6: confirmed processed -> Granger context params ---------------


def test_granger_uses_processed_context_params(tmp_path, monkeypatch):
    config = _run_lowpass_workflow(tmp_path)
    confirm_initial_screening_branch(tmp_path, branch="processed")

    captured = _spy_scaled_config(monkeypatch)
    _patch_granger_core(monkeypatch)

    run_granger_for_active_branch(tmp_path, base_config=config)

    downstream = captured["config"]
    assert downstream.preprocess_mode == "lowpass_diff"
    assert downstream.lowpass_tau_minutes == 7.5
    assert downstream.diff_interval_minutes == 5.0
    assert downstream.resample_rule == "2min"


# --- Test 7: raw workflow runs downstream with raw -----------------------


def test_raw_workflow_downstream_runs_raw_for_enhanced_and_granger(
    tmp_path, monkeypatch
):
    config = _run_raw_workflow(tmp_path)
    context = _read_context(tmp_path)
    assert context["branch_selection_status"] == "not_required"
    assert context["active_screening_branch"] == "raw"

    captured = _spy_scaled_config(monkeypatch)
    _patch_enhanced_core(monkeypatch)
    run_enhanced_screening_for_active_branch(tmp_path, base_config=config)
    assert captured["config"].preprocess_mode == "raw"

    captured = _spy_scaled_config(monkeypatch)
    _patch_granger_core(monkeypatch)
    run_granger_for_active_branch(tmp_path, base_config=config)
    assert captured["config"].preprocess_mode == "raw"

    assert (tmp_path / DOWNSTREAM_LOCK_FILENAME).exists()


# --- Test 8: context is source of truth over caller config ---------------


def test_context_is_source_of_truth_over_caller_config(tmp_path, monkeypatch):
    config = _run_raw_workflow(tmp_path)
    caller_config = replace(
        config,
        preprocess_mode="lowpass_diff",
        lowpass_tau_minutes=3.0,
        diff_interval_minutes=2.0,
    )

    captured = _spy_scaled_config(monkeypatch)
    _patch_enhanced_core(monkeypatch)
    run_enhanced_screening_for_active_branch(tmp_path, base_config=caller_config)
    downstream = captured["config"]
    assert downstream.preprocess_mode == "raw"
    assert downstream.diff_interval_minutes is None

    captured = _spy_scaled_config(monkeypatch)
    _patch_granger_core(monkeypatch)
    run_granger_for_active_branch(tmp_path, base_config=caller_config)
    downstream = captured["config"]
    assert downstream.preprocess_mode == "raw"
    assert downstream.diff_interval_minutes is None


# --- Test 9: formal root is the only candidate source --------------------


def test_enhanced_screening_candidates_only_from_formal_root(tmp_path, monkeypatch):
    from chem_ts_corr import web

    config = _run_lowpass_workflow(tmp_path)
    confirm_initial_screening_branch(tmp_path, branch="raw")
    _inject_sentinel_into_processed_branch(tmp_path)

    monkeypatch.setattr(
        web, "_scaled_frame_for_secondary", lambda config, protected_columns=None: _full_secondary_frame()
    )
    monkeypatch.setattr(web, "_target_segment_mask", lambda frame: None)
    captured = _patch_enhanced_core(monkeypatch)

    run_enhanced_screening_for_active_branch(tmp_path, base_config=config)

    assert captured["variables"]
    assert "sentinel_only_in_processed" not in captured["variables"]


def test_granger_candidates_only_from_formal_root(tmp_path, monkeypatch):
    from chem_ts_corr import web

    config = _run_lowpass_workflow(tmp_path)
    confirm_initial_screening_branch(tmp_path, branch="raw")
    _inject_sentinel_into_processed_branch(tmp_path)

    monkeypatch.setattr(
        web, "_scaled_frame_for_secondary", lambda config, protected_columns=None: _full_secondary_frame()
    )
    monkeypatch.setattr(web, "_target_segment_mask", lambda frame: None)
    captured = _patch_granger_core(monkeypatch)

    run_granger_for_active_branch(tmp_path, base_config=config)

    assert captured["variables"]
    assert "sentinel_only_in_processed" not in captured["variables"]


# --- Test 10: enhanced screening keeps formal root byte-identical --------


def test_enhanced_screening_keeps_formal_root_byte_identical(tmp_path):
    config = _run_raw_workflow(tmp_path)
    before = _formal_root_bytes(tmp_path)

    result = run_enhanced_screening_for_active_branch(tmp_path, base_config=config)

    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
    assert result["active_screening_branch"] == "raw"
    assert result["active_preprocessing_mode"] == "raw"
    for name in ENHANCED_OUTPUT_FILES:
        assert (tmp_path / name).exists()


# --- Test 11: Granger keeps formal root byte-identical -------------------


def test_granger_keeps_formal_root_byte_identical(tmp_path):
    config = _run_raw_workflow(tmp_path)
    before = _formal_root_bytes(tmp_path)
    ranked_before = pd.read_csv(
        tmp_path / "ranked_features.csv", encoding="utf-8-sig"
    )

    result = run_granger_for_active_branch(tmp_path, base_config=config)

    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
    ranked_after = pd.read_csv(
        tmp_path / "ranked_features.csv", encoding="utf-8-sig"
    )
    assert ranked_after["final_score"].tolist() == ranked_before["final_score"].tolist()
    assert ranked_after["driver_rank"].tolist() == ranked_before["driver_rank"].tolist()
    assert ranked_after["variable"].tolist() == ranked_before["variable"].tolist()
    assert result["active_screening_branch"] == "raw"
    assert result["active_preprocessing_mode"] == "raw"
    assert (tmp_path / "granger_tests.csv").exists()


# --- Test 12: enhanced screening creates the downstream lock -------------


def test_enhanced_screening_creates_downstream_lock(tmp_path, monkeypatch):
    config = _run_raw_workflow(tmp_path)
    _assert_no_lock(tmp_path)
    _spy_scaled_config(monkeypatch)
    _patch_enhanced_core(monkeypatch)

    run_enhanced_screening_for_active_branch(tmp_path, base_config=config)

    lock = tmp_path / DOWNSTREAM_LOCK_FILENAME
    assert lock.exists()
    assert lock.read_text(encoding="utf-8") == "downstream-locked"


# --- Test 13: Granger creates the downstream lock ------------------------


def test_granger_creates_downstream_lock(tmp_path, monkeypatch):
    config = _run_raw_workflow(tmp_path)
    _assert_no_lock(tmp_path)
    _spy_scaled_config(monkeypatch)
    _patch_granger_core(monkeypatch)

    run_granger_for_active_branch(tmp_path, base_config=config)

    lock = tmp_path / DOWNSTREAM_LOCK_FILENAME
    assert lock.exists()
    assert lock.read_text(encoding="utf-8") == "downstream-locked"


# --- Test 14: enhanced then Granger with existing lock -------------------


def test_granger_runs_after_enhanced_screening_lock(tmp_path, monkeypatch):
    config = _run_raw_workflow(tmp_path)
    _spy_scaled_config(monkeypatch)
    _patch_enhanced_core(monkeypatch)
    run_enhanced_screening_for_active_branch(tmp_path, base_config=config)
    assert (tmp_path / DOWNSTREAM_LOCK_FILENAME).exists()

    captured = _spy_scaled_config(monkeypatch)
    _patch_granger_core(monkeypatch)
    result = run_granger_for_active_branch(tmp_path, base_config=config)

    assert (tmp_path / "granger_tests.csv").exists()
    assert result["active_screening_branch"] == "raw"


# --- Test 15: formal root missing input fails without lock ---------------


def test_formal_root_missing_input_fails_without_branch_fallback(tmp_path):
    config = _run_raw_workflow(tmp_path)
    (tmp_path / "recommended_candidates.csv").unlink()

    with pytest.raises(ValueError, match="initial_screening_formal_output_missing"):
        run_enhanced_screening_for_active_branch(tmp_path, base_config=config)
    with pytest.raises(ValueError, match="initial_screening_formal_output_missing"):
        run_granger_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)
    _assert_no_downstream_outputs(tmp_path)


# --- Test 16: missing / invalid context blocks downstream ----------------


def test_missing_context_blocks_enhanced_and_granger(tmp_path):
    config = _run_raw_workflow(tmp_path)
    (tmp_path / "preprocessing_context.json").unlink()

    with pytest.raises(ValueError, match="initial_screening_context_missing"):
        run_enhanced_screening_for_active_branch(tmp_path, base_config=config)
    with pytest.raises(ValueError, match="initial_screening_context_missing"):
        run_granger_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)
    _assert_no_downstream_outputs(tmp_path)


def test_invalid_context_blocks_enhanced_and_granger(tmp_path):
    config = _run_raw_workflow(tmp_path)
    (tmp_path / "preprocessing_context.json").write_text(
        "{not-json", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="initial_screening_context_invalid"):
        run_enhanced_screening_for_active_branch(tmp_path, base_config=config)
    with pytest.raises(ValueError, match="initial_screening_context_invalid"):
        run_granger_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)
    _assert_no_downstream_outputs(tmp_path)


def test_invalid_status_context_blocks_downstream(tmp_path):
    config = _run_raw_workflow(tmp_path)
    context = _read_context(tmp_path)
    context["branch_selection_status"] = "bogus"
    (tmp_path / "preprocessing_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="initial_screening_context_invalid"):
        run_enhanced_screening_for_active_branch(tmp_path, base_config=config)
    with pytest.raises(ValueError, match="initial_screening_context_invalid"):
        run_granger_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)


# --- Test 17: Granger preserves signed lag -------------------------------


def test_granger_output_preserves_signed_lag(tmp_path):
    config = _config(tmp_path, preprocess_mode="raw", max_lag=4, top_k=15)
    _write_input(config, _lagged_granger_frame())
    run_initial_screening_workflow(config)

    run_granger_for_active_branch(tmp_path, base_config=config)

    granger = pd.read_csv(tmp_path / "granger_tests.csv", encoding="utf-8-sig")
    row = granger[granger["variable"] == "candidate_0"].iloc[0]
    assert row["status"] == "ok"
    assert int(row["best_granger_lag"]) == 2
    assert int(row["best_granger_lag"]) > 0


def test_pr9_runner_source_does_not_abs_lag_direction():
    source = Path("chem_ts_corr/pipeline.py").read_text(encoding="utf-8")
    start = source.index("def run_enhanced_screening_for_active_branch")
    end = source.index("\ndef _resolve_base_config", start)
    body = source[start:end]
    for marker in ("abs(lag", "abs(best_lag", "abs(granger_lag", ".abs()"):
        assert marker not in body


# --- Test 18: unexecuted stages produce no files -------------------------


def test_unexecuted_stages_do_not_generate_files(tmp_path, monkeypatch):
    config = _run_raw_workflow(tmp_path)
    _spy_scaled_config(monkeypatch)
    _patch_enhanced_core(monkeypatch)
    _patch_granger_core(monkeypatch)

    run_enhanced_screening_for_active_branch(tmp_path, base_config=config)
    run_granger_for_active_branch(tmp_path, base_config=config)

    for name in [
        "conditional_granger_scores.csv",
        "causal_review_report.csv",
        "causal_review_evidence.csv",
        "final_review_summary.csv",
        "shap_or_importance.csv",
        "model_variable_importance.csv",
        "model_discovered_candidates.csv",
    ]:
        assert not (tmp_path / name).exists(), name
    assert not (tmp_path / "xgb_validation").exists()
