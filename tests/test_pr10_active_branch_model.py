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
    run_model_for_active_branch,
)


MODEL_OUTPUT_FILES = [
    "shap_or_importance.csv",
    "model_variable_importance.csv",
    "model_discovered_candidates.csv",
]
FORMAL_ROOT_FILES = [
    "ranked_features.csv",
    "recommended_candidates.csv",
    "causal_review_candidates.csv",
]


def _config(tmp_path: Path, **overrides) -> AnalysisConfig:
    kwargs = {
        "input_path": tmp_path / "input.csv",
        "time_column": "time",
        "target": "target",
        "output_dir": tmp_path,
        "max_lag": 3,
        "top_k": 15,
        "residual_control_columns": [],
        "force_include_variables": [],
        "enable_model": False,
    }
    kwargs.update(overrides)
    return AnalysisConfig(**kwargs)


def _ranked_frame(
    n_variables: int = 20, *, include_forced: bool = False
) -> pd.DataFrame:
    names = [f"variable_{index}" for index in range(n_variables)]
    scores = [0.95 - 0.06 * index for index in range(n_variables)]
    if include_forced:
        names.append("forced_include_variable")
        scores.append(0.01)
    rows = []
    for rank, (name, score) in enumerate(zip(names, scores), start=1):
        rows.append(
            {
                "variable": name,
                "final_score": score,
                "driver_rank": rank,
                "lag": (rank % 3) + 1,
            }
        )
    return pd.DataFrame(rows)


def _write_formal_root(run_dir: Path, ranked: pd.DataFrame) -> None:
    ranked.to_csv(
        run_dir / "ranked_features.csv", index=False, encoding="utf-8-sig"
    )
    recommended = ranked[["variable", "final_score", "driver_rank"]].copy()
    recommended["candidate_source"] = "formal"
    recommended["recommended_use"] = ""
    recommended["recommended_action"] = ""
    recommended.to_csv(
        run_dir / "recommended_candidates.csv", index=False, encoding="utf-8-sig"
    )
    causal = pd.DataFrame(
        {
            "variable": ranked["variable"].head(5).tolist(),
            "candidate_source": ["formal"] * 5,
            "reason": [""] * 5,
        }
    )
    causal.to_csv(
        run_dir / "causal_review_candidates.csv", index=False, encoding="utf-8-sig"
    )
    risk = pd.DataFrame(
        {
            "variable": ranked["variable"].tolist(),
            "risk_flags": ["formal_root_risk"] * len(ranked),
            "recommended_use": [""] * len(ranked),
            "recommended_action": [""] * len(ranked),
        }
    )
    risk.to_csv(run_dir / "risk_flags.csv", index=False, encoding="utf-8-sig")


def _write_context(run_dir: Path, **overrides) -> None:
    context = {
        "selected_preprocessing_mode": "raw",
        "active_screening_branch": "raw",
        "active_preprocessing_mode": "raw",
        "lowpass_tau_minutes": None,
        "requested_diff_interval_minutes": None,
        "effective_diff_points": None,
        "effective_diff_interval_minutes": None,
        "resample_rule": None,
        "branch_selection_status": "not_required",
    }
    context.update(overrides)
    (run_dir / "preprocessing_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _make_raw_run(tmp_path: Path, *, force_include: bool = False) -> AnalysisConfig:
    config = _config(
        tmp_path,
        force_include_variables=(
            ["forced_include_variable"] if force_include else []
        ),
    )
    _write_formal_root(tmp_path, _ranked_frame(include_forced=force_include))
    _write_context(tmp_path)
    return config


def _make_processed_run(
    tmp_path: Path, *, branch: str = "processed"
) -> AnalysisConfig:
    config = _config(
        tmp_path,
        preprocess_mode="lowpass_diff",
        lowpass_tau_minutes=7.5,
        diff_interval_minutes=5.0,
        resample_rule="2min",
    )
    _write_formal_root(tmp_path, _ranked_frame())
    if branch == "processed":
        _write_context(
            tmp_path,
            selected_preprocessing_mode="lowpass_diff",
            active_screening_branch="processed",
            active_preprocessing_mode="lowpass_diff",
            lowpass_tau_minutes=7.5,
            requested_diff_interval_minutes=5.0,
            effective_diff_points=5,
            effective_diff_interval_minutes=5.0,
            resample_rule="2min",
            branch_selection_status="confirmed",
        )
    else:
        _write_context(
            tmp_path,
            selected_preprocessing_mode="lowpass_diff",
            active_screening_branch="raw",
            active_preprocessing_mode="raw",
            branch_selection_status="confirmed",
        )
    return config


def _make_awaiting_run(tmp_path: Path) -> AnalysisConfig:
    config = _config(tmp_path, preprocess_mode="lowpass_diff")
    _write_context(
        tmp_path,
        selected_preprocessing_mode="lowpass_diff",
        active_screening_branch=None,
        active_preprocessing_mode=None,
        branch_selection_status="awaiting_confirmation",
    )
    return config


def _read_context(run_dir: Path) -> dict[str, object]:
    return json.loads(
        (Path(run_dir) / "preprocessing_context.json").read_text(encoding="utf-8")
    )


def _full_secondary_frame(run_dir: Path, n_rows: int = 60) -> pd.DataFrame:
    ranked = pd.read_csv(run_dir / "ranked_features.csv", encoding="utf-8-sig")
    columns = ranked["variable"].astype(str).tolist()
    return pd.DataFrame(
        {
            "target": np.arange(n_rows, dtype=float),
            **{
                name: np.arange(n_rows, dtype=float) + index
                for index, name in enumerate(columns, start=1)
            },
        }
    )


def _sample_importance(run_dir: Path, limit: int = 5) -> pd.DataFrame:
    ranked = pd.read_csv(run_dir / "ranked_features.csv", encoding="utf-8-sig")
    rows = []
    for variable in ranked["variable"].astype(str).tolist()[:limit]:
        for lag in (1, 2):
            rows.append(
                {
                    "feature": f"{variable}__lag_{lag}",
                    "importance": 0.1,
                    "method": "random_forest_feature_importance",
                    "variable": variable,
                    "lag": float(lag),
                }
            )
    return pd.DataFrame(rows)


def _spy_scaled_config(monkeypatch, run_dir: Path) -> dict[str, object]:
    from chem_ts_corr import web

    captured: dict[str, object] = {}
    frame = _full_secondary_frame(run_dir)

    def capture_scaled(config, protected_columns=None):
        captured["config"] = config
        return frame

    monkeypatch.setattr(web, "_scaled_frame_for_secondary", capture_scaled)
    monkeypatch.setattr(web, "_target_segment_mask", lambda frame: None)
    return captured


def _patch_model_core(monkeypatch, importance: pd.DataFrame | None = None):
    from chem_ts_corr import modeling

    captured: dict[str, object] = {}
    if importance is None:
        importance = pd.DataFrame(
            columns=["feature", "importance", "method", "variable", "lag"]
        )

    def capture_fit(
        frame,
        target,
        max_lag,
        candidate_variables,
        max_features,
        random_state,
        best_lags,
        lag_mode,
        target_mask,
    ):
        captured.update(
            {
                "target": target,
                "max_lag": max_lag,
                "candidate_variables": list(candidate_variables),
                "max_features": max_features,
                "random_state": random_state,
                "best_lags": dict(best_lags or {}),
                "lag_mode": lag_mode,
                "target_mask": target_mask,
            }
        )
        return (
            importance.copy(deep=True),
            {"model_status": "ok", "r2_holdout": 0.5},
        )

    monkeypatch.setattr(modeling, "fit_explainable_model", capture_fit)
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
    processed_dir = Path(run_dir) / "screening_branches" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_csv(run_dir / "ranked_features.csv", encoding="utf-8-sig")
    sentinel = ranked.iloc[[0]].copy()
    sentinel["variable"] = "sentinel_only_in_processed"
    ranked = pd.concat([ranked, sentinel], ignore_index=True)
    ranked.to_csv(
        processed_dir / "ranked_features.csv", index=False, encoding="utf-8-sig"
    )


def _write_input_csv(config: AnalysisConfig) -> None:
    rows = 120
    time = np.arange(rows, dtype=float)
    ranked = pd.read_csv(
        config.output_dir / "ranked_features.csv", encoding="utf-8-sig"
    )
    columns = ranked["variable"].astype(str).tolist()
    frame = pd.DataFrame(
        {
            "target": np.sin(time / 7),
            **{
                name: np.sin((time + index + 1) / 7)
                for index, name in enumerate(columns)
            },
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="min"),
    )
    config.input_path.parent.mkdir(parents=True, exist_ok=True)
    table = frame.copy()
    table[config.time_column] = frame.index
    table[[config.time_column, *frame.columns]].to_csv(
        config.input_path, index=False, encoding=config.encoding
    )


def _assert_no_lock(run_dir: Path) -> None:
    assert not (Path(run_dir) / DOWNSTREAM_LOCK_FILENAME).exists()


def _assert_no_model_outputs(run_dir: Path) -> None:
    for name in MODEL_OUTPUT_FILES:
        assert not (Path(run_dir) / name).exists(), name


def _assert_model_outputs(run_dir: Path) -> None:
    for name in MODEL_OUTPUT_FILES:
        assert (Path(run_dir) / name).exists(), name


def _formal_root_bytes(run_dir: Path) -> dict[str, bytes]:
    return {
        name: (Path(run_dir) / name).read_bytes() for name in FORMAL_ROOT_FILES
    }


# --- Test 1: awaiting blocks the formal model runner ----------------------


def test_awaiting_confirmation_blocks_model(tmp_path):
    _make_awaiting_run(tmp_path)
    assert _read_context(tmp_path)["branch_selection_status"] == "awaiting_confirmation"

    with pytest.raises(ValueError, match="initial_screening_branch_not_confirmed"):
        run_model_for_active_branch(tmp_path)

    _assert_no_lock(tmp_path)
    _assert_no_model_outputs(tmp_path)


# --- Test 2: confirmed raw wins over selected mode ------------------------


def test_model_uses_confirmed_raw_over_selected_mode(tmp_path, monkeypatch):
    config = _make_processed_run(tmp_path, branch="raw")
    context = _read_context(tmp_path)
    assert context["active_screening_branch"] == "raw"
    assert context["active_preprocessing_mode"] == "raw"

    captured = _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch)

    run_model_for_active_branch(tmp_path, base_config=config)

    assert captured["config"].preprocess_mode == "raw"
    assert captured["config"].preprocess_mode != "lowpass_diff"


# --- Test 3: confirmed processed uses context parameters ------------------


def test_model_uses_processed_context_params(tmp_path, monkeypatch):
    config = _make_processed_run(tmp_path, branch="processed")
    assert _read_context(tmp_path)["active_preprocessing_mode"] == "lowpass_diff"

    captured = _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch)

    run_model_for_active_branch(tmp_path, base_config=config)

    downstream = captured["config"]
    assert downstream.preprocess_mode == "lowpass_diff"
    assert downstream.lowpass_tau_minutes == 7.5
    assert downstream.diff_interval_minutes == 5.0
    assert downstream.resample_rule == "2min"


# --- Test 4: raw workflow runs model directly with raw --------------------


def test_raw_workflow_model_runs_raw(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    context = _read_context(tmp_path)
    assert context["branch_selection_status"] == "not_required"
    assert context["active_screening_branch"] == "raw"
    assert context["active_preprocessing_mode"] == "raw"

    captured = _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch)
    result = run_model_for_active_branch(tmp_path, base_config=config)

    assert captured["config"].preprocess_mode == "raw"
    assert result["active_screening_branch"] == "raw"
    assert result["active_preprocessing_mode"] == "raw"


# --- Test 5: context is source of truth over caller config ----------------


def test_context_overrides_caller_preprocessing_config(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    caller_config = replace(
        config,
        preprocess_mode="lowpass_diff",
        lowpass_tau_minutes=3.0,
        diff_interval_minutes=2.0,
    )

    captured = _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch)
    run_model_for_active_branch(tmp_path, base_config=caller_config)

    downstream = captured["config"]
    assert downstream.preprocess_mode == "raw"
    assert downstream.diff_interval_minutes is None


# --- Test 6: candidates only come from the formal root --------------------


def test_model_candidates_only_from_formal_root(tmp_path, monkeypatch):
    config = _make_processed_run(tmp_path, branch="raw")
    _inject_sentinel_into_processed_branch(tmp_path)

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_model_core(monkeypatch)
    run_model_for_active_branch(tmp_path, base_config=config)

    assert captured["candidate_variables"]
    assert "sentinel_only_in_processed" not in captured["candidate_variables"]


# --- Test 7: model candidates cover Top-K without re-thresholding ---------


def test_model_candidates_cover_top_k_without_score_rethreshold(
    tmp_path, monkeypatch
):
    config = _make_raw_run(tmp_path)
    ranked = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")
    low_score_top = ranked[
        (ranked["driver_rank"] <= config.top_k)
        & (pd.to_numeric(ranked["final_score"], errors="coerce") < 0.30)
    ]["variable"].astype(str).tolist()
    assert low_score_top, "fixture must contain Top-K variables below 0.30"

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_model_core(monkeypatch)
    run_model_for_active_branch(tmp_path, base_config=config)

    candidates = captured["candidate_variables"]
    assert candidates
    assert all(variable in candidates for variable in low_score_top)


# --- Test 8: force include variables stay in the model candidates ---------


def test_force_include_variables_preserved(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path, force_include=True)
    ranked = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")
    top_k = ranked["variable"].astype(str).tolist()[: config.top_k]
    assert "forced_include_variable" not in top_k

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_model_core(monkeypatch)
    run_model_for_active_branch(tmp_path, base_config=config)

    candidates = captured["candidate_variables"]
    assert candidates[: len(top_k)] == top_k
    assert "forced_include_variable" in candidates
    assert len(candidates) == len(set(candidates))


# --- Test 9: model calls the existing core with downstream parameters -----


def test_model_calls_existing_core_with_downstream_parameters(
    tmp_path, monkeypatch
):
    config = _make_raw_run(tmp_path)
    ranked = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_model_core(monkeypatch)
    run_model_for_active_branch(tmp_path, base_config=config)

    expected_variables = ranked["variable"].astype(str).tolist()[: config.top_k]
    assert captured["target"] == config.target
    assert captured["max_lag"] == config.max_lag
    assert captured["candidate_variables"] == expected_variables
    assert captured["max_features"] == config.max_model_features
    assert captured["random_state"] == config.random_state
    assert captured["lag_mode"] == "best_only"
    assert captured["target_mask"] is None
    expected_best_lags = {
        str(row["variable"]): int(row["lag"])
        for _, row in ranked[["variable", "lag"]].dropna().iterrows()
    }
    assert captured["best_lags"] == expected_best_lags


# --- Test 10: exactly the three model outputs are generated ---------------


def test_model_generates_three_outputs_only(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    before = {path.name for path in tmp_path.glob("*.csv")}

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch, importance=_sample_importance(tmp_path))
    run_model_for_active_branch(tmp_path, base_config=config)

    after = {path.name for path in tmp_path.glob("*.csv")}
    assert after - before == set(MODEL_OUTPUT_FILES)
    assert "secondary_candidate_context.csv" not in after
    _assert_model_outputs(tmp_path)


# --- Model must not touch the shared secondary candidate context ----------


def test_existing_secondary_candidate_context_stays_byte_identical(
    tmp_path, monkeypatch
):
    config = _make_raw_run(tmp_path)
    secondary_path = tmp_path / "secondary_candidate_context.csv"
    secondary_path.write_text(
        "variable\nexisting_secondary_only_candidate\n", encoding="utf-8-sig"
    )
    before = secondary_path.read_bytes()

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch)
    run_model_for_active_branch(tmp_path, base_config=config)

    assert secondary_path.exists()
    assert secondary_path.read_bytes() == before


def test_model_does_not_create_secondary_candidate_context(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    assert not (tmp_path / "secondary_candidate_context.csv").exists()

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch)
    run_model_for_active_branch(tmp_path, base_config=config)

    assert not (tmp_path / "secondary_candidate_context.csv").exists()
    _assert_model_outputs(tmp_path)


def test_secondary_candidate_context_read_stays_original_after_model(
    tmp_path, monkeypatch
):
    from chem_ts_corr import web

    config = _make_raw_run(tmp_path)
    (tmp_path / "secondary_candidate_context.csv").write_text(
        "variable\nsecondary_candidate_A\nsecondary_candidate_B\n",
        encoding="utf-8-sig",
    )

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch)
    run_model_for_active_branch(tmp_path, base_config=config)

    assert web._load_secondary_candidate_context(tmp_path) == [
        "secondary_candidate_A",
        "secondary_candidate_B",
    ]


# --- Test 11: formal root screening files stay byte-identical -------------


def test_model_keeps_formal_root_byte_identical(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    before = _formal_root_bytes(tmp_path)
    ranked_before = pd.read_csv(
        tmp_path / "ranked_features.csv", encoding="utf-8-sig"
    )

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch, importance=_sample_importance(tmp_path))
    result = run_model_for_active_branch(tmp_path, base_config=config)

    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
    ranked_after = pd.read_csv(
        tmp_path / "ranked_features.csv", encoding="utf-8-sig"
    )
    assert ranked_after["final_score"].tolist() == ranked_before["final_score"].tolist()
    assert ranked_after["driver_rank"].tolist() == ranked_before["driver_rank"].tolist()
    assert ranked_after["variable"].tolist() == ranked_before["variable"].tolist()
    assert result["active_screening_branch"] == "raw"


# --- Test 12: model discovery never writes back into screening ------------


def test_model_discovery_does_not_enter_screening(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    sentinel = "only_in_model_discovery"
    pd.DataFrame({"variable": [sentinel], "max_importance": [0.99]}).to_csv(
        tmp_path / "model_discovered_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch, importance=_sample_importance(tmp_path))
    run_model_for_active_branch(tmp_path, base_config=config)

    for name in FORMAL_ROOT_FILES:
        frame = pd.read_csv(tmp_path / name, encoding="utf-8-sig")
        assert sentinel not in frame["variable"].astype(str).tolist(), name


# --- Test 13: risk information comes from the formal root -----------------


def test_model_risk_comes_from_formal_root(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    processed_dir = tmp_path / "screening_branches" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_risk = pd.DataFrame(
        {
            "variable": ["variable_0"],
            "risk_flags": ["processed_branch_risk"],
            "recommended_use": [""],
            "recommended_action": [""],
        }
    )
    processed_risk.to_csv(
        processed_dir / "risk_flags.csv", index=False, encoding="utf-8-sig"
    )

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch, importance=_sample_importance(tmp_path, limit=1))
    run_model_for_active_branch(tmp_path, base_config=config)

    var_importance = pd.read_csv(
        tmp_path / "model_variable_importance.csv", encoding="utf-8-sig"
    )
    discovered = pd.read_csv(
        tmp_path / "model_discovered_candidates.csv", encoding="utf-8-sig"
    )
    for frame in (var_importance, discovered):
        risk_text = frame["risk_flags"].astype(str).str.cat(sep=" ")
        assert "formal_root_risk" in risk_text
        assert "processed_branch_risk" not in risk_text


# --- Test 14: model as the first downstream stage creates the lock --------


def test_model_creates_downstream_lock_as_first_stage(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    _assert_no_lock(tmp_path)
    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch)

    run_model_for_active_branch(tmp_path, base_config=config)

    lock = tmp_path / DOWNSTREAM_LOCK_FILENAME
    assert lock.exists()
    assert lock.read_text(encoding="utf-8") == "downstream-locked"


# --- Test 15: existing lock still allows the model stage ------------------


@pytest.mark.parametrize("prior_stage", ["enhanced", "granger"])
def test_model_runs_after_existing_downstream_lock(
    tmp_path, monkeypatch, prior_stage
):
    config = _make_raw_run(tmp_path)
    _spy_scaled_config(monkeypatch, tmp_path)
    if prior_stage == "enhanced":
        _patch_enhanced_core(monkeypatch)
        run_enhanced_screening_for_active_branch(tmp_path, base_config=config)
    else:
        _patch_granger_core(monkeypatch)
        run_granger_for_active_branch(tmp_path, base_config=config)
    assert (tmp_path / DOWNSTREAM_LOCK_FILENAME).exists()

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch)
    result = run_model_for_active_branch(tmp_path, base_config=config)

    _assert_model_outputs(tmp_path)
    assert result["active_screening_branch"] == "raw"


# --- Test 16: after model starts the branch cannot switch ----------------


def test_model_locks_branch_switch(tmp_path, monkeypatch):
    config = _make_processed_run(tmp_path, branch="raw")
    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch)
    run_model_for_active_branch(tmp_path, base_config=config)
    assert (tmp_path / DOWNSTREAM_LOCK_FILENAME).exists()

    with pytest.raises(ValueError, match="initial_screening_branch_locked"):
        confirm_initial_screening_branch(tmp_path, branch="processed")


# --- Test 17: missing formal root input fails without fallback ------------


@pytest.mark.parametrize("missing_name", ["ranked_features.csv", "risk_flags.csv"])
def test_model_formal_root_missing_fails_without_fallback(tmp_path, missing_name):
    config = _make_raw_run(tmp_path)
    (tmp_path / missing_name).unlink()

    with pytest.raises(ValueError, match="initial_screening_formal_output_missing"):
        run_model_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)
    _assert_no_model_outputs(tmp_path)


# --- Test 18: missing / invalid context keeps frozen errors ---------------


def test_model_missing_context_fails(tmp_path):
    config = _make_raw_run(tmp_path)
    (tmp_path / "preprocessing_context.json").unlink()

    with pytest.raises(ValueError, match="initial_screening_context_missing"):
        run_model_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)
    _assert_no_model_outputs(tmp_path)


def test_model_invalid_context_fails(tmp_path):
    config = _make_raw_run(tmp_path)
    (tmp_path / "preprocessing_context.json").write_text(
        "{not-json", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="initial_screening_context_invalid"):
        run_model_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)
    _assert_no_model_outputs(tmp_path)


def test_model_invalid_status_context_fails(tmp_path):
    config = _make_raw_run(tmp_path)
    context = _read_context(tmp_path)
    context["branch_selection_status"] = "bogus"
    (tmp_path / "preprocessing_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="initial_screening_context_invalid"):
        run_model_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)


# --- Test 19: SHAP fallback keeps the formal runner working ---------------


def test_model_runner_succeeds_with_rf_importance_fallback(tmp_path, monkeypatch):
    from chem_ts_corr import modeling
    import sklearn.ensemble as sklearn_ensemble

    real_rf = sklearn_ensemble.RandomForestRegressor

    def sequential_rf(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["n_jobs"] = 1
        return real_rf(*args, **kwargs)

    monkeypatch.setattr(sklearn_ensemble, "RandomForestRegressor", sequential_rf)

    config = _make_raw_run(tmp_path)
    _write_input_csv(config)
    monkeypatch.setattr(
        modeling, "_try_shap_importance", lambda model, x_train, random_state: None
    )

    result = run_model_for_active_branch(tmp_path, base_config=config)

    _assert_model_outputs(tmp_path)
    assert result["model_metrics"]["model_status"] == "ok"
    importance = pd.read_csv(
        tmp_path / "shap_or_importance.csv", encoding="utf-8-sig"
    )
    assert (importance["method"] == "random_forest_feature_importance").all()


# --- Test 20: unexecuted stages produce no files --------------------------


def test_unexecuted_stages_do_not_generate_files(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_model_core(monkeypatch)

    run_model_for_active_branch(tmp_path, base_config=config)

    for name in [
        "conditional_granger_scores.csv",
        "causal_review_report.csv",
        "causal_review_evidence.csv",
        "final_review_summary.csv",
        "model_lift_scores.csv",
        "rolling_corr_scores.csv",
        "enhanced_validation_summary.csv",
        "granger_tests.csv",
    ]:
        assert not (tmp_path / name).exists(), name
    assert not (tmp_path / "xgb_validation").exists()


# --- Source constraints: direction, candidate source, no re-threshold -----


def test_pr10_runner_source_preserves_direction_and_candidate_source():
    source = Path("chem_ts_corr/pipeline.py").read_text(encoding="utf-8")
    start = source.index("def run_model_for_active_branch")
    end = source.index("\ndef _resolve_base_config", start)
    body = source[start:end]
    for marker in ("abs(lag", "abs(best_lag", ".abs()"):
        assert marker not in body
    assert "screening_branches" not in body
    assert "preprocessing_comparison.csv" not in body
    assert "final_score >= 0.30" not in body
    assert "_save_secondary_candidate_context" not in body
