from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import (
    CAUSAL_REVIEW_FORMAL_INPUT_FILES,
    DOWNSTREAM_LOCK_FILENAME,
    confirm_initial_screening_branch,
    run_causal_review_for_active_branch,
    run_enhanced_screening_for_active_branch,
    run_granger_for_active_branch,
    run_model_for_active_branch,
)


THREE_TIER_OUTPUT_FILES = [
    "conditional_granger_scores.csv",
    "causal_review_report.csv",
    "causal_review_evidence.csv",
    "final_review_summary.csv",
]
FORMAL_ROOT_FILES = [
    "ranked_features.csv",
    "recommended_candidates.csv",
    "causal_review_candidates.csv",
    "risk_flags.csv",
]
OTHER_STAGE_FILES = [
    "model_lift_scores.csv",
    "rolling_corr_scores.csv",
    "enhanced_validation_summary.csv",
    "granger_tests.csv",
    "shap_or_importance.csv",
    "model_variable_importance.csv",
    "model_discovered_candidates.csv",
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
    n_variables: int = 20,
    *,
    include_forced: bool = False,
    negative_lag_variable: str | None = None,
) -> pd.DataFrame:
    names = [f"variable_{index}" for index in range(n_variables)]
    scores = [0.95 - 0.06 * index for index in range(n_variables)]
    if include_forced:
        names.append("forced_include_variable")
        scores.append(0.01)
    rows = []
    for rank, (name, score) in enumerate(zip(names, scores), start=1):
        lag = (rank % 3) + 1
        if negative_lag_variable is not None and name == negative_lag_variable:
            lag = -lag
        rows.append(
            {
                "variable": name,
                "final_score": score,
                "driver_rank": rank,
                "lag": lag,
            }
        )
    return pd.DataFrame(rows)


def _write_formal_root(
    run_dir: Path,
    ranked: pd.DataFrame,
    *,
    causal_candidates: pd.DataFrame | None = None,
) -> None:
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
    if causal_candidates is None:
        causal_candidates = pd.DataFrame(
            {
                "variable": ranked["variable"].head(5).tolist(),
                "candidate_source": ["formal"] * 5,
                "reason": [""] * 5,
            }
        )
    causal_candidates.to_csv(
        run_dir / "causal_review_candidates.csv",
        index=False,
        encoding="utf-8-sig",
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


def _make_raw_run(
    tmp_path: Path, *, negative_lag_variable: str | None = None
) -> AnalysisConfig:
    config = _config(tmp_path)
    _write_formal_root(
        tmp_path,
        _ranked_frame(negative_lag_variable=negative_lag_variable),
    )
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
    _write_formal_root(tmp_path, _ranked_frame())
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


def _spy_scaled_config(monkeypatch, run_dir: Path) -> dict[str, object]:
    from chem_ts_corr import web

    captured: dict[str, object] = {}
    frame = _full_secondary_frame(run_dir)

    def capture_scaled(config, protected_columns=None):
        captured["config"] = config
        captured["protected_columns"] = list(protected_columns or [])
        return frame

    monkeypatch.setattr(web, "_scaled_frame_for_secondary_causal", capture_scaled)
    monkeypatch.setattr(web, "_target_segment_mask", lambda frame: None)
    return captured


def _patch_causal_review_stage(monkeypatch) -> dict[str, object]:
    from chem_ts_corr import causal_review_runner

    captured: dict[str, object] = {}

    def capture_stage(
        *,
        frame,
        target,
        ranked_features,
        causal_review_candidates,
        risk_flags,
        output_dir,
        control_columns,
        maxlag,
        min_rows,
        top_n,
        conditional_lag_mode,
        conditional_lag_window,
        conditional_fallback_maxlag,
        conditional_baseline_maxlag,
        target_mask,
        prefer_ranked_lag,
    ):
        captured.update(
            {
                "frame": frame,
                "target": target,
                "ranked_features": ranked_features,
                "causal_review_candidates": causal_review_candidates,
                "risk_flags": risk_flags,
                "output_dir": output_dir,
                "control_columns": list(control_columns or []),
                "maxlag": maxlag,
                "min_rows": min_rows,
                "top_n": top_n,
                "conditional_lag_mode": conditional_lag_mode,
                "conditional_lag_window": conditional_lag_window,
                "conditional_fallback_maxlag": conditional_fallback_maxlag,
                "conditional_baseline_maxlag": conditional_baseline_maxlag,
                "target_mask": target_mask,
                "prefer_ranked_lag": prefer_ranked_lag,
            }
        )
        return {
            key: pd.DataFrame({"variable": []})
            for key in (
                "conditional_granger_scores",
                "causal_review_report",
                "causal_review_evidence",
                "final_review_summary",
            )
        }

    monkeypatch.setattr(
        causal_review_runner, "run_causal_review_stage", capture_stage
    )
    return captured


def _patch_enhanced_core(monkeypatch) -> None:
    from chem_ts_corr import screening, web

    monkeypatch.setattr(
        screening,
        "prepare_best_lag_evidence",
        lambda *args, **kwargs: ({}, {}),
    )
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


def _patch_granger_core(monkeypatch) -> None:
    from chem_ts_corr import causality

    monkeypatch.setattr(
        causality,
        "run_granger_tests",
        lambda *args, **kwargs: pd.DataFrame({"variable": []}),
    )


def _patch_model_core(monkeypatch) -> None:
    from chem_ts_corr import modeling

    monkeypatch.setattr(
        modeling,
        "fit_explainable_model",
        lambda *args, **kwargs: (
            pd.DataFrame(
                columns=["feature", "importance", "method", "variable", "lag"]
            ),
            {"model_status": "ok", "r2_holdout": 0.5},
        ),
    )


def _write_input_csv(config: AnalysisConfig) -> None:
    rows = 160
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


def _assert_no_three_tier_outputs(run_dir: Path) -> None:
    for name in THREE_TIER_OUTPUT_FILES:
        assert not (Path(run_dir) / name).exists(), name


def _assert_three_tier_outputs(run_dir: Path) -> None:
    for name in THREE_TIER_OUTPUT_FILES:
        assert (Path(run_dir) / name).exists(), name


def _formal_root_bytes(run_dir: Path) -> dict[str, bytes]:
    return {
        name: (Path(run_dir) / name).read_bytes()
        for name in FORMAL_ROOT_FILES
    }


def _candidate_variables(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty or "variable" not in frame.columns:
        return []
    return [str(value) for value in frame["variable"].dropna()]


# --- Test 1: awaiting blocks the formal causal review runner --------------


def test_awaiting_confirmation_blocks_causal_review(tmp_path, monkeypatch):
    _make_awaiting_run(tmp_path)
    assert _read_context(tmp_path)["branch_selection_status"] == "awaiting_confirmation"
    _patch_causal_review_stage(monkeypatch)

    with pytest.raises(ValueError, match="initial_screening_branch_not_confirmed"):
        run_causal_review_for_active_branch(tmp_path)

    _assert_no_lock(tmp_path)
    _assert_no_three_tier_outputs(tmp_path)


# --- Test 2: confirmed raw wins over selected mode ------------------------


def test_causal_review_uses_confirmed_raw_over_selected_mode(tmp_path, monkeypatch):
    config = _make_processed_run(tmp_path, branch="raw")
    context = _read_context(tmp_path)
    assert context["active_screening_branch"] == "raw"
    assert context["active_preprocessing_mode"] == "raw"

    captured = _spy_scaled_config(monkeypatch, tmp_path)
    _patch_causal_review_stage(monkeypatch)

    run_causal_review_for_active_branch(tmp_path, base_config=config)

    assert captured["config"].preprocess_mode == "raw"
    assert captured["config"].preprocess_mode != "lowpass_diff"


def test_causal_review_uses_in_memory_causal_ranked_lags(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    ranked_path = tmp_path / "ranked_features.csv"
    ranked_before = ranked_path.read_bytes()
    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)

    run_causal_review_for_active_branch(tmp_path, base_config=config)

    ranked = pd.read_csv(ranked_path, encoding="utf-8-sig")
    assert ranked_path.read_bytes() == ranked_before
    assert captured["ranked_features"]["lag"].tolist() == [-config.max_lag] * len(ranked)
    assert captured["ranked_features"]["final_score"].tolist() == ranked["final_score"].tolist()
    assert captured["prefer_ranked_lag"] is True


# --- Test 3: confirmed processed uses context parameters ------------------


def test_causal_review_uses_processed_context_params(tmp_path, monkeypatch):
    config = _make_processed_run(tmp_path, branch="processed")
    assert _read_context(tmp_path)["active_preprocessing_mode"] == "lowpass_diff"

    captured = _spy_scaled_config(monkeypatch, tmp_path)
    _patch_causal_review_stage(monkeypatch)

    run_causal_review_for_active_branch(tmp_path, base_config=config)

    downstream = captured["config"]
    assert downstream.preprocess_mode == "lowpass_diff"
    assert downstream.lowpass_tau_minutes == 7.5
    assert downstream.diff_interval_minutes == 5.0
    assert downstream.resample_rule == "2min"


# --- Test 4: context is source of truth over caller config ----------------


def test_context_overrides_caller_preprocessing_config(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    caller_config = replace(
        config,
        preprocess_mode="lowpass_diff",
        lowpass_tau_minutes=3.0,
        diff_interval_minutes=2.0,
    )

    captured = _spy_scaled_config(monkeypatch, tmp_path)
    _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(tmp_path, base_config=caller_config)

    downstream = captured["config"]
    assert downstream.preprocess_mode == "raw"
    assert downstream.diff_interval_minutes is None


# --- Test 5: candidates only come from the formal root --------------------


def test_causal_review_candidates_come_from_formal_root(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    formal_candidates = pd.DataFrame(
        {
            "variable": ["candidate_A", "candidate_B", "candidate_C"],
            "candidate_source": ["formal", "formal", "formal"],
            "review_priority": [1, 2, 3],
        }
    )
    formal_candidates.to_csv(
        tmp_path / "causal_review_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    before = (tmp_path / "causal_review_candidates.csv").read_bytes()

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(tmp_path, base_config=config)

    candidates = _candidate_variables(captured["causal_review_candidates"])
    assert candidates == ["candidate_A", "candidate_B", "candidate_C"]
    assert (tmp_path / "causal_review_candidates.csv").read_bytes() == before


# --- Test 6: secondary candidate context is ignored -----------------------


def test_secondary_candidate_context_ignored(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    formal_candidates = pd.DataFrame(
        {"variable": ["candidate_A", "candidate_B"], "candidate_source": ["formal"] * 2}
    )
    formal_candidates.to_csv(
        tmp_path / "causal_review_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (tmp_path / "secondary_candidate_context.csv").write_text(
        "variable\nsentinel_secondary_only\n", encoding="utf-8-sig"
    )

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(tmp_path, base_config=config)

    candidates = _candidate_variables(captured["causal_review_candidates"])
    assert "sentinel_secondary_only" not in candidates
    assert candidates == ["candidate_A", "candidate_B"]


# --- Test 7: non-active branch candidates are ignored ---------------------


def test_non_active_branch_candidates_ignored(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    processed_dir = tmp_path / "screening_branches" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"variable": ["sentinel_processed_only"]}).to_csv(
        processed_dir / "causal_review_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(tmp_path, base_config=config)

    candidates = _candidate_variables(captured["causal_review_candidates"])
    assert "sentinel_processed_only" not in candidates


# --- Test 8: model discovered candidates are ignored ----------------------


def test_model_discovered_candidates_ignored(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    pd.DataFrame({"variable": ["sentinel_model_only"], "max_importance": [0.99]}).to_csv(
        tmp_path / "model_discovered_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(tmp_path, base_config=config)

    candidates = _candidate_variables(captured["causal_review_candidates"])
    assert "sentinel_model_only" not in candidates


# --- Test 9: default top_n=None covers the full formal candidate set -----


def test_default_covers_full_formal_candidate_set(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    ranked = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")
    full_candidates = ranked.head(config.top_k)[
        ["variable", "final_score", "driver_rank"]
    ].copy()
    full_candidates["candidate_source"] = "formal"
    full_candidates.to_csv(
        tmp_path / "causal_review_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    low_score = ranked[
        (ranked["driver_rank"] <= config.top_k)
        & (pd.to_numeric(ranked["final_score"], errors="coerce") < 0.30)
    ]["variable"].astype(str).tolist()
    assert low_score, "fixture must contain Top-K variables below 0.30"

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(tmp_path, base_config=config)

    candidates = _candidate_variables(captured["causal_review_candidates"])
    expected = ranked["variable"].astype(str).head(config.top_k).tolist()
    assert candidates == expected
    assert all(variable in candidates for variable in low_score)


# --- Test 10: explicit top_n preserves the stage's existing order --------


def test_explicit_top_n_preserves_stage_order(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    ordered_candidates = pd.DataFrame(
        {"variable": ["A", "B", "C", "D"], "candidate_source": ["formal"] * 4}
    )
    ordered_candidates.to_csv(
        tmp_path / "causal_review_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(
        tmp_path, base_config=config, top_n=2
    )

    candidates = _candidate_variables(captured["causal_review_candidates"])
    assert candidates == ["A", "B", "C", "D"]
    assert captured["top_n"] == 2


# --- Test 11: control column resolution keeps the existing priority ------


@pytest.mark.parametrize(
    ("explicit", "residual", "capacity", "expected"),
    [
        (["explicit_control"], ["residual_control"], [], ["explicit_control"]),
        (None, ["residual_control"], ["capacity_control"], ["residual_control"]),
        (None, [], ["capacity_control"], ["capacity_control"]),
        ([], ["residual_control"], ["capacity_control"], []),
    ],
)
def test_control_columns_resolution(
    tmp_path, monkeypatch, explicit, residual, capacity, expected
):
    config = _make_raw_run(tmp_path)
    config = replace(
        config,
        residual_control_columns=residual,
        capacity_columns=capacity,
    )

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(
        tmp_path, base_config=config, control_columns=explicit
    )

    assert captured["control_columns"] == expected


# --- Test 12: explicit control columns are protected in the frame --------


def test_control_columns_protected_in_scaled_frame(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)

    captured = _spy_scaled_config(monkeypatch, tmp_path)
    _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(
        tmp_path,
        base_config=config,
        control_columns=["explicit_control"],
    )

    assert captured["protected_columns"] == ["explicit_control"]


# --- Test 13: conditional parameters are fully passed through ------------


def test_conditional_parameters_passed_through(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(
        tmp_path,
        base_config=config,
        control_columns=["explicit_control"],
        maxlag=4,
        min_rows=80,
        top_n=3,
        conditional_lag_mode="best_only",
        conditional_lag_window=2,
        conditional_fallback_maxlag=10,
        conditional_baseline_maxlag=8,
    )

    assert captured["target"] == config.target
    assert captured["control_columns"] == ["explicit_control"]
    assert captured["maxlag"] == 4
    assert captured["min_rows"] == 80
    assert captured["top_n"] == 3
    assert captured["conditional_lag_mode"] == "best_only"
    assert captured["conditional_lag_window"] == 2
    assert captured["conditional_fallback_maxlag"] == 10
    assert captured["conditional_baseline_maxlag"] == 8
    assert captured["target_mask"] is None
    assert captured["output_dir"] == tmp_path


# --- Test 14: maxlag=None uses the existing resolved config --------------


@pytest.mark.parametrize(
    ("granger_maxlag", "max_lag", "expected"),
    [(None, 3, 3), (5, 3, 5)],
)
def test_maxlag_none_uses_resolved_config(
    tmp_path, monkeypatch, granger_maxlag, max_lag, expected
):
    config = _make_raw_run(tmp_path)
    config = replace(config, granger_maxlag=granger_maxlag, max_lag=max_lag)

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(tmp_path, base_config=config)

    assert captured["maxlag"] == expected


# --- Test 15: formal screening files stay byte-identical -----------------


def test_causal_review_keeps_formal_root_byte_identical(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    before = _formal_root_bytes(tmp_path)
    ranked_before = pd.read_csv(
        tmp_path / "ranked_features.csv", encoding="utf-8-sig"
    )

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_causal_review_stage(monkeypatch)
    result = run_causal_review_for_active_branch(tmp_path, base_config=config)

    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
    ranked_after = pd.read_csv(
        tmp_path / "ranked_features.csv", encoding="utf-8-sig"
    )
    assert ranked_after["final_score"].tolist() == ranked_before["final_score"].tolist()
    assert ranked_after["driver_rank"].tolist() == ranked_before["driver_rank"].tolist()
    assert ranked_after["variable"].tolist() == ranked_before["variable"].tolist()
    assert result["active_screening_branch"] == "raw"


# --- Test 16: exactly the four three-tier outputs are generated ----------


def test_causal_review_generates_four_outputs_only(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    before = {path.name for path in tmp_path.glob("*.csv")}

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(tmp_path, base_config=config)

    after = {path.name for path in tmp_path.glob("*.csv")}
    assert after - before == set(THREE_TIER_OUTPUT_FILES)
    assert "secondary_candidate_context.csv" not in after
    _assert_three_tier_outputs(tmp_path)


# --- Test 17: no optional evidence is required ---------------------------


def test_causal_review_runs_without_optional_evidence(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    for name in (
        "enhanced_validation_summary.csv",
        "granger_tests.csv",
        "model_variable_importance.csv",
    ):
        assert not (tmp_path / name).exists(), name

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_causal_review_stage(monkeypatch)
    result = run_causal_review_for_active_branch(tmp_path, base_config=config)

    _assert_three_tier_outputs(tmp_path)
    assert result["active_screening_branch"] == "raw"


def test_causal_review_runs_real_stage_without_optional_evidence(
    tmp_path, monkeypatch
):
    config = _make_raw_run(tmp_path)
    _write_input_csv(config)

    result = run_causal_review_for_active_branch(tmp_path, base_config=config)

    _assert_three_tier_outputs(tmp_path)
    conditional = pd.read_csv(
        tmp_path / "conditional_granger_scores.csv", encoding="utf-8-sig"
    )
    assert "variable" in conditional.columns
    assert len(conditional) == 5
    assert result["active_preprocessing_mode"] == "raw"


# --- Test 18: existing optional evidence is read by the stage ------------


def test_optional_evidence_loader_receives_run_dir(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(tmp_path, base_config=config)

    assert captured["output_dir"] == tmp_path


def test_causal_review_reads_existing_optional_evidence(tmp_path):
    config = _make_raw_run(tmp_path)
    _write_input_csv(config)
    pd.DataFrame(
        [
            {
                "variable": "variable_0",
                "model_lift": 0.06,
                "rolling_stability": 0.8,
            }
        ]
    ).to_csv(
        tmp_path / "enhanced_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "variable": "variable_0",
                "importance_rank": 2,
                "max_importance": 0.3,
            }
        ]
    ).to_csv(
        tmp_path / "model_variable_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    run_causal_review_for_active_branch(tmp_path, base_config=config)

    evidence = pd.read_csv(
        tmp_path / "causal_review_evidence.csv", encoding="utf-8-sig"
    )
    row = evidence[evidence["variable"].astype(str) == "variable_0"].iloc[0]
    assert row["model_lift"] == 0.06
    assert row["rolling_stability"] == 0.8
    assert row["model_importance_rank"] == 2


# --- Test 19: causal review as the first downstream stage creates lock ---


def test_causal_review_creates_downstream_lock_as_first_stage(
    tmp_path, monkeypatch
):
    config = _make_raw_run(tmp_path)
    _assert_no_lock(tmp_path)
    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_causal_review_stage(monkeypatch)

    run_causal_review_for_active_branch(tmp_path, base_config=config)

    lock = tmp_path / DOWNSTREAM_LOCK_FILENAME
    assert lock.exists()
    assert lock.read_text(encoding="utf-8") == "downstream-locked"


# --- Test 20: existing lock still allows the causal review stage ---------


@pytest.mark.parametrize("prior_stage", ["enhanced", "granger", "model"])
def test_causal_review_runs_after_existing_downstream_lock(
    tmp_path, monkeypatch, prior_stage
):
    config = _make_raw_run(tmp_path)
    _spy_scaled_config(monkeypatch, tmp_path)
    if prior_stage == "enhanced":
        _patch_enhanced_core(monkeypatch)
        run_enhanced_screening_for_active_branch(tmp_path, base_config=config)
    elif prior_stage == "granger":
        _patch_granger_core(monkeypatch)
        run_granger_for_active_branch(tmp_path, base_config=config)
    else:
        _patch_model_core(monkeypatch)
        run_model_for_active_branch(tmp_path, base_config=config)
    assert (tmp_path / DOWNSTREAM_LOCK_FILENAME).exists()

    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_causal_review_stage(monkeypatch)
    result = run_causal_review_for_active_branch(tmp_path, base_config=config)

    _assert_three_tier_outputs(tmp_path)
    assert result["active_screening_branch"] == "raw"


# --- Test 21: after causal review the branch cannot switch ---------------


def test_causal_review_locks_branch_switch(tmp_path, monkeypatch):
    config = _make_processed_run(tmp_path, branch="raw")
    confirm_initial_screening_branch(tmp_path, branch="raw")
    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(tmp_path, base_config=config)
    assert (tmp_path / DOWNSTREAM_LOCK_FILENAME).exists()

    with pytest.raises(ValueError, match="initial_screening_branch_locked"):
        confirm_initial_screening_branch(tmp_path, branch="processed")


# --- Test 22: missing formal root input fails without fallback -----------


@pytest.mark.parametrize(
    "missing_name",
    [
        "ranked_features.csv",
        "recommended_candidates.csv",
        "causal_review_candidates.csv",
        "risk_flags.csv",
    ],
)
def test_causal_review_formal_root_missing_fails_without_fallback(
    tmp_path, missing_name
):
    config = _make_raw_run(tmp_path)
    (tmp_path / missing_name).unlink()

    with pytest.raises(ValueError, match="initial_screening_formal_output_missing"):
        run_causal_review_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)
    _assert_no_three_tier_outputs(tmp_path)


# --- Test 23: missing / invalid context keeps frozen errors --------------


def test_causal_review_missing_context_fails(tmp_path):
    config = _make_raw_run(tmp_path)
    (tmp_path / "preprocessing_context.json").unlink()

    with pytest.raises(ValueError, match="initial_screening_context_missing"):
        run_causal_review_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)
    _assert_no_three_tier_outputs(tmp_path)


def test_causal_review_invalid_context_fails(tmp_path):
    config = _make_raw_run(tmp_path)
    (tmp_path / "preprocessing_context.json").write_text(
        "{not-json", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="initial_screening_context_invalid"):
        run_causal_review_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)
    _assert_no_three_tier_outputs(tmp_path)


def test_causal_review_invalid_status_context_fails(tmp_path):
    config = _make_raw_run(tmp_path)
    context = _read_context(tmp_path)
    context["branch_selection_status"] = "bogus"
    (tmp_path / "preprocessing_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="initial_screening_context_invalid"):
        run_causal_review_for_active_branch(tmp_path, base_config=config)

    _assert_no_lock(tmp_path)
    _assert_no_three_tier_outputs(tmp_path)


# --- Test 24: no other downstream stages are auto-generated --------------


def test_causal_review_does_not_generate_other_stages(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path)
    _spy_scaled_config(monkeypatch, tmp_path)
    _patch_causal_review_stage(monkeypatch)

    run_causal_review_for_active_branch(tmp_path, base_config=config)

    for name in OTHER_STAGE_FILES:
        assert not (tmp_path / name).exists(), name
    assert not (tmp_path / "xgb_validation").exists()
    assert not (tmp_path / "preprocessing_comparison.csv").exists()
    assert not (tmp_path / "secondary_candidate_context.csv").exists()


# --- Signed lag direction is preserved from the formal root --------------


def test_negative_signed_lag_preserved(tmp_path, monkeypatch):
    config = _make_raw_run(tmp_path, negative_lag_variable="variable_0")
    ranked_before = pd.read_csv(
        tmp_path / "ranked_features.csv", encoding="utf-8-sig"
    )
    assert ranked_before.loc[
        ranked_before["variable"] == "variable_0", "lag"
    ].iloc[0] < 0

    _spy_scaled_config(monkeypatch, tmp_path)
    captured = _patch_causal_review_stage(monkeypatch)
    run_causal_review_for_active_branch(tmp_path, base_config=config)

    passed = captured["ranked_features"]
    assert passed.loc[passed["variable"] == "variable_0", "lag"].iloc[0] < 0


# --- Test 25: source constraints for the formal runner -------------------


def test_pr11_runner_source_preserves_direction_and_candidate_source():
    source = Path("chem_ts_corr/pipeline.py").read_text(encoding="utf-8")
    start = source.index("def run_causal_review_for_active_branch")
    end = source.index("\ndef _resolve_base_config", start)
    body = source[start:end]
    for marker in (
        "_save_secondary_candidate_context",
        "_load_secondary_candidate_context",
        "_build_causal_review_candidate_table",
        "build_causal_review_candidates",
        "screening_branches",
        "preprocessing_comparison.csv",
        "model_discovered_candidates",
        "final_score >= 0.30",
        "abs(lag",
        "abs(best_lag",
        "abs(granger_lag",
    ):
        assert marker not in body
    assert "required_formal_files=CAUSAL_REVIEW_FORMAL_INPUT_FILES" in body
    for name in THREE_TIER_OUTPUT_FILES:
        assert name in body
