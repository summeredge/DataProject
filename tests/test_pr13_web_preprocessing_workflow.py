from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chem_ts_corr.pipeline as pipeline
import chem_ts_corr.web as web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import (
    begin_downstream_stage,
    confirm_initial_screening_branch,
    run_analysis,
    run_initial_screening_workflow,
)


FORMAL_MODES = ["raw", "lowpass", "lowpass_detrend", "lowpass_diff"]
LEGACY_MODES = ["detrend", "diff", "detrend_diff"]
PROCESSED_MODES = ["lowpass", "lowpass_detrend", "lowpass_diff"]


@pytest.fixture(autouse=True)
def _web_runs_dir(tmp_path, monkeypatch):
    """Point the Web RUNS_DIR at the test directory root."""
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path.parent)


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


def _write_run(
    tmp_path: Path,
    *,
    mode: str,
    tau: float = 5.0,
    diff_interval: float | None = None,
) -> tuple[AnalysisConfig, dict[str, object]]:
    config = _raw_config(
        tmp_path,
        preprocess_mode=mode,
        lowpass_tau_minutes=tau,
        diff_interval_minutes=diff_interval,
    )
    _write_input(config, _raw_frame())
    web._write_run_config(config.output_dir, config, "file-id")
    result = run_initial_screening_workflow(config)
    return config, result


def _read_context(run_dir: Path) -> dict[str, object]:
    return json.loads(
        (Path(run_dir) / "preprocessing_context.json").read_text(encoding="utf-8")
    )


def _write_stale_validation_evidence(output_dir: Path) -> None:
    pd.DataFrame(
        [{
            "variable": "candidate_0",
            "status": "ok",
            "predictive_contribution": 0.9,
        }]
    ).to_csv(output_dir / "granger_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{
            "variable": "stale_conditional",
            "status": "ok",
            "best_lag": 7,
        }]
    ).to_csv(
        output_dir / "conditional_granger_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for name in [
        "enhanced_validation_summary.csv",
        "model_lift_scores.csv",
        "rolling_corr_scores.csv",
        "model_variable_importance.csv",
        "shap_or_importance.csv",
        "model_discovered_candidates.csv",
    ]:
        pd.DataFrame(
            [{
                "variable": "candidate_0",
                "status": "ok",
                "model_lift": 0.2,
                "rolling_stability": 0.8,
                "max_importance": 0.7,
            }]
        ).to_csv(output_dir / name, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{
            "variable": "candidate_0",
            "validation_status": "supported",
            "evidence_consistency": "consistent",
            "supporting_methods": "granger",
            "limiting_factors": "",
        }]
    ).to_csv(output_dir / "validation_summary.csv", index=False, encoding="utf-8-sig")


# --- Test 1: Web only offers the four formal preprocess modes --------------


def test_web_form_offers_only_four_formal_modes():
    select = web.INDEX_HTML.split('<select id="preprocessMode">', 1)[1].split(
        "</select>", 1
    )[0]
    for mode in FORMAL_MODES:
        assert f'value="{mode}"' in select
    for mode in LEGACY_MODES:
        assert f'value="{mode}"' not in select


# --- Tests 2/3: tau / diff parameters -------------------------------------


def test_analyze_response_passes_tau_default_and_diff_none(monkeypatch, tmp_path):
    form = {
        "file_id": "file-1",
        "time_column": "time",
        "target": "target",
        "preprocess_mode": "lowpass_diff",
        "resample_rule": "",
    }
    threads: list[object] = []

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            self.args = args
            self.started = False
            threads.append(self)

        def start(self):
            self.started = True

    (tmp_path / "input.csv").write_text(
        "time,target,x\n2025-01-01 00:00:00,1,2\n", encoding="utf-8"
    )
    monkeypatch.setattr(web, "_multipart_form", lambda handler: form)
    monkeypatch.setattr(web, "_resolve_upload", lambda file_id: tmp_path / "input.csv")
    monkeypatch.setattr(web, "_resolve_encoding", lambda path, encoding: "utf-8-sig")
    monkeypatch.setattr(web, "_validate_analysis_excluded_columns", lambda *args, **kwargs: None)
    monkeypatch.setattr(web.threading, "Thread", FakeThread)
    monkeypatch.setattr(web, "_cleanup_tasks_locked", lambda **kwargs: None)

    web._analyze_response(object())

    config = threads[0].args[1]
    assert config.preprocess_mode == "lowpass_diff"
    assert config.lowpass_tau_minutes == 5.0
    assert config.diff_interval_minutes is None


def test_analyze_response_passes_positive_diff_interval(monkeypatch, tmp_path):
    form = {
        "file_id": "file-1",
        "time_column": "time",
        "target": "target",
        "preprocess_mode": "lowpass_diff",
        "lowpass_tau_minutes": "7.5",
        "diff_interval_minutes": "5",
        "resample_rule": "",
    }
    threads: list[object] = []

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            self.args = args
            self.started = False
            threads.append(self)

        def start(self):
            self.started = True

    (tmp_path / "input.csv").write_text(
        "time,target,x\n2025-01-01 00:00:00,1,2\n", encoding="utf-8"
    )
    monkeypatch.setattr(web, "_multipart_form", lambda handler: form)
    monkeypatch.setattr(web, "_resolve_upload", lambda file_id: tmp_path / "input.csv")
    monkeypatch.setattr(web, "_resolve_encoding", lambda path, encoding: "utf-8-sig")
    monkeypatch.setattr(web, "_validate_analysis_excluded_columns", lambda *args, **kwargs: None)
    monkeypatch.setattr(web.threading, "Thread", FakeThread)
    monkeypatch.setattr(web, "_cleanup_tasks_locked", lambda **kwargs: None)

    web._analyze_response(object())

    config = threads[0].args[1]
    assert config.lowpass_tau_minutes == 7.5
    assert config.diff_interval_minutes == 5.0


@pytest.mark.parametrize("bad_diff", ["0", "0.0", "-1"])
def test_analyze_response_rejects_non_positive_diff_interval(
    monkeypatch, tmp_path, bad_diff: str
):
    form = {
        "file_id": "file-1",
        "time_column": "time",
        "target": "target",
        "preprocess_mode": "lowpass_diff",
        "diff_interval_minutes": bad_diff,
        "resample_rule": "",
    }
    (tmp_path / "input.csv").write_text(
        "time,target,x\n2025-01-01 00:00:00,1,2\n", encoding="utf-8"
    )
    monkeypatch.setattr(web, "_multipart_form", lambda handler: form)
    monkeypatch.setattr(web, "_resolve_upload", lambda file_id: tmp_path / "input.csv")
    monkeypatch.setattr(web, "_resolve_encoding", lambda path, encoding: "utf-8-sig")
    monkeypatch.setattr(web, "_validate_analysis_excluded_columns", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        web.threading,
        "Thread",
        lambda **kwargs: pytest.fail("invalid diff interval must not create a thread"),
    )

    with pytest.raises(ValueError, match="diff_interval_minutes"):
        web._analyze_response(object())


# --- Test 4: raw analyze uses the unified workflow -------------------------


def test_analyze_task_uses_unified_workflow_not_legacy_run_analysis():
    source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")
    task_source = source.split("def _analyze_task", 1)[1].split("def _workflow_timings", 1)[0]
    assert "run_initial_screening_workflow(" in task_source
    assert "run_analysis(" not in task_source
    assert not hasattr(web, "run_analysis")


def test_raw_workflow_payload_is_formal_and_auto_promoted(tmp_path):
    config, _ = _write_run(tmp_path, mode="raw")

    payload = web._build_result_payload(config.output_dir.name, config.output_dir, config)

    assert payload["branchSelectionStatus"] == "not_required"
    assert payload["activeScreeningBranch"] == "raw"
    assert payload["activePreprocessingMode"] == "raw"
    assert payload["selectedPreprocessingMode"] == "raw"
    assert payload["analysisContext"]["preprocess_mode"] == "raw"
    assert payload["rankedFeatures"]
    assert payload["recommendedCandidates"]


def test_raw_workflow_publishes_a_separate_verification_review_pool(tmp_path):
    config, _ = _write_run(tmp_path, mode="raw")
    ranked = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")

    pool = pd.read_csv(tmp_path / "verification_review_pool.csv", encoding="utf-8-sig")

    assert list(pool.columns) == [
        "variable",
        "candidate_source",
        "source_rank",
        "include_reason",
    ]
    assert len(pool) == min(config.top_k, len(ranked))
    assert set(pool["candidate_source"]) == {"initial_screening"}
    assert pool["variable"].tolist() == ranked.head(config.top_k)["variable"].tolist()
    assert pool["source_rank"].tolist() == ranked.head(config.top_k)["driver_rank"].tolist()


def test_reused_output_dir_clears_stale_validation_summary(tmp_path):
    config, _ = _write_run(tmp_path, mode="raw")
    _write_stale_validation_evidence(tmp_path)

    run_initial_screening_workflow(config)

    stale_files = [
        "validation_summary.csv",
        "granger_tests.csv",
        "conditional_granger_scores.csv",
        "enhanced_validation_summary.csv",
        "model_lift_scores.csv",
        "rolling_corr_scores.csv",
        "model_variable_importance.csv",
        "shap_or_importance.csv",
        "model_discovered_candidates.csv",
    ]
    assert all(not (tmp_path / name).exists() for name in stale_files)
    payload = web._build_result_payload(config.output_dir.name, config.output_dir, config)
    assert payload["validationSummary"]
    assert all(
        row["validation_status"] == "not_run"
        for row in payload["validationSummary"]
    )
    validation_fields = payload["validationFields"]
    assert "stale_conditional" not in {
        row["variable"] for row in validation_fields
    }
    current = next(row for row in validation_fields if row["variable"] == "candidate_0")
    assert current["conditional_validation_lag"] is None
    assert payload["importance"] == []
    assert payload["modelDiscoveredCandidates"] == []


def test_legacy_run_analysis_clears_stale_validation_evidence(tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="raw")
    _write_input(config, _raw_frame())
    _write_stale_validation_evidence(tmp_path)

    run_analysis(config)

    stale_files = [
        "validation_summary.csv",
        "granger_tests.csv",
        "conditional_granger_scores.csv",
        "enhanced_validation_summary.csv",
        "model_lift_scores.csv",
        "rolling_corr_scores.csv",
        "model_variable_importance.csv",
        "shap_or_importance.csv",
        "model_discovered_candidates.csv",
    ]
    assert all(not (tmp_path / name).exists() for name in stale_files)
    payload = web._build_result_payload(config.output_dir.name, config.output_dir, config)
    assert payload["validationSummary"]
    assert all(
        row["validation_status"] == "not_run"
        for row in payload["validationSummary"]
    )
    assert payload["importance"] == []
    assert payload["modelDiscoveredCandidates"] == []


# --- Test 5: processed workflow returns pending payload --------------------


@pytest.mark.parametrize("mode", PROCESSED_MODES)
def test_processed_workflow_payload_is_pending(tmp_path, mode: str):
    config, _ = _write_run(tmp_path, mode=mode)

    payload = web._build_result_payload(config.output_dir.name, config.output_dir, config)

    assert payload["branchSelectionStatus"] == "awaiting_confirmation"
    assert payload["activeScreeningBranch"] is None
    assert payload["activePreprocessingMode"] is None
    assert payload["selectedPreprocessingMode"] == mode
    assert payload["preprocessingComparison"]
    assert "rankedFeatures" not in payload
    assert "recommendedCandidates" not in payload
    assert "summary.md" not in {item["name"] for item in payload["downloads"]}
    assert {item["name"] for item in payload["downloads"]} <= {
        "preprocessing_comparison.csv",
        "preprocessing_context.json",
    }


# --- Test 6: comparison records come directly from the frozen CSV ----------


def test_payload_comparison_reads_frozen_csv_verbatim(tmp_path):
    config, _ = _write_run(tmp_path, mode="lowpass_diff")
    sentinel = pd.DataFrame(
        [{"variable": "sentinel_var", "processed_mode": "lowpass_diff", "rank_delta": 7}]
    )
    sentinel.to_csv(
        tmp_path / "preprocessing_comparison.csv", index=False, encoding="utf-8-sig"
    )

    payload = web._build_result_payload(config.output_dir.name, config.output_dir, config)

    assert payload["preprocessingComparison"][0]["variable"] == "sentinel_var"
    assert payload["preprocessingComparison"][0]["rank_delta"] == 7


# --- Tests 7-9: confirmation API -------------------------------------------


def _confirm_form(run_id: str, branch: str) -> dict[str, str]:
    return {"run_id": run_id, "branch": branch}


def _confirm_payload(tmp_path: Path, branch: str) -> dict[str, object]:
    patch = pytest.MonkeyPatch()
    patch.setattr(
        web,
        "_multipart_form",
        lambda handler: {"run_id": tmp_path.name, "branch": branch},
    )
    try:
        return web._confirm_initial_screening_branch_response(SimpleNamespace())
    finally:
        patch.undo()


class SimpleNamespace:
    pass


def test_confirm_api_spies_backend_call_only(monkeypatch, tmp_path):
    config = _raw_config(tmp_path, preprocess_mode="lowpass")
    monkeypatch.setattr(web, "_multipart_form", lambda handler: _confirm_form("run-1", "raw"))
    monkeypatch.setattr(web, "_resolve_run_dir", lambda run_id: tmp_path)
    monkeypatch.setattr(web, "_read_run_config", lambda output_dir: config)
    monkeypatch.setattr(
        web, "_build_result_payload", lambda *args, **kwargs: {"ok": True}
    )
    calls: list[tuple[object, ...]] = []

    def spy(output_dir, *, branch):
        calls.append((output_dir, branch))

    monkeypatch.setattr(web, "confirm_initial_screening_branch", spy)

    result = _confirm_payload(tmp_path, "raw")

    assert result == {"ok": True}
    assert calls == [(tmp_path, "raw")]


def test_confirm_raw_promotes_and_payload_reflects_active_mode(tmp_path):
    config, _ = _write_run(tmp_path, mode="lowpass_diff")

    payload = _confirm_payload(tmp_path, "raw")

    assert payload["branchSelectionStatus"] == "confirmed"
    assert payload["activeScreeningBranch"] == "raw"
    assert payload["activePreprocessingMode"] == "raw"
    assert payload["selectedPreprocessingMode"] == "lowpass_diff"
    assert payload["analysisContext"]["preprocess_mode"] == "raw"
    assert payload["rankedFeatures"]


def test_confirm_processed_promotes_selected_mode(tmp_path):
    config, _ = _write_run(tmp_path, mode="lowpass_diff")

    payload = _confirm_payload(tmp_path, "processed")

    assert payload["branchSelectionStatus"] == "confirmed"
    assert payload["activeScreeningBranch"] == "processed"
    assert payload["activePreprocessingMode"] == "lowpass_diff"
    assert payload["selectedPreprocessingMode"] == "lowpass_diff"
    assert payload["analysisContext"]["preprocess_mode"] == "lowpass_diff"


def test_confirmation_never_reruns_screening(monkeypatch, tmp_path):
    config, _ = _write_run(tmp_path, mode="lowpass")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("confirmation must not re-run screening")

    monkeypatch.setattr(web, "run_initial_screening_workflow", fail_if_called)
    monkeypatch.setattr(pipeline, "run_initial_screening_branch", fail_if_called)
    monkeypatch.setattr(pipeline, "run_initial_screening_comparison", fail_if_called)

    payload = _confirm_payload(tmp_path, "processed")

    assert payload["branchSelectionStatus"] == "confirmed"


# --- Test 10/11: branch switching and lock semantics -----------------------


def test_switch_branch_via_api_before_downstream(tmp_path):
    _write_run(tmp_path, mode="lowpass")
    _confirm_payload(tmp_path, "raw")
    payload = _confirm_payload(tmp_path, "processed")

    assert payload["activeScreeningBranch"] == "processed"
    assert payload["activePreprocessingMode"] == "lowpass"
    assert _read_context(tmp_path)["active_screening_branch"] == "processed"


def test_downstream_lock_blocks_branch_switch_via_api(tmp_path):
    _write_run(tmp_path, mode="lowpass")
    _confirm_payload(tmp_path, "raw")
    begin_downstream_stage(tmp_path)

    with pytest.raises(ValueError, match="initial_screening_branch_locked"):
        _confirm_payload(tmp_path, "processed")


# --- Test 12: pending state blocks all downstream endpoints ----------------


@pytest.mark.parametrize(
    "endpoint_name",
    [
        "_run_enhanced_screening_response",
        "_run_granger_response",
        "_run_model_response",
        "_run_causal_review_response",
    ],
)
def test_pending_state_blocks_downstream_endpoints(
    tmp_path, monkeypatch, endpoint_name: str
):
    config, _ = _write_run(tmp_path, mode="lowpass")
    monkeypatch.setattr(
        web, "_multipart_form", lambda handler: {"run_id": config.output_dir.name}
    )
    with pytest.raises(ValueError, match="initial_screening_branch_not_confirmed"):
        getattr(web, endpoint_name)(object())


def test_pending_state_blocks_xgb_endpoint_without_raw_fallback(
    tmp_path, monkeypatch
):
    config, _ = _write_run(tmp_path, mode="lowpass")
    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda handler: {
            "run_id": config.output_dir.name,
            "enable_xgb_validation": "true",
        },
    )
    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == "invalid_input"
    assert "initial_screening_branch_not_confirmed" in payload["error_message"]


# --- Test 13/53: downstream endpoints call the formal runners --------------


def test_xgb_endpoint_uses_fold_safe_formal_runner(tmp_path, monkeypatch):
    config, _ = _write_run(tmp_path, mode="raw")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda handler: {
            "run_id": config.output_dir.name,
            "enable_xgb_validation": "true",
            "top_n": "4",
            "max_lag": "5",
            "whitelist": "w1, w2",
            "control_columns": "c1",
        },
    )

    def fake_runner(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        (output_dir / "xgb_validation").mkdir(exist_ok=True)
        pd.DataFrame([{"model_name": "M2"}]).to_csv(
            output_dir / "xgb_validation" / "xgb_model_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([{"variable": "x"}]).to_csv(
            output_dir / "xgb_validation" / "xgb_candidate_uplift.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (output_dir / "xgb_validation" / "xgb_validation_summary.json").write_text(
            json.dumps({"status": "success"}), encoding="utf-8"
        )
        return {"status": "success", "error_message": None}

    monkeypatch.setattr(web, "run_xgb_for_active_branch", fake_runner)

    payload = web._run_xgb_validation_response(object())

    assert payload["status"] == "success"
    assert captured["output_dir"] == config.output_dir
    assert captured["top_n"] == 4
    assert captured["max_lag"] == 5
    assert captured["whitelist"] == ["w1", "w2"]
    assert captured["control_columns"] == ["c1"]
    assert captured["base_config"].preprocess_mode == "raw"
    assert captured["base_config"].output_dir == config.output_dir


# --- Test 52: causal review candidates stay byte-identical -----------------


def test_causal_review_endpoint_does_not_rewrite_candidate_csv(tmp_path, monkeypatch):
    config, _ = _write_run(tmp_path, mode="lowpass")
    _confirm_payload(tmp_path, "processed")
    candidate_path = tmp_path / "causal_review_candidates.csv"
    candidates_before = candidate_path.read_bytes()

    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda handler: {"run_id": config.output_dir.name},
    )

    def fake_runner(output_dir, **kwargs):
        for name in [
            "conditional_granger_scores.csv",
            "causal_review_report.csv",
            "causal_review_evidence.csv",
            "final_review_summary.csv",
        ]:
            pd.DataFrame([{"variable": "candidate_0"}]).to_csv(
                output_dir / name, index=False, encoding="utf-8-sig"
            )
        return {"run_dir": output_dir}

    monkeypatch.setattr(web, "run_causal_review_for_active_branch", fake_runner)

    result = web._run_causal_review_response(object())

    assert candidate_path.read_bytes() == candidates_before
    assert result["finalReviewSummary"][0]["variable"] == "candidate_0"


# --- Test 15/55: selected vs active mode regression ------------------------


def test_payload_distinguishes_selected_and_active_modes(tmp_path):
    _write_run(tmp_path, mode="lowpass_diff")
    payload = _confirm_payload(tmp_path, "raw")

    assert payload["selectedPreprocessingMode"] == "lowpass_diff"
    assert payload["activePreprocessingMode"] == "raw"
    assert payload["analysisContext"]["preprocess_mode"] == "raw"
    assert payload["activePreprocessingMode"] != payload["selectedPreprocessingMode"]


def test_chart_query_uses_active_context_and_form_only_for_preview():
    body = web.INDEX_HTML.split("function appendChartQueryParams", 1)[1].split(
        "async function drawTrend", 1
    )[0]

    assert 'currentAnalysisContext.preprocess_mode' in body
    assert 'currentAnalysisContext.lowpass_tau_minutes' in body
    assert 'currentAnalysisContext.diff_interval_minutes' in body
    assert 'currentAnalysisContext.detrend_window' in body
    assert 'hasActiveContext ? activeMode : el("preprocessMode").value' in body


def test_confirmed_payload_preserves_formal_chart_parameters(tmp_path):
    _write_run(tmp_path, mode="lowpass_diff", tau=7.0, diff_interval=5.0)
    payload = _confirm_payload(tmp_path, "processed")

    assert payload["analysisContext"] == {
        "preprocess_mode": "lowpass_diff",
        "lowpass_tau_minutes": 7.0,
        "diff_interval_minutes": 5.0,
        "detrend_window": 24,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("lowpass_tau_minutes", ""),
        ("lowpass_tau_minutes", "0"),
        ("lowpass_tau_minutes", "nan"),
        ("diff_interval_minutes", "0"),
        ("diff_interval_minutes", "nan"),
    ],
)
def test_chart_rejects_invalid_positive_transform_parameters(name, value):
    params = {name: [value]}
    if name == "lowpass_tau_minutes":
        with pytest.raises(ValueError, match=name):
            web._positive_query_float(params, name, default=5.0)
    else:
        with pytest.raises(ValueError, match=name):
            web._optional_positive_query_float(params, name)


# --- Test 57: downloads ----------------------------------------------------


def test_download_whitelist_contains_comparison_and_context_but_not_branch_files():
    assert "preprocessing_comparison.csv" in web.DOWNLOAD_FILES
    assert "preprocessing_context.json" in web.DOWNLOAD_FILES
    assert "screening_branches/raw/ranked_features.csv" not in web.DOWNLOAD_FILES
    assert "screening_branches/processed/ranked_features.csv" not in web.DOWNLOAD_FILES


def test_download_rejects_branch_internal_and_traversal_paths(monkeypatch, tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "preprocessing_comparison.csv").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path)

    for forbidden in [
        "screening_branches/raw/ranked_features.csv",
        "screening_branches/processed/ranked_features.csv",
        "../input.csv",
    ]:
        with pytest.raises(ValueError, match="Unsupported download file"):
            web._Handler._send_download(SimpleNamespace(), "run-1", forbidden)


# --- Test 46: LLM/report gate ----------------------------------------------


@pytest.mark.parametrize("endpoint_name", ["_llm_prompt_response", "_llm_report_response"])
def test_llm_outputs_reject_awaiting_branch(tmp_path, monkeypatch, endpoint_name):
    config, _ = _write_run(tmp_path, mode="lowpass")
    monkeypatch.setattr(
        web, "_multipart_form", lambda handler: {"run_id": config.output_dir.name}
    )
    with pytest.raises(ValueError, match="initial_screening_branch_not_confirmed"):
        getattr(web, endpoint_name)(object())


@pytest.mark.parametrize("endpoint_name", ["_llm_prompt_response", "_llm_report_response"])
def test_llm_outputs_lock_confirmed_branch_before_generation(
    tmp_path, monkeypatch, endpoint_name
):
    config, _ = _write_run(tmp_path, mode="lowpass")
    _confirm_payload(tmp_path, "raw")
    lock_path = tmp_path / "screening_downstream.lock"
    ranked_path = tmp_path / "ranked_features.csv"
    ranked_before = ranked_path.read_bytes()
    assert not lock_path.exists()
    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda handler: {"run_id": config.output_dir.name, "api_key": "test-key"},
    )
    if endpoint_name == "_llm_prompt_response":
        original_builder = web.build_llm_analysis_package

        def build_locked_package(*args, **kwargs):
            assert lock_path.exists()
            return original_builder(*args, **kwargs)

        monkeypatch.setattr(web, "build_llm_analysis_package", build_locked_package)
    else:
        def generate_locked_report(*args, **kwargs):
            assert lock_path.exists()
            return {"report": "report", "prompt": "prompt"}

        monkeypatch.setattr(
            web,
            "generate_llm_report",
            generate_locked_report,
        )

    getattr(web, endpoint_name)(object())

    assert lock_path.exists()
    assert ranked_path.read_bytes() == ranked_before
    with pytest.raises(ValueError, match="initial_screening_branch_locked"):
        confirm_initial_screening_branch(tmp_path, branch="processed")


# --- Test 54: UI button gate (static) --------------------------------------


def test_ui_downstream_button_gate_static_contract():
    source = web.INDEX_HTML
    assert "function setDownstreamGate" in source
    assert "downstreamGateHint" in source
    assert "awaiting_confirmation" in source
    assert "请先确认正式初筛分支" in source
    assert "branchLocked" in source
    assert "后续验证已开始，当前初筛分支已锁定；如需切换请重新分析。" in source


# --- Test 60/61: raw and processed workflow regressions --------------------


def test_raw_workflow_does_not_create_comparison_or_processed_branch(tmp_path):
    config, _ = _write_run(tmp_path, mode="raw")

    assert not (tmp_path / "preprocessing_comparison.csv").exists()
    assert not (tmp_path / "screening_branches" / "processed").exists()
    assert (tmp_path / "screening_branches" / "raw" / "ranked_features.csv").exists()
    assert (tmp_path / "ranked_features.csv").exists()


@pytest.mark.parametrize("mode", PROCESSED_MODES)
def test_processed_workflow_runs_both_branches_and_waits(tmp_path, mode: str):
    config, result = _write_run(tmp_path, mode=mode)

    assert (tmp_path / "screening_branches" / "raw" / "ranked_features.csv").exists()
    assert (
        tmp_path / "screening_branches" / "processed" / "ranked_features.csv"
    ).exists()
    assert (tmp_path / "preprocessing_comparison.csv").exists()
    assert not (tmp_path / "ranked_features.csv").exists()
    context = _read_context(tmp_path)
    assert context["branch_selection_status"] == "awaiting_confirmation"
    assert context["active_screening_branch"] is None
