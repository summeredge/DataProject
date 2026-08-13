from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr.web import (
    INDEX_HTML,
    AnalysisConfig,
    _enhanced_validation_summary,
    _secondary_best_lags_for_missing_variables,
    _secondary_config_from_form,
    _secondary_lag_search_changed,
    _secondary_variables_from_ranked,
)


def test_secondary_variables_take_top_k_then_forced_then_extra():
    ranked = pd.DataFrame(
        {
            "variable": ["A", "B", "C", "D"],
            "final_score": [0.1, 0.9, 0.5, 0.3],
        }
    )
    config = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        top_k=2,
        force_include_variables=["Z"],
    )

    variables = _secondary_variables_from_ranked(
        ranked,
        config,
        extra_variables=["X", "B", "Y"],
    )

    assert variables == ["B", "C", "Z", "X", "Y"]


def test_secondary_variables_append_config_forced_outside_top_k():
    ranked = pd.DataFrame(
        {
            "variable": ["A", "B", "C"],
            "final_score": [0.3, 0.2, 0.1],
        }
    )
    config = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        top_k=1,
        force_include_variables=["C"],
    )

    variables = _secondary_variables_from_ranked(
        ranked,
        config,
        extra_variables=["D", "A"],
    )

    assert variables == ["A", "C", "D"]


def test_secondary_variables_order_by_final_score_descending():
    ranked = pd.DataFrame(
        {
            "variable": ["low", "high", "mid"],
            "final_score": [0.2, 0.9, 0.5],
        }
    )
    config = AnalysisConfig(Path("dummy.csv"), "time", "Y", Path("out"))

    variables = _secondary_variables_from_ranked(ranked, config)

    assert variables == ["high", "mid", "low"]


def _secondary_ranked_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable": [f"v{i}" for i in range(1, 11)],
            "final_score": [0.60, 0.45, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22],
            "lag": [1] * 10,
        }
    )


def _secondary_endpoint_config(tmp_path: Path) -> AnalysisConfig:
    return AnalysisConfig(
        input_path=tmp_path / "input.csv",
        time_column="time",
        target="target",
        output_dir=tmp_path,
        max_lag=2,
        top_k=5,
        force_include_variables=["v8"],
        max_model_features=123,
    )


def _secondary_scaled_frame(columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or [f"v{i}" for i in range(1, 11)]
    return pd.DataFrame(
        {
            "target": pd.Series(range(60), dtype=float),
            **{name: pd.Series(range(60), dtype=float) + i for i, name in enumerate(columns, start=1)},
        }
    )


def test_secondary_variables_top_k_ignores_min_score_threshold():
    ranked = _secondary_ranked_frame()
    config = AnalysisConfig(Path("dummy.csv"), "time", "target", Path("out"), top_k=5)

    variables = _secondary_variables_from_ranked(ranked, config)

    assert variables == ["v1", "v2", "v3", "v4", "v5"]
    assert float(ranked.loc[2, "final_score"]) < 0.30
    assert float(ranked.loc[4, "final_score"]) < 0.30


def test_secondary_variables_whitelist_outside_top_k_appended_without_replacing_top_k():
    ranked = _secondary_ranked_frame()
    config = AnalysisConfig(
        Path("dummy.csv"),
        "time",
        "target",
        Path("out"),
        top_k=5,
        force_include_variables=["v8"],
    )

    variables = _secondary_variables_from_ranked(ranked, config)

    assert variables == ["v1", "v2", "v3", "v4", "v5", "v8"]
    assert len(variables) == 6
    assert variables[:5] == ["v1", "v2", "v3", "v4", "v5"]


def test_secondary_variables_extra_outside_top_k_appended_and_deduplicated():
    ranked = _secondary_ranked_frame()
    config = AnalysisConfig(
        Path("dummy.csv"),
        "time",
        "target",
        Path("out"),
        top_k=5,
        force_include_variables=["v8"],
    )

    variables = _secondary_variables_from_ranked(
        ranked, config, extra_variables=["v9", "v1", "v8", "v9"]
    )

    assert variables == ["v1", "v2", "v3", "v4", "v5", "v8", "v9"]
    assert variables.count("v8") == 1
    assert variables.count("v9") == 1


# --- PR-13: formal Web endpoints delegate to active-branch runners ---------


def _endpoint_bodies() -> dict[str, str]:
    web_source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")
    bodies: dict[str, str] = {}
    for function_name in [
        "_run_enhanced_screening_response",
        "_run_granger_response",
        "_run_model_response",
        "_run_causal_review_response",
        "_run_xgb_validation_response",
    ]:
        function_start = web_source.index(f"def {function_name}")
        function_end = web_source.index("\ndef ", function_start + 1)
        bodies[function_name] = web_source[function_start:function_end]
    return bodies


def test_formal_web_endpoints_call_active_branch_runners_only():
    bodies = _endpoint_bodies()

    assert "run_enhanced_screening_for_active_branch(" in bodies["_run_enhanced_screening_response"]
    assert "run_granger_for_active_branch(" in bodies["_run_granger_response"]
    assert "run_model_for_active_branch(" in bodies["_run_model_response"]
    assert "run_causal_review_for_active_branch(" in bodies["_run_causal_review_response"]
    assert "run_xgb_for_active_branch(" in bodies["_run_xgb_validation_response"]


def test_formal_web_endpoints_do_not_reimplement_old_orchestration():
    bodies = _endpoint_bodies()
    forbidden = [
        "run_analysis(",
        "run_xgb_analysis(",
        "run_causal_review_stage(",
        "fit_explainable_model(",
        "run_granger_tests(",
        "_build_causal_review_candidate_table(",
        "_load_secondary_candidate_context(",
        "_prepared_frame_for_validation(",
        "_secondary_config_from_form(",
        "_secondary_extra_variables_from_form(",
    ]
    for body in bodies.values():
        for token in forbidden:
            assert token not in body, f"{token} must not appear in a formal endpoint body"


def test_enhanced_screening_endpoint_delegates_and_reads_formal_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from chem_ts_corr import web

    config = _secondary_endpoint_config(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(web, "_multipart_form", lambda handler: {"run_id": "run-1"})
    monkeypatch.setattr(web, "_resolve_run_dir", lambda run_id: tmp_path)
    monkeypatch.setattr(web, "_read_run_config", lambda output_dir: config)

    def fake_runner(output_dir, base_config=None):
        captured["output_dir"] = output_dir
        captured["base_config"] = base_config
        pd.DataFrame([{"variable": "v1", "model_lift": 0.1}]).to_csv(
            output_dir / "model_lift_scores.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([{"variable": "v1", "rolling_stability": 0.7}]).to_csv(
            output_dir / "rolling_corr_scores.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([{"variable": "v1", "interpretation": "enhanced screening only"}]).to_csv(
            output_dir / "enhanced_validation_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return {"run_dir": output_dir}

    monkeypatch.setattr(web, "run_enhanced_screening_for_active_branch", fake_runner)
    monkeypatch.setattr(web, "_download_links", lambda *args, **kwargs: [])

    result = web._run_enhanced_screening_response(object())

    assert captured["output_dir"] == tmp_path
    assert captured["base_config"] is config
    assert result["enhancedValidationSummary"][0]["variable"] == "v1"
    assert result["modelLiftScores"][0]["model_lift"] == 0.1
    assert result["rollingCorrScores"][0]["rolling_stability"] == 0.7
    assert "timings" in result


def test_granger_endpoint_delegates_and_reads_formal_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from chem_ts_corr import web

    config = _secondary_endpoint_config(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(web, "_multipart_form", lambda handler: {"run_id": "run-1"})
    monkeypatch.setattr(web, "_resolve_run_dir", lambda run_id: tmp_path)
    monkeypatch.setattr(web, "_read_run_config", lambda output_dir: config)

    def fake_runner(output_dir, base_config=None):
        captured["output_dir"] = output_dir
        captured["base_config"] = base_config
        pd.DataFrame([{"variable": "v1", "best_granger_lag": 3}]).to_csv(
            output_dir / "granger_tests.csv", index=False, encoding="utf-8-sig"
        )
        return {"run_dir": output_dir}

    monkeypatch.setattr(web, "run_granger_for_active_branch", fake_runner)
    monkeypatch.setattr(web, "_download_links", lambda *args, **kwargs: [])

    result = web._run_granger_response(object())

    assert captured["output_dir"] == tmp_path
    assert captured["base_config"] is config
    assert result["grangerTests"][0]["best_granger_lag"] == 3


def test_model_endpoint_delegates_and_reads_formal_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from chem_ts_corr import web

    config = _secondary_endpoint_config(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(web, "_multipart_form", lambda handler: {"run_id": "run-1"})
    monkeypatch.setattr(web, "_resolve_run_dir", lambda run_id: tmp_path)
    monkeypatch.setattr(web, "_read_run_config", lambda output_dir: config)

    def fake_runner(output_dir, base_config=None):
        captured["output_dir"] = output_dir
        captured["base_config"] = base_config
        pd.DataFrame([{"variable": "v1", "importance": 0.3}]).to_csv(
            output_dir / "shap_or_importance.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([{"variable": "v1", "importance_rank": 1}]).to_csv(
            output_dir / "model_variable_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([{"variable": "v9", "discovery": True}]).to_csv(
            output_dir / "model_discovered_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return {"run_dir": output_dir, "model_metrics": {"r2": 0.5}}

    monkeypatch.setattr(web, "run_model_for_active_branch", fake_runner)
    monkeypatch.setattr(web, "_download_links", lambda *args, **kwargs: [])

    result = web._run_model_response(object())

    assert captured["output_dir"] == tmp_path
    assert captured["base_config"] is config
    assert result["importance"][0]["importance"] == 0.3
    assert result["modelVariableImportance"][0]["importance_rank"] == 1
    assert result["modelDiscoveredCandidates"][0]["variable"] == "v9"
    assert result["modelMetrics"]["r2"] == 0.5


def test_causal_review_endpoint_delegates_and_reads_four_formal_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from chem_ts_corr import web

    config = AnalysisConfig(
        tmp_path / "input.csv",
        "time",
        "target",
        tmp_path,
        top_k=5,
        max_lag=2,
        residual_control_columns=["c1"],
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda handler: {
            "run_id": "run-1",
            "control_columns": "c1",
            "maxlag": "3",
            "min_rows": "60",
            "top_n": "4",
            "conditional_lag_mode": "full_scan",
        },
    )
    monkeypatch.setattr(web, "_resolve_run_dir", lambda run_id: tmp_path)
    monkeypatch.setattr(web, "_read_run_config", lambda output_dir: config)

    def fake_runner(output_dir, **kwargs):
        captured.update(kwargs)
        pd.DataFrame([{"variable": "v1", "lag": 1}]).to_csv(
            output_dir / "conditional_granger_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([{"variable": "v1", "recommendation": "priority_review"}]).to_csv(
            output_dir / "causal_review_report.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([{"variable": "v1", "evidence": 1}]).to_csv(
            output_dir / "causal_review_evidence.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([{"variable": "v1", "final_rank": 1}]).to_csv(
            output_dir / "final_review_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return {"run_dir": output_dir}

    monkeypatch.setattr(web, "run_causal_review_for_active_branch", fake_runner)
    monkeypatch.setattr(web, "_download_links", lambda *args, **kwargs: [])

    result = web._run_causal_review_response(object())

    assert captured["control_columns"] == ["c1"]
    assert captured["maxlag"] == 3
    assert captured["top_n"] == 4
    assert captured["conditional_lag_mode"] == "full_scan"
    assert result["conditionalGrangerScores"][0]["variable"] == "v1"
    assert result["causalReviewReport"][0]["recommendation"] == "priority_review"
    assert result["causalReviewEvidence"][0]["evidence"] == 1
    assert result["finalReviewSummary"][0]["final_rank"] == 1


def test_causal_review_risk_filter_is_display_only_and_candidate_csv_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from chem_ts_corr import web

    config = AnalysisConfig(
        tmp_path / "input.csv",
        "time",
        "target",
        tmp_path,
        top_k=5,
        max_lag=2,
    )
    risk = pd.DataFrame(
        [
            {"variable": "v1", "risk_flags": "common_capacity_driver", "risk_count": 1},
            {"variable": "v2", "risk_flags": "", "risk_count": 0},
        ]
    )
    risk.to_csv(tmp_path / "risk_flags.csv", index=False, encoding="utf-8-sig")
    candidates = pd.DataFrame([{"variable": "v1"}, {"variable": "v2"}])
    candidates.to_csv(
        tmp_path / "causal_review_candidates.csv", index=False, encoding="utf-8-sig"
    )
    candidates_bytes = (tmp_path / "causal_review_candidates.csv").read_bytes()
    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda handler: {
            "run_id": "run-1",
            "risk_flag_filter": "共同负荷驱动",
        },
    )
    monkeypatch.setattr(web, "_resolve_run_dir", lambda run_id: tmp_path)
    monkeypatch.setattr(web, "_read_run_config", lambda output_dir: config)

    def fake_runner(output_dir, **kwargs):
        pd.DataFrame([{"variable": "v1"}, {"variable": "v2"}]).to_csv(
            output_dir / "final_review_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([{"variable": "v1"}, {"variable": "v2"}]).to_csv(
            output_dir / "causal_review_report.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame().to_csv(
            output_dir / "conditional_granger_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame().to_csv(
            output_dir / "causal_review_evidence.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return {"run_dir": output_dir}

    monkeypatch.setattr(web, "run_causal_review_for_active_branch", fake_runner)
    monkeypatch.setattr(web, "_download_links", lambda *args, **kwargs: [])

    result = web._run_causal_review_response(object())

    assert (tmp_path / "causal_review_candidates.csv").read_bytes() == candidates_bytes
    assert [row["variable"] for row in result["finalReviewSummary"]] == ["v1"]
    assert [row["variable"] for row in result["causalReviewReport"]] == ["v1"]


# --- Legacy helpers are retained for compatibility unit tests --------------


def test_secondary_lag_search_changed_normalizes_resample_rule():
    base = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule="5min",
        max_lag=72,
    )
    same = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule=" 5MIN ",
        max_lag=72,
    )
    raw = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule=None,
        max_lag=72,
    )

    assert not _secondary_lag_search_changed(base, same)
    assert _secondary_lag_search_changed(base, raw)


def test_secondary_lag_search_changed_when_max_lag_changes():
    base = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule=None,
        max_lag=72,
    )
    secondary = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule=None,
        max_lag=360,
    )

    assert _secondary_lag_search_changed(base, secondary)


def test_secondary_config_defaults_to_raw_resample_and_custom_max_lag():
    config = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule="5min",
        max_lag=72,
    )

    secondary = _secondary_config_from_form(config, {"secondary_max_lag": "360"})

    assert secondary.resample_rule is None
    assert secondary.max_lag == 360
    assert config.resample_rule == "5min"
    assert config.max_lag == 72


def test_secondary_config_merges_extra_variables_into_force_include():
    config = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule="5min",
        max_lag=72,
        force_include_variables=["A"],
    )

    secondary = _secondary_config_from_form(
        config,
        {
            "secondary_include_variables": "B,C,A",
            "secondary_max_lag": "360",
        },
    )

    assert secondary.resample_rule is None
    assert secondary.max_lag == 360
    assert secondary.force_include_variables == ["A", "B", "C"]
    assert config.force_include_variables == ["A"]


def test_secondary_config_can_inherit_resample_rule():
    config = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule="5min",
        max_lag=72,
    )

    secondary = _secondary_config_from_form(
        config,
        {"secondary_resample_mode": "inherit", "secondary_max_lag": "90"},
    )

    assert secondary.resample_rule == "5min"
    assert secondary.max_lag == 90


def test_secondary_config_can_use_custom_resample_rule():
    config = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule="5min",
        max_lag=72,
    )

    secondary = _secondary_config_from_form(
        config,
        {
            "secondary_resample_mode": "custom",
            "secondary_resample_rule": "2",
            "secondary_max_lag": "180",
        },
    )

    assert secondary.resample_rule == "2min"
    assert secondary.max_lag == 180


def test_secondary_config_custom_resample_rule_is_required():
    config = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
    )

    with pytest.raises(ValueError, match="^重采样间隔必须是大于 0 的整数分钟$"):
        _secondary_config_from_form(
            config,
            {"secondary_resample_mode": "custom", "secondary_resample_rule": ""},
        )


def test_secondary_config_raw_ignores_custom_resample_value():
    config = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule="5min",
    )

    secondary = _secondary_config_from_form(
        config,
        {"secondary_resample_mode": "raw", "secondary_resample_rule": "2"},
    )

    assert secondary.resample_rule is None


def test_secondary_config_caps_secondary_max_lag_to_ui_limit():
    config = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        max_lag=72,
    )

    secondary = _secondary_config_from_form(config, {"secondary_max_lag": "50000"})

    assert secondary.max_lag == 5000


def test_enhanced_validation_summary_includes_secondary_whitelist_variables():
    ranked = pd.DataFrame(
        {
            "variable": ["A"],
            "final_score": [0.8],
            "lag": [12],
        }
    )
    model_lift = pd.DataFrame(
        {
            "variable": ["A", "WHITELIST_ONLY"],
            "status": ["ok", "ok"],
            "model_lift": [0.1, 0.2],
        }
    )
    rolling = pd.DataFrame(
        {
            "variable": ["WHITELIST_ONLY"],
            "rolling_stability": [0.7],
        }
    )

    summary = _enhanced_validation_summary(ranked, model_lift, rolling)

    assert summary["variable"].tolist() == ["A", "WHITELIST_ONLY"]
    whitelist_row = summary[summary["variable"] == "WHITELIST_ONLY"].iloc[0]
    assert whitelist_row["model_lift"] == 0.2
    assert whitelist_row["rolling_stability"] == 0.7


def test_enhanced_validation_summary_keeps_variables_only_in_secondary_outputs():
    ranked = pd.DataFrame(
        {
            "variable": ["A"],
            "final_score": [0.8],
            "lag": [10],
            "direction": ["positive"],
            "risk_flags": [""],
            "recommended_use": ["strong_screening_candidate"],
        }
    )
    model_lift = pd.DataFrame(
        {
            "variable": ["B"],
            "status": ["ok"],
            "model_lift": [0.12],
            "ar_baseline_rmse": [1.0],
            "candidate_rmse": [0.88],
        }
    )
    rolling = pd.DataFrame(
        {
            "variable": ["B"],
            "rolling_stability": [0.6],
            "rolling_corr_median": [0.5],
            "rolling_sign_consistency": [1.0],
            "valid_window_count": [20],
        }
    )

    summary = _enhanced_validation_summary(ranked, model_lift, rolling)

    assert summary["variable"].tolist() == ["A", "B"]
    row_b = summary.loc[summary["variable"] == "B"].iloc[0]
    assert row_b["status"] == "ok"
    assert row_b["model_lift"] == 0.12
    assert row_b["rolling_stability"] == 0.6
    assert row_b["interpretation"] == "enhanced screening only; not a causal conclusion"


def test_secondary_best_lags_for_missing_variables_adds_lag_for_extra_candidate():
    import numpy as np

    n = 80
    x = np.arange(n, dtype=float)
    y = pd.Series(x).shift(3).bfill()

    frame = pd.DataFrame({"Y": y, "X": x})
    result = _secondary_best_lags_for_missing_variables(
        frame,
        target="Y",
        variables=["X"],
        existing_best_lags={},
        max_lag=8,
    )

    assert "X" in result
    assert isinstance(result["X"], int)


def test_secondary_best_lags_limits_bulk_missing_lag_scan_without_skipping_all():
    import numpy as np

    n = 80
    data = {"Y": np.arange(n, dtype=float)}
    data.update({f"X{i}": np.arange(n, dtype=float) for i in range(21)})
    frame = pd.DataFrame(data)

    result = _secondary_best_lags_for_missing_variables(
        frame,
        target="Y",
        variables=[f"X{i}" for i in range(21)],
        existing_best_lags={},
        max_lag=8,
    )

    assert result
    assert len(result) <= 20


def test_secondary_best_lags_recomputes_all_when_limit_is_none():
    import numpy as np

    n = 80
    data = {"Y": np.arange(n, dtype=float)}
    data.update({f"X{i}": np.arange(n, dtype=float) for i in range(21)})
    frame = pd.DataFrame(data)

    result = _secondary_best_lags_for_missing_variables(
        frame,
        target="Y",
        variables=[f"X{i}" for i in range(21)],
        existing_best_lags={},
        max_lag=8,
        recompute_limit=None,
    )

    assert len(result) == 21
    assert all(isinstance(value, int) for value in result.values())


# --- PR-13: formal Web UI no longer exposes secondary override -------------


def test_index_html_hides_secondary_validation_override_controls():
    for token in [
        "secondaryIncludeDropdown",
        "secondaryIncludeOptions",
        "secondaryIncludeSummary",
        "secondaryResampleMode",
        "secondaryResampleRule",
        "secondaryMaxLag",
        "二次验证重采样",
        "二次验证最大滞后点数",
        "二次验证补充变量",
    ]:
        assert token not in INDEX_HTML
    assert "appendSecondaryValidationOptions" not in INDEX_HTML


def test_index_html_offers_only_the_four_formal_preprocess_modes():
    select = INDEX_HTML.split('<select id="preprocessMode">', 1)[1].split("</select>", 1)[0]
    options = [
        value
        for value in ["raw", "lowpass", "lowpass_detrend", "lowpass_diff"]
        if f'value="{value}"' in select
    ]
    assert options == ["raw", "lowpass", "lowpass_detrend", "lowpass_diff"]
    for legacy in ["detrend", "diff", "detrend_diff"]:
        assert f'value="{legacy}"' not in select


def test_index_html_submits_tau_and_diff_parameters():
    analyze_body = INDEX_HTML.split("async function analyze()", 1)[1].split(
        "async function waitForAnalysisResult", 1
    )[0]
    assert 'form.append("lowpass_tau_minutes", el("lowpassTauMinutes").value);' in analyze_body
    assert 'form.append("diff_interval_minutes", el("diffIntervalMinutes").value.trim());' in analyze_body
    assert 'id="lowpassTauMinutes" type="number" min="0.1" step="0.1" value="5.0"' in INDEX_HTML
    assert 'id="diffIntervalMinutes"' in INDEX_HTML


def test_index_html_has_branch_confirmation_ui_and_downstream_gate():
    for token in [
        "branchSelectionSection",
        "branchSelectionStatus",
        "preprocessingComparisonTable",
        "confirmRawBranch",
        "confirmProcessedBranch",
        "branchLockedHint",
        "downstreamGateHint",
        "请先确认正式初筛分支",
        "后续验证已开始，当前初筛分支已锁定；如需切换请重新分析。",
        "Raw vs Processed 对比",
        "confirmInitialScreeningBranch",
    ]:
        assert token in INDEX_HTML
    assert 'confirmInitialScreeningBranch("raw")' in INDEX_HTML
    assert 'confirmInitialScreeningBranch("processed")' in INDEX_HTML


def test_index_html_reset_restores_formal_preprocess_controls():
    function_start = INDEX_HTML.index("function reset()")
    reset_body = INDEX_HTML[function_start:]

    assert 'el("preprocessMode").value = "raw"' in reset_body
    assert 'el("lowpassTauMinutes").value = "5.0"' in reset_body
    assert 'el("diffIntervalMinutes").value = ""' in reset_body
    assert "updatePreprocessControls();" in reset_body


def test_web_source_does_not_keep_old_lag_scale_changed_helper():
    web_source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")

    assert "_secondary_lag_scale_changed" not in web_source


def test_index_html_defines_secondary_validation_grid_styles():
    assert ".secondary-validation-params, .causal-review-params" in INDEX_HTML
    assert ".grid {" in INDEX_HTML
    assert "grid-template-columns:repeat(2, minmax(160px, 1fr))" in INDEX_HTML
