from pathlib import Path

import numpy as np
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


def test_secondary_variables_include_topk_forced_and_extra():
    ranked = pd.DataFrame(
        {
            "variable": ["A", "B", "C", "D"],
            "force_included": [False, True, False, False],
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

    assert variables == ["A", "B", "X", "Y"]


def test_secondary_variables_include_config_forced_when_no_force_column():
    ranked = pd.DataFrame({"variable": ["A", "B", "C"]})
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


def test_index_html_contains_secondary_validation_options():
    for token in [
        "secondaryIncludeDropdown",
        "secondaryIncludeOptions",
        "secondaryIncludeSummary",
        "secondaryResampleMode",
        "secondaryResampleRule",
        "secondaryMaxLag",
        "二次验证补充变量",
        "二次验证重采样",
        "二次验证最大滞后点数",
        "原始数据（不重采样）",
        "继承主筛查",
        "自定义",
    ]:
        assert token in INDEX_HTML
    assert "仅白名单" not in INDEX_HTML
    assert "whitelist_only" not in INDEX_HTML


def test_secondary_validation_buttons_append_options():
    assert "function appendSecondaryValidationOptions" in INDEX_HTML
    assert INDEX_HTML.count("appendSecondaryValidationOptions(form)") >= 3
    for field in [
        "secondary_include_variables",
        "secondary_resample_mode",
        "secondary_resample_rule",
        "secondary_max_lag",
    ]:
        assert field in INDEX_HTML
    for function_name in ["runEnhancedScreening", "runGranger", "runModel"]:
        function_start = INDEX_HTML.index(f"async function {function_name}()")
        function_end = INDEX_HTML.index("async function", function_start + 1) if "async function" in INDEX_HTML[function_start + 1 :] else len(INDEX_HTML)
        assert "appendSecondaryValidationOptions(form)" in INDEX_HTML[function_start:function_end]


def test_backend_secondary_endpoints_use_secondary_form_helpers():
    web_source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")
    assert 'read_text(encoding="utf-8")' in Path("tests/test_secondary_validation_options.py").read_text(encoding="utf-8")
    for function_name in [
        "_run_enhanced_screening_response",
        "_run_granger_response",
        "_run_model_response",
    ]:
        function_start = web_source.index(f"def {function_name}")
        function_end = web_source.index("\ndef ", function_start + 1)
        function_body = web_source[function_start:function_end]
        assert "_secondary_config_from_form" in function_body
        assert "_secondary_extra_variables_from_form" in function_body
        assert "secondary_config.max_lag" in function_body
        assert "_scaled_frame_for_secondary(secondary_config, protected_columns=extra_variables)" in function_body


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
    import pandas as pd

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


def test_index_html_reset_restores_secondary_validation_defaults():
    function_start = INDEX_HTML.index("function reset()")
    reset_body = INDEX_HTML[function_start:]

    assert 'el("secondaryResampleMode").value = "raw"' in reset_body
    assert 'el("secondaryResampleRule").value = ""' in reset_body
    assert 'el("secondaryMaxLag").value = ""' in reset_body


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


def test_run_granger_response_clamps_zero_max_lag_for_granger_api():
    web_source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")
    function_start = web_source.index("def _run_granger_response")
    function_end = web_source.index("\ndef ", function_start + 1)
    function_body = web_source[function_start:function_end]

    assert "maxlag=max(1, secondary_config.max_lag)" in function_body


def test_secondary_best_lags_limits_bulk_missing_lag_scan_without_skipping_all():
    import numpy as np
    import pandas as pd

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
    import pandas as pd

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


def test_enhanced_screening_recomputes_best_lags_when_secondary_resample_changes():
    web_source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")
    function_start = web_source.index("def _run_enhanced_screening_response")
    function_end = web_source.index("\ndef ", function_start + 1)
    function_body = web_source[function_start:function_end]

    assert "lag_search_changed = _secondary_lag_search_changed(base_config, secondary_config)" in function_body
    assert "prepare_best_lag_evidence(" in function_body
    assert "ranked_source_scaled = _scaled_frame_for_secondary(base_config)" in function_body
    assert "ranked_source_frame=ranked_source_scaled" in function_body
    assert "allow_ranked_reuse=not lag_search_changed" in function_body
    assert "best_lag_evidence=best_lag_evidence" in function_body


def _run_enhanced_screening_case(tmp_path, monkeypatch, lag_search_changed):
    from chem_ts_corr import screening, web
    from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags

    n = 120
    rng = np.random.default_rng(17)
    target = pd.Series(rng.normal(size=n))
    variable = target.shift(-2).ffill().bfill() + rng.normal(scale=1e-4, size=n)
    frame = pd.DataFrame(
        {"target": target.to_numpy(), "x": variable.to_numpy()},
        index=pd.date_range("2026-01-01", periods=n, freq="min"),
    )
    base_config = AnalysisConfig(
        input_path=tmp_path / "unused.csv",
        time_column="timestamp",
        target="target",
        output_dir=tmp_path,
        max_lag=3,
        top_k=1,
    )
    secondary_config = AnalysisConfig(
        input_path=tmp_path / "unused.csv",
        time_column="timestamp",
        target="target",
        output_dir=tmp_path,
        max_lag=4 if lag_search_changed else 3,
        top_k=1,
    )
    ranked_best = summarize_best_lags(compute_lag_scores(frame, "target", 3)).iloc[0]
    ranked = pd.DataFrame(
        {
            "variable": ["x"],
            "lag": [int(ranked_best["lag"])],
            "raw_corr": [float(ranked_best["score"])],
        }
    )

    monkeypatch.setattr(web, "_multipart_form", lambda handler: {})
    monkeypatch.setattr(web, "_field", lambda form, name: "run-1")
    monkeypatch.setattr(web, "_resolve_run_dir", lambda run_id: tmp_path)
    monkeypatch.setattr(web, "_read_run_config", lambda output_dir: base_config)
    monkeypatch.setattr(web, "_secondary_config_from_form", lambda config, form: secondary_config)
    monkeypatch.setattr(web, "_secondary_extra_variables_from_form", lambda form: [])
    monkeypatch.setattr(web, "_safe_read_result_csv", lambda path: ranked)
    monkeypatch.setattr(web, "_secondary_variables_from_ranked", lambda *args, **kwargs: ["x"])
    scaled_calls = []

    def scaled_for_config(config, protected_columns=None):
        scaled_calls.append((config, tuple(protected_columns or [])))
        return frame

    monkeypatch.setattr(web, "_scaled_frame_for_secondary", scaled_for_config)
    monkeypatch.setattr(web, "_download_links", lambda *args, **kwargs: {})

    original_compute = screening.compute_lag_scores
    scan_calls = []

    def counted_compute(pair, target_name, max_lag):
        scan_calls.append(pair.columns[-1])
        return original_compute(pair, target_name, max_lag)

    captured = {}
    original_lift = screening.model_lift_scores
    original_rolling = screening.rolling_corr_scores

    def capture_lift(*args, best_lags=None, **kwargs):
        captured["best_lags"] = best_lags
        return original_lift(*args, best_lags=best_lags, **kwargs)

    def capture_rolling(*args, best_lag_evidence=None, **kwargs):
        captured["evidence"] = best_lag_evidence
        captured["rolling"] = original_rolling(
            *args,
            best_lag_evidence=best_lag_evidence,
            **kwargs,
        )
        return captured["rolling"]

    monkeypatch.setattr(screening, "compute_lag_scores", counted_compute)
    monkeypatch.setattr(screening, "model_lift_scores", capture_lift)
    monkeypatch.setattr(screening, "rolling_corr_scores", capture_rolling)

    result = web._run_enhanced_screening_response(object())
    captured["scaled_calls"] = scaled_calls
    captured["base_config"] = base_config
    captured["secondary_config"] = secondary_config
    return result, captured, scan_calls


def test_enhanced_screening_reuses_ranked_evidence_and_reports_timings(tmp_path, monkeypatch):
    result, captured, scan_calls = _run_enhanced_screening_case(
        tmp_path, monkeypatch, lag_search_changed=False
    )

    assert scan_calls == []
    assert captured["evidence"]["x"]["source"] == "ranked"
    assert captured["scaled_calls"] == [
        (captured["secondary_config"], ()),
        (captured["base_config"], ()),
    ]
    assert captured["best_lags"] == {
        variable: item["best_lag"] for variable, item in captured["evidence"].items()
    }
    assert set(result["timings"]) == {
        "lag_evidence_seconds",
        "model_lift_seconds",
        "rolling_seconds",
        "output_seconds",
        "total_seconds",
    }
    assert all(value >= 0 for value in result["timings"].values())


def test_enhanced_screening_changed_parameters_scan_each_variable_once(tmp_path, monkeypatch):
    result, captured, scan_calls = _run_enhanced_screening_case(
        tmp_path, monkeypatch, lag_search_changed=True
    )

    assert scan_calls == ["x"]
    assert captured["evidence"]["x"]["source"] == "recomputed"
    assert captured["best_lags"]["x"] == captured["evidence"]["x"]["best_lag"]
    assert captured["scaled_calls"] == [(captured["secondary_config"], ())]
    assert result["timings"]["total_seconds"] >= max(result["timings"].values())


def test_enhanced_screening_extra_variable_index_change_recomputes_ranked_candidate(
    tmp_path, monkeypatch
):
    from chem_ts_corr import screening, web
    from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags

    n = 120
    rng = np.random.default_rng(23)
    target = pd.Series(rng.normal(size=n))
    x = target.shift(-2).ffill().bfill() + rng.normal(scale=1e-4, size=n)
    z = target.shift(1).ffill().bfill()
    full_index = pd.date_range("2026-02-01", periods=n, freq="min")
    raw = pd.DataFrame(
        {
            "timestamp": full_index,
            "target": target.to_numpy(),
            "x": x.to_numpy(),
            "z": z.to_numpy(),
        }
    )
    raw.loc[20:89, "z"] = np.nan
    input_path = tmp_path / "input.csv"
    raw.to_csv(input_path, index=False, encoding="utf-8-sig")
    base_config = AnalysisConfig(
        input_path=input_path,
        time_column="timestamp",
        target="target",
        output_dir=tmp_path,
        max_lag=3,
        top_k=1,
    )
    secondary_config = AnalysisConfig(
        input_path=input_path,
        time_column="timestamp",
        target="target",
        output_dir=tmp_path,
        max_lag=3,
        top_k=1,
        force_include_variables=["z"],
    )
    monkeypatch.setattr(web, "SCALED_FRAME_CACHE", {})
    source_frame = web._scaled_frame_for_secondary(base_config)
    current_frame = web._scaled_frame_for_secondary(
        secondary_config, protected_columns=["z"]
    )
    assert "z" not in source_frame.columns
    assert "z" in current_frame.columns
    assert len(current_frame) < len(source_frame)
    assert current_frame[["target", "x"]].notna().all().all()
    expected = screening.rolling_corr_scores(current_frame, "target", ["x"], 3)
    ranked_best = summarize_best_lags(compute_lag_scores(source_frame, "target", 3)).iloc[0]
    ranked = pd.DataFrame(
        {
            "variable": ["x"],
            "lag": [int(ranked_best["lag"])],
            "raw_corr": [float(ranked_best["score"])],
        }
    )
    web.SCALED_FRAME_CACHE.clear()

    monkeypatch.setattr(web, "_multipart_form", lambda handler: {})
    monkeypatch.setattr(web, "_field", lambda form, name: "run-1")
    monkeypatch.setattr(web, "_resolve_run_dir", lambda run_id: tmp_path)
    monkeypatch.setattr(web, "_read_run_config", lambda output_dir: base_config)
    monkeypatch.setattr(web, "_secondary_config_from_form", lambda config, form: secondary_config)
    monkeypatch.setattr(web, "_secondary_extra_variables_from_form", lambda form: ["z"])
    monkeypatch.setattr(web, "_safe_read_result_csv", lambda path: ranked)
    monkeypatch.setattr(web, "_download_links", lambda *args, **kwargs: {})

    scaled_calls = []
    original_scaled = web._scaled_frame_for_secondary

    def capture_scaled(config, protected_columns=None):
        scaled_calls.append((config, tuple(protected_columns or [])))
        return original_scaled(config, protected_columns)

    monkeypatch.setattr(web, "_scaled_frame_for_secondary", capture_scaled)
    original_compute = screening.compute_lag_scores
    scan_calls = []

    def counted_compute(pair, target_name, max_lag):
        scan_calls.append(pair.columns[-1])
        return original_compute(pair, target_name, max_lag)

    captured = {}
    original_rolling = screening.rolling_corr_scores

    def capture_rolling(*args, best_lag_evidence=None, **kwargs):
        captured["evidence"] = best_lag_evidence
        captured["rolling"] = original_rolling(
            *args,
            best_lag_evidence=best_lag_evidence,
            **kwargs,
        )
        return captured["rolling"]

    monkeypatch.setattr(screening, "compute_lag_scores", counted_compute)
    monkeypatch.setattr(screening, "rolling_corr_scores", capture_rolling)

    web._run_enhanced_screening_response(object())

    assert not _secondary_lag_search_changed(base_config, secondary_config)
    assert scan_calls.count("x") == 1
    assert captured["evidence"]["x"]["source"] == "recomputed"
    assert scaled_calls == [
        (secondary_config, ("z",)),
        (base_config, ()),
    ]
    actual = captured["rolling"].loc[
        captured["rolling"]["variable"].eq("x")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_model_response_recomputes_best_lags_when_secondary_resample_changes():
    web_source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")
    function_start = web_source.index("def _run_model_response")
    function_end = web_source.index("\ndef ", function_start + 1)
    function_body = web_source[function_start:function_end]

    assert "lag_search_changed = _secondary_lag_search_changed(base_config, secondary_config)" in function_body
    assert "if lag_search_changed:" in function_body
    assert "best_lags = {}" in function_body
    assert "else:" in function_body
    assert "best_lags = _best_lags_from_ranked(ranked)" in function_body
    assert "best_lags = _merge_near_miss_lags(best_lags, near_miss)" in function_body
    assert "recompute_limit=None if lag_search_changed else 20" in function_body


def test_web_source_does_not_keep_old_lag_scale_changed_helper():
    web_source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")

    assert "_secondary_lag_scale_changed" not in web_source


def test_secondary_max_lag_does_not_fallback_to_primary_max_lag_in_frontend():
    function_start = INDEX_HTML.index("function appendSecondaryValidationOptions")
    function_end = INDEX_HTML.index("async function", function_start)
    function_body = INDEX_HTML[function_start:function_end]

    assert 'form.append("secondary_max_lag", el("secondaryMaxLag").value);' in function_body
    assert 'secondaryMaxLag").value || el("maxLag").value' not in function_body


def test_index_html_defines_secondary_validation_grid_styles():
    assert ".secondary-validation-params, .causal-review-params" in INDEX_HTML
    assert ".grid {" in INDEX_HTML
    assert "grid-template-columns:repeat(2, minmax(160px, 1fr))" in INDEX_HTML
