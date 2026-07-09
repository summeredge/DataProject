from pathlib import Path

import pandas as pd

from chem_ts_corr.web import (
    INDEX_HTML,
    AnalysisConfig,
    _enhanced_validation_summary,
    _secondary_best_lags_for_missing_variables,
    _secondary_config_from_form,
    _secondary_lag_scale_changed,
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


def test_secondary_lag_scale_changed_normalizes_resample_rule():
    base = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule="5min",
    )
    same = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule=" 5MIN ",
    )
    raw = AnalysisConfig(
        input_path=Path("dummy.csv"),
        time_column="time",
        target="Y",
        output_dir=Path("out"),
        resample_rule=None,
    )

    assert not _secondary_lag_scale_changed(base, same)
    assert _secondary_lag_scale_changed(base, raw)


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
            "secondary_resample_rule": "2min",
            "secondary_max_lag": "180",
        },
    )

    assert secondary.resample_rule == "2min"
    assert secondary.max_lag == 180


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

    assert "lag_scale_changed = _secondary_lag_scale_changed(base_config, secondary_config)" in function_body
    assert "best_lags = {} if lag_scale_changed else _best_lags_from_ranked(ranked)" in function_body
    assert "recompute_limit=None if lag_scale_changed else 20" in function_body


def test_model_response_recomputes_best_lags_when_secondary_resample_changes():
    web_source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")
    function_start = web_source.index("def _run_model_response")
    function_end = web_source.index("\ndef ", function_start + 1)
    function_body = web_source[function_start:function_end]

    assert "lag_scale_changed = _secondary_lag_scale_changed(base_config, secondary_config)" in function_body
    assert "if lag_scale_changed:" in function_body
    assert "best_lags = {}" in function_body
    assert "else:" in function_body
    assert "best_lags = _best_lags_from_ranked(ranked)" in function_body
    assert "best_lags = _merge_near_miss_lags(best_lags, near_miss)" in function_body
    assert "recompute_limit=None if lag_scale_changed else 20" in function_body


def test_secondary_max_lag_does_not_fallback_to_primary_max_lag_in_frontend():
    function_start = INDEX_HTML.index("function appendSecondaryValidationOptions")
    function_end = INDEX_HTML.index("async function", function_start)
    function_body = INDEX_HTML[function_start:function_end]

    assert 'form.append("secondary_max_lag", el("secondaryMaxLag").value);' in function_body
    assert 'secondaryMaxLag").value || el("maxLag").value' not in function_body


def test_index_html_defines_secondary_validation_card_grid_styles():
    assert ".card {" in INDEX_HTML
    assert ".grid {" in INDEX_HTML
    assert "grid-template-columns:repeat(2, minmax(160px, 1fr))" in INDEX_HTML
