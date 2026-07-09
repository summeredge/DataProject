from pathlib import Path

import pandas as pd

from chem_ts_corr.web import (
    INDEX_HTML,
    AnalysisConfig,
    _secondary_config_from_form,
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
    web_source = Path("chem_ts_corr/web.py").read_text()
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
        assert "_scaled_frame_for_secondary(secondary_config)" in function_body
