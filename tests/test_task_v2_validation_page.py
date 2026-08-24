import json
import shutil
import subprocess

import pytest

from chem_ts_corr.web import INDEX_HTML


_NODE = shutil.which("node")
requires_node = pytest.mark.skipif(_NODE is None, reason="node is required for UI behavior tests")


def _validation_panel() -> str:
    return INDEX_HTML.split('<div id="validationTab"', 1)[1].split(
        '<div id="causalReviewTab"', 1
    )[0]


def test_validation_page_defaults_to_the_unified_summary():
    panel = _validation_panel()
    summary_position = panel.index('id="validationSummarySection"')
    enhanced_position = panel.index('id="enhancedValidationDetails"')
    assert summary_position < enhanced_position
    assert 'id="validationSummaryTable"' in panel
    for column_label in ["验证状态", "证据一致性", "主要支持证据", "限制因素"]:
        assert column_label in panel or column_label in INDEX_HTML
    assert "未执行、未计算或失败的分析会明确标记" in panel


def test_validation_detail_sections_are_collapsed_by_default():
    panel = _validation_panel()
    for detail_id, title in [
        ("enhancedValidationDetails", "Enhanced Validation"),
        ("grangerValidationDetails", "Granger"),
        ("modelExplanationDetails", "Model Explanation"),
    ]:
        start = panel.index(f'<details id="{detail_id}"')
        opening_end = panel.index(">", start) + 1
        opening_tag = panel[start:opening_end]
        assert " open" not in opening_tag
        assert title in panel[start:]


def test_validation_summary_rendering_is_updated_after_initial_and_secondary_results():
    required = [
        "let lastValidationSummaryRows = [];",
        "lastValidationSummaryRows = data.validationSummary || [];",
        "function validationSummaryColumns()",
        "function renderValidationSummaryTable(rows)",
        "validationSummaryTable: [\"variable\", \"validation_status\", \"evidence_consistency\", \"supporting_methods\", \"limiting_factors\"]",
        "lastValidationSummaryRows = data.validationSummary || lastValidationSummaryRows;",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_unified_summary_does_not_add_a_second_ranking_contract():
    columns = INDEX_HTML.split("function validationSummaryColumns()", 1)[1].split("}", 1)[0]
    assert "validation_score" not in columns
    assert "validation_rank" not in columns


def test_validation_limiting_factor_mapping_is_declared_in_the_ui():
    source = INDEX_HTML.split("function validationSummaryStateLabel", 1)[1].split(
        "function displayCellValue", 1
    )[0]
    for backend_state, display_text in [
        ("variable_missing", "变量缺失"),
        ("zero_evidence", "零支持证据"),
        ("computed_no_support", "已计算但未形成支持"),
        ("missing", "证据缺失"),
        ("not_computed", "不可计算"),
        ("skipped", "已跳过"),
        ("failed", "执行失败"),
    ]:
        assert backend_state in source
        assert display_text in source
    display_source = INDEX_HTML.split("function displayCellValue", 1)[1].split(
        "function openTrendForCandidate", 1
    )[0]
    assert "validationSummaryStateLabel(state)" in display_source
    assert "validationSummarySupportingMethods(value)" in display_source


@requires_node
def test_validation_limiting_factor_mapping_returns_chinese_labels():
    function_source = (
        "function validationSummaryStateLabel"
        + INDEX_HTML.split("function validationSummaryStateLabel", 1)[1].split(
            "function displayCellValue", 1
        )[0]
    )
    script = f"""
{function_source}
const values = [
  "variable_missing", "zero_evidence", "computed_no_support", "missing",
  "not_computed", "skipped", "failed", "failed:solver", "unavailable"
];
process.stdout.write(JSON.stringify(values.map(validationSummaryStateLabel)));
"""
    result = subprocess.run(
        [_NODE, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert json.loads(result.stdout) == [
        "变量缺失",
        "零支持证据",
        "已计算但未形成支持",
        "证据缺失",
        "不可计算",
        "已跳过",
        "执行失败",
        "执行失败",
        "不可计算",
    ]


@requires_node
def test_empty_supporting_methods_do_not_render_as_other_validation():
    function_source = (
        "function validationSummarySupportingMethods"
        + INDEX_HTML.split("function validationSummarySupportingMethods", 1)[1].split(
            "function displayCellValue", 1
        )[0]
    )
    script = f"""
{function_source}
const values = [null, "", " ; ", "granger", "unknown_method"];
process.stdout.write(JSON.stringify(values.map(validationSummarySupportingMethods)));
"""
    result = subprocess.run(
        [_NODE, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert json.loads(result.stdout) == [
        "无支持证据",
        "无支持证据",
        "无支持证据",
        "Granger",
        "其他验证",
    ]
