from chem_ts_corr.web import INDEX_HTML


def test_final_summary_uses_core_columns_and_modal_review():
    required = [
        "FINAL_SUMMARY_CORE_COLUMNS",
        "detailModal",
        "renderSingleVariableReview",
        "selectFinalReviewRow",
        "查看详情",
            "候选解释",
        "主要原因",
        "建议下一步",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_final_summary_default_view_hides_long_explanation_columns():
    required = [
        "main_reason",
        "suggested_next_action",
        "evidence_conflict_explanation",
        "interpretation_boundary",
    ]
    for marker in required:
        assert marker in INDEX_HTML
    assert "FINAL_SUMMARY_DETAIL_COLUMNS" in INDEX_HTML


def test_display_label_translates_common_evidence_markers():
    required = [
        "translateDisplayValue",
        "候选等级D",
        "条件格兰杰支持",
        "预测贡献为正",
        "格兰杰辅助支持",
        "模型提升弱支持",
        "模型解释支持",
        "滞后触及边界",
        "变量滞后目标",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_core_table_width_and_wrapping_rules_exist():
    compact = INDEX_HTML.replace(" ", "")
    required = [
        "table-layout:fixed",
        "overflow-wrap:anywhere",
        "word-break:break-word",
        "white-space:normal",
    ]
    for marker in required:
        assert marker in compact
