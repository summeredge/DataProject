from chem_ts_corr.web import INDEX_HTML


def test_table_ui_has_compact_summary_and_detail_panel_hooks():
    required = [
        "compact-result-table",
        "detail-panel",
        "renderRowDetails",
        "selectTableRow",
        "核心列",
        "详情",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_table_ui_prefers_page_width_and_wrapping():
    required = [
        "table-layout:fixed",
        "overflow-wrap:anywhere",
        "word-break:break-word",
        "white-space:normal",
    ]
    for marker in required:
        assert marker in INDEX_HTML.replace(" ", "")


def test_common_evidence_and_risk_labels_are_chinese_ready():
    required = [
        "多证据支持",
        "建议优先复核",
        "条件 Granger 显示存在独立预测贡献证据",
        "预测贡献为正",
        "格兰杰辅助支持",
        "模型提升弱支持",
        "模型解释支持",
        "候选等级D",
        "滞后触及边界",
        "变量滞后目标",
        "跨工况不稳定",
        "数据质量需关注",
        "数据质量严重不足",
    ]
    for marker in required:
        assert marker in INDEX_HTML
