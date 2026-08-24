from chem_ts_corr.web import INDEX_HTML


def test_render_generic_table_uses_compact_modal_path():
    required = [
        "GENERIC_TABLE_CORE_COLUMNS",
        "genericTableCoreColumns",
        "genericTableDetailColumns",
        "renderGenericTable",
        "renderCompactDetailTable",
        "openDetailModal",
        "clickable-row",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_remaining_major_table_ids_are_covered_by_generic_modal_config():
    required = [
        "overviewTop",
        "grangerTable",
        "modelVariableImportanceTable",
        "importanceTable",
        "modelDiscoveredTable",
        "enhancedSummaryTable",
        "enhancedLiftTable",
        "enhancedRollingTable",
        "conditionalGrangerTable",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_enhanced_screening_tables_have_metric_explanations_without_generic_table_hint():
    required = [
        "术语说明：模型提升表示加入该候选变量后",
        "滚动稳定性表示固定最佳滞后后",
        "模型提升得分为分段提升中位数",
        "自回归基准 RMSE 是只使用目标变量自身历史值时",
        "候选变量模型 RMSE 是在同一基准上加入该候选变量滞后值后",
        "滚动稳定性为固定最佳滞后后",
    ]

    for marker in required:
        assert marker in INDEX_HTML
    assert "表格按内容宽度展示；超出页面时横向滚动" not in INDEX_HTML


def test_generic_table_no_longer_renders_full_width_rows_directly():
    forbidden = [
        "body.innerHTML += `<tr>${columns.map",
    ]
    for marker in forbidden:
        assert marker not in INDEX_HTML


def test_full_data_remains_available_in_modal_raw_fields():
    required = [
        "展开完整原始字段",
        "renderGenericDetailModalBody",
        "detail-grid",
    ]
    for marker in required:
        assert marker in INDEX_HTML
