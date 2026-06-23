from chem_ts_corr.web import INDEX_HTML


def test_render_generic_table_uses_compact_modal_path():
    required = [
        "GENERIC_TABLE_CORE_COLUMNS",
        "genericTableCoreColumns",
        "genericTableDetailColumns",
        "renderGenericTable",
        "renderCompactDetailTable",
        "openDetailModal",
        "查看详情",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_remaining_major_table_ids_are_covered_by_generic_modal_config():
    required = [
        "overviewTop",
        "nearMissTable",
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


def test_generic_table_no_longer_renders_full_width_rows_directly():
    forbidden = [
        "body.innerHTML += `<tr>${columns.map",
        "<thead><tr>${columns.map((c) => sortableHeaderHtml(targetId, c)).join(\"\")}</tr></thead>",
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
