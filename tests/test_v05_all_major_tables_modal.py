from chem_ts_corr.web import INDEX_HTML


def test_common_compact_detail_table_renderer_exists():
    required = [
        "renderCompactDetailTable",
        "buildDetailModalBody",
        "openDetailModal",
        "detailModal",
        "clickable-row",
        "展开完整原始字段",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_candidate_table_uses_compact_modal_detail_view():
    required = [
        "candidateCoreColumns",
        "candidateDetailColumns",
        "renderCandidateTable",
        "candidateTable",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_final_and_evidence_tables_use_compact_modal_detail_view():
    required = [
        "FINAL_SUMMARY_CORE_COLUMNS",
        "FINAL_SUMMARY_DETAIL_COLUMNS",
        "causalEvidenceCoreColumns",
        "causalEvidenceDetailColumns",
        "finalReviewSummaryTable",
        "causalReviewEvidenceTable",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_major_result_tables_do_not_default_to_full_wide_columns():
    forbidden = [
        "finalReviewDetailPanel",
        "singleVariableReviewCard",
        "完整字段：",
    ]
    for marker in forbidden:
        assert marker not in INDEX_HTML


def test_row_click_detail_tables_do_not_render_duplicate_detail_actions():
    compact = INDEX_HTML.split("function renderCompactDetailTable", 1)[1].split(
        "function shouldOpenRowDetail", 1
    )[0]
    final_summary = INDEX_HTML.split("function renderFinalReviewSummaryTable", 1)[1].split(
        "function selectFinalReviewRow", 1
    )[0]
    row_guard = INDEX_HTML.split("function shouldOpenRowDetail", 1)[1].split(
        "function selectCompactDetailRow", 1
    )[0]

    assert "查看详情" not in INDEX_HTML
    assert "detail_action" not in INDEX_HTML
    assert "<th scope=\"col\">${escapeHtml(columnLabel(\"detail_action\"))}</th>" not in compact
    assert "button.textContent = \"查看详情\"" not in compact
    assert "button.textContent = \"查看详情\"" not in final_summary
    assert "shouldOpenRowDetail(event)" in compact
    assert "button, a, input, select, textarea, label" in row_guard
    assert "window.getSelection" in row_guard
