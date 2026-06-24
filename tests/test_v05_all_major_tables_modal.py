from chem_ts_corr.web import INDEX_HTML


def test_common_compact_detail_table_renderer_exists():
    required = [
        "renderCompactDetailTable",
        "buildDetailModalBody",
        "openDetailModal",
        "detailModal",
        "查看详情",
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
