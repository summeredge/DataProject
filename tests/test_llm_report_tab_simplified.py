from chem_ts_corr.web import INDEX_HTML


def test_report_page_hides_prompt_controls_and_raw_editor():
    """Regression guard for the simplified report page UI."""
    removed_markup = [
        'id="generateLlmPrompt"',
        'id="llmPrompt"',
        'id="llmReport"',
        "Raw Markdown",
        "raw Markdown",
        "原始 Markdown",
        "原始报告",
    ]

    for marker in removed_markup:
        assert marker not in INDEX_HTML


def test_three_layer_review_keeps_formal_result_download_areas():
    """The cleanup must keep the formal result tables and download targets."""
    kept_markup = [
        'id="conditionalGrangerTable"',
        'id="finalReviewSummaryTable"',
        'id="causalReviewEvidenceTable"',
        'id="conditionalDownload"',
        'id="finalReviewSummaryDownload"',
        'id="causalEvidenceDownload"',
        "renderReviewDownloads",
    ]

    for marker in kept_markup:
        assert marker in INDEX_HTML


def test_validation_headers_match_and_legacy_review_ui_is_removed():
    assert '<h2>二次验证</h2>' in INDEX_HTML
    assert '<h2>三层复核</h2>' in INDEX_HTML
    assert ".secondary-validation-params, .causal-review-params" in INDEX_HTML

    removed_markup = [
        "二次验证" + "参数",
        "保守复核" + "报告",
        "旧版保守复核" + "报告",
        'id="causalReviewTable"',
        'id="causalReportDownload"',
    ]
    for marker in removed_markup:
        assert marker not in INDEX_HTML
