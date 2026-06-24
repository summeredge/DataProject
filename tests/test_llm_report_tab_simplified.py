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


def test_report_page_keeps_rendered_report_download_area():
    """The cleanup must keep the rendered report table and report download target."""
    kept_markup = [
        'id="causalReviewTable"',
        'id="causalReportDownload"',
        "renderCausalReviewTable",
        "renderReviewDownloads",
    ]

    for marker in kept_markup:
        assert marker in INDEX_HTML
