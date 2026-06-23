from chem_ts_corr.web import INDEX_HTML


def test_v05_llm_report_page_removes_prompt_ui_and_raw_editor():
    forbidden = [
        'id="generateLlmPrompt"',
        'id="llmPrompt"',
        'id="llmReport"',
        "生成 Prompt",
        "生成后将在这里显示 Prompt",
        "查看 / 编辑 Markdown 原文",
        "LLM 报告 Markdown 原文",
        "raw-report",
    ]
    for marker in forbidden:
        assert marker not in INDEX_HTML


def test_v05_llm_report_page_keeps_direct_report_controls():
    required = [
        'id="testLlmConnection"',
        'id="llmConnectionStatus"',
        'id="generateLlmReport"',
        'id="copyLlmReport"',
        'id="llmReportRendered"',
        'id="llmReportDownload"',
        "/api/llm_report",
        "/api/llm_connection",
        "function setLlmReport",
        "function renderMarkdownReport",
    ]
    for marker in required:
        assert marker in INDEX_HTML
