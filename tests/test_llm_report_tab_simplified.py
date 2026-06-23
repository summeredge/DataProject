from chem_ts_corr.web import INDEX_HTML


def test_llm_tab_hides_prompt_controls_from_ui():
    assert 'id="generateLlmPrompt"' not in INDEX_HTML
    assert 'id="llmPrompt"' not in INDEX_HTML
    assert "生成 Prompt" not in INDEX_HTML
    assert "生成后将在这里显示 Prompt" not in INDEX_HTML


def test_llm_tab_hides_raw_markdown_editor():
    assert "查看 / 编辑 Markdown 原文" not in INDEX_HTML
    assert "Markdown 原文" not in INDEX_HTML
    assert 'id="llmReport"' not in INDEX_HTML
    assert "raw-report" not in INDEX_HTML


def test_llm_tab_keeps_report_generation_and_rendering():
    assert 'id="testLlmConnection"' in INDEX_HTML
    assert 'id="generateLlmReport"' in INDEX_HTML
    assert 'id="copyLlmReport"' in INDEX_HTML
    assert 'id="llmReportRendered"' in INDEX_HTML
    assert 'id="llmReportDownload"' in INDEX_HTML
    assert "/api/llm_report" in INDEX_HTML


def test_prompt_endpoint_is_not_exposed_from_simplified_ui():
    assert "/api/llm_prompt" not in INDEX_HTML
