from chem_ts_corr.web import INDEX_HTML


def test_llm_page_has_api_connection_controls():
    assert "testLlmConnection" in INDEX_HTML
    assert "测试 API 连接" in INDEX_HTML or "连接 API" in INDEX_HTML
    assert "llmConnectionStatus" in INDEX_HTML
    assert "/api/llm_connection" in INDEX_HTML


def test_llm_page_removes_prompt_copy_and_download_controls():
    assert "copyLlmPrompt" not in INDEX_HTML
    assert "llmPromptDownload" not in INDEX_HTML
    assert "复制 Prompt" not in INDEX_HTML
    assert "下载 llm_prompt.md" not in INDEX_HTML


def test_llm_config_grid_uses_two_rows_four_columns_marker():
    assert "llm-config-grid" in INDEX_HTML
    assert "grid-template-columns: repeat(4" in INDEX_HTML or "repeat(4, 1fr)" in INDEX_HTML


def test_prompt_generation_is_not_exposed_in_simplified_ui():
    assert "generateLlmPrompt" not in INDEX_HTML
    assert "llmPrompt" not in INDEX_HTML
    assert "renderDownloadTarget(\"llmPromptDownload\"" not in INDEX_HTML
