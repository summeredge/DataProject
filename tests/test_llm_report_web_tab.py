from chem_ts_corr.web import DOWNLOAD_FILES, INDEX_HTML


def test_ai_interpretation_tab_exists_with_prompt_actions():
    assert "AI 综合解读" in INDEX_HTML
    assert "生成 Prompt" in INDEX_HTML
    assert "复制 Prompt" not in INDEX_HTML
    assert "下载 llm_prompt.md" not in INDEX_HTML
    assert "llmPrompt" in INDEX_HTML
    assert "/api/llm_prompt" in INDEX_HTML


def test_llm_prompt_download_is_registered():
    assert "llm_prompt.md" in DOWNLOAD_FILES
