from chem_ts_corr.web import DOWNLOAD_FILES, INDEX_HTML


def test_ai_interpretation_tab_exists_without_prompt_actions():
    assert "AI 综合解读" in INDEX_HTML
    assert "生成 Prompt" not in INDEX_HTML
    assert "复制 Prompt" not in INDEX_HTML
    assert "下载 llm_prompt.md" not in INDEX_HTML
    assert "llmPrompt" not in INDEX_HTML
    assert "/api/llm_prompt" not in INDEX_HTML


def test_llm_prompt_download_is_not_registered():
    assert "llm_prompt.md" not in DOWNLOAD_FILES
