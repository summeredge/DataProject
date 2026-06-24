from chem_ts_corr.web import INDEX_HTML


def test_table_cell_class_helper_is_defined_for_table_renderers():
    assert "function tableCellClass" in INDEX_HTML
    assert "td.numeric" in INDEX_HTML
    assert "td.wrap-cell" in INDEX_HTML


def test_llm_markdown_renderer_supports_tables():
    assert "function markdownTableToHtml" in INDEX_HTML or "parseMarkdownTable" in INDEX_HTML
    assert "<table" in INDEX_HTML
    assert "<thead" in INDEX_HTML
    assert "<tbody" in INDEX_HTML


def test_llm_markdown_renderer_keeps_html_escaping():
    assert "escapeHtml" in INDEX_HTML
    assert "inlineMarkdown" in INDEX_HTML
    assert "markdownToHtml" in INDEX_HTML
