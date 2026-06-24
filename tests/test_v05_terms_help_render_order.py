from chem_ts_corr.web import INDEX_HTML


def test_terms_help_rows_defined_before_render_call():
    rows_pos = INDEX_HTML.index("const termsHelpRows")
    fn_pos = INDEX_HTML.index("function renderTermsHelpTab")
    call_pos = INDEX_HTML.rindex("renderTermsHelpTab();")
    assert rows_pos < fn_pos < call_pos


def test_terms_help_loading_placeholder_is_replaced_by_static_table():
    required = [
        "termsHelpTable",
        "术语说明加载中。",
        "replaceChildren(wrap)",
        "termsHelpRows",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_terms_help_render_call_is_not_in_early_event_setup_block():
    early_block_end = INDEX_HTML.index("function fillCapacityOptions")
    early_block = INDEX_HTML[:early_block_end]
    assert "renderTermsHelpTab();" not in early_block
