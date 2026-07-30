from chem_ts_corr.web import INDEX_HTML


def test_terms_help_page_has_no_internal_vertical_scroll():
    required = [
        "terms-help-table-wrap",
        "termsHelpTable",
    ]
    for marker in required:
        assert marker in INDEX_HTML
    forbidden = [
        "max-height",
        "overflow-y: auto",
        "overflow-y:auto",
    ]
    terms_section_start = INDEX_HTML.index("termsHelpTab")
    terms_section = INDEX_HTML[terms_section_start:]
    for marker in forbidden:
        assert marker not in terms_section


def test_terms_help_category_cells_are_grouped():
    required = [
        "renderTermsHelpGroupedRows",
        "rowspan",
        "terms-help-category-cell",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_direction_wording_is_globally_consistent():
    required = [
        "变量领先目标",
        "变量滞后目标",
        "目标领先变量",
        "target_leads_variable",
        "target_leads_candidate",
    ]
    for marker in required:
        assert marker in INDEX_HTML
    forbidden = ["更可能表现为响应变量、反馈动作或下游状态"]
    for marker in forbidden:
        assert marker not in INDEX_HTML


def test_terms_help_uses_lagging_target_wording():
    required = [
        "变量领先目标",
        "变量滞后目标",
        "变量变化早于目标",
        "变量变化晚于目标",
    ]
    for marker in required:
        assert marker in INDEX_HTML
