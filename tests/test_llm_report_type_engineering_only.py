from chem_ts_corr.web import INDEX_HTML


def test_llm_report_type_shows_engineering_advice_only():
    assert '<option value="apc_advice">工程建议</option>' in INDEX_HTML
    assert 'APC/DCS 工程建议' not in INDEX_HTML
    assert '通用综合解读' not in INDEX_HTML
    assert '<option value="general">' not in INDEX_HTML


def test_llm_report_type_keeps_existing_internal_value_for_scope_safety():
    assert 'id="llmReportType"' in INDEX_HTML
    assert 'value="apc_advice"' in INDEX_HTML
