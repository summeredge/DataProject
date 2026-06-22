from chem_ts_corr.web import INDEX_HTML


def test_ai_interpretation_page_has_no_anonymize_control():
    assert "llmAnonymize" not in INDEX_HTML
    assert "匿名化变量名" not in INDEX_HTML


def test_ai_interpretation_requests_do_not_send_anonymize_field():
    assert 'form.append("anonymize"' not in INDEX_HTML
