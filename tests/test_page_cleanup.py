from chem_ts_corr.web import INDEX_HTML


def test_page_no_inactive_option():
    target = "llm" + "Anonymize"
    assert target not in INDEX_HTML
