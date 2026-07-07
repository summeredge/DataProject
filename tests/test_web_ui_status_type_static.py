from chem_ts_corr.web import INDEX_HTML


def test_background_polling_status_remains_loading():
    assert 'setStatus(formatTaskStatus(statusData), "loading")' in INDEX_HTML


def test_llm_connection_error_sets_error_status():
    assert 'setStatus(appendElapsed(message, startedAt), "error")' in INDEX_HTML


def test_trend_error_updates_global_status_as_error():
    assert 'setStatus(error.message || String(error), "error")' in INDEX_HTML
