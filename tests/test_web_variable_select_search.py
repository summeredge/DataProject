from chem_ts_corr.web import INDEX_HTML


def test_variable_selectors_share_case_insensitive_search_component():
    assert "function searchableSelect" in INDEX_HTML
    assert 'String(value).toLowerCase().includes(query)' in INDEX_HTML
    assert 'placeholder = "筛选位号（可选）"' in INDEX_HTML
    assert 'option.textContent = "未找到匹配的位号"' in INDEX_HTML


def test_all_variable_single_and_multi_selectors_use_search():
    for select_id in [
        "timeColumn",
        "targetColumn",
        "segmentColumn",
        "trendVar1",
        "trendVar4",
        "scatterX1",
        "scatterY3",
    ]:
        assert f'el("{select_id}")' in INDEX_HTML
    assert INDEX_HTML.count("searchableMultiOptions(box);") == 4
    assert "function clearVariableFilters" in INDEX_HTML
