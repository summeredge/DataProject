from chem_ts_corr.web import INDEX_HTML


def test_variable_selectors_share_case_insensitive_search_component():
    assert "function searchableSelect" in INDEX_HTML
    assert 'String(value).toLowerCase().includes(query)' in INDEX_HTML
    assert 'placeholder = "筛选位号（可选）"' in INDEX_HTML
    assert 'option.textContent = "未找到匹配的位号"' in INDEX_HTML
    assert 'dropdown.className = "single-dropdown"' in INDEX_HTML
    assert "dropdown.append(summary, filter, empty, options, select)" in INDEX_HTML
    assert 'select.classList.add("variable-select-native")' in INDEX_HTML
    assert "if (currentValue || allowEmpty) select.value = currentValue;" in INDEX_HTML
    assert "for (const value of values)" in INDEX_HTML


def test_all_variable_single_and_multi_selectors_use_search():
    for select_id in [
        "timeColumn",
        "targetColumn",
        "segmentColumn",
        "trendVar1",
        "trendVar4",
        "trendVar8",
        "scatterX1",
        "scatterY3",
    ]:
        assert f'el("{select_id}")' in INDEX_HTML
    assert INDEX_HTML.count("searchableMultiOptions(box);") == 3
    assert "function clearVariableFilters" in INDEX_HTML


def test_trend_variable_selectors_cover_eight_slots_everywhere():
    load_body = INDEX_HTML.split("async function loadColumns()", 1)[1].split(
        "async function uploadFile", 1
    )[0]
    refresh_body = INDEX_HTML.split("function refreshColumnSelectors()", 1)[1].split(
        "function setSecondaryIncludeSelection", 1
    )[0]
    draw_body = INDEX_HTML.split("async function drawTrend()", 1)[1].split(
        "async function drawScatterMatrix()", 1
    )[0]
    reset_body = INDEX_HTML.split("function reset()", 1)[1].split(
        "function updateDrawButtons", 1
    )[0]
    open_body = INDEX_HTML.split("function openTrendForCandidate", 1)[1].split(
        "function setSelectValueIfExists", 1
    )[0]

    for index in range(1, 9):
        assert f"trendVar{index}" in refresh_body
        assert f'el("trendVar{index}").value' in draw_body
        assert f'el("trendVar{index}").innerHTML = ""' in reset_body
    for index in range(5, 9):
        assert f'fillSelect(el("trendVar{index}")' in load_body
        assert f'setSelectValueIfExists("trendVar{index}", "")' in open_body


def test_variable_select_layout_keeps_rows_aligned():
    assert ".single-dropdown {" in INDEX_HTML
    assert ".single-dropdown > .select-filter {" in INDEX_HTML
    assert ".variable-select-native { display:none; }" in INDEX_HTML
    assert ".row > label { align-self:start; width:100%; }" in INDEX_HTML
    assert ".multi-options label[hidden] { display:none; }" in INDEX_HTML


def test_single_select_filter_only_appears_inside_expanded_dropdown():
    assert 'dropdown = document.createElement("details")' in INDEX_HTML
    assert 'const summary = document.createElement("summary")' in INDEX_HTML
    assert "dropdown.append(summary, filter, empty, options, select)" in INDEX_HTML
    assert 'document.querySelector(`[data-select-dropdown-for="${select.id}"]`).open = false' in INDEX_HTML
