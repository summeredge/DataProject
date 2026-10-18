import shutil
import subprocess

import pytest

from chem_ts_corr.web import INDEX_HTML


def test_variable_selectors_share_case_insensitive_search_component():
    assert "function searchableSelect" in INDEX_HTML
    assert 'String(value).toLowerCase().includes(query)' in INDEX_HTML
    assert 'placeholder = "筛选位号（可选）"' in INDEX_HTML
    assert 'option.textContent = "未找到匹配的位号"' in INDEX_HTML
    assert 'dropdown.className = "single-dropdown"' in INDEX_HTML
    assert "dropdown.append(summary, select)" in INDEX_HTML
    assert "setupDropdownPanel(dropdown, [filter, empty, options])" in INDEX_HTML
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
    assert ".dropdown-panel > .select-filter {" in INDEX_HTML
    assert ".variable-select-native { display:none; }" in INDEX_HTML
    assert ".row > label { align-self:start; width:100%; }" in INDEX_HTML
    assert ".multi-options label[hidden] { display:none; }" in INDEX_HTML


def test_single_select_filter_only_appears_inside_expanded_dropdown():
    assert 'dropdown = document.createElement("details")' in INDEX_HTML
    assert 'const summary = document.createElement("summary")' in INDEX_HTML
    assert "setupDropdownPanel(dropdown, [filter, empty, options])" in INDEX_HTML
    assert 'document.querySelector(`[data-select-dropdown-for="${select.id}"]`).open = false' in INDEX_HTML


def test_single_and_multi_dropdowns_share_viewport_constrained_top_layer():
    assert 'panel.setAttribute("popover", "auto")' in INDEX_HTML
    assert 'setupDropdownPanel(box.closest(".multi-dropdown"), [filter, empty, box])' in INDEX_HTML
    assert '.dropdown-panel { position:fixed;' in INDEX_HTML
    assert 'const upwards = below < 240 && above > below;' in INDEX_HTML
    assert 'window.addEventListener("resize", position)' in INDEX_HTML
    assert 'document.removeEventListener("scroll", position, true)' in INDEX_HTML


def test_dropdown_positioning_and_dismissal_execute_in_javascript():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the dropdown behavior check")
    component = "function setupDropdownPanel" + INDEX_HTML.split(
        "function setupDropdownPanel", 1
    )[1].split("function searchableSelect", 1)[0]
    script = r'''
const assert = require("node:assert/strict");
const listeners = {}, panelListeners = {}, active = new Set();
let rect = {left:100, top:100, bottom:140, width:250};
const panel = {
  style:{}, setAttribute(){}, append(){},
  addEventListener(type, callback){ panelListeners[type] = callback; },
  showPopover(){ this.visible = true; }, hidePopover(){ this.visible = false; },
  matches(){ return this.visible; },
  getBoundingClientRect(){ return {height:Math.min(240, parseFloat(this.style.maxHeight))}; }
};
global.document = {
  createElement(){ return panel; },
  addEventListener(type){ active.add(type); },
  removeEventListener(type){ active.delete(type); }
};
global.window = {
  innerWidth:800, innerHeight:600,
  addEventListener(type){ active.add(type); },
  removeEventListener(type){ active.delete(type); }
};
const dropdown = {
  open:true, appendChild(){},
  querySelector(){return {getBoundingClientRect(){return rect;}};},
  addEventListener(type, callback){ listeners[type] = callback; }
};
setupDropdownPanel(dropdown, []);
listeners.toggle();
assert.equal(panel.style.top, "144px");
assert.equal(panel.style.width, "250px");
rect = {left:740, top:550, bottom:590, width:250};
panelListeners.input();
assert.equal(panel.style.top, "306px");
assert.equal(panel.style.left, "542px");
window.innerHeight = 300;
rect = {left:0, top:140, bottom:180, width:900};
panelListeners.input();
assert.equal(panel.style.maxHeight, "128px");
assert.equal(panel.style.width, "784px");
panel.visible = false;
panelListeners.toggle();
assert.equal(dropdown.open, false);
listeners.toggle();
assert.equal(active.size, 0);
'''
    subprocess.run([node, "-e", component + script], check=True, capture_output=True, text=True)
