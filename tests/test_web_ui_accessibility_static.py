from chem_ts_corr.web import INDEX_HTML


def test_status_tabs_and_panels_have_accessibility_semantics():
    required = [
        'role="status"',
        'aria-live="polite"',
        'role="tablist"',
        'role="tab"',
        'role="tabpanel"',
        'aria-selected',
        'aria-controls',
    ]
    for token in required:
        assert token in INDEX_HTML


def test_focus_status_tokens_and_small_font_token_are_present():
    required = [
        ':focus-visible',
        '--font-xs:11px',
        '--focus:',
        '.status.error',
        '.status.loading',
        '.status.success',
    ]
    for token in required:
        assert token in INDEX_HTML


def test_small_font_sizes_are_unified_to_font_xs_token():
    assert 'font-size:10px' not in INDEX_HTML
    assert 'font-size:10.5px' not in INDEX_HTML
    assert '--font-xs:11px' in INDEX_HTML


def test_template_like_visual_decoration_is_not_introduced():
    forbidden = [
        'linear-gradient',
        'radial-gradient',
        'blob',
        'glow',
        'backdrop-filter',
    ]
    lowered = INDEX_HTML.lower()
    for token in forbidden:
        assert token not in lowered


def test_table_and_modal_accessibility_enhancements_are_present():
    required = [
        'scope="col"',
        'aria-sort=',
        'tabindex="0"',
        'let lastModalTrigger = null;',
        'lastModalTrigger.focus()',
    ]
    for token in required:
        assert token in INDEX_HTML


def test_workbench_hierarchy_defaults_to_basic_parameters_and_result_summary():
    assert '<details id="advancedParameters" class="advanced-parameters">' in INDEX_HTML
    assert 'id="timeColumn"' in INDEX_HTML.split('<details id="advancedParameters"', 1)[0]
    assert 'activateTab("overviewTab");' in INDEX_HTML
    assert 'class="results-heading"' in INDEX_HTML
    assert 'class="status-panel"' in INDEX_HTML


def test_status_columns_render_text_labels_and_mobile_layout_has_horizontal_tabs():
    assert 'class="status-label status-label-${tone}"' in INDEX_HTML
    assert 'const STATUS_COLUMNS = new Set([' in INDEX_HTML
    assert '.tabs { flex-wrap:nowrap; overflow-x:auto;' in INDEX_HTML
    assert 'table { min-width:680px; }' in INDEX_HTML
