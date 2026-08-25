from pathlib import Path

from chem_ts_corr.web import INDEX_HTML


def test_section_names_are_clearer():
    required = [
        "<h3>可信度审查概览</h3>",
        "<h2>逐变量可信度审查证据表</h2>",
        'id="finalReviewQualityOverview"',
        'id="causalReviewEvidenceTable"',
    ]
    for marker in required:
        assert marker in INDEX_HTML
    forbidden = [
        "<h2>最终推荐质量总览</h2>",
        "<h2>综合证据复核</h2>",
    ]
    for marker in forbidden:
        assert marker not in INDEX_HTML


def test_index_html_uses_plain_string_semantics():
    source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")

    assert type(INDEX_HTML) is str
    assert "class _IndexHtml" not in source
    assert "def __contains__" not in source
    assert "INDEX_HTML = _IndexHtml" not in source


def test_raw_risk_tag_display_is_chinese_readable():
    required = [
        "formatRawRiskTags",
        "变量滞后目标",
        "滞后触及边界",
        "数据质量需关注",
        "数据质量严重不足",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_remaining_visible_values_are_translated():
    required = [
        "non-positive screening lag",
        "非正主筛查滞后",
        "false",
        "否",
    ]
    for marker in required:
        assert marker in INDEX_HTML
