from chem_ts_corr.web import INDEX_HTML


def test_section_names_are_clearer():
    required = [
        "最终推荐结果质检总览",
        "逐变量综合证据复核表",
    ]
    for marker in required:
        assert marker in INDEX_HTML
    forbidden = [
        "最终推荐质量总览",
        "综合证据复核",
    ]
    for marker in forbidden:
        assert marker not in INDEX_HTML


def test_raw_risk_tag_display_is_chinese_readable():
    required = [
        "formatRawRiskTags",
        "目标领先变量",
        "滞后触及边界",
        "数据质量差",
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
