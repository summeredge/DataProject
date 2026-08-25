from pathlib import Path

from chem_ts_corr.web import FINAL_REVIEW_SUMMARY_FIELD_NOTES, INDEX_HTML


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


def test_third_layer_priority_copy_is_not_an_algorithmic_ranking():
    required = [
        "人工复核优先级（展示序号；不参与初筛评分或排序）",
        "人工复核优先级仅用于第三层展示和复核建议，不参与算法评分或初筛排序",
        "人工复核优先级仅用于展示，不改变初筛顺序",
    ]
    for marker in required:
        assert marker in INDEX_HTML
    for forbidden in ["最终排名", "最终因果排名", "综合因果评分", "第三层排名"]:
        assert forbidden not in INDEX_HTML


def test_api_summary_field_notes_define_manual_review_semantics():
    assert FINAL_REVIEW_SUMMARY_FIELD_NOTES == {
        "final_rank": "人工复核优先级（展示序号）；仅用于第三层人工复核，不参与初筛评分或排序。",
        "review_priority": "人工复核优先级。",
        "review_reason": "证据摘要。",
        "final_recommendation": "复核建议。",
    }
    assert all(
        forbidden not in " ".join(FINAL_REVIEW_SUMMARY_FIELD_NOTES.values())
        for forbidden in ["最终排名", "最终因果排名", "综合因果评分"]
    )


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
