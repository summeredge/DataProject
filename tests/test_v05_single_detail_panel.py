from chem_ts_corr.web import INDEX_HTML


def test_single_variable_review_is_rendered_inside_modal_only():
    required = [
        "detailModal",
        "renderSingleVariableReview",
        "showRawFieldsToggle",
        "rawFieldsCollapsed",
        "modal-card",
        "展开完整原始字段",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_full_raw_fields_are_collapsed_not_duplicated_by_default():
    forbidden = [
        "完整字段：",
        "finalReviewDetailPanel",
    ]
    for marker in forbidden:
        assert marker not in INDEX_HTML


def test_duplicate_long_reason_cards_are_avoided():
    assert INDEX_HTML.count("主要原因") <= 2
    assert "renderFinalReviewDetails" not in INDEX_HTML or "renderSingleVariableReview" in INDEX_HTML
