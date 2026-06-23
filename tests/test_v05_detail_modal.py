from chem_ts_corr.web import INDEX_HTML


def test_detail_view_uses_modal_instead_of_inline_panel():
    required = [
        "detailModal",
        "detailModalTitle",
        "detailModalBody",
        "openDetailModal",
        "closeDetailModal",
        "modal-card",
        "modal-backdrop",
        "查看详情",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_main_page_does_not_show_inline_detail_panel_by_default():
    forbidden = [
        "finalReviewDetailPanel",
        "singleVariableReviewCard",
        "完整字段：",
    ]
    for marker in forbidden:
        assert marker not in INDEX_HTML


def test_raw_fields_are_collapsed_inside_modal():
    required = [
        "展开完整原始字段",
        "renderRawFields",
        "renderSingleVariableReview",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_modal_supports_basic_accessibility_and_close_actions():
    required = [
        "role=\"dialog\"",
        "aria-modal=\"true\"",
        "Escape",
        "关闭",
    ]
    for marker in required:
        assert marker in INDEX_HTML
