from chem_ts_corr.web import INDEX_HTML


def test_modal_raw_fields_are_expanded_by_default():
    assert "展开完整原始字段" in INDEX_HTML
    assert "rawFieldsCollapsed" not in INDEX_HTML
    assert "<details class=\"raw-fields\" open>" in INDEX_HTML or "<details class='raw-fields' open>" in INDEX_HTML


def test_remaining_detail_field_names_are_chinese_ready():
    required = [
        "raw_corr_score",
        "residual_corr_score",
        "residual_status",
        "regime_stability_final",
        "regime_status",
        "rolling_status",
        "lag_quality_status",
        "model_lift_score",
        "model_lift_status",
        "risk_penalty",
        "force_included",
        "原始相关得分",
        "残差相关得分",
        "残差状态",
        "工况稳定性",
        "工况状态",
        "滚动状态",
        "滞后峰值质量状态",
        "模型提升得分",
        "模型提升状态",
        "风险惩罚",
        "强制纳入",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_remaining_status_values_are_chinese_ready():
    required = [
        "not_computed",
        "ranked_window",
        "non-positive screening lag",
        "未计算",
        "排序窗口",
        "非正主筛查滞后",
        "true",
        "false",
        "是",
        "否",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_table_width_uses_content_fit_with_page_maximum():
    compact = INDEX_HTML.replace(" ", "")
    required = [
        "width:max-content",
        "max-width:100%",
        "overflow-x:auto",
    ]
    for marker in required:
        assert marker in compact
    assert "width:100%;table-layout:fixed" not in compact
