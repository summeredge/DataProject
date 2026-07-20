from chem_ts_corr.web import INDEX_HTML


def test_modal_raw_fields_are_collapsed_by_default():
    assert "展开完整原始字段" in INDEX_HTML
    assert "rawFieldsCollapsed" in INDEX_HTML
    assert '<details class="raw-fields ${rawFieldsCollapsed}">' in INDEX_HTML


def test_remaining_detail_field_names_are_chinese_ready():
    required = [
        "association_score",
        "independent_signal_score",
        "correlation_evidence_score",
        "correlation_evidence_status",
        "residual_status",
        "regime_stability_final",
        "regime_status",
        "rolling_status",
        "lag_quality_status",
        "model_lift_score",
        "model_lift_status",
        "risk_penalty",
        "force_included",
        "原始关联规范化得分",
        "独立残差信号得分",
        "关联证据综合得分",
        "关联证据状态",
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


def test_innovation_statuses_and_sign_values_are_chinese_ready():
    required = [
        "non_predictive_lag",
        "innovation_verified",
        "innovation_lag_conflict",
        "innovation_sign_conflict",
        "innovation_sign_unknown",
        "非预测性滞后",
        "变化量验证通过",
        "变化量滞后冲突",
        "变化量符号冲突",
        "变化量符号未知",
        'innovation_sign: {',
        '"1": "正向"',
        '"-1": "负向"',
        '"0": "零相关"',
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


def test_overview_recommendations_default_to_driver_rank_ascending():
    assert 'tableSortStates["overviewTop"] = { column: "driver_rank", direction: "asc" };' in INDEX_HTML
    assert 'overviewTop: ["variable", "driver_rank", "driver_priority_score", "candidate_class", "evidence_coverage_status"' in INDEX_HTML
    overview_columns = INDEX_HTML.split("overviewTop:", 1)[1].split("],", 1)[0]
    assert "final_score" not in overview_columns
    assert "evidence_confidence" not in overview_columns
