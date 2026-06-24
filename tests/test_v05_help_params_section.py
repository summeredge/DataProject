from chem_ts_corr.web import INDEX_HTML


def test_parameter_settings_section_exists_and_is_first():
    params_pos = INDEX_HTML.index("参数设置说明")
    risk_pos = INDEX_HTML.index("风险标签说明")
    assert params_pos < risk_pos


def test_parameter_settings_section_covers_main_controls():
    required = [
        "参数设置说明",
        "时间列",
        "目标列",
        "最大滞后点数",
        "输出前 K 个",
        "最小有效比例",
        "重采样规则",
        "预处理模式",
        "去趋势窗口点数",
        "负荷代表列",
        "工况分段",
        "自定义下限",
        "自定义上限",
        "残差控制列",
        "强制复核变量",
        "三层复核候选数量",
        "风险标签包含过滤",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_parameter_settings_explain_result_impact():
    required = [
        "参数说明用于解释页面设置项的含义和对结果的影响",
        "参数设置会影响候选筛选、滞后搜索、风险标签和复核范围",
        "不改变分析结果，也不参与计算",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_parameter_settings_keep_user_facing_columns_only():
    required = [
        "分类",
        "页面显示名称",
        "具体表征",
        "工程解读",
        "建议动作",
    ]
    for marker in required:
        assert marker in INDEX_HTML
    forbidden = [
        "原始字段 / 标签示例",
        "原始字段或标签",
        "原始标签示例",
    ]
    for marker in forbidden:
        assert marker not in INDEX_HTML
