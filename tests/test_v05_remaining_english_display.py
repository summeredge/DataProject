from chem_ts_corr.web import INDEX_HTML


def test_remaining_detail_field_labels_are_chinese_ready():
    required = [
        "feature",
        "importance",
        "valid_window_count",
        "fallback_maxlag",
        "模型特征",
        "重要性",
        "有效窗口数",
        "回退最大滞后",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_remaining_detail_values_are_chinese_ready():
    required = [
        "mean_abs_shap",
        "enhanced screening only",
        "non-positive screening lag",
        "not_computed",
        "ranked_window",
        "SHAP平均绝对值",
        "仅作增强筛查",
        "非正主筛查滞后",
        "未计算",
        "排序窗口",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_remaining_display_values_translate_in_split_text():
    required = [
        "enhanced screening only; not a causal conclusion",
        "仅作增强筛查；不是因果结论",
        "split(/[;,，；]/)",
        "formatValue",
    ]
    for marker in required:
        assert marker in INDEX_HTML
