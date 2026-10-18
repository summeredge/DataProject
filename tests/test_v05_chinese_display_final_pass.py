from chem_ts_corr.web import INDEX_HTML


def test_static_llm_ui_labels_are_chinese():
    forbidden = [
        "<label>Top N",
        "<label>Provider",
        "<label>Base URL",
        "<label>Model",
        "<label>API Key",
        "<label>temperature",
        "<label>max_tokens",
        "Chat Completions",
    ]
    for marker in forbidden:
        assert marker not in INDEX_HTML
    required = [
        "分析变量数量",
        "模型服务",
        "接口地址",
        "模型名称",
        "API 密钥",
        "温度参数",
        "最大输出 Token 数",
        "聊天补全接口",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_candidate_grades_and_common_status_values_are_mapped():
    required = [
        "candidate_grade_A",
        "candidate_grade_B",
        "candidate_grade_C",
        "candidate_grade_D",
        "candidate_grade_E",
        "候选等级A",
        "候选等级B",
        "候选等级C",
        "候选等级D",
        "候选等级E",
        "failed",
        "error",
        "no_data",
        "insufficient_data",
        "not_run",
        "失败",
        "错误",
        "无数据",
        "数据不足",
        "未运行",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_common_risk_decision_direction_markers_are_mapped():
    required = [
        "high_collinearity_risk",
        "formula_leakage_risk",
        "no_positive_lag",
        "non_positive_screening_lag",
        "variable_leads_target",
        "candidate_leads_target",
        "target_leads_candidate",
        "priority_review_with_statistical_limit",
        "secondary_review_with_statistical_limit",
        "高共线性风险",
        "公式泄漏风险",
        "无正向滞后",
        "非正主筛查滞后",
        "变量领先目标",
        "目标领先变量",
        "优先复核但统计受限",
        "二级复核但统计受限",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_translation_function_handles_delimited_english_tokens():
    required = [
        "split(/[;,，；]/)",
        "translateDisplayValue",
        "formatCellValue",
    ]
    for marker in required:
        assert marker in INDEX_HTML
