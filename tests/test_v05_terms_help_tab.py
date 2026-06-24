from chem_ts_corr.web import INDEX_HTML


def test_terms_help_tab_exists():
    required = [
        "术语与标签说明",
        "本页用于解释分析结果中的标签、风险、证据等级和模型指标",
        "这些说明仅用于辅助工程复核，不改变分析结果，也不参与计算",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_terms_help_table_columns_do_not_expose_raw_fields():
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


def test_terms_help_sections_cover_main_result_concepts():
    required = [
        "风险标签说明",
        "证据等级与复核建议",
        "滞后与方向解释",
        "模型验证指标",
        "工程使用建议",
        "滞后边界风险",
        "目标领先风险",
        "公式泄漏 / 计算耦合风险",
        "数据质量风险",
        "闭环反馈风险",
        "共线性风险",
        "强预测证据",
        "风险受限证据",
        "优先复核",
        "仅人工复核",
        "变量领先目标",
        "目标领先变量",
        "模型提升",
        "预测贡献",
        "滚动稳定性",
        "工况稳定性",
        "可能 MV 候选",
        "可能前馈 / 干扰变量",
        "仅作预测验证，不是因果结论",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_terms_help_tab_is_static_and_not_backend_dependent():
    required = [
        "renderTermsHelpTab",
        "termsHelpTable",
    ]
    for marker in required:
        assert marker in INDEX_HTML
