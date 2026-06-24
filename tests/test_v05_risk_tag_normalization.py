from chem_ts_corr.web import INDEX_HTML


def test_canonical_risk_tag_normalizer_exists():
    required = [
        "CANONICAL_RISK_GROUPS",
        "normalizeRiskTags",
        "formatRiskFlags",
        "renderRiskTagDetails",
        "标准风险",
        "原始风险标签",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_lag_boundary_variants_are_grouped():
    required = [
        "lag_boundary",
        "lag_boundary_flag",
        "lag_boundary_risk",
        "lag_reaches_boundary",
        "boundary_lag_uncertain",
        "screening_lag_boundary_risk",
        "model_lag_boundary_risk",
        "滞后边界风险",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_direction_and_formula_variants_are_grouped():
    required = [
        "target_leads_variable",
        "target_leads_candidate",
        "target_lead_risk",
        "no_positive_lag",
        "non_positive_screening_lag",
        "strong_formula_leakage",
        "formula_leakage_risk",
        "formula_coupled_reference",
        "formula_like",
        "变量滞后目标风险",
        "公式泄漏/计算耦合风险",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_data_regime_driver_loop_and_collinearity_variants_are_grouped():
    required = [
        "poor_data_quality",
        "poor_quality_variable",
        "unstable_across_regimes",
        "unstable_over_time",
        "unstable_candidate",
        "stability_risk",
        "capacity_driven",
        "common_capacity_driver",
        "closed_loop_suspect",
        "residual_collinearity",
        "high_collinearity_risk",
        "数据质量风险",
        "工况/时变不稳定风险",
        "共同负荷驱动风险",
        "闭环反馈风险",
        "共线性风险",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_main_display_uses_canonical_risks_but_raw_tags_remain_available():
    required = [
        "displayRisks",
        "rawRiskTags",
        "risk_flags",
        "完整原始字段",
    ]
    for marker in required:
        assert marker in INDEX_HTML
