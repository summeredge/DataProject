from chem_ts_corr.web import INDEX_HTML


def test_remaining_risk_aliases_are_grouped():
    required = [
        "non-positive screening lag",
        "data_or_formula_risk",
        "synchronous_or_leakage_risk",
        "synchronous",
        "变量滞后目标风险",
        "公式泄漏/计算耦合风险",
        "同步变化风险",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_risk_flags_are_not_duplicated_in_raw_field_grid():
    required = [
        "rawFieldColumnsWithoutRiskFlags",
        "filter((column) => column !== \"risk_flags\")",
        "renderRiskTagDetails(row.risk_flags)",
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_canonical_risk_details_still_keep_raw_tags():
    required = [
        "标准风险",
        "原始风险标签",
        "rawRiskTags",
        "displayRisks",
    ]
    for marker in required:
        assert marker in INDEX_HTML
