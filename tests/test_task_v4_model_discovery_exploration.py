from chem_ts_corr import web


def test_model_discovery_ui_is_exploration_only_not_a_recommendation():
    assert "随机森林模型遗漏探索" in web.INDEX_HTML
    assert "不属于二级验证结论" in web.INDEX_HTML
    assert "不会自动加入推荐、候选池或任何排序" in web.INDEX_HTML

    columns = web.INDEX_HTML.split("function modelDiscoveredColumns()", 1)[1].split("\n}", 1)[0]
    assert '"recommended_use"' not in columns
    assert '"recommended_action"' not in columns


def test_model_discovery_interpretation_has_chinese_exploration_boundary():
    assert (
        "仅作模型遗漏探索；不是验证结论或因果结论"
        in web.INDEX_HTML
    )
