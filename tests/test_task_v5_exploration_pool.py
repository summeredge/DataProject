from chem_ts_corr import web


def test_model_exploration_ui_declares_the_fixed_bounded_pool():
    assert "初筛 Rank K+1~K+10" in web.INDEX_HTML
    assert "最多显示 5 个并保持初筛顺序" in web.INDEX_HTML
    assert "不会自动加入推荐、候选池或任何排序" in web.INDEX_HTML


def test_model_exploration_table_preserves_the_initial_screening_order():
    render = web.INDEX_HTML.split("function renderCompactDetailTable", 1)[1].split(
        "function renderCompactDetailModal", 1
    )[0]
    assert 'targetId === "modelDiscoveredTable"' in render
