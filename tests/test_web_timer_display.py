from chem_ts_corr.web import INDEX_HTML


def test_background_status_messages_do_not_use_static_timer_placeholder():
    """后台执行中的状态文案应显示实时秒数，而不是固定“计时中”。"""
    assert "（计时中）" not in INDEX_HTML


def test_background_status_has_live_runtime_timer_and_total_elapsed_label():
    """增强筛选、Granger、模型解释、三层复核等前端后台动作应具备运行中计时与完成总耗时文案。"""
    assert "总耗时" in INDEX_HTML
    assert "已运行" in INDEX_HTML
    assert "setInterval" in INDEX_HTML or "requestAnimationFrame" in INDEX_HTML
