from chem_ts_corr.web import INDEX_HTML


def test_trend_stats_and_axis_helpers_are_present():
    required = [
        'trendStats',
        'renderTrendStats',
        'trend-stat-card',
        'axisTicks',
        'formatAxisValue',
        'formatCountRatio',
    ]
    for token in required:
        assert token in INDEX_HTML


def test_trend_stats_container_exists_near_chart():
    assert 'id="trendStats"' in INDEX_HTML
    assert 'class="trend-stats empty"' in INDEX_HTML


def test_trend_stats_labels_are_present():
    required = ['均值', '标准差', '最大值', '最小值', '极差', '中位数', '有效点数']
    for token in required:
        assert token in INDEX_HTML


def test_template_like_visual_decoration_is_not_introduced():
    forbidden = ['linear-gradient', 'radial-gradient', 'blob', 'glow', 'backdrop-filter']
    lowered = INDEX_HTML.lower()
    for token in forbidden:
        assert token not in lowered


def test_trend_api_contract_remains_get_query_only():
    assert 'fetch(`/api/trend?${params.toString()}`)' in INDEX_HTML
    assert '/api/trend/stats' not in INDEX_HTML
    assert '/api/trend_stats' not in INDEX_HTML
    trend_fetch = INDEX_HTML.split("fetch(`/api/trend?${params.toString()}`)", 1)[0].rsplit("const response = await ", 1)[-1]
    assert "method" not in trend_fetch


def test_trend_chart_width_uses_container_measurement():
    assert 'container.getBoundingClientRect().width' in INDEX_HTML
    assert 'Math.max(640, measuredWidth)' in INDEX_HTML
    assert 'const width = 960' not in INDEX_HTML


def test_effective_point_count_includes_ratio():
    assert 'validRatio' in INDEX_HTML
    assert '${stats.count} / ${(stats.validRatio * 100).toFixed(1)}%' in INDEX_HTML
