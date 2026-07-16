from chem_ts_corr.web import INDEX_HTML


def _function_source(name, next_name):
    return INDEX_HTML.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def _css_rule(selector):
    return INDEX_HTML.split(f"{selector} {{", 1)[1].split("}", 1)[0]


def test_trend_stats_grid_is_fixed_four_columns_on_desktop():
    trend_stats_rule = _css_rule(".trend-stats")

    assert "display:grid" in trend_stats_rule
    assert "grid-template-columns:repeat(4, minmax(0, 1fr))" in trend_stats_rule
    assert "auto-fit" not in trend_stats_rule
    assert "auto-fill" not in trend_stats_rule


def test_trend_stats_and_axis_helpers_are_present():
    required = [
        'trendStats',
        'trendHistogram',
        'trendNormalCurve',
        'renderTrendHistogram',
        'renderTrendStats',
        'trend-stat-card',
        'axisTicks',
        'formatAxisValue',
        'trendChartWidth',
        'formatCountRatio',
    ]
    for token in required:
        assert token in INDEX_HTML


def test_trend_stats_container_exists_near_chart():
    assert 'id="trendStats"' in INDEX_HTML
    assert 'class="trend-stats empty"' in INDEX_HTML


def test_trend_stats_labels_are_present():
    required = ['均值', '标准差', '最大值', '最小值', '极差', '中位数', '有效点数/占比']
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


def test_trend_chart_uses_measured_width_and_clears_cached_series():
    assert 'const width = 960' not in INDEX_HTML
    assert 'const width = trendChartWidth(container)' in INDEX_HTML
    assert 'viewBox="0 0 ${width} ${height}"' in INDEX_HTML
    assert 'lastTrendSeries = []' in INDEX_HTML
    assert 'lastTrendSeries = series' in INDEX_HTML


def test_trend_stats_count_ratio_contract_is_static():
    assert '["有效点数/占比", "countRatio"]' in INDEX_HTML
    assert 'formatCountRatio(stats.count, stats.ratio)' in INDEX_HTML
    assert 'ratio: total ? values.length / total : 0' in INDEX_HTML


def test_each_trend_stat_card_contains_histogram_from_item_points_and_line_color():
    source = _function_source("renderTrendStats", "formatCountRatio")

    assert "series.map((item, index)" in source
    assert "renderTrendHistogram(item.points || [], trendColors[index % trendColors.length], item.name)" in source
    assert '<dl>${rows}</dl>${histogram}' in source


def test_trend_histogram_filters_numeric_finite_values():
    source = _function_source("trendHistogram", "trendNormalCurve")

    assert "(points || [])" in source
    assert ".map((point) => Number(point.y))" in source
    assert ".filter((value) => Number.isFinite(value))" in source


def test_trend_histogram_handles_empty_constant_and_max_value_boundaries():
    source = _function_source("trendHistogram", "trendNormalCurve")
    render_source = _function_source("renderTrendHistogram", "renderTrendStats")

    assert "if (!values.length)" in source
    assert "无有效数据" in render_source
    assert "if (min === max)" in source
    assert "bins: [{ min, max, count: values.length }]" in source
    assert "const counts = Array(binCount).fill(0)" in source
    assert "for (const value of values)" in source
    assert "counts[index] += 1" in source
    assert "Math.min(binCount - 1" in source
    assert "Math.ceil(Math.sqrt(values.length))" in source
    assert "requestedBinCount = 12" in INDEX_HTML


def test_trend_normal_curve_uses_fitted_mean_and_population_stddev():
    histogram_source = _function_source("trendHistogram", "trendNormalCurve")
    curve_source = _function_source("trendNormalCurve", "renderTrendHistogram")

    assert "const mean = values.reduce" in histogram_source
    assert "(value - mean) ** 2" in histogram_source
    assert "histogram.stddev <= 0" in curve_source
    assert "histogram.min === histogram.max" in curve_source
    assert "Math.sqrt(2 * Math.PI)" in curve_source
    assert "Math.exp(-0.5 * z ** 2)" in curve_source
    assert "sampleCount = 40" in INDEX_HTML


def test_trend_histogram_has_safe_accessible_labels_and_formatted_bounds():
    source = _function_source("renderTrendHistogram", "renderTrendStats")

    assert "数值分布" in source
    assert 'role="img"' in source
    assert 'aria-label="${safeName}' in source
    assert "const safeName = escapeHtml(variableName)" in source
    assert "formatAxisValue(histogram.min)" in source
    assert "formatAxisValue(histogram.max)" in source
    assert 'title="${title}"' in source
    assert 'class="trend-histogram-curve"' in source
    assert '<polyline points="${curvePoints}" stroke="${color}"/>' in source
    assert "含拟合正态分布曲线" in source


def test_trend_histogram_width_and_flex_contract():
    histogram_rule = _css_rule(".trend-histogram")
    bars_rule = _css_rule(".trend-histogram-bars")
    bar_rule = _css_rule(".trend-histogram-bar")
    curve_rule = _css_rule(".trend-histogram-curve")
    card_rule = _css_rule(".trend-stat-card")

    assert "width:100%" in histogram_rule
    assert "min-width:0" in histogram_rule
    assert "width:100%" in bars_rule
    assert "min-width:0" in bars_rule
    assert "display:flex" in bars_rule
    assert "height:72px" in bars_rule
    assert "flex:1 1 0" in bar_rule
    assert "position:absolute" in curve_rule
    assert "width:100%" in curve_rule
    assert "min-width:0" in curve_rule
    assert "height:100%" in curve_rule
    assert "min-width:0" in card_rule
    assert "overflow:hidden" in card_rule


def test_trend_histogram_uses_native_markup_without_new_api_or_dependency():
    source = _function_source("trendHistogram", "renderTrendStats").lower()

    assert "fetch(" not in source
    assert "canvas" not in source
    assert "/api/" not in source
    for dependency in ["plotly", "chart.js", "echarts", "d3.js"]:
        assert dependency not in INDEX_HTML.lower()


def test_existing_trend_stats_guards_remain_present():
    render_source = _function_source("renderTrendStats", "formatCountRatio")
    clear_source = _function_source("clearTrendStats", "trendHistogram")

    assert "if (!series.length)" in render_source
    assert "clearTrendStats()" in render_source
    assert 'node.className = "trend-stats empty"' in clear_source
    assert '[el("trendVar1").value, el("trendVar2").value, el("trendVar3").value, el("trendVar4").value]' in INDEX_HTML
