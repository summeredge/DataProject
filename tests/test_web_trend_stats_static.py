import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from chem_ts_corr.web import INDEX_HTML


_NODE = shutil.which("node")
requires_node = pytest.mark.skipif(_NODE is None, reason="node is required for trend JS behavior tests")


def _function_source(name, next_name):
    return INDEX_HTML.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def _css_rule(selector):
    return INDEX_HTML.split(f"{selector} {{", 1)[1].split("}", 1)[0]


def _trend_js_block():
    start = INDEX_HTML.index("function renderTrendChart")
    end = INDEX_HTML.index("const candidateTable", start)
    helpers_start = INDEX_HTML.index("function timestampMilliseconds")
    return INDEX_HTML[helpers_start:start] + INDEX_HTML[start:end]


def _run_trend_js(script_body):
    script = "\n".join(
        [
            "const elements = {};",
            "const el = (id) => { if (!elements[id]) elements[id] = { className: '', innerHTML: '', textContent: '', getBoundingClientRect: () => ({ width: 960 }), clientWidth: 960 }; return elements[id]; };",
            "const escapeHtml = (value) => String(value == null ? '' : value).replace(/[&<>\"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[ch]));",
            "const trendColors = ['#176b87', '#c2410c', '#6d28d9', '#15803d', '#b91c1c', '#ca8a04', '#a21caf', '#475569'];",
            "let trendSelection = null;",
            "let lastTrendSeries = [];",
            _trend_js_block(),
            script_body,
            "console.log(JSON.stringify(moduleResult));",
        ]
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(script)
        script_path = handle.name
    try:
        result = subprocess.run(
            [_NODE, script_path],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        os.unlink(script_path)
    return json.loads(result.stdout)


def test_trend_stats_grid_is_fixed_four_columns_on_desktop():
    trend_stats_rule = _css_rule(".trend-stats")

    assert "display:grid" in trend_stats_rule
    assert "grid-template-columns:repeat(4, minmax(0, 1fr))" in trend_stats_rule
    assert "auto-fit" not in trend_stats_rule
    assert "auto-fill" not in trend_stats_rule


def test_trend_controls_layout_is_four_columns_on_desktop_and_responsive():
    controls_rule = _css_rule(".chart-controls")

    assert "display:grid" in controls_rule
    assert "grid-template-columns:repeat(4,minmax(120px,1fr))" in controls_rule
    assert "@media (max-width:900px)" in INDEX_HTML
    assert ".chart-controls { grid-template-columns:repeat(2,minmax(120px,1fr)); }" in INDEX_HTML
    assert ".chart-controls { grid-template-columns:1fr; }" in INDEX_HTML


def test_max_trend_total_points_constant_limits_trend_response():
    web_source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")

    assert "MAX_TREND_TOTAL_POINTS = 300000" in web_source
    trend_source = web_source.split("def _trend_response", 1)[1].split(
        "def _finite_json_number", 1
    )[0]
    assert "total_points_cap=MAX_TREND_TOTAL_POINTS" in trend_source


def test_trend_colors_extend_to_eight_distinct_colors_with_stable_first_four():
    match = re.search(r"const trendColors = \[([^\]]+)\];", INDEX_HTML)
    assert match is not None
    colors = re.findall(r'"([^"]+)"', match.group(1))

    assert len(colors) == 8
    assert len(set(colors)) == 8
    assert colors[:4] == ["#176b87", "#c2410c", "#6d28d9", "#15803d"]


def test_trend_legend_stats_and_curve_share_series_color():
    chart_source = INDEX_HTML.split("function renderTrendChart", 1)[1].split(
        "function trendChartWidth", 1
    )[0]
    legend_source = INDEX_HTML.split("function renderTrendChart", 1)[1].split(
        "function trendChartWidth", 1
    )[0]
    stats_source = _function_source("renderTrendStats", "formatCountRatio")

    assert 'stroke="${trendColors[idx % trendColors.length]}"' in chart_source
    assert 'style="background:${trendColors[idx % trendColors.length]}"' in legend_source
    assert "renderTrendHistogram(item.points || [], trendColors[index % trendColors.length], item.name)" in stats_source


def test_trend_stats_renders_all_returned_series_without_truncation():
    stats_source = _function_source("renderTrendStats", "formatCountRatio")

    assert "series.map((item, index)" in stats_source
    assert "series.slice" not in stats_source


def test_trend_page_contains_eight_variable_selectors():
    for index in range(1, 9):
        assert f'<label>数据 {index}<select id="trendVar{index}"></select></label>' in INDEX_HTML


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
    assert 'ratio: total ? count / total : 0' in INDEX_HTML


def test_each_trend_stat_card_contains_histogram_from_item_points_and_line_color():
    source = _function_source("renderTrendStats", "formatCountRatio")

    assert "series.map((item, index)" in source
    assert "renderTrendHistogram(item.points || [], trendColors[index % trendColors.length], item.name)" in source
    assert '<dl>${rows}</dl>${histogram}' in source


def test_trend_numeric_summary_filters_numeric_finite_values_in_one_pass():
    histogram_source = _function_source("trendHistogram", "trendNormalCurve")
    summary_source = _function_source("trendNumericSummary", "trendHistogram")

    assert "trendNumericSummary(points)" in histogram_source
    assert "trendFiniteValue(point)" in summary_source
    assert "Number.isFinite(value)" in summary_source
    assert "sumSquares += value * value" in summary_source
    assert "values.push(value)" in summary_source


def test_trend_histogram_handles_empty_constant_and_max_value_boundaries():
    source = _function_source("trendHistogram", "trendNormalCurve")
    render_source = _function_source("renderTrendHistogram", "renderTrendStats")

    assert "if (!count)" in source
    assert "无有效数据" in render_source
    assert "if (min === max)" in source
    assert "bins: [{ min, max, count }]" in source
    assert "const counts = Array(binCount).fill(0)" in source
    assert "for (const value of values)" in source
    assert "counts[index] += 1" in source
    assert "Math.min(binCount - 1" in source
    assert "Math.ceil(Math.sqrt(count))" in source
    assert "requestedBinCount = 12" in INDEX_HTML


def test_trend_normal_curve_uses_fitted_mean_and_population_stddev():
    histogram_source = _function_source("trendHistogram", "trendNormalCurve")
    curve_source = _function_source("trendNormalCurve", "renderTrendHistogram")

    assert "const mean = sum / count" in histogram_source
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
    assert 'el("trendVar1").value, el("trendVar2").value, el("trendVar3").value, el("trendVar4").value, el("trendVar5").value, el("trendVar6").value, el("trendVar7").value, el("trendVar8").value' in INDEX_HTML


def test_trend_selection_uses_physical_timestamps_and_has_clear_feedback():
    assert 'id="clearTrendSelection"' in INDEX_HTML
    assert 'id="trendSelectionInfo"' in INDEX_HTML
    assert 'el("clearTrendSelection").addEventListener("click", clearTrendSelection);' in INDEX_HTML
    trend_source = INDEX_HTML.split("function renderTrendChart", 1)[1].split(
        "function trendChartWidth", 1
    )[0]

    for marker in (
        "let timeStart = Infinity;",
        "let timeEnd = -Infinity;",
        "const timeToX = (milliseconds)",
        "const xToTime = (position)",
        'id="trendSelectionHitbox"',
        "data-trend-selection",
        "setTrendWindowFromSelection(xToTime(start), xToTime(dragEnd));",
        "formatTrendTimestamp(timeStart)",
        "formatTrendTimestamp(timeEnd)",
    ):
        assert marker in trend_source

    assert "function formatTrendDuration(milliseconds)" in INDEX_HTML
    assert "已选择时间窗口" in INDEX_HTML
    assert "趋势数据缺少可解析的时间戳" in INDEX_HTML


def test_trend_selection_reset_restores_defaults_and_redraws():
    reset_source = INDEX_HTML.split("function clearTrendSelection()", 1)[1].split(
        "function timestampMilliseconds", 1
    )[0]

    assert "trendSelection = null;" in reset_source
    assert 'trendTimeRangeMode = "manual";' in reset_source
    assert "trendDefaultStart" in reset_source
    assert "trendDefaultEnd" in reset_source
    assert 'el("trendMaxPoints").value = "10000";' in reset_source
    assert "updateTrendSelectionInfo();" in reset_source
    assert "void drawTrend();" in reset_source


def test_trend_js_avoids_large_array_spreads():
    block = _trend_js_block()

    for forbidden in ["Math.min(...", "Math.max(...", "series.flatMap", ".push(..."]:
        assert forbidden not in block
    assert "function trendSharedRange" in block
    assert "function trendNumericSummary" in block
    assert "function valueRange" in block


@requires_node
def test_trend_js_renders_eight_series_20000_points_without_stack_overflow():
    payload = _run_trend_js(
        """
const series = Array.from({ length: 8 }, (_, i) => ({
  name: `v${i + 1}`,
  points: Array.from({ length: 20000 }, (_, j) => ({ x: String(j), y: (j + i) % 100 })),
}));
renderTrendChart(series, "shared");
const moduleResult = {
  legendCards: (el("trendLegend").innerHTML.match(/class="swatch"/g) || []).length,
  statCards: (el("trendStats").innerHTML.match(/trend-stat-card/g) || []).length,
  polylines: (el("trendChart").innerHTML.match(/<polyline/g) || []).length,
};
"""
    )

    assert payload == {"legendCards": 8, "statCards": 8, "polylines": 8}


@requires_node
def test_trend_js_renders_eight_series_37500_points_without_stack_overflow():
    payload = _run_trend_js(
        """
const series = Array.from({ length: 8 }, (_, i) => ({
  name: `v${i + 1}`,
  points: Array.from({ length: 37500 }, (_, j) => ({ x: String(j), y: (j + i) % 97 })),
}));
renderTrendChart(series, "shared");
const moduleResult = {
  legendCards: (el("trendLegend").innerHTML.match(/class="swatch"/g) || []).length,
  statCards: (el("trendStats").innerHTML.match(/trend-stat-card/g) || []).length,
};
"""
    )

    assert payload == {"legendCards": 8, "statCards": 8}


@requires_node
def test_trend_js_shared_range_covers_all_series_valid_values():
    payload = _run_trend_js(
        """
const series = [
  { name: "a", points: [{ x: "0", y: 1 }, { x: "1", y: 5 }] },
  { name: "b", points: [{ x: "0", y: 10 }, { x: "1", y: 2 }] },
  { name: "c", points: [{ x: "0", y: null }, { x: "1", y: "7" }] },
];
const shared = trendSharedRange(series);
const moduleResult = { min: shared.min, max: shared.max };
"""
    )

    assert payload == {"min": pytest.approx(0.28), "max": pytest.approx(10.72)}


@requires_node
def test_trend_js_independent_ranges_use_each_series_values():
    payload = _run_trend_js(
        """
const series = [
  { name: "a", points: [{ x: "0", y: 1 }, { x: "1", y: 5 }] },
  { name: "b", points: [{ x: "0", y: 10 }, { x: "1", y: 20 }] },
];
const moduleResult = series.map((item) => valueRange(item.points));
"""
    )

    assert payload == [
        {"min": pytest.approx(0.68), "max": pytest.approx(5.32)},
        {"min": pytest.approx(9.2), "max": pytest.approx(20.8)},
    ]


@requires_node
def test_trend_js_stats_match_manual_small_sample():
    payload = _run_trend_js(
        """
const stats = trendStats([{ y: 1 }, { y: 2 }, { y: 3 }, { y: 4 }]);
const moduleResult = {
  mean: stats.mean,
  stddev: stats.stddev,
  min: stats.min,
  max: stats.max,
  range: stats.range,
  median: stats.median,
  count: stats.count,
  ratio: stats.ratio,
};
"""
    )

    assert payload["mean"] == 2.5
    assert payload["stddev"] == pytest.approx(1.25 ** 0.5)
    assert payload["min"] == 1
    assert payload["max"] == 4
    assert payload["range"] == 3
    assert payload["median"] == 2.5
    assert payload["count"] == 4
    assert payload["ratio"] == 1


@requires_node
def test_trend_js_ignores_invalid_values_in_stats_and_range():
    payload = _run_trend_js(
        """
const points = [
  { y: 1 }, { y: null }, { y: "abc" }, { y: NaN }, { y: Infinity }, { y: -Infinity }, { y: 3 },
];
const stats = trendStats(points);
const shared = trendSharedRange([{ name: "a", points }]);
const moduleResult = {
  count: stats.count,
  ratio: stats.ratio,
  min: stats.min,
  max: stats.max,
  mean: stats.mean,
  median: stats.median,
  rangeMin: shared.min,
  rangeMax: shared.max,
};
"""
    )

    assert payload == {
        "count": 2,
        "ratio": 2 / 7,
        "min": 1,
        "max": 3,
        "mean": 2,
        "median": 2,
        "rangeMin": pytest.approx(0.84),
        "rangeMax": pytest.approx(3.16),
    }


@requires_node
def test_trend_js_single_constant_and_empty_series_keep_semantics():
    payload = _run_trend_js(
        """
const single = trendStats([{ y: 7 }]);
const constant = trendStats([{ y: 5 }, { y: 5 }]);
const empty = trendStats([]);
const constRange = valueRange([{ y: 5 }, { y: 5 }]);
const emptyRange = valueRange([]);
const moduleResult = {
  singleStddev: single.stddev,
  singleMedian: single.median,
  constantStddev: constant.stddev,
  constantRange: constant.range,
  emptyCount: empty.count,
  emptyRatio: empty.ratio,
  emptyMin: empty.min,
  constRangeMin: constRange.min,
  constRangeMax: constRange.max,
  emptyRangeMin: emptyRange.min,
  emptyRangeMax: emptyRange.max,
};
"""
    )

    assert payload == {
        "singleStddev": 0,
        "singleMedian": 7,
        "constantStddev": 0,
        "constantRange": 0,
        "emptyCount": 0,
        "emptyRatio": 0,
        "emptyMin": None,
        "constRangeMin": pytest.approx(3.84),
        "constRangeMax": pytest.approx(6.16),
        "emptyRangeMin": 0,
        "emptyRangeMax": 1,
    }


@requires_node
def test_trend_js_histogram_handles_large_inputs():
    payload = _run_trend_js(
        """
const points20000 = Array.from({ length: 20000 }, (_, j) => ({ y: j % 100 }));
const points37500 = Array.from({ length: 37500 }, (_, j) => ({ y: j % 97 }));
const h20 = trendHistogram(points20000);
const h375 = trendHistogram(points37500);
const moduleResult = {
  bins20: h20.bins.length,
  count20: h20.count,
  min20: h20.min,
  max20: h20.max,
  bins375: h375.bins.length,
  count375: h375.count,
};
"""
    )

    assert payload == {
        "bins20": 12,
        "count20": 20000,
        "min20": 0,
        "max20": 99,
        "bins375": 12,
        "count375": 37500,
    }


@requires_node
def test_trend_js_four_series_behavior_is_unchanged():
    payload = _run_trend_js(
        """
const series = Array.from({ length: 4 }, (_, i) => ({
  name: `v${i + 1}`,
  points: Array.from({ length: 100 }, (_, j) => ({ x: String(j), y: j + i })),
}));
renderTrendChart(series, "shared");
const stats = trendStats(series[0].points);
const moduleResult = {
  legendCards: (el("trendLegend").innerHTML.match(/class="swatch"/g) || []).length,
  statCards: (el("trendStats").innerHTML.match(/trend-stat-card/g) || []).length,
  firstCardName: el("trendStats").innerHTML.match(/<h3>([^<]+)<\\/h3>/)[1],
  mean: stats.mean,
  count: stats.count,
};
"""
    )

    assert payload["legendCards"] == 4
    assert payload["statCards"] == 4
    assert payload["firstCardName"] == "v1"
    assert payload["mean"] == 49.5
    assert payload["count"] == 100
