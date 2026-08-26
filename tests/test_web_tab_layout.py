from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser

from chem_ts_corr.web import INDEX_HTML


class _TabLayoutParser(HTMLParser):
    _VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[dict[str, object]] = []
        self.active_panels: list[str] = []
        self.buttons: list[dict[str, object]] = []
        self.panels: list[dict[str, object]] = []
        self.ids: list[str] = []
        self._current_button: dict[str, object] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids.append(element_id)

        inside_tablist = any(entry["starts_tablist"] for entry in self.stack)
        classes = set(attributes.get("class", "").split())
        panel_id = element_id if tag == "div" and "tab-panel" in classes else None
        if panel_id:
            panel = {"id": panel_id, "attrs": attributes, "content_ids": []}
            self.panels.append(panel)
            self.active_panels.append(panel_id)
        elif element_id and self.active_panels:
            panel = next(item for item in self.panels if item["id"] == self.active_panels[-1])
            panel["content_ids"].append(element_id)

        if tag == "button" and inside_tablist and "tab-button" in classes:
            self._current_button = {"attrs": attributes, "text": ""}
            self.buttons.append(self._current_button)

        if tag not in self._VOID_TAGS:
            self.stack.append(
                {
                    "tag": tag,
                    "starts_tablist": attributes.get("role") == "tablist",
                    "panel_id": panel_id,
                }
            )

    def handle_data(self, data: str) -> None:
        if self._current_button is not None:
            self._current_button["text"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "button":
            self._current_button = None
        if not self.stack:
            return
        entry = self.stack.pop()
        assert entry["tag"] == tag
        panel_id = entry["panel_id"]
        if panel_id:
            assert self.active_panels.pop() == panel_id


def _layout() -> _TabLayoutParser:
    parser = _TabLayoutParser()
    parser.feed(INDEX_HTML)
    return parser


def test_result_tab_buttons_have_required_order_and_no_candidate_tab():
    layout = _layout()
    labels = [str(button["text"]).strip() for button in layout.buttons]
    targets = [button["attrs"]["data-tab"] for button in layout.buttons]

    assert labels[:2] == ["初步分析", "趋势图"]
    assert targets[:2] == ["overviewTab", "trendTab"]
    assert labels[2:] == ["二次验证", "可信度审查", "时间外预测验证", "AI 综合解读", "下载", "术语与标签说明"]
    assert labels.count("初步分析") == 1
    assert "候选变量" not in labels
    assert "candidatesTab" not in targets


def test_overview_tab_and_panel_are_the_only_default_active_items():
    layout = _layout()
    first_button = layout.buttons[0]["attrs"]
    other_buttons = [button["attrs"] for button in layout.buttons[1:]]
    panels = {panel["id"]: panel["attrs"] for panel in layout.panels}

    assert "active" in first_button["class"].split()
    assert first_button["aria-selected"] == "true"
    assert first_button["tabindex"] == "0"
    assert all("active" not in attrs["class"].split() for attrs in other_buttons)
    assert all(attrs["aria-selected"] == "false" for attrs in other_buttons)
    assert "active" in panels["overviewTab"]["class"].split()
    assert "hidden" not in panels["overviewTab"]
    assert all(
        "active" not in attrs["class"].split() and "hidden" in attrs
        for panel_id, attrs in panels.items()
        if panel_id != "overviewTab"
    )


def test_overview_panel_contains_analysis_then_candidate_content():
    layout = _layout()
    panels = {panel["id"]: panel for panel in layout.panels}
    content_ids = panels["overviewTab"]["content_ids"]

    for element_id in [
        "overview",
        "analysisTimingBreakdown",
        "overviewTop",
        "candidatesTab",
        "controlReferenceTable",
        "screeningQualityHints",
        "table",
    ]:
        assert element_id in content_ids
    assert content_ids.index("overviewTop") < content_ids.index("candidatesTab")
    assert content_ids.index("candidatesTab") < content_ids.index("controlReferenceTable")
    assert content_ids.index("controlReferenceTable") < content_ids.index("screeningQualityHints")


def test_all_tab_targets_are_reachable_without_orphans_or_duplicate_ids():
    layout = _layout()
    button_targets = [button["attrs"]["data-tab"] for button in layout.buttons]
    button_controls = [button["attrs"]["aria-controls"] for button in layout.buttons]
    panel_ids = [panel["id"] for panel in layout.panels]

    assert button_targets == button_controls
    assert set(button_targets) == set(panel_ids)
    assert "tab-candidatesTab" not in layout.ids
    assert not [element_id for element_id, count in Counter(layout.ids).items() if count > 1]
    assert 'activateTab("candidatesTab")' not in INDEX_HTML


def test_default_overview_activation_keeps_trend_render_path_available():
    initialization = INDEX_HTML.split(
        'for (const button of document.querySelectorAll(".tab-button")) {', 1
    )[1].split('el("drawTrend").addEventListener', 1)[0]
    activation = INDEX_HTML.split("function activateTab(tabId) {", 1)[1].split(
        "function handleTabKeydown", 1
    )[0]

    assert 'activateTab("overviewTab");' in initialization
    assert initialization.count('button.addEventListener("click"') == 1
    assert "requestAnimationFrame" in activation
    assert "renderTrendChart(lastTrendSeries, lastTrendAxisMode)" in activation
    assert "renderScatterMatrix(lastScatterMatrixPayload)" in activation
    assert "fetch(" not in activation
    assert "drawTrend()" not in activation
