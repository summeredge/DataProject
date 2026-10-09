import shutil
import subprocess

import pytest

from chem_ts_corr.web import INDEX_HTML


def test_compact_review_preserves_full_text_values_and_statuses():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    # Run the real display helpers; compact labels must not alter detail text.
    functions = INDEX_HTML.split("function statusTone(", 1)[1].split(
        "function tableCellClass(", 1
    )[0]
    script = r'''
const assert = require('node:assert/strict');
const STATUS_COLUMNS = new Set(['confounder_assessment','control_relation_assessment','direction_assessment','statistical_limitation']);
function escapeHtml(v) {return String(v ?? '').replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;');}
function displayCellValue(c,v) {
  return ({no_flagged_confounder:'未发现明显混杂风险',possible_control_response:'可能属于控制响应信号',not_run:'未执行',not_computed:'未计算',missing:'证据缺失'})[v] ?? String(v ?? '-');
}
''' + "function statusTone(" + functions + r'''
const original = renderTableCell('confounder_assessment','no_flagged_confounder');
assert.ok(original.includes('未发现明显混杂风险'));
const compact = renderFinalReviewCell('confounder_assessment','no_flagged_confounder');
assert.ok(compact.includes('title="未发现明显混杂风险"'));
assert.ok(compact.includes('>未见明显风险</span>'));
assert.ok(compact.includes('status-label-positive'));
assert.ok(renderFinalReviewCell('control_relation_assessment','possible_control_response').includes('>疑似控制响应</span>'));
assert.ok(renderFinalReviewCell('statistical_limitation','high_collinearity_limitation').includes('status-label-caution'));
assert.ok(renderFinalReviewCell('statistical_limitation','insufficient_sample_limitation').includes('status-label-negative'));
assert.ok(renderFinalReviewCell('statistical_limitation','failed_statistical_limitation').includes('status-label-negative'));
const distinct = ['not_run','not_computed','missing',0].map(v => renderFinalReviewCell('direction_assessment',v));
assert.equal(new Set(distinct).size,4);
assert.ok(distinct[0].includes('status-label-neutral'));
assert.ok(renderFinalReviewCell('screening_lag',-3).includes('>-3</span>'));
assert.ok(renderFinalReviewCell('variable','<tag>"').includes('&lt;tag>&quot;'));
'''
    result = subprocess.run([node], input=script, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


def test_review_layout_is_scoped_and_keeps_existing_detail_access():
    assert '#finalReviewSummaryTable .table-wrap { max-height:none;' in INDEX_HTML
    assert '[data-column="final_rank"] { width:65px;' in INDEX_HTML
    assert 'td:nth-child(2) { left:65px;' in INDEX_HTML
    assert 'title="${escapeHtml(title)}"' in INDEX_HTML
    assert 'final_rank: "复核序号"' in INDEX_HTML
    assert 'id="finalReviewSummaryDownload"' in INDEX_HTML
    assert "attachFinalSummaryRowClick(rows)" in INDEX_HTML
    assert "openTrendForCandidate(row)" in INDEX_HTML
