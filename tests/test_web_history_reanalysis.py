from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time

import pytest

from chem_ts_corr import web
from chem_ts_corr.history import query_history


@pytest.fixture
def stored_file(tmp_path, monkeypatch):
    uploads, runs = tmp_path / "uploads", tmp_path / "runs"
    uploads.mkdir()
    runs.mkdir()
    file_id = "a" * 32
    path = uploads / f"{file_id}.csv"
    path.write_text("time,target,feature\n" + "\n".join(
        f"2025-01-01 {i // 60:02d}:{i % 60:02d}:00,{math.sin(i / 5)},{math.sin((i - 1) / 5)}"
        for i in range(120)
    ), encoding="utf-8")
    monkeypatch.setattr(web, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(web, "RUNS_DIR", runs)
    monkeypatch.setattr(web, "TASKS", {})
    monkeypatch.setattr(web, "EXCLUDE_WINDOW_CONTEXTS", {})
    return file_id, path, runs


def post(monkeypatch, response, **fields):
    monkeypatch.setattr(web, "_multipart_form", lambda handler: fields)
    return response(object())


@pytest.mark.parametrize("mode", ["raw", "lowpass"])
def test_reanalysis_reuses_file_and_keeps_each_run_independent(stored_file, monkeypatch, mode):
    file_id, path, runs = stored_file
    columns = web._columns_response(file_id, "auto")
    assert columns["timeColumn"] == "time"
    assert columns["numericColumns"] == ["target", "feature"]
    original = path.read_bytes()
    snapshots = []
    for index in range(2):
        reply = post(monkeypatch, web._analyze_response, file_id=file_id, time_column="time",
                     target="target" if index == 0 else "feature", preprocess_mode=mode,
                     max_lag="2", top_k="1")
        deadline = time.monotonic() + 30
        while web.TASKS[reply["task_id"]]["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert web.TASKS[reply["task_id"]]["status"] == "done", web.TASKS[reply["task_id"]]
        directory = runs / reply["run_id"]
        config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        assert config["file_id"] == file_id and config["input_path"] == str(path)
        assert config["target"] == ("target" if index == 0 else "feature")
        context = json.loads((directory / "preprocessing_context.json").read_text(encoding="utf-8"))
        assert context["branch_selection_status"] == ("not_required" if mode == "raw" else "awaiting_confirmation")
        for snapshot in snapshots:
            assert all(p.read_bytes() == content for p, content in snapshot.items())
        snapshots.append({p: p.read_bytes() for p in directory.rglob("*") if p.is_file()})
    assert len(list(runs.iterdir())) == 2
    history = query_history(web.UPLOADS_DIR, runs)
    assert history["uploads"][0]["analysis_count"] == 2
    assert len({row["run_id"] for row in history["analyses"]}) == 2
    assert path.read_bytes() == original
    assert list(web.UPLOADS_DIR.iterdir()) == [path]


def test_selection_resets_all_time_column_contexts_only_for_selected_file(stored_file, monkeypatch):
    file_id, path, _ = stored_file
    window = {"start": "2025-01-01T00:00:00", "end": "2025-01-01T00:01:00"}
    for time_column in ["time", "alternative_time"]:
        web.EXCLUDE_WINDOW_CONTEXTS[(file_id, time_column)] = {"exclude_windows": [window]}
    other = ("b" * 32, "time")
    web.EXCLUDE_WINDOW_CONTEXTS[other] = {"exclude_windows": [window]}
    original = path.read_bytes()
    result = post(monkeypatch, web._restore_all_exclude_windows_response, file_id=file_id)
    assert result == {"excludeWindows": [], "excludeWindowStats": None}
    assert set(web.EXCLUDE_WINDOW_CONTEXTS) == {other}
    assert path.read_bytes() == original
    config_calls = []
    monkeypatch.setattr(web, "_analyze_task", lambda task_id, config, selected: config_calls.append(config))
    post(monkeypatch, web._analyze_response, file_id=file_id, time_column="time", target="target")
    deadline = time.monotonic() + 2
    while not config_calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert config_calls[0].exclude_windows == []


@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_invalid_historical_file_does_not_create_task(stored_file, monkeypatch, failure):
    file_id, path, runs = stored_file
    if failure == "missing":
        path.unlink()
    else:
        def unreadable(*args, **kwargs):
            raise PermissionError("历史文件不可读取")
        monkeypatch.setattr(web, "read_timeseries_table", unreadable)
    with pytest.raises((FileNotFoundError, PermissionError)):
        web._columns_response(file_id, "auto")
    with pytest.raises((FileNotFoundError, PermissionError)):
        post(monkeypatch, web._analyze_response, file_id=file_id, time_column="time", target="target")
    assert web.TASKS == {}
    assert not list(runs.iterdir())


def test_frontend_selection_reset_busy_guard_and_failure():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the frontend state check")
    html = web.INDEX_HTML
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    globals_source = script.split('for (const button of document.querySelectorAll(".tab-button"))', 1)[0]
    def function_source(name):
        match = re.search(rf"^(?:async )?function {name}\(", script, re.MULTILINE)
        assert match is not None
        end = re.search(r"^}", script[match.end():], re.MULTILINE)
        assert end is not None
        return script[match.start():match.end() + end.end()]
    functions = "\n".join(function_source(name) for name in [
        "lockDataSelection", "prepareFileSelection", "selectHistoryFile", "uploadFile", "loadColumns",
        "postForm", "analyze", "reset", "clearVariableFilters", "clearLagProfileCache", "updateExcludeWindowState",
        "restoreAnalysis", "restoreRunFromUrl", "historyTime", "setDownstreamGate", "updateXgbRunAvailability",
        "isRestoredRunReadOnly", "updateBranchSelectionUi", "processedBranchLabel",
        "confirmInitialScreeningBranch", "addToVerificationReviewPool",
    ])
    harness = r'''
const assert = require('node:assert/strict');
const elements = new Map();
const buttons = [{dataset:{fileId:'a'.repeat(32)},disabled:false}];
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {value:'',innerHTML:'',textContent:'',files:[],
      disabled:false,hidden:false,checked:false,scrollIntoView(){},
      options:[{value:'raw'},{value:'lowpass'}],appendChild(option){this.options.push(option)},
      classList:{contains(){return true}}});
    return elements.get(id);
  },
  querySelectorAll(selector) { return selector === '.history-reanalyze, .history-restore' ? buttons : []; },
  createElement(){return {};}
};
const window = {location:{href:'http://localhost/?run_id=old',get search(){return new URL(this.href).search}},
  history:{replaceState(a,b,url){window.location.href=String(url)}}};
let activeTab = '';
let requests = [];
let failColumns = false;
let pauseColumns = null;
async function fetch(url) {
  requests.push(url);
  if (url.startsWith('/api/columns')) {
    if (pauseColumns) await pauseColumns;
    return {ok:!failColumns,json:async()=>failColumns ? {error:'上传文件不存在，请重新上传'} : {
      columns:['time','target','feature'],numericColumns:['target','feature'],timeColumn:'time',
      sampleRows:120,encoding:'utf-8',timeEnd:'2025-01-01T02:00',trendStartDefault:'2025-01-01T00:00'
    }};
  }
  return {ok:true,json:async()=>({excludeWindows:[],excludeWindowStats:null})};
}
function fillSelect(node, values, allowEmpty=false) {node.innerHTML=values.join(',');node.value=allowEmpty?'':values[0];}
function setStatus(text){document.getElementById('status').textContent=text;}
function activateTab(id){activeTab=id;}
function resetOptionalTable(id,text){document.getElementById(id).textContent=text;}
function clearOptionalElement(id){document.getElementById(id).innerHTML='';}
function renderValidationSummaryTable(){document.getElementById('validationSummaryTable').textContent='';}
function renderValidationFieldsTable(){document.getElementById('validationFieldsTable').textContent='';}
function setLlmReport(text){document.getElementById('llmReport').textContent=text;}
function validateAnalysisColumnSelection(){return '';}
function getCapacitySelection(){return [];}
function getForceIncludeSelection(){return [];}
function getExcludedColumnSelection(){return [];}
let finishAnalysis;
function waitForAnalysisResult(){return new Promise(resolve=>{finishAnalysis=resolve});}
let renderedResult;
function renderAnalysisResult(data){currentRunId=data.run_id;currentAnalysisContext=data.analysisContext||{};
  renderedResult=data;updateBranchSelectionUi(data);
  REVIEW_BUTTON_STATE
  if(data.branchSelectionStatus==='awaiting_confirmation') {
    el('addManualReviewPool').disabled=true;el('addModelDiscoveryReviewPool').disabled=true;
  }
  setDownstreamGate(data.branchSelectionStatus==='awaiting_confirmation');}
function setForceIncludeSelection(){}
function updateExcludedColumnDisabledState(){}
function formatCompletedAnalysisStatus(){return '完成';}
function startStatusTimer(){return 0;}
function stopStatusTimer(){}
function appendElapsed(text){return text;}
function renderVerificationReviewPool(){}
function syncModelDiscoveryReviewPoolOptions(){}
function renderDownloads(){}
'''
    review_button_state = re.search(
        r'  el\("addManualReviewPool"\)\.disabled = !currentRunId[^\n]*\n'
        r'  el\("addModelDiscoveryReviewPool"\)\.disabled = !currentRunId[^\n]*', script
    )
    assert review_button_state is not None
    harness = harness.replace("REVIEW_BUTTON_STATE", review_button_state.group())
    stubs = "\n".join(f"function {name}() {{}}" for name in [
        "updatePreprocessControls", "fillCapacityOptions", "fillForceIncludeOptions", "syncSearchableSelect",
        "updateTrendSelectionInfo", "setCapacitySelection", "fillExcludedColumnOptions", "refreshColumnSelectors",
        "clearTrendStats", "clearScatterMatrix", "closeDetailModal", "clearXgbFoldContext",
        "clearXgbCandidateFoldDetails", "renderExcludeWindows", "setExcludedColumnSelection",
    ])
    assertions = r'''
(async()=>{
  fileId='old';currentRunId='old-run';currentAnalysisContext={old:true};
  lastRows=[{old:true}];lastValidationSummaryRows=[{old:true}];lastValidationFieldsRows=[{old:true}];
  lastFinalReviewSummaryRows=[{old:true}];lastXgbFoldContextRows=[{old:true}];
  lastTrendSeries=[{old:true}];lastScatterMatrixPayload={old:true};
  excludeWindows=[{start:'old',end:'old'}];lagProfileCache.set('old',{});
  el('downloads').innerHTML='old-link';el('validationSummaryTable').textContent='old-stage';
  el('preprocessMode').value='lowpass';el('maxLag').value='7';
  await selectHistoryFile({file_id:'a'.repeat(32),original_filename:'历史.csv'});
  assert.equal(fileId,'a'.repeat(32));assert.equal(currentRunId,'');
  assert.deepEqual(currentAnalysisContext,{});assert.deepEqual(lastRows,[]);
  assert.deepEqual(lastValidationSummaryRows,[]);assert.deepEqual(lastValidationFieldsRows,[]);
  assert.deepEqual(lastFinalReviewSummaryRows,[]);assert.deepEqual(lastXgbFoldContextRows,[]);
  assert.deepEqual(excludeWindows,[]);assert.deepEqual(lastTrendSeries,[]);
  assert.equal(lastScatterMatrixPayload,null);assert.equal(lagProfileCache.size,0);
  assert.equal(el('downloads').innerHTML,'');assert.equal(el('validationSummaryTable').textContent,'');
  assert(!new URL(window.location.href).searchParams.has('run_id'));
  assert.equal(el('preprocessMode').value,'lowpass');assert.equal(el('maxLag').value,'7');
  assert.equal(el('timeColumn').value,'time');assert.equal(el('targetColumn').value,'target');
  assert.equal(el('selectedDataFile').textContent,'历史数据：历史.csv');
  assert.equal(el('analyze').disabled,false);assert.equal(el('runGranger').disabled,true);
  assert.equal(activeTab,'overviewTab');assert.equal(dataSelectionLocks,0);
  assert(requests.includes('/api/restore_all_exclude_windows'));assert(!requests.includes('/api/upload'));
  let release;
  pauseColumns=new Promise(resolve=>{release=resolve});
  const selecting=selectHistoryFile({file_id:'b'.repeat(32),original_filename:null});
  assert.equal(buttons[0].disabled,true);assert.equal(el('upload').disabled,true);assert.equal(el('reset').disabled,true);
  const before=requests.length;
  await selectHistoryFile({file_id:'c'.repeat(32)});await uploadFile();reset();
  assert.equal(requests.length,before);assert.equal(fileId,'b'.repeat(32));
  release();await selecting;pauseColumns=null;
  assert.equal(dataSelectionLocks,0);assert.equal(buttons[0].disabled,false);
  failColumns=true;
  await selectHistoryFile({file_id:'c'.repeat(32),original_filename:'缺失.csv'});
  assert.equal(fileId,'');assert.equal(el('analyze').disabled,true);
  assert.equal(dataSelectionLocks,0);assert(el('status').textContent.includes('不存在'));
  failColumns=false;
  el('fileInput').files=[new Blob(['time,target\n1,2'])];
  fetch=async url=>({ok:true,json:async()=>url==='/api/upload'?{file_id:'d'.repeat(32),filename:'新文件.csv'}:{
    columns:['time','target'],numericColumns:['target'],sampleRows:1,encoding:'utf-8',timeColumn:'time'
  }});
  await uploadFile();
  assert.equal(fileId,'d'.repeat(32));assert.equal(el('selectedDataFile').textContent,'新上传数据：新文件.csv');
  assert.equal(dataSelectionLocks,0);
  fetch=async()=>({ok:true,json:async()=>({task_id:'task',run_id:'new-run'})});
  const running=analyze();
  for (let i=0;i<10&&!finishAnalysis;i++) await Promise.resolve();
  assert(finishAnalysis);assert.equal(dataSelectionLocks,1);
  assert.equal(buttons[0].disabled,true);assert.equal(el('upload').disabled,true);
  await selectHistoryFile({file_id:'a'.repeat(32)});reset();
  assert.equal(fileId,'d'.repeat(32));assert.equal(currentRunId,'new-run');
  finishAnalysis({run_id:'new-run'});await running;
  assert.equal(dataSelectionLocks,0);assert.equal(el('upload').disabled,false);
  assert.equal(currentRunId,'new-run');
  fetch=async()=>({ok:false,json:async()=>({error:'请求失败'})});
  await assert.rejects(postForm('/api/run_model',new FormData()));
  assert.equal(dataSelectionLocks,0);assert.equal(el('upload').disabled,false);
  const restored={run_id:'saved-run',branchSelectionStatus:'confirmed',activeScreeningBranch:'raw',
    branchLocked:true,analysisContext:{preprocess_mode:'raw'},xgbResult:{status:'not_run'},restoration:{
      file_id:'a'.repeat(32),original_filename:'保存.csv',created_at:'2025-01-01T00:00:00Z',time_source:'metadata',
      input_available:true,context_available:true,input_error:'',stage_issues:{},excludeWindows:[{start:'saved',end:'saved'}],
      parameters:{time_column:'time',target:'target',max_lag:4,top_k:3,min_valid_ratio:0.8,
        resample_rule:'2min',preprocess_mode:'lowpass',lowpass_tau_minutes:2,diff_interval_minutes:1,
        detrend_window:30,segment_mode:'all',excluded_columns:[],force_include_variables:[]}
  }};
  fetch=async url=>({ok:true,json:async()=>url.startsWith('/api/result')?restored:{
    columns:['time','target','feature'],numericColumns:['target','feature'],sampleRows:120,encoding:'utf-8',timeColumn:'time'
  }});
  window.location.href='http://localhost/?run_id=saved-run';
  await restoreRunFromUrl();
  assert.equal(currentRunId,'saved-run');assert.equal(el('maxLag').value,4);
  assert.equal(el('resampleRule').value,'2');assert.equal(el('preprocessMode').value,'lowpass');
  assert.equal(currentAnalysisContext.preprocess_mode,'raw');
  assert.equal(renderedResult.branchLocked,true);assert.deepEqual(excludeWindows,[{start:'saved',end:'saved'}]);
  assert.equal(restoredRunInfo,restored.restoration);assert.equal(el('runGranger').disabled,false);
  assert.equal(el('addManualReviewPool').disabled,false);
  assert.equal(el('addModelDiscoveryReviewPool').disabled,false);
  assert(new URL(window.location.href).searchParams.get('run_id')==='saved-run'||renderedResult.run_id==='saved-run');
  restored.restoration.context_available=false;
  await restoreAnalysis('saved-run');
  assert.equal(el('runGranger').disabled,true);assert.equal(el('analyze').disabled,false);
  assert(el('status').textContent.includes('预处理上下文'));
  assert.equal(el('addManualReviewPool').disabled,true);
  assert.equal(el('addModelDiscoveryReviewPool').disabled,true);
  restored.restoration.context_available=true;
  restored.restoration.input_available=false;restored.restoration.input_error='原始数据缺失';
  restored.restoration.parameters.segment_column='load';
  restored.restoration.parameters.residual_control_columns=['control'];
  restored.restoration.parameters.force_include_variables=['feature'];
  await restoreAnalysis('saved-run');
  assert.equal(currentRunId,'saved-run');assert.equal(el('analyze').disabled,true);
  assert.equal(el('drawTrend').disabled,true);assert.equal(el('runGranger').disabled,true);
  assert.equal(el('runXgbValidation').disabled,true);assert(el('restoredAnalysis').textContent.includes('原始数据缺失'));
  restored.branchSelectionStatus='awaiting_confirmation';restored.branchLocked=false;
  for (const info of [
    {input_available:false,context_available:true},
    {input_available:true,context_available:false}
  ]) {
    restoredRunInfo=info;renderAnalysisResult(restored);
    assert.equal(el('confirmRawBranch').disabled,true);
    assert.equal(el('confirmProcessedBranch').disabled,true);
    assert.equal(el('addManualReviewPool').disabled,true);
    assert.equal(el('addModelDiscoveryReviewPool').disabled,true);
    let mutations=0;
    const originalPostForm=postForm;postForm=async()=>{mutations++;return restored;};
    el('manualReviewPoolVariable').value='feature';el('modelDiscoveryReviewPoolVariable').value='feature';
    await confirmInitialScreeningBranch('raw');await confirmInitialScreeningBranch('processed');
    await addToVerificationReviewPool('manual_include');await addToVerificationReviewPool('model_discovery');
    assert.equal(mutations,0);postForm=originalPostForm;
  }
  restoredRunInfo={input_available:true,context_available:true};renderAnalysisResult(restored);
  assert.equal(el('confirmRawBranch').disabled,false);assert.equal(el('confirmProcessedBranch').disabled,false);
  assert.equal(el('addManualReviewPool').disabled,true);
  let validMutations=0;
  const savedPostForm=postForm;postForm=async()=>{validMutations++;return restored;};
  await confirmInitialScreeningBranch('raw');assert.equal(validMutations,1);
  restored.branchSelectionStatus='confirmed';renderAnalysisResult(restored);
  assert.equal(el('addManualReviewPool').disabled,false);assert.equal(el('addModelDiscoveryReviewPool').disabled,false);
  el('manualReviewPoolVariable').value='feature';el('modelDiscoveryReviewPoolVariable').value='feature';
  await addToVerificationReviewPool('manual_include');await addToVerificationReviewPool('model_discovery');
  assert.equal(validMutations,3);postForm=savedPostForm;
  assert(recognizedNumericColumns.includes('load')&&recognizedNumericColumns.includes('control')&&recognizedNumericColumns.includes('feature'));
  lockDataSelection(1);const previous=renderedResult;await restoreAnalysis('different-run');
  assert.equal(renderedResult,previous);lockDataSelection(-1);
  fetch=async()=>({ok:false,json:async()=>({error:'运行配置无法恢复'})});
  await restoreAnalysis('broken-run');
  assert.equal(currentRunId,'');assert.equal(el('analyze').disabled,true);
  assert.equal(dataSelectionLocks,0);assert(el('status').textContent.includes('运行配置无法恢复'));
  assert.equal(isRestoredRunReadOnly(),false);
  assert.equal(el('confirmRawBranch').disabled,true);assert.equal(el('confirmProcessedBranch').disabled,true);
  assert.equal(el('addManualReviewPool').disabled,true);assert.equal(el('addModelDiscoveryReviewPool').disabled,true);
  fetch=async()=>({ok:true,json:async()=>({columns:['time','target'],numericColumns:['target'],
    sampleRows:120,encoding:'utf-8',timeColumn:'time'})});
  await selectHistoryFile({file_id:'a'.repeat(32),original_filename:'重新分析.csv'});
  assert.equal(isRestoredRunReadOnly(),false);assert.equal(el('analyze').disabled,false);
  renderAnalysisResult({run_id:'new-run',branchSelectionStatus:'awaiting_confirmation'});
  assert.equal(el('confirmRawBranch').disabled,false);assert.equal(el('confirmProcessedBranch').disabled,false);
  reset();assert.equal(currentRunId,'');assert.equal(restoredRunInfo,null);
  assert.equal(el('confirmRawBranch').disabled,true);assert.equal(el('addManualReviewPool').disabled,true);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run([node], input=harness + globals_source + stubs + functions + assertions,
                            text=True, encoding="utf-8", capture_output=True, timeout=15, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
