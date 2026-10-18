from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.data import apply_exclude_windows
from chem_ts_corr import pipeline, web
from test_tr3_exclude_window_flow import _config, _write_input
import test_tr3_exclude_window_flow as window_flow
from test_web_exclude_windows import _write_upload, _post


def rule(operator="=", value=2, column="target"):
    return {"column": column, "operator": operator, "value": value}


@pytest.mark.parametrize("operator,removed", [
    (">", [3]), (">=", [2, 3]), ("<", [-1.5, 0]),
    ("<=", [-1.5, 0, 2]), ("=", [2]), ("!=", [-1.5, 0, 3]),
])
def test_comparisons_ignore_missing_and_nonfinite(operator, removed):
    frame = pd.DataFrame({"target": [-1.5, 0, 2, 3, np.nan, np.inf, -np.inf]},
                         index=pd.date_range("2026-01-01", periods=7, freq="min"))
    result = apply_exclude_windows(frame, [], [rule(operator)])
    assert frame.loc[~frame.index.isin(result.index), "target"].tolist() == removed
    assert len(result) == 7 - len(removed)


@pytest.mark.parametrize("conditions", [
    [rule(column="missing")], [rule(operator="==")], [rule(value=None)],
    [rule(value="2")], [rule(value=True)], [rule(value=np.nan)], [rule(value=np.inf)],
    [{}], [None], {}, "bad",
])
def test_invalid_rules_fail_explicitly(conditions):
    frame = pd.DataFrame({"target": [1, 2]}, index=pd.date_range("2026-01-01", periods=2))
    with pytest.raises(ValueError, match="数值排除条件"):
        apply_exclude_windows(frame, [], conditions)


def test_union_deduplicates_and_does_not_mutate_source(tmp_path):
    config = _config(tmp_path)
    times = _write_input(config)
    before = config.input_path.read_bytes()
    config = replace(config, excluded_columns=["predictor"],
                     exclude_windows=[{"start": str(times[2]), "end": str(times[4])}],
                     exclude_conditions=[rule("<=", 6, "predictor"), rule("=", 2)])
    source = pipeline.load_analysis_source_frame(config)
    stats = pipeline._analysis_exclude_window_stats(config)
    assert stats["excluded_rows"] == 5
    assert stats["remaining_rows"] == 75
    assert stats["excluded_ratio"] == 5 / 80
    assert source.index.equals(times[5:])
    assert config.input_path.read_bytes() == before
    assert "predictor" not in source.columns
    with_column = pipeline.load_analysis_source_frame(replace(config, excluded_columns=[]))
    assert "predictor" in with_column.columns


@pytest.mark.parametrize("case", [
    "test_exclusion_gap_resets_lowpass_and_prevents_cross_gap_difference",
    "test_no_resample_exclusion_forces_all_lags_to_stay_within_segments",
    "test_innovation_difference_retains_exclusion_gap_metadata",
    "test_resample_keeps_legal_boundary_bins_without_merging_segments",
    "test_rolling_correlation_does_not_cross_exclusion_gap",
    "test_rolling_correlation_restarts_at_resampled_forced_break",
])
def test_numeric_exclusion_reuses_physical_break_guarantees(case, tmp_path, monkeypatch):
    # Use the existing behavioral assertions with equivalent raw-value conditions.
    def numeric_config(path, **overrides):
        windows = overrides.pop("exclude_windows", [])
        origin = pd.Timestamp("2026-08-14 08:00")
        conditions = [rule("=", int((time - origin).total_seconds() / 60))
                      for window in windows
                      for time in pd.date_range(window["start"], window["end"], freq="min")]
        return _config(path, exclude_conditions=conditions, **overrides)
    monkeypatch.setattr(window_flow, "_config", numeric_config)
    getattr(window_flow, case)(tmp_path)


def test_detrend_does_not_use_values_before_removed_point(tmp_path):
    from chem_ts_corr.preprocess import preprocess_frame, transform_frame
    config = _config(tmp_path, exclude_conditions=[rule("=", 25)])
    times = _write_input(config)
    source = pipeline.load_analysis_source_frame(config)
    cleaned = preprocess_frame(source, "target", None, 0.7)
    changed = cleaned.copy(deep=True)
    changed.loc[changed.index < times[25], "predictor"] += 10000
    for mode in ["lowpass", "lowpass_detrend", "lowpass_diff"]:
        original = transform_frame(cleaned, mode, 12)
        other = transform_frame(changed, mode, 12)
        pd.testing.assert_frame_equal(original.loc[times[26]:], other.loc[times[26]:])


def test_branches_freeze_conditions_and_old_context_defaults(tmp_path, monkeypatch):
    config = _config(tmp_path, preprocess_mode="lowpass", exclude_conditions=[rule("=", 20)])
    times = _write_input(config)
    captured = {}
    original = pipeline.analyze_initial_screening_branch_frame
    def capture(frame, config_arg, progress_callback=None):
        captured[config_arg.preprocess_mode] = frame.index.copy()
        return original(frame, config_arg, progress_callback=progress_callback)
    monkeypatch.setattr(pipeline, "analyze_initial_screening_branch_frame", capture)
    pipeline.run_initial_screening_workflow(config)
    assert captured["raw"].equals(captured["lowpass"])
    assert times[20] not in captured["raw"]
    context = pipeline._read_preprocessing_context(config.output_dir)
    assert context["exclude_conditions"] == config.exclude_conditions
    assert context["excluded_rows"] == 1
    pipeline.confirm_initial_screening_branch(config.output_dir, branch="raw")
    _, downstream = pipeline.prepare_downstream_analysis_context(
        config.output_dir, base_config=replace(config, exclude_conditions=[rule(">", 10)]),
        required_formal_files=[])
    assert downstream.exclude_conditions == config.exclude_conditions
    assert times[20] not in pipeline.load_analysis_source_frame(downstream).index
    web._write_run_config(config.output_dir, config, "a" * 32)
    assert web._read_run_config(config.output_dir).exclude_conditions == config.exclude_conditions
    stored = json.loads((config.output_dir / "run_config.json").read_text(encoding="utf-8"))
    stored.pop("exclude_conditions")
    (config.output_dir / "run_config.json").write_text(json.dumps(stored), encoding="utf-8")
    assert web._read_run_config(config.output_dir).exclude_conditions == []
    context.pop("exclude_conditions")
    pipeline._write_context(config.output_dir, context)
    assert pipeline._read_preprocessing_context(config.output_dir)["exclude_conditions"] == []


def test_all_excluded_fails(tmp_path):
    config = _config(tmp_path, exclude_conditions=[rule(">=", 0)])
    _write_input(config)
    with pytest.raises(ValueError, match="没有可用于分析的数据"):
        pipeline.load_analysis_source_frame(config)


def test_preview_matches_analysis_and_new_file_is_empty(tmp_path, monkeypatch):
    web.EXCLUDE_WINDOW_CONTEXTS.clear()
    file_id = "c" * 32
    _write_upload(tmp_path, monkeypatch, file_id)
    context = web._exclude_window_context(file_id, "time", "auto")
    context["exclude_windows"] = [{"start": str(context["frame"].index[1]),
                                   "end": str(context["frame"].index[3])}]
    conditions = [rule("<", 3), rule("=", 1)]
    response = _post(monkeypatch, web._preview_exclusions_response, file_id=file_id,
                     time_column="time", exclude_conditions=json.dumps(conditions))
    config = _config(tmp_path, input_path=web._resolve_upload(file_id),
                     exclude_windows=context["exclude_windows"], exclude_conditions=conditions)
    source = pipeline.load_analysis_source_frame(config)
    assert response["stats"]["excluded_rows"] == 4
    assert response["stats"]["remaining_rows"] == len(source)
    _write_upload(tmp_path, monkeypatch, "d" * 32)
    fresh = _post(monkeypatch, web._preview_exclusions_response, file_id="d" * 32,
                  time_column="time")
    assert fresh["stats"]["excluded_rows"] == 0
    web.EXCLUDE_WINDOW_CONTEXTS.clear()


def test_ui_position_and_operators():
    html = web.INDEX_HTML
    assert html.index("原始上传文件不会被修改。") < html.index('id="excludeConditions"') < html.index(
        '<div class="control-group-title">工况与残差控制</div>')
    assert 'step="any"' in html
    assert 'form.append("exclude_conditions", JSON.stringify(getExcludeConditions()))' in html


def test_analyze_snapshots_rules_and_rejects_invalid_before_task(tmp_path, monkeypatch):
    file_id = "e" * 32
    frame = _write_upload(tmp_path, monkeypatch, file_id)
    frame["predictor"] = range(len(frame))
    frame.to_csv(web._resolve_upload(file_id), index=False, encoding="utf-8-sig")
    configs = []
    class Thread:
        def __init__(self, *, target, args, daemon):
            configs.append(args[1])
        def start(self):
            pass
    monkeypatch.setattr(web.threading, "Thread", Thread)
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(web, "_cleanup_tasks_locked", lambda **kwargs: None)
    form = dict(file_id=file_id, time_column="time", target="target", resample_rule="")
    conditions = [rule("<", 2)]
    response = _post(monkeypatch, web._analyze_response, **form,
                     exclude_conditions=json.dumps(conditions))
    assert configs[0].exclude_conditions == conditions
    web.TASKS.pop(response["task_id"])
    for bad in [[rule(column="missing")], [rule(">=", 0)], [rule(value=np.inf)]]:
        with pytest.raises(ValueError):
            _post(monkeypatch, web._analyze_response, **form, exclude_conditions=json.dumps(bad))
    assert len(configs) == 1
    web.EXCLUDE_WINDOW_CONTEXTS.clear()


def test_preview_only_latest_success_or_error_updates_stats():
    import re
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required")
    script = web.INDEX_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    start = script.index("async function previewExclusions()")
    end = re.search(r"^}", script[start:], re.MULTILINE).end() + start
    harness = r'''
const assert = require('node:assert/strict');
let fileId = 'file', exclusionPreviewRevision = 0, conditions = [];
const nodes = {timeColumn:{value:'time'},excludeConditionStats:{textContent:''}};
const el = id => nodes[id];
const getExcludeConditions = () => conditions;
const pending = [];
const postForm = () => new Promise((resolve,reject)=>pending.push({resolve,reject}));
const stats = n => ({stats:{original_rows:100,excluded_rows:n,remaining_rows:100-n,excluded_ratio:n/100}});
'''
    assertions = r'''
(async()=>{
  for (const reverse of [true,false]) {
    const first = previewExclusions(), second = previewExclusions();
    const [a,b] = pending.splice(0);
    if(reverse){b.resolve(stats(20));await second;a.resolve(stats(10));await first;}
    else{a.resolve(stats(10));await first;b.resolve(stats(20));await second;}
    assert(nodes.excludeConditionStats.textContent.includes('20 / 100'));
  }
  const old = previewExclusions(), latest = previewExclusions();
  const [a,b] = pending.splice(0);b.resolve(stats(30));await latest;
  a.reject(new Error('stale error'));await old;
  assert(nodes.excludeConditionStats.textContent.includes('30 / 100'));
  const request = previewExclusions();const previous=pending.shift();
  fileId='';await previewExclusions();previous.resolve(stats(40));await request;
  assert.equal(nodes.excludeConditionStats.textContent,'请先上传并识别数据。');
  fileId='new';const failed=previewExclusions();pending.shift().reject(new Error('预览失败'));await failed;
  assert.equal(nodes.excludeConditionStats.textContent,'预览失败');
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run([node], input=harness + script[start:end] + assertions,
                            text=True, encoding="utf-8", capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
