from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
from http.client import HTTPException
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from chem_ts_corr import history, web


@pytest.fixture
def records(tmp_path, monkeypatch):
    uploads, runs = tmp_path / "uploads", tmp_path / "web_runs"
    uploads.mkdir()
    runs.mkdir()
    monkeypatch.setattr(web, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(web, "RUNS_DIR", runs)
    monkeypatch.setattr(web, "TASKS", {})
    for file_id in ["a" * 32, "b" * 32]:
        (uploads / f"{file_id}.csv").write_bytes(b"time,target\n1,2\n")
        (uploads / f"{file_id}.json").write_text(json.dumps({"original_filename": "历史.csv"}), encoding="utf-8")
    for run_id, file_id in [("1" * 32, "a" * 32), ("2" * 32, "a" * 32)]:
        path = runs / run_id
        path.mkdir()
        (path / "run_config.json").write_text(json.dumps({"file_id": file_id, "target": "target"}), encoding="utf-8")
        (path / "result.csv").write_bytes(b"saved-result")
    return uploads, runs


def cleanup(records, *, files=(), runs=(), mode="selected", execute=True, active=()):
    return history.cleanup_history(*records, mode=mode, file_ids=list(files), run_ids=list(runs),
                                   execute=execute, active=active)


def test_delete_analysis_keeps_upload_and_refreshes_associations(records):
    uploads, runs = records
    result = cleanup(records, runs=["1" * 32])
    assert result["complete"]
    assert result["deleted_run_ids"] == ["1" * 32]
    assert (uploads / f"{'a' * 32}.csv").exists()
    assert not (runs / ("1" * 32)).exists()
    remaining = history.query_history(*records)
    assert next(row for row in remaining["uploads"] if row["file_id"] == "a" * 32)["analysis_count"] == 1


def test_unreferenced_upload_and_metadata_deleted(records):
    result = cleanup(records, files=["b" * 32])
    assert result["complete"] and result["deleted_file_ids"] == ["b" * 32]
    assert not list(records[0].glob(f"{'b' * 32}.*"))


def test_reference_conflict_and_complete_batch(records):
    blocked = cleanup(records, files=["a" * 32], execute=False)
    assert not blocked["allowed"]
    assert blocked["conflicts"][0]["related_analysis_count"] == 2
    assert blocked["storage"]["total_size"] == 0
    partial = cleanup(records, files=["a" * 32], runs=["1" * 32])
    assert not partial["complete"] and not partial["deleted_file_ids"]
    assert (records[0] / f"{'a' * 32}.csv").exists()
    complete = cleanup(records, files=["a" * 32, "b" * 32], runs=["2" * 32])
    assert complete["complete"]
    assert len(complete["deleted_file_ids"]) == 2
    assert history.query_history(*records)["storage"]["total_size"] == 0


def test_preview_cancel_and_all_preserve_roots(records):
    before = {p: p.read_bytes() for root in records for p in root.rglob("*") if p.is_file()}
    preview = cleanup(records, mode="all", execute=False)
    assert preview["storage"] == history.query_history(*records)["storage"]
    assert all(p.read_bytes() == content for p, content in before.items())
    result = cleanup(records, mode="all")
    assert result["complete"] and result["released_size"] == preview["storage"]["total_size"]
    assert all(root.is_dir() and not list(root.iterdir()) for root in records)
    assert history.query_history(*records)["storage"]["total_size"] == 0
    (records[0] / f"{'c' * 32}.csv").write_text("time,target\n1,2", encoding="utf-8")
    assert web._resolve_upload("c" * 32).exists()


def test_execute_rechecks_new_reference_after_preview(records):
    assert cleanup(records, files=["b" * 32], execute=False)["allowed"]
    path = records[1] / ("3" * 32)
    path.mkdir()
    (path / "run_config.json").write_text(json.dumps({"file_id": "b" * 32}), encoding="utf-8")
    assert not cleanup(records, files=["b" * 32])["deleted_file_ids"]
    assert (records[0] / f"{'b' * 32}.csv").exists()


@pytest.mark.parametrize("identifier", ["../outside", "a" * 31, "A" * 32, "", "a" * 32 + "/x"])
def test_invalid_ids_rejected_without_deletion(records, identifier):
    before = history.query_history(*records)
    with pytest.raises(ValueError, match="Invalid history id"):
        cleanup(records, files=[identifier])
    assert history.query_history(*records) == before


def test_missing_unmanaged_and_broken_configuration(records):
    (records[0] / "unmanaged.txt").write_text("keep", encoding="utf-8")
    path = records[1] / ("3" * 32)
    path.mkdir()
    (path / "run_config.json").write_text("broken", encoding="utf-8")
    assert not cleanup(records, files=["b" * 32])["allowed"]
    missing = cleanup(records, runs=["f" * 32])
    assert not missing["complete"] and missing["conflicts"]
    result = cleanup(records, mode="all")
    assert not result["complete"]
    assert (records[0] / "unmanaged.txt").read_text(encoding="utf-8") == "keep"
    assert not list(records[1].iterdir())


def test_legacy_reference_uses_input_path(records):
    config = records[1] / ("1" * 32) / "run_config.json"
    config.write_text(json.dumps({"input_path": str(records[0] / f"{'b' * 32}.csv")}), encoding="utf-8")
    assert not cleanup(records, files=["b" * 32])["allowed"]


def test_failed_run_deletion_does_not_delete_its_input(records, monkeypatch):
    real_remove = shutil.rmtree

    def denied(path):
        if path.name == "1" * 32:
            raise PermissionError("locked result")
        real_remove(path)

    monkeypatch.setattr(history.shutil, "rmtree", denied)
    result = cleanup(records, files=["a" * 32, "b" * 32], runs=["1" * 32, "2" * 32])
    assert not result["complete"]
    assert result["deleted_run_ids"] == ["2" * 32]
    assert result["deleted_file_ids"] == ["b" * 32]
    assert (records[0] / f"{'a' * 32}.csv").exists()
    assert any("locked result" in row["reason"] for row in result["conflicts"])
    assert result["released_size"] > 0


def test_links_cannot_delete_external_files(records, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    saved = external / "keep.txt"
    saved.write_text("keep", encoding="utf-8")
    link = records[1] / ("1" * 32) / "outside"
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires Windows privileges")
    result = cleanup(records, runs=["1" * 32])
    assert not result["complete"]
    assert saved.read_text(encoding="utf-8") == "keep"
    assert link.is_symlink()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction safety")
@pytest.mark.parametrize("location", ["nested", "run", "root"])
def test_windows_junctions_are_rejected(records, tmp_path, location):
    import _winapi

    external = tmp_path / "external"
    external.mkdir()
    saved = external / "keep.txt"
    saved.write_text("keep", encoding="utf-8")
    identifier = "3" * 32
    if location == "nested":
        link = records[1] / ("1" * 32) / "junction"
        identifier = "1" * 32
    elif location == "run":
        link = records[1] / identifier
    else:
        link = tmp_path / "linked-runs"
        records = (records[0], link)
        (external / identifier).mkdir()
    _winapi.CreateJunction(str(external), str(link))
    result = cleanup(records, runs=[identifier])
    assert not result["complete"]
    assert saved.read_text(encoding="utf-8") == "keep"
    assert link.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction history scan")
def test_history_scan_and_storage_do_not_follow_windows_junctions(records, tmp_path):
    import _winapi

    before = history.query_history(*records)
    external = tmp_path / "external"
    external.mkdir()
    (external / "large.csv").write_bytes(b"outside" * 100)
    (external / "run_config.json").write_text(json.dumps({"target": "outside"}), encoding="utf-8")
    _winapi.CreateJunction(str(external), str(records[1] / ("3" * 32)))
    _winapi.CreateJunction(str(external), str(records[1] / ("1" * 32) / "nested"))
    assert history.query_history(*records) == before
    linked_root = tmp_path / "linked-root"
    _winapi.CreateJunction(str(external), str(linked_root))
    assert history.disk_size(linked_root) == 0
    assert not history.query_history(records[0], linked_root)["analyses"]
    assert (external / "large.csv").read_bytes() == b"outside" * 100


def test_metadata_failure_reports_partial_and_lost_input(records, monkeypatch):
    real_unlink = type(records[0]).unlink

    def denied(path, *args, **kwargs):
        if path.name == f"{'b' * 32}.json":
            raise PermissionError("metadata locked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(records[0]), "unlink", denied)
    before = history.query_history(*records)["storage"]["total_size"]
    result = cleanup(records, files=["b" * 32])
    assert not result["complete"] and result["deleted_file_ids"] == ["b" * 32]
    assert (records[0] / f"{'b' * 32}.json").exists()
    assert not (records[0] / f"{'b' * 32}.csv").exists()
    assert result["released_size"] == before - history.query_history(*records)["storage"]["total_size"]


@pytest.mark.parametrize("suffix", [".csv", ".json"])
def test_upload_and_metadata_links_preserve_external_file(records, tmp_path, suffix):
    external = tmp_path / "keep.txt"
    external.write_text("outside", encoding="utf-8")
    path = records[0] / f"{'b' * 32}{suffix}"
    path.unlink()
    try:
        path.symlink_to(external)
    except OSError:
        pytest.skip("Creating symlinks requires Windows privileges")
    result = cleanup(records, mode="all")
    assert not result["complete"]
    assert path.is_symlink() and external.read_text(encoding="utf-8") == "outside"


def test_active_tasks_protect_dependencies_but_allow_other_records(records, monkeypatch):
    monkeypatch.setattr(web, "_multipart_form", lambda handler: handler)
    web.TASKS["active"] = {"status": "running", "file_id": "a" * 32, "run_id": "1" * 32}
    result = web._history_cleanup_response({"mode": "selected", "phase": "execute",
                                            "file_ids": ",".join(["a" * 32, "b" * 32]), "run_ids": "1" * 32})
    assert result["deleted_file_ids"] == ["b" * 32]
    assert not result["deleted_run_ids"]
    assert not web._history_cleanup_response({"mode": "all"})["allowed"]
    web.TASKS["active"]["status"] = "done"
    assert web._history_cleanup_response({"mode": "all", "phase": "execute"})["complete"]


def test_synchronous_operation_registration_and_cleanup_lock(records, monkeypatch):
    monkeypatch.setattr(web, "_multipart_form", lambda handler: handler)
    entered, release = threading.Event(), threading.Event()

    @web._protect_history_operation
    def operation(handler, *, form):
        entered.set()
        assert release.wait(5)

    thread = threading.Thread(target=operation, args=({"run_id": "1" * 32},))
    thread.start()
    try:
        assert entered.wait(5)
        result = web._history_cleanup_response({"mode": "selected", "phase": "execute", "run_ids": "1" * 32})
        assert not result["allowed"]
        assert not web._history_cleanup_response({"mode": "all"})["allowed"]
    finally:
        release.set()
        thread.join(5)
    assert not web.TASKS
    entered.clear()
    release.set()
    with web.TASKS_LOCK:
        thread = threading.Thread(target=operation, args=({"run_id": "1" * 32},))
        thread.start()
        assert not entered.wait(0.05)
    thread.join(5)
    assert entered.is_set() and not web.TASKS


def test_http_preview_and_execute(records):
    server = ThreadingHTTPServer(("127.0.0.1", 0), web._Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def request(phase):
            body = urlencode({"mode": "all", "phase": phase}).encode()
            with urlopen(Request(f"http://127.0.0.1:{server.server_port}/api/history/cleanup", data=body), timeout=5) as response:
                return json.load(response)
        assert request("preview")["allowed"]
        assert history.query_history(*records)["storage"]["analysis_count"] == 2
        assert request("execute")["complete"]
        assert not history.query_history(*records)["uploads"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.mark.parametrize("route,callback,dependency", [
    ("/api/columns?file_id=" + "b" * 32, "_columns_response", "upload"),
    ("/api/columns?file_id=" + "b" * 32 + "&run_id=" + "1" * 32, "_columns_response", "upload"),
    ("/api/trend?file_id=" + "b" * 32, "_trend_response", "upload"),
    ("/api/scatter_matrix?file_id=" + "b" * 32, "_scatter_matrix_response", "upload"),
    ("/api/lag_profile?run_id=" + "1" * 32, "_lag_profile_response", "analysis"),
    ("/api/result?run_id=" + "1" * 32, "_build_result_payload", "analysis"),
    ("/api/result?run_id=" + "1" * 32 + "&restore=1", "_restore_run_payload", "analysis"),
    ("/api/result?task_id=saved-task", "_build_result_payload", "analysis"),
    ("/download?run_id=" + "1" * 32 + "&file=ranked_features.csv", "_send_download", "analysis"),
])
def test_get_reads_protect_dependencies_and_allow_unrelated_cleanup(records, monkeypatch, route, callback, dependency):
    entered, release = threading.Event(), threading.Event()
    path = records[0] / f"{'b' * 32}.csv" if dependency == "upload" else records[1] / ("1" * 32) / "ranked_features.csv"
    if dependency == "analysis":
        path.write_bytes(b"saved-ranking")
    web.TASKS["saved-task"] = {"status": "done", "run_id": "1" * 32, "result": {"run_id": "1" * 32}}
    monkeypatch.setattr(web, "_read_run_config", lambda directory: object())

    def blocked_read(*args):
        entered.set()
        assert release.wait(5)
        assert path.read_bytes()
        return {}

    if callback == "_send_download":
        original = web._Handler._send_download

        def download(handler, run_id, filename):
            blocked_read()
            return original(handler, run_id, filename)

        monkeypatch.setattr(web._Handler, callback, download)
    else:
        monkeypatch.setattr(web, callback, blocked_read)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web._Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    responses = []
    errors = []

    def read_request():
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}{route}", timeout=5) as response:
                responses.append(response.read())
        except (OSError, HTTPException) as exc:
            errors.append(exc)

    reader = threading.Thread(target=read_request)
    reader.start()
    try:
        assert entered.wait(5)
        monkeypatch.setattr(web, "_multipart_form", lambda handler: handler)
        selected = {"mode": "selected", "phase": "execute"}
        selected["file_ids" if dependency == "upload" else "run_ids"] = "b" * 32 if dependency == "upload" else "1" * 32
        assert not web._history_cleanup_response(selected)["allowed"]
        assert not web._history_cleanup_response({"mode": "all"})["allowed"]
        unrelated = {"mode": "selected", "phase": "execute"}
        unrelated["run_ids" if dependency == "upload" else "file_ids"] = "2" * 32 if dependency == "upload" else "b" * 32
        assert web._history_cleanup_response(unrelated)["complete"]
        assert path.exists()
    finally:
        release.set()
        reader.join(5)
        server.shutdown()
        server.server_close()
        serving.join(5)
    assert responses and not errors and not reader.is_alive()
    assert not any(task.get("status") == "running" for task in web.TASKS.values())
    assert web._history_cleanup_response(selected)["complete"]


@pytest.mark.parametrize("failure", [ValueError("read failed"), BrokenPipeError("disconnected"), ConnectionResetError("reset")])
def test_get_failure_and_disconnect_release_activity(records, monkeypatch, failure):
    from types import SimpleNamespace

    def fail_read(params):
        assert any(task.get("status") == "running" for task in web.TASKS.values())
        raise failure

    errors = []
    monkeypatch.setattr(web, "_trend_response", fail_read)
    handler = SimpleNamespace(path="/api/trend?file_id=" + "b" * 32,
                              _send_json=lambda payload, **kwargs: errors.append(payload))
    web._Handler.do_GET(handler)
    assert not web.TASKS
    assert bool(errors) == isinstance(failure, ValueError)
    assert cleanup(records, files=["b" * 32])["complete"]


def test_activity_registration_precedes_metadata_read_and_does_not_hold_lock(records, monkeypatch):
    original = web.read_metadata

    def read_config(path):
        assert web.TASKS_LOCK.acquire(blocking=False)
        try:
            assert any(task.get("run_id") == "1" * 32 for task in web.TASKS.values())
        finally:
            web.TASKS_LOCK.release()
        return original(path)

    monkeypatch.setattr(web, "read_metadata", read_config)
    with web._history_activity("/api/result", run_id="1" * 32):
        assert web.TASKS_LOCK.acquire(blocking=False)
        web.TASKS_LOCK.release()
    assert not web.TASKS


def test_frontend_cancel_execute_partial_and_reset():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for frontend behavior tests")
    script = web.INDEX_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    functions = []
    for name in ["cleanupHistory", "historySize", "renderHistorySelection"]:
        match = re.search(rf"^(?:async )?function {name}\(", script, re.MULTILINE)
        end = re.search(r"^}", script[match.end():], re.MULTILINE)
        functions.append(script[match.start():match.end() + end.end()])
    harness = r'''
const assert=require('node:assert/strict');
const historySelectedFiles=new Set(['a']),historySelectedRuns=new Set(['r']);
let dataSelectionLocks=0,fileId='a',currentRunId='r',confirmed=false,requests=[],resets=0,refreshes=0;
const elements=new Map();
function el(id){if(!elements.has(id))elements.set(id,{textContent:'',disabled:false});return elements.get(id);}
function lockDataSelection(delta){dataSelectionLocks+=delta;}
function setStatus(){}
function reset(){resets++;fileId='';currentRunId='';}
async function refreshHistory(){refreshes++;}
const window={confirm(message){assert(message.includes('预计释放'));return confirmed;}};
let result={complete:false,deleted_file_ids:['a'],deleted_run_ids:[],released_size:10,conflicts:[{id:'r',reason:'操作正在执行'}]};
async function postForm(url,form){requests.push(form.get('phase'));return form.get('phase')==='preview'?{
  allowed:true,storage:{upload_count:1,analysis_count:1,upload_size:5,analysis_size:5,total_size:10},conflicts:[]
}:result;}
function node(){return {checked:false,disabled:false,events:{},setAttribute(){},addEventListener(name,fn){this.events[name]=fn;}};}
const document={createElement:node};
const heading={replaceChildren(value){this.box=value;}};
const rows=[0,1].map(()=>({firstElementChild:{replaceChildren(value){this.box=value;}}}));
elements.set('table',{querySelector(){return heading;},querySelectorAll(){return rows;}});
'''
    checks = r'''
(async()=>{
  const selected=new Set();const ids=['a'.repeat(32),'b'.repeat(32)];
  renderHistorySelection('table',ids.map(file_id=>({file_id})),'file_id',selected);
  heading.box.checked=true;heading.box.events.change();assert.equal(selected.size,2);
  rows[0].firstElementChild.box.checked=false;rows[0].firstElementChild.box.events.change();
  assert.equal(selected.size,1);assert.equal(heading.box.indeterminate,true);
  await cleanupHistory('selected');assert.deepEqual(requests,['preview']);assert.equal(resets,0);
  confirmed=true;await cleanupHistory('selected');assert.deepEqual(requests,['preview','preview','execute']);
  assert.equal(resets,1);assert.equal(refreshes,1);assert(!historySelectedFiles.has('a'));assert(historySelectedRuns.has('r'));
  assert(el('historyStatus').textContent.includes('未全部完成'));assert(el('historyStatus').textContent.includes('操作正在执行'));
  dataSelectionLocks=1;const before=requests.length;await cleanupHistory('all');assert.equal(requests.length,before);
  dataSelectionLocks=0;confirmed=false;await cleanupHistory('all');assert.equal(requests.at(-1),'preview');
  assert.equal(dataSelectionLocks,0);
  assert(el('historyStatus').textContent.includes('已取消'));
  confirmed=true;fileId='retained';currentRunId='r';
  result={complete:true,deleted_file_ids:[],deleted_run_ids:['r'],released_size:5,conflicts:[]};
  await cleanupHistory('selected');assert.equal(resets,2);assert.equal(refreshes,2);
  assert.equal(historySelectedRuns.size,0);assert(el('historyStatus').textContent.includes('清理完成'));
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run([node], input=harness + "\n".join(functions) + checks,
                            text=True, encoding="utf-8", capture_output=True, timeout=15, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
