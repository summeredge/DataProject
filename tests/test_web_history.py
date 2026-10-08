from __future__ import annotations

import json
import math
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from chem_ts_corr import web
from chem_ts_corr.history import query_history
from chem_ts_corr.pipeline import _load_run_config


def test_storage_overview_uses_three_column_table_and_api_values():
    refresh = web.INDEX_HTML.split("async function refreshHistory() {", 1)[1].split(
        "function activateTab", 1
    )[0]
    assert '<div id="historyStorage"></div>' in web.INDEX_HTML
    assert 'renderHistoryTable("historyStorage", ["类别", "数量", "磁盘占用"], [' in refresh
    assert '["上传数据", `${storage.upload_count} 个文件`, historySize(storage.upload_size)]' in refresh
    assert '["历史分析", `${storage.analysis_count} 条记录`, historySize(storage.analysis_size)]' in refresh
    assert '["合计", "—", historySize(storage.total_size)]' in refresh
    assert "metric-card" not in refresh
    assert 'class="metric-card"' in web.INDEX_HTML


def test_storage_table_is_compact_right_aligned_and_keeps_refresh_entrypoints():
    assert "#historyStorage table { min-width:0; width:100%; }" in web.INDEX_HTML
    assert "#historyStorage th:nth-child(n+2), #historyStorage td:nth-child(n+2) { text-align:right;" in web.INDEX_HTML
    assert 'el("refreshHistory").addEventListener("click", refreshHistory)' in web.INDEX_HTML
    assert 'if (tabId === "historyTab") refreshHistory();' in web.INDEX_HTML
    assert 'fetch("/api/history", { cache: "no-store" })' in web.INDEX_HTML


@pytest.fixture
def history_dirs(tmp_path, monkeypatch):
    uploads, runs = tmp_path / "uploads", tmp_path / "web_runs"
    uploads.mkdir()
    runs.mkdir()
    monkeypatch.setattr(web, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(web, "RUNS_DIR", runs)
    return uploads, runs


def test_empty_history_does_not_create_directories(tmp_path):
    result = query_history(tmp_path / "uploads", tmp_path / "runs")
    assert result == {"storage": {"upload_count": 0, "analysis_count": 0,
                                  "upload_size": 0, "analysis_size": 0, "total_size": 0},
                      "uploads": [], "analyses": []}
    assert not list(tmp_path.iterdir())


def test_legacy_incomplete_association_sorting_and_disk_size(history_dirs):
    uploads, runs = history_dirs
    file_id = "a" * 32
    (uploads / f"{file_id}.csv").write_bytes(b"time,target\n1,2\n")
    (uploads / f"{file_id}.json").write_text("broken", encoding="utf-8")
    for index, date in [(1, "2025-01-01T00:00:00+00:00"), (2, "2025-02-01T00:00:00+00:00")]:
        directory = runs / (str(index) * 32)
        directory.mkdir()
        (directory / "run_config.json").write_text(json.dumps({
            "file_id": file_id, "target": "温度", "created_at": date,
        }), encoding="utf-8")
        (directory / "nested").mkdir()
        (directory / "nested" / "result.csv").write_bytes(b"123")
    broken = runs / ("b" * 32)
    broken.mkdir()
    (broken / "run_config.json").write_text("[]", encoding="utf-8")
    (runs / ("c" * 32)).mkdir()
    missing = runs / ("d" * 32)
    missing.mkdir()
    (missing / "run_config.json").write_text(json.dumps({"file_id": "e" * 32, "target": "压力"}))
    before = {p: p.read_bytes() for root in history_dirs for p in root.rglob("*") if p.is_file()}
    result = query_history(*history_dirs)
    upload = result["uploads"][0]
    assert upload["original_filename"] is None
    assert upload["time_source"] == "filesystem"
    assert upload["analysis_count"] == 2
    assert set(upload["run_ids"]) == {"1" * 32, "2" * 32}
    associated = [item for item in result["analyses"] if item["file_id"] == file_id]
    assert [item["run_id"] for item in associated] == ["2" * 32, "1" * 32]
    assert all(item["target"] == "温度" and item["original_filename"] is None for item in associated)
    assert next(item for item in result["analyses"] if item["run_id"] == "d" * 32)["upload_exists"] is False
    assert result["storage"]["analysis_count"] == 5
    for root, key in [(uploads, "upload_size"), (runs, "analysis_size")]:
        assert result["storage"][key] == sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    assert result["storage"]["total_size"] == sum(len(content) for content in before.values())
    assert before == {p: p.read_bytes() for root in history_dirs for p in root.rglob("*") if p.is_file()}


def test_upload_http_persistence_repeated_runs_and_service_restart(history_dirs):
    uploads, runs = history_dirs
    def start():
        server = ThreadingHTTPServer(("127.0.0.1", 0), web._Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, f"http://127.0.0.1:{server.server_port}"

    server, thread, url = start()
    try:
        raw = b"time,target,feature\n2025-01-01,1,2\n"
        body = ('--boundary\r\nContent-Disposition: form-data; name="file"; filename="工艺数据.csv"\r\nContent-Type: text/csv\r\n\r\n').encode() + raw + b"\r\n--boundary--\r\n"
        with urlopen(Request(url + "/api/upload", data=body, headers={"Content-Type": "multipart/form-data; boundary=boundary"})) as response:
            uploaded = json.load(response)
        assert set(uploaded) == {"file_id", "filename"}
        file_id = uploaded["file_id"]
        assert uploaded["filename"] == "工艺数据.csv"
        assert (uploads / f"{file_id}.csv").read_bytes() == raw
        metadata = json.loads((uploads / f"{file_id}.json").read_text(encoding="utf-8"))
        assert metadata["original_filename"] == "工艺数据.csv"
        assert metadata["file_size"] == len(raw)
        for index in [1, 2]:
            directory = runs / (str(index) * 32)
            config = web.AnalysisConfig(input_path=uploads / f"{file_id}.csv", time_column="time", target="target", output_dir=directory)
            web._write_run_config(directory, config, file_id)
            assert web._read_run_config(directory) == config
            assert _load_run_config(directory) == config
        with urlopen(url + "/api/history") as response:
            result = json.load(response)
        assert result["uploads"][0]["analysis_count"] == 2
        assert result["uploads"][0]["time_source"] == "metadata"
        assert all(item["original_filename"] == "工艺数据.csv" and item["time_source"] == "metadata" for item in result["analyses"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    server, thread, url = start()
    try:
        with urlopen(url + "/api/history") as response:
            assert json.load(response) == result
        (runs / ("3" * 32)).mkdir()
        with urlopen(url + "/api/history") as response:
            assert json.load(response)["storage"]["analysis_count"] == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_upload_time_sorting_and_invalid_metadata_fields(history_dirs):
    uploads, _ = history_dirs
    for file_id, date in [("1" * 32, "2025-01-01T00:00:00+00:00"),
                          ("2" * 32, "2025-02-01T00:00:00+00:00")]:
        path = uploads / f"{file_id}.xlsx"
        path.write_bytes(b"data")
        path.with_suffix(".json").write_text(json.dumps({
            "uploaded_at": date, "original_filename": "数据.xlsx", "file_size": 999,
        }), encoding="utf-8")
    result = query_history(*history_dirs)
    assert [item["file_id"] for item in result["uploads"]] == ["2" * 32, "1" * 32]
    assert all(item["file_size"] == 4 for item in result["uploads"])
    metadata = uploads / ("1" * 32 + ".json")
    metadata.write_text(json.dumps({"uploaded_at": "invalid", "original_filename": []}))
    item = next(item for item in query_history(*history_dirs)["uploads"] if item["file_id"] == "1" * 32)
    assert item["time_source"] == "filesystem" and item["original_filename"] is None
    metadata.write_bytes(b"\xff\xfe")
    assert len(query_history(*history_dirs)["uploads"]) == 2


def test_real_screening_adds_independent_history_records(history_dirs, monkeypatch):
    uploads, runs = history_dirs
    file_id = "f" * 32
    source = uploads / f"{file_id}.csv"
    source.write_text("time,target,feature\n" + "\n".join(
        f"2025-01-01 {index // 60:02d}:{index % 60:02d}:00,{math.sin(index / 5)},{math.sin((index - 1) / 5)}"
        for index in range(120)
    ), encoding="utf-8")
    source.with_suffix(".json").write_text(json.dumps({"original_filename": "装置.csv"}), encoding="utf-8")
    monkeypatch.setattr(web, "TASKS", {})
    for index in [1, 2]:
        task_id = str(index)
        web.TASKS[task_id] = {"status": "running"}
        config = web.AnalysisConfig(input_path=source, time_column="time", target="target",
                                    output_dir=runs / (str(index) * 32), max_lag=2, top_k=1)
        web._analyze_task(task_id, config, file_id)
        assert web.TASKS[task_id]["status"] == "done", web.TASKS[task_id]
        assert (config.output_dir / "ranked_features.csv").exists()
    result = query_history(*history_dirs)
    assert result["uploads"][0]["analysis_count"] == 2
    assert len(result["analyses"]) == 2
    assert all(item["original_filename"] == "装置.csv" and item["target"] == "target" for item in result["analyses"])
