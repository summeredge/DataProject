import inspect
import io
import re
import time

import pandas as pd
import pytest

from chem_ts_corr import web


def test_multipart_form_rejects_large_content_length_before_body_read():
    source = inspect.getsource(web._multipart_form)
    assert "MAX_REQUEST_BODY_BYTES" in source
    read_pos = source.index(".read(")
    limit_pos = source.index("MAX_REQUEST_BODY_BYTES")
    assert limit_pos < read_pos
    assert "content_length" in source

    class UnreadableBody(io.BytesIO):
        def read(self, *args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("request body was read before Content-Length validation")

    handler = type(
        "Handler",
        (),
        {
            "headers": {
                "Content-Length": str(web.MAX_REQUEST_BODY_BYTES + 1),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            "rfile": UnreadableBody(),
        },
    )()
    with pytest.raises(ValueError, match="上传文件过大"):
        web._multipart_form(handler)

    for invalid_length in ["not-an-int", "-1"]:
        handler.headers["Content-Length"] = invalid_length
        with pytest.raises(ValueError, match="Content-Length"):
            web._multipart_form(handler)


def test_resolve_upload_requires_uuid_hex_file_id():
    validator = getattr(web, "_validate_file_id", None)
    assert validator is not None
    valid = "0123456789abcdef0123456789abcdef"
    assert validator(valid) == valid
    assert validator(valid.upper()) == valid
    for invalid in ["*", "../abc", "abc?", "[abc]", "not-a-uuid", ""]:
        with pytest.raises(ValueError):
            validator(invalid)

    resolve_source = inspect.getsource(web._resolve_upload)
    assert "_validate_file_id" in resolve_source
    assert "glob(f\"{file_id}.*\")" not in resolve_source


def test_tasks_have_cleanup_policy_and_size_bound(monkeypatch):
    assert hasattr(web, "TASK_TTL_SECONDS")
    assert hasattr(web, "MAX_TASKS")
    cleanup = getattr(web, "_cleanup_tasks", None)
    assert cleanup is not None

    now = time.time()
    monkeypatch.setattr(web, "TASK_TTL_SECONDS", 60)
    monkeypatch.setattr(web, "MAX_TASKS", 3)
    with web.TASKS_LOCK:
        web.TASKS.clear()
        web.TASKS.update(
            {
                "old_done": {"status": "done", "updated_at": now - 120, "created_at": now - 130},
                "old_error": {"status": "error", "updated_at": now - 120, "created_at": now - 130},
                "running": {"status": "running", "updated_at": now - 120, "created_at": now - 130},
                "new_done": {"status": "done", "updated_at": now, "created_at": now},
            }
        )
    cleanup(now=now)
    with web.TASKS_LOCK:
        assert "old_done" not in web.TASKS
        assert "old_error" not in web.TASKS
        assert "running" in web.TASKS
        assert len(web.TASKS) <= web.MAX_TASKS
        web.TASKS.clear()


def test_secondary_scaled_frame_is_cached_per_run_config(monkeypatch, tmp_path):
    cache_clear = getattr(web, "_clear_scaled_frame_cache", None)
    assert cache_clear is not None
    cache_clear()

    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=20, freq="min"),
            "target": range(20),
            "x": range(20, 40),
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    config = web.AnalysisConfig(
        input_path=csv_path,
        time_column="time",
        target="target",
        output_dir=tmp_path / "out",
        min_valid_ratio=0.0,
    )
    calls = {"n": 0}
    original_loader = web.load_timeseries_csv

    def counting_loader(*args, **kwargs):
        calls["n"] += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(web, "load_timeseries_csv", counting_loader)
    first = web._scaled_frame_for_secondary(config)
    second = web._scaled_frame_for_secondary(config)
    assert calls["n"] == 1
    assert first.equals(second)
    assert first is not second


def test_pr_b_keeps_scope_to_web_module():
    source = inspect.getsource(web)
    required_markers = [
        "MAX_REQUEST_BODY_BYTES",
        "_validate_file_id",
        "_cleanup_tasks",
        "_clear_scaled_frame_cache",
    ]
    for marker in required_markers:
        assert marker in source
