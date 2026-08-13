import pandas as pd
import pytest

import chem_ts_corr.web as web
from chem_ts_corr.config import AnalysisConfig


@pytest.fixture(autouse=True)
def clear_tasks():
    with web.TASKS_LOCK:
        web.TASKS.clear()
    yield
    with web.TASKS_LOCK:
        web.TASKS.clear()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5", "5min"),
        (" 10 ", "10min"),
        (5, "5min"),
        ("5min", "5min"),
        (" 10min ", "10min"),
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_normalize_minute_resample_rule(value, expected):
    assert web._normalize_minute_resample_rule(value) == expected


@pytest.mark.parametrize(
    "value",
    [0, -1, 1.5, "0", "-1", "1.5", "abc", "5h", "300s", "5m", "min", True, False],
)
def test_normalize_minute_resample_rule_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="^重采样间隔必须是大于 0 的整数分钟$"):
        web._normalize_minute_resample_rule(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_normalize_minute_resample_rule_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="^重采样间隔必须是大于 0 的整数分钟$"):
        web._normalize_minute_resample_rule(value)


@pytest.mark.parametrize(("raw_rule", "expected"), [("5", "5min"), ("", None), ("5min", "5min")])
def test_analyze_response_normalizes_resample_before_starting_thread(
    raw_rule, expected, monkeypatch, tmp_path
):
    (tmp_path / "input.csv").write_text(
        "time,target,x\n2025-01-01 00:00:00,1,2\n",
        encoding="utf-8",
    )
    form = {
        "file_id": "file-1",
        "time_column": "time",
        "target": "target",
        "resample_rule": raw_rule,
    }
    threads = []

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            self.args = args
            self.started = False
            threads.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(web, "_multipart_form", lambda handler: form)
    monkeypatch.setattr(web, "_resolve_upload", lambda file_id: tmp_path / "input.csv")
    monkeypatch.setattr(web, "_resolve_encoding", lambda path, encoding: encoding)
    monkeypatch.setattr(web.threading, "Thread", FakeThread)
    monkeypatch.setattr(web, "_cleanup_tasks_locked", lambda **kwargs: None)
    web._analyze_response(object())

    assert len(threads) == 1
    assert threads[0].started
    assert threads[0].args[1].resample_rule == expected


@pytest.mark.parametrize("raw_rule", ["0", "-1", "1.5", "abc", "5h", "300s", "5m"])
def test_invalid_primary_resample_fails_before_thread_or_analysis(
    raw_rule, monkeypatch
):
    form = {
        "file_id": "file-1",
        "time_column": "time",
        "target": "target",
        "resample_rule": raw_rule,
    }
    monkeypatch.setattr(web, "_multipart_form", lambda handler: form)
    monkeypatch.setattr(
        web.threading,
        "Thread",
        lambda **kwargs: pytest.fail("invalid input must not create a thread"),
    )
    monkeypatch.setattr(
        web,
        "run_initial_screening_workflow",
        lambda *args, **kwargs: pytest.fail("invalid input must not run analysis"),
    )

    with pytest.raises(ValueError, match="^重采样间隔必须是大于 0 的整数分钟$"):
        web._analyze_response(object())


@pytest.mark.parametrize(
    "endpoint_name",
    ["_run_enhanced_screening_response", "_run_granger_response", "_run_model_response"],
)
def test_stale_secondary_resample_params_are_ignored_and_formal_runner_is_called(
    endpoint_name, monkeypatch, tmp_path
):
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path / "run")
    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda handler: {
            "run_id": "run",
            "secondary_resample_mode": "custom",
            "secondary_resample_rule": "",
            "secondary_include_variables": "stale_whitelist",
        },
    )
    monkeypatch.setattr(web, "_resolve_run_dir", lambda run_id: tmp_path / "run")
    monkeypatch.setattr(web, "_read_run_config", lambda output_dir: config)
    captured: dict[str, object] = {}

    def fake_runner(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        return {}

    if endpoint_name == "_run_enhanced_screening_response":
        monkeypatch.setattr(web, "run_enhanced_screening_for_active_branch", fake_runner)
    elif endpoint_name == "_run_granger_response":
        monkeypatch.setattr(web, "run_granger_for_active_branch", fake_runner)
    else:
        monkeypatch.setattr(web, "run_model_for_active_branch", fake_runner)
    monkeypatch.setattr(web, "_safe_read_result_csv", lambda path: pd.DataFrame())
    monkeypatch.setattr(web, "_download_links", lambda *args, **kwargs: [])

    getattr(web, endpoint_name)(object())

    assert captured["output_dir"] == tmp_path / "run"


def test_resample_inputs_are_integer_minute_controls():
    primary = web.INDEX_HTML.split('id="resampleRule"', 1)[1].split(">", 1)[0]

    assert 'type="number"' in primary
    assert 'min="1"' in primary
    assert 'step="1"' in primary
    assert 'inputmode="numeric"' in primary
    assert 'placeholder="可留空，例如 5"' in primary
    assert "例如 5min" not in web.INDEX_HTML
    assert "例如 2min / 5min" not in web.INDEX_HTML
    assert 'id="secondaryResampleRule"' not in web.INDEX_HTML
