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
        "run_analysis",
        lambda *args, **kwargs: pytest.fail("invalid input must not run analysis"),
    )

    with pytest.raises(ValueError, match="^重采样间隔必须是大于 0 的整数分钟$"):
        web._analyze_response(object())


@pytest.mark.parametrize(
    "endpoint_name",
    ["_run_enhanced_screening_response", "_run_granger_response", "_run_model_response"],
)
def test_empty_custom_secondary_resample_fails_before_validation_work(
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
        },
    )
    monkeypatch.setattr(web, "_resolve_run_dir", lambda run_id: tmp_path / "run")
    monkeypatch.setattr(web, "_read_run_config", lambda output_dir: config)
    monkeypatch.setattr(
        web,
        "_safe_read_result_csv",
        lambda path: pytest.fail("validation data must not be read after invalid input"),
    )

    with pytest.raises(ValueError, match="^重采样间隔必须是大于 0 的整数分钟$"):
        getattr(web, endpoint_name)(object())


def test_resample_inputs_are_integer_minute_controls():
    primary = web.INDEX_HTML.split('id="resampleRule"', 1)[1].split(">", 1)[0]
    secondary = web.INDEX_HTML.split('id="secondaryResampleRule"', 1)[1].split(">", 1)[0]

    for control in [primary, secondary]:
        assert 'type="number"' in control
        assert 'min="1"' in control
        assert 'step="1"' in control
        assert 'inputmode="numeric"' in control
    assert 'placeholder="可留空，例如 5"' in primary
    assert "例如 5min" not in web.INDEX_HTML
    assert "例如 2min / 5min" not in web.INDEX_HTML
