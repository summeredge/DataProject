from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
START_APP = ROOT / "start_app.bat"


def _start_app_text() -> str:
    return START_APP.read_text(encoding="utf-8")


def _non_echo_command_text() -> str:
    lines = []
    for line in _start_app_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("echo "):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def test_start_app_does_not_import_heavy_model_packages_on_startup():
    text = _start_app_text()
    assert 'import sklearn, shap' not in text
    assert 'import shap' not in text
    assert 'import sklearn' not in text


def test_start_app_does_not_auto_install_optional_model_packages_on_startup():
    text = _start_app_text()
    assert 'pip install scikit-learn shap' not in text
    assert 'pip install sklearn shap' not in text
    assert 'Installing model explanation packages' not in text


def test_start_app_uses_lightweight_optional_dependency_check_or_clear_message():
    text = _start_app_text()
    assert 'importlib.util.find_spec' in text
    assert 'optional' in text.lower() or '可选' in text


def test_start_app_required_dependency_check_uses_find_spec_not_imports():
    text = _start_app_text()
    assert 'import pandas, numpy' not in text
    assert 'import numpy, pandas' not in text
    assert re.search(r"find_spec\(['\"]pandas['\"]\)", text)
    assert re.search(r"find_spec\(['\"]numpy['\"]\)", text)


def test_start_app_does_not_auto_install_project_on_startup():
    text = _start_app_text()
    command_text = _non_echo_command_text()
    assert 'Installing required packages' not in text
    assert 'pip install -e .' not in command_text


def test_start_app_missing_required_dependencies_instructs_manual_install():
    text = _start_app_text()
    assert 'Required Python packages are missing' in text
    assert 'Please run' in text
    assert 'pip install -e .' in text


def test_start_app_still_uses_selected_python_command_for_checks():
    text = _start_app_text()
    assert '%PYTHON_CMD%' in text
    assert 'Checking Python dependencies' in text
    assert 'Starting local web app' in text


def test_start_app_uses_dynamic_port_and_python_browser_launch():
    text = _start_app_text()
    assert re.search(r"cli serve --host 127\.0\.0\.1 --port 0", text)
    assert "--no-open" not in text
    assert "APP_URL" not in text
    assert "Invoke-WebRequest" not in text


def test_serve_command_defaults_to_dynamic_port():
    from chem_ts_corr.cli import build_parser

    assert build_parser().parse_args(["serve"]).port == 0
    assert build_parser().parse_args(["serve", "--port", "8765"]).port == 8765


def test_web_server_uses_actual_bound_port_for_url(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    from chem_ts_corr import web

    opened_urls = []

    class StubServer:
        server_port = 52341

        def __init__(self, address, _handler):
            assert address == ("127.0.0.1", 0)

        def serve_forever(self):
            pass

    class ImmediateTimer:
        def __init__(self, _delay, callback):
            self.callback = callback

        def start(self):
            self.callback()

    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "ThreadingHTTPServer", StubServer)
    monkeypatch.setattr(web, "threading", SimpleNamespace(Timer=ImmediateTimer))
    monkeypatch.setattr(web.webbrowser, "open", opened_urls.append)

    web.run_server()

    assert opened_urls == ["http://127.0.0.1:52341"]
    assert "http://127.0.0.1:52341" in capsys.readouterr().out
    assert (tmp_path / "runs").is_dir()
    assert (tmp_path / "uploads").is_dir()

