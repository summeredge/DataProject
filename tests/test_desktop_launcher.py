import sys
import socket
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

import chem_ts_corr.desktop as desktop


class _Process:
    pid = 12345

    def poll(self):
        return None


def test_available_port_is_loopback_only():
    assert isinstance(desktop._available_port(), int)
    assert desktop.HOST == "127.0.0.1"


def test_frozen_launcher_starts_packaged_service_mode(monkeypatch, tmp_path):
    captured = {}

    class Process:
        pass

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "C:/ChemTsCorr/ChemTsCorr.exe")
    monkeypatch.setattr(desktop, "SERVICE_LOG_PATH", tmp_path / "desktop-launcher.log")
    monkeypatch.setattr(
        desktop.subprocess,
        "Popen",
        lambda command, **kwargs: captured.update(command=command, kwargs=kwargs) or Process(),
    )

    desktop._start_service(43210)

    assert captured["command"] == [
        "C:/ChemTsCorr/ChemTsCorr.exe",
        "--desktop-service",
        "--host",
        "127.0.0.1",
        "--port",
        "43210",
    ]


def test_main_stops_service_after_window_closes(monkeypatch):
    calls = []
    webview = SimpleNamespace(
        create_window=lambda *args, **kwargs: calls.append((args, kwargs)),
        start=lambda: calls.append(("start", {})),
    )
    process = _Process()
    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setattr(desktop, "_available_port", lambda: 43210)
    monkeypatch.setattr(desktop, "_start_service", lambda port: process)
    monkeypatch.setattr(desktop, "_wait_until_ready", lambda url, child: calls.append((url, child)))
    monkeypatch.setattr(desktop, "_stop_service", lambda child: calls.append(("stop", child)))

    desktop.main()

    assert calls == [
        ("http://127.0.0.1:43210/", process),
        (("化工时序相关性分析", "http://127.0.0.1:43210/"), {"width": 1440, "height": 900}),
        ("start", {}),
        ("stop", process),
    ]


def test_main_stops_service_before_showing_startup_error(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace())
    monkeypatch.setattr(desktop, "_available_port", lambda: 43210)
    monkeypatch.setattr(desktop, "_start_service", lambda port: _Process())
    def fail_readiness(*args):
        raise RuntimeError("bind failed")

    monkeypatch.setattr(desktop, "_wait_until_ready", fail_readiness)
    monkeypatch.setattr(desktop, "_stop_service", lambda child: calls.append("stop"))

    def show_error(webview, message):
        calls.append(message)
        raise SystemExit(1)

    monkeypatch.setattr(desktop, "_show_error", show_error)

    with pytest.raises(SystemExit):
        desktop.main()

    assert calls == ["stop", f"bind failed\n\n启动日志：{desktop.SERVICE_LOG_PATH}"]


def test_main_shows_error_when_service_creation_fails_without_stopping_process(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace())
    monkeypatch.setattr(desktop, "_available_port", lambda: 43210)

    def fail_start(port):
        raise OSError("cannot create process")

    monkeypatch.setattr(desktop, "_start_service", fail_start)
    monkeypatch.setattr(desktop, "_stop_service", lambda child: calls.append("stop"))

    def show_error(webview, message):
        calls.append(message)
        raise SystemExit(1)

    monkeypatch.setattr(desktop, "_show_error", show_error)

    with pytest.raises(SystemExit):
        desktop.main()

    assert calls == [f"cannot create process\n\n启动日志：{desktop.SERVICE_LOG_PATH}"]


def test_main_retries_with_new_port_after_bind_conflict(monkeypatch):
    calls = []
    first, second = _Process(), _Process()
    processes = iter([first, second])
    ports = iter([43210, 43211])
    webview = SimpleNamespace(
        create_window=lambda *args, **kwargs: calls.append((args, kwargs)),
        start=lambda: calls.append("start"),
    )
    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setattr(desktop, "_available_port", lambda: next(ports))
    monkeypatch.setattr(desktop, "_start_service", lambda port: next(processes))

    def wait_until_ready(url, process):
        if process is first:
            raise RuntimeError("OSError: [Errno 98] Address already in use")
        calls.append((url, process))

    monkeypatch.setattr(desktop, "_wait_until_ready", wait_until_ready)
    monkeypatch.setattr(desktop, "_stop_service", lambda child: calls.append(("stop", child)))

    desktop.main()

    assert calls == [
        ("stop", first),
        ("http://127.0.0.1:43211/", second),
        (("化工时序相关性分析", "http://127.0.0.1:43211/"), {"width": 1440, "height": 900}),
        "start",
        ("stop", second),
    ]


def test_service_starts_serves_homepage_and_releases_port(tmp_path, monkeypatch):
    port = desktop._available_port()
    monkeypatch.setattr(desktop, "SERVICE_LOG_PATH", tmp_path / "desktop-launcher.log")
    process = desktop._start_service(port)
    try:
        desktop._wait_until_ready(f"http://{desktop.HOST}:{port}/", process)
        with urlopen(f"http://{desktop.HOST}:{port}/", timeout=2) as response:
            assert response.status == 200
    finally:
        desktop._stop_service(process)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((desktop.HOST, port))


def test_packaged_service_mode_forces_loopback_host():
    source = (Path(desktop.__file__)).read_text(encoding="utf-8")

    assert "run_server(host=HOST, port=int(sys.argv[5]), open_browser=False)" in source
    assert "run_server(host=sys.argv[3]" not in source
