import sys
from types import SimpleNamespace

import pytest

import chem_ts_corr.desktop as desktop


class _Process:
    pid = 12345

    def poll(self):
        return None


def test_available_port_is_loopback_only():
    assert isinstance(desktop._available_port(), int)
    assert desktop.HOST == "127.0.0.1"


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

    assert calls == ["stop", "bind failed"]
