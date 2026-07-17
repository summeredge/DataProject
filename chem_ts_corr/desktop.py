from __future__ import annotations

import html
import importlib
import os
import signal
import socket
import subprocess
import sys
import time
from typing import NoReturn
from urllib.request import urlopen

from chem_ts_corr.paths import desktop_log_path


HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 15
SERVICE_START_ATTEMPTS = 3
SERVICE_LOG_PATH = desktop_log_path()
MODULE_CHECKS = (
    "xgboost",
    "shap",
    "openpyxl",
    "xlrd",
)


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def _start_service(port: int) -> subprocess.Popen[bytes]:
    kwargs: dict[str, object] = {
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    SERVICE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "--desktop-service",
        "--host",
        HOST,
        "--port",
        str(port),
    ] if getattr(sys, "frozen", False) else [
        sys.executable,
        "-m",
        "chem_ts_corr.cli",
        "serve",
        "--host",
        HOST,
        "--port",
        str(port),
        "--no-open",
    ]
    with SERVICE_LOG_PATH.open("wb") as log_file:
        kwargs["stdout"] = log_file
        return subprocess.Popen(command, **kwargs)


def _service_error(process: subprocess.Popen[bytes]) -> str:
    try:
        output = SERVICE_LOG_PATH.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        output = ""
    return output or f"服务进程以退出码 {process.returncode} 退出。"


def _is_port_conflict(message: str) -> bool:
    text = message.lower()
    return "address already in use" in text or "winerror 10048" in text


def _wait_until_ready(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(_service_error(process))
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"等待本地服务就绪超时（{STARTUP_TIMEOUT_SECONDS} 秒）。")


def _stop_service(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _show_error(webview: object, message: str) -> NoReturn:
    webview.create_window(
        "化工时序相关性分析 - 启动失败",
        html=f"<h2>本地服务启动失败</h2><pre>{html.escape(message)}</pre>",
        width=720,
        height=360,
    )
    webview.start()
    raise SystemExit(1)


def _module_check() -> None:
    for module in MODULE_CHECKS:
        importlib.import_module(module)
        print(f"Module check passed: {module}")


def main() -> None:
    import webview

    process: subprocess.Popen[bytes] | None = None
    try:
        for attempt in range(SERVICE_START_ATTEMPTS):
            port = _available_port()
            url = f"http://{HOST}:{port}/"
            process = _start_service(port)
            try:
                _wait_until_ready(url, process)
                break
            except RuntimeError as exc:
                _stop_service(process)
                process = None
                if _is_port_conflict(str(exc)) and attempt < SERVICE_START_ATTEMPTS - 1:
                    continue
                raise
        else:
            raise RuntimeError("本地服务未能启动。")
        webview.create_window("化工时序相关性分析", url, width=1440, height=900)
        webview.start()
    except Exception as exc:
        if process is not None:
            _stop_service(process)
        _show_error(webview, f"{exc}\n\n启动日志：{SERVICE_LOG_PATH}")
    else:
        if process is not None:
            _stop_service(process)


if __name__ == "__main__":
    if sys.argv[1:] == ["--module-check"]:
        _module_check()
    elif len(sys.argv) == 6 and sys.argv[1] == "--desktop-service":
        from chem_ts_corr.web import run_server

        run_server(host=sys.argv[3], port=int(sys.argv[5]), open_browser=False)
    else:
        main()
