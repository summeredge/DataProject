from __future__ import annotations

import html
import os
import signal
import socket
import subprocess
import sys
import time
from typing import NoReturn
from urllib.request import urlopen


HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 15


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def _start_service(port: int) -> subprocess.Popen[bytes]:
    kwargs: dict[str, object] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "chem_ts_corr.cli",
            "serve",
            "--host",
            HOST,
            "--port",
            str(port),
            "--no-open",
        ],
        **kwargs,
    )


def _service_error(process: subprocess.Popen[bytes]) -> str:
    output = process.stdout.read().decode("utf-8", errors="replace") if process.stdout else ""
    return output.strip() or f"服务进程以退出码 {process.returncode} 退出。"


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


def main() -> None:
    import webview

    port = _available_port()
    url = f"http://{HOST}:{port}/"
    process = _start_service(port)
    try:
        _wait_until_ready(url, process)
        webview.create_window("化工时序相关性分析", url, width=1440, height=900)
        webview.start()
    except Exception as exc:
        _stop_service(process)
        _show_error(webview, str(exc))
    else:
        _stop_service(process)


if __name__ == "__main__":
    main()
