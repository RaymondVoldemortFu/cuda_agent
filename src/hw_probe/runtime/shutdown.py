"""跨线程协作的优雅终止：子进程轮询 + 信号置位，使 Ctrl+C / SIGTERM 尽快生效。"""

from __future__ import annotations

import signal
import subprocess
import threading
import time
from typing import Any

import _thread

_shutdown = threading.Event()
_installed = False


def shutdown_requested() -> bool:
    return _shutdown.is_set()


def _on_signal(signum: int, frame: Any) -> None:  # noqa: ARG001
    """置位全局标志，并尝试在主线程抛出 KeyboardInterrupt 以打断 LangGraph / IO 等阻塞。"""
    _shutdown.set()
    try:
        _thread.interrupt_main()
    except (RuntimeError, ValueError):
        pass


def install_shutdown_handlers() -> None:
    """在进程入口尽早调用。注册 SIGINT；在支持的平台注册 SIGTERM。"""
    global _installed
    if _installed:
        return
    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _on_signal)
        except (ValueError, OSError):
            pass
    _installed = True


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass
    t0 = time.monotonic()
    while proc.poll() is None and time.monotonic() - t0 < 8.0:
        time.sleep(0.1)
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def interruptible_communicate(
    proc: subprocess.Popen[str],
    *,
    total_timeout_sec: float | None,
    poll_sec: float = 0.25,
) -> tuple[str | None, str | None]:
    """在轮询 ``poll`` 中响应 ``shutdown_requested`` 与总超时，最后 ``communicate`` 回收管道。"""
    deadline: float | None = None
    if total_timeout_sec is not None:
        deadline = time.monotonic() + float(total_timeout_sec)

    try:
        while proc.poll() is None:
            if _shutdown.is_set():
                _terminate_process_tree(proc)
                _drain_pipes(proc)
                raise KeyboardInterrupt("用户请求终止（Ctrl+C 或 SIGTERM）")
            if deadline is not None and time.monotonic() > deadline:
                _terminate_process_tree(proc)
                _drain_pipes(proc)
                raise subprocess.TimeoutExpired(cmd=proc.args, timeout=total_timeout_sec)
            time.sleep(poll_sec)

        return proc.communicate()
    except KeyboardInterrupt:
        if proc.poll() is None:
            _terminate_process_tree(proc)
        _drain_pipes(proc)
        raise


def _drain_pipes(proc: subprocess.Popen[str]) -> None:
    try:
        proc.communicate(timeout=5)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        try:
            proc.kill()
        except OSError:
            pass
