from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

from hw_probe.config.settings import AppSettings
from hw_probe.observability.logging_setup import get_hw_probe_logger

_LOG = get_hw_probe_logger("status")


def _safe_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for dist in ("langchain-core", "langchain-openai", "langgraph", "pydantic"):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = "unknown"
    return out


def collect_system_status(settings: AppSettings) -> dict[str, Any]:
    """收集可打印/可落库的系统状态（不含密钥明文）。"""
    ws = settings.resolved_workspace()
    spec = settings.target_spec_path.expanduser().resolve()
    log_dir = settings.resolved_log_dir()
    base = str(settings.base_url or "").strip()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd().resolve()),
        "environment": settings.environment,
        "workspace_root": str(ws),
        "target_spec_path": str(spec),
        "log_dir": str(log_dir),
        "model": settings.model,
        "base_url_configured": bool(base),
        "base_url_host": base[:80] + ("..." if len(base) > 80 else "") if base else None,
        "api_key_configured": bool(settings.api_key.strip()),
        "package_versions": _safe_versions(),
    }


def print_system_status(settings: AppSettings) -> None:
    """在命令行打印人类可读的系统状态（同时 DEBUG 写入日志）。"""
    s = collect_system_status(settings)
    lines = [
        "",
        "=" * 72,
        " HW_PROBE 系统状态",
        "=" * 72,
        f"  Python        : {s['python']}",
        f"  Platform      : {s['platform']}",
        f"  CWD           : {s['cwd']}",
        f"  Environment   : {s['environment']}",
        f"  Workspace     : {s['workspace_root']}",
        f"  Target spec   : {s['target_spec_path']}",
        f"  Log directory : {s['log_dir']}",
        f"  Model         : {s['model']}",
        f"  BASE_URL set  : {s['base_url_configured']}"
        + (f" ({s['base_url_host']})" if s["base_url_host"] else ""),
        f"  API_KEY set   : {s['api_key_configured']}",
        "  Package ver.  : "
        + ", ".join(f"{k}={v}" for k, v in s["package_versions"].items()),
        "=" * 72,
        "",
    ]
    text = "\n".join(lines)
    print(text, flush=True)
    _LOG.debug("system_status %s", s)


def log_system_status(settings: AppSettings) -> None:
    """仅写入日志（不打印控制台）。"""
    _LOG.debug("system_status %s", collect_system_status(settings))
