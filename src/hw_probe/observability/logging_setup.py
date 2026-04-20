from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Literal

from hw_probe.config.settings import AppSettings

_LOG_NAMESPACE = "hw_probe"


def get_hw_probe_logger(name: str | None = None) -> logging.Logger:
    """获取 ``hw_probe`` 命名空间下的 logger（便于分级：agents / tools / graph）。"""
    if name:
        return logging.getLogger(f"{_LOG_NAMESPACE}.{name}")
    return logging.getLogger(_LOG_NAMESPACE)


def configure_logging(settings: AppSettings, *, console_level_override: str | None = None) -> None:
    """配置控制台 + ``<workspace>/log/hw_probe.debug.log`` 的 DEBUG 文件日志。"""
    log_dir = settings.resolved_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    raw_console = (console_level_override or settings.console_log_level).upper()
    console_level = getattr(logging, raw_console, logging.INFO)
    file_level = getattr(logging, settings.file_log_level.upper(), logging.DEBUG)

    root_hw = logging.getLogger(_LOG_NAMESPACE)
    root_hw.setLevel(logging.DEBUG)
    root_hw.handlers.clear()

    fmt_detailed = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_console = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = RotatingFileHandler(
        log_dir / settings.debug_log_filename,
        maxBytes=settings.log_file_max_bytes,
        backupCount=settings.log_file_backup_count,
        encoding="utf-8",
    )
    fh.setLevel(file_level)
    fh.setFormatter(fmt_detailed)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt_console)

    root_hw.addHandler(fh)
    root_hw.addHandler(ch)
    root_hw.propagate = False

    # 压低第三方库噪声；需要时可改为 DEBUG
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    get_hw_probe_logger("logging").debug(
        "日志已初始化: log_dir=%s console_level=%s file_level=%s",
        log_dir,
        raw_console,
        settings.file_log_level.upper(),
    )


def parse_console_level(s: str) -> Literal["DEBUG", "INFO", "WARNING", "ERROR"]:
    u = s.strip().upper()
    if u in ("DEBUG", "INFO", "WARNING", "ERROR"):
        return u  # type: ignore[return-value]
    return "INFO"
