from __future__ import annotations

import subprocess
from typing import Annotated

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from hw_probe.config.settings import AppSettings
from hw_probe.observability.logging_setup import get_hw_probe_logger
from hw_probe.runtime.shutdown import interruptible_communicate

_LOG = get_hw_probe_logger("tools.shell")

_BLOCKED_SUBSTRINGS = (
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=/dev/",
    ":(){:|:&};:",
    "> /dev/sd",
    "chmod -R 777 /",
    "apt-get",
    "apt install",
    "yum install",
    "dnf install",
    "pip install",
    "uv pip install",
    "npm install",
    "chown ",
    "mount ",
)


class RunShellArgs(BaseModel):
    command: Annotated[str, Field(description="在固定工作目录下执行的 shell 命令（勿使用交互式程序）")]
    cwd_relative: Annotated[
        str,
        Field(
            default=".",
            description="相对工作区的 cwd，默认工作区根目录",
        ),
    ]


def make_run_shell_tool(settings: AppSettings) -> StructuredTool:
    ws = settings.resolved_workspace()
    timeout = settings.shell_timeout_sec
    max_chars = settings.max_tool_output_chars

    def run_shell(command: str, cwd_relative: str = ".") -> str:
        _LOG.debug("run_shell cwd_relative=%r command=%r", cwd_relative, command[:2000])
        lowered = command.strip().lower()
        for bad in _BLOCKED_SUBSTRINGS:
            if bad.lower() in lowered:
                raise ValueError(
                    f"命令被静态策略拒绝（禁止修改系统级环境或拉取网络安装）: 包含 {bad!r}。"
                    "如需依赖，请在评测环境由管理员预装或由用户在本机自行安装。"
                )
        from hw_probe.tools.pathutil import assert_under_workspace

        cwd = assert_under_workspace(ws, cwd_relative)
        cwd.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = interruptible_communicate(proc, total_timeout_sec=float(timeout))
        _LOG.debug("run_shell done exit=%s", proc.returncode)
        out = (
            f"exit_code={proc.returncode}\n--- stdout ---\n{stdout or ''}\n--- stderr ---\n{stderr or ''}"
        )
        if len(out) > max_chars:
            out = out[: max_chars // 2] + "\n...[truncated]...\n" + out[-max_chars // 2 :]
        return out

    return StructuredTool.from_function(
        name="run_shell",
        description=(
            "在工作区内的固定 cwd 下执行 shell 命令（如 which nvcc、nvidia-smi、nvcc --version）。"
            "禁止用于安装系统包或修改全局环境。"
        ),
        args_schema=RunShellArgs,
        func=lambda command, cwd_relative=".": run_shell(command, cwd_relative),
    )
