from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from hw_probe.config.settings import AppSettings
from hw_probe.observability.logging_setup import get_hw_probe_logger
from hw_probe.tools.pathutil import assert_under_workspace

_LOG = get_hw_probe_logger("tools.fs")


class ReadFileArgs(BaseModel):
    relative_path: Annotated[str, Field(description="工作区内的相对路径，例如 probes/foo.cu")]


class WriteFileArgs(BaseModel):
    relative_path: Annotated[str, Field(description="工作区内的相对路径")]
    content: Annotated[str, Field(description="写入文件的完整文本")]


class ListDirArgs(BaseModel):
    relative_path: Annotated[str, Field(default=".", description="要列出的子目录（相对工作区）")]


def make_filesystem_tools(settings: AppSettings) -> list[StructuredTool]:
    ws = settings.resolved_workspace()

    def read_file(relative_path: str) -> str:
        _LOG.debug("read_workspace_file %r", relative_path)
        p = assert_under_workspace(ws, relative_path)
        if not p.is_file():
            raise FileNotFoundError(f"不是文件或不存在: {p}")
        return p.read_text(encoding="utf-8")

    def write_file(relative_path: str, content: str) -> str:
        _LOG.debug("write_workspace_file %r bytes=%s", relative_path, len(content.encode("utf-8")))
        p = assert_under_workspace(ws, relative_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已写入 {p.relative_to(ws)}，字节数={len(content.encode('utf-8'))}"

    def list_dir(relative_path: str = ".") -> str:
        d = assert_under_workspace(ws, relative_path)
        if not d.is_dir():
            raise NotADirectoryError(f"不是目录: {d}")
        names = sorted(x.name for x in d.iterdir())
        return json.dumps(names, ensure_ascii=False, indent=2)

    return [
        StructuredTool.from_function(
            name="read_workspace_file",
            description="读取工作区内文本文件内容。路径为相对工作区根目录。",
            args_schema=ReadFileArgs,
            func=lambda relative_path: read_file(relative_path),
        ),
        StructuredTool.from_function(
            name="write_workspace_file",
            description="在工作区内创建或覆盖文本文件。自动创建父目录。",
            args_schema=WriteFileArgs,
            func=lambda relative_path, content: write_file(relative_path, content),
        ),
        StructuredTool.from_function(
            name="list_workspace_dir",
            description="列出工作区某子目录下的文件名（仅一层）。",
            args_schema=ListDirArgs,
            func=lambda relative_path=".": list_dir(relative_path),
        ),
    ]
