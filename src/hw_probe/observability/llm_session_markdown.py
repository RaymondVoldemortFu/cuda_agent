from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult

from hw_probe.observability.llm_trace import _utc_iso
from hw_probe.observability.logging_setup import get_hw_probe_logger

_LOG = get_hw_probe_logger("llm_session_md")

_TAG_TO_AGENT: dict[str, str] = {
    "graph:planner": "Planner",
    "graph:programmer": "Programmer",
    "graph:supervisor": "Supervisor",
    "graph:synthesize": "Synthesizer",
}


def _line_count(s: str) -> int:
    if not s:
        return 0
    return len(s.splitlines())


def _take_first_n_lines(s: str, n: int) -> tuple[str, str]:
    """取前 n 行（按 splitlines）为 head，其余为 tail；keepends 保持原样换行。"""
    if n <= 0:
        return "", s
    parts = s.splitlines(keepends=True)
    if len(parts) <= n:
        return s, ""
    return "".join(parts[:n]), "".join(parts[n:])


def _agent_from_tags(tags: list[str] | None) -> str:
    for t in tags or []:
        if t in _TAG_TO_AGENT:
            return _TAG_TO_AGENT[t]
    if tags:
        return tags[0]
    return "Unknown"


def _tool_name(serialized: dict[str, Any]) -> str:
    n = serialized.get("name")
    if isinstance(n, str) and n.strip():
        return n.strip()
    kid = serialized.get("id")
    if isinstance(kid, list) and kid:
        return str(kid[-1])
    return "tool"


def _trim(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    head = max_chars // 2
    tail = max_chars - head
    return s[:head] + "\n…(已截断)…\n" + s[-tail:]


def _message_body_text(m: BaseMessage, *, max_chars: int) -> str:
    c = m.content
    if isinstance(c, str):
        return _trim(c, max_chars)
    return _trim(str(c), max_chars)


def _format_block(*, agent: str, kind: str, body: str) -> str:
    body = (body or "").strip("\n")
    ts = _utc_iso()
    header = f"来源: {agent}  |  {kind}  |  {ts}\n" + ("-" * 72) + "\n"
    if body:
        return header + body.rstrip() + "\n\n"
    return header + "\n"


def _tool_calls_as_text(msg: AIMessage, *, max_chars: int) -> str:
    tcs = msg.tool_calls or []
    if not tcs:
        return ""
    lines: list[str] = []
    for tc in tcs:
        name = tc.get("name", "")
        args = tc.get("args", {})
        try:
            arg_s = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            arg_s = str(args)
        lines.append(f"工具提议: {name}  参数: {_trim(arg_s, max_chars)}")
    return "\n".join(lines)


class MarkdownLlmSessionLog(BaseCallbackHandler):
    """仅将纯文本与来源 Agent 写入 Markdown；按行数切分多文件，避免单文件过大。"""

    def __init__(
        self,
        md_path: Path,
        *,
        session_id: str,
        block_max_chars: int,
        max_file_lines: int,
    ) -> None:
        super().__init__()
        self._base_path = md_path
        self._session_id = session_id
        self._block_max = max(4_000, int(block_max_chars))
        self._max_lines = int(max_file_lines)
        self._lock = threading.Lock()
        self._base_path.parent.mkdir(parents=True, exist_ok=True)
        self._part = 0
        self._current_path = self._path_for_part(0)
        self._lines_in_file = 0
        self._bootstrap_part0()

    def _path_for_part(self, part: int) -> Path:
        p = self._base_path
        if part == 0:
            return p
        return p.parent / f"{p.stem}.part{part + 1}{p.suffix}"

    def _bootstrap_part0(self) -> None:
        text = (
            f"hw_probe LLM 文本日志（纯文本）\n"
            f"session_id={self._session_id}\n"
            f"单文件最多 {self._max_lines} 行，超出后自动写入续文件（*.part2.md 起）\n"
            + ("-" * 72)
            + "\n\n"
        )
        self._write_raw_to_path(self._current_path, text, append=False)
        self._lines_in_file = _line_count(text)

    def _rotate_file_unlocked(self) -> None:
        self._part += 1
        self._current_path = self._path_for_part(self._part)
        cont = (
            f"续篇 part {self._part + 1}  (session_id={self._session_id})\n"
            + ("-" * 72)
            + "\n\n"
        )
        self._write_raw_to_path(self._current_path, cont, append=False)
        self._lines_in_file = _line_count(cont)

    def _write_raw_to_path(self, path: Path, text: str, *, append: bool) -> None:
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as f:
            f.write(text)
            f.flush()

    def _append_block(self, block: str) -> None:
        if not block:
            return
        if not block.endswith("\n"):
            block += "\n"
        remaining = block
        with self._lock:
            while remaining:
                while self._lines_in_file >= self._max_lines:
                    self._rotate_file_unlocked()
                space = self._max_lines - self._lines_in_file
                if space <= 0:
                    self._rotate_file_unlocked()
                    continue
                lines_rem = _line_count(remaining)
                if lines_rem <= space:
                    self._write_raw_to_path(self._current_path, remaining, append=True)
                    self._lines_in_file += lines_rem
                    remaining = ""
                else:
                    head, tail = _take_first_n_lines(remaining, space)
                    if head:
                        self._write_raw_to_path(self._current_path, head, append=True)
                    self._lines_in_file += space
                    remaining = tail
                    if remaining:
                        self._rotate_file_unlocked()

    def emit_custom(self, event: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        agent = "会话"
        if event == "session_start":
            lines = [
                "事件: session_start",
                f"workspace={payload.get('workspace', '')}",
                f"target_spec={payload.get('target_spec', '')}",
                f"argv={' '.join(str(a) for a in (payload.get('argv') or []))}",
            ]
            body = "\n".join(lines)
        elif event == "session_end":
            body = f"事件: session_end\nresults_keys={payload.get('results_keys', [])!r}"
        elif event == "session_interrupted":
            body = f"事件: session_interrupted\nreason={payload.get('reason', '')}"
        elif event == "session_error":
            body = f"事件: session_error\n{payload.get('traceback', '')}"
        else:
            body = f"事件: {event}\n{_trim(str(payload), self._block_max)}"
        self._append_block(_format_block(agent=agent, kind="系统", body=_trim(body, self._block_max)))
        _LOG.debug("llm_session_md 写入: %s", event)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        agent = _agent_from_tags(tags)
        parts: list[str] = []
        for turn in messages:
            for m in turn:
                role = type(m).__name__
                parts.append(f"[{role}]\n{_message_body_text(m, max_chars=self._block_max)}")
        body = "\n\n".join(parts)
        self._append_block(_format_block(agent=agent, kind="LLM 输入", body=body))

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Completion 类模型走此路径；与 chat 互斥场景下补充输入文本。"""
        if not prompts or all(not str(p).strip() for p in prompts):
            return
        agent = _agent_from_tags(tags)
        body = "\n\n".join(_trim(p, self._block_max) for p in prompts)
        self._append_block(_format_block(agent=agent, kind="LLM 输入(prompt)", body=body))

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        agent = _agent_from_tags(tags)
        chunks: list[str] = []
        for row in response.generations:
            for g in row:
                msg = getattr(g, "message", None)
                if isinstance(msg, AIMessage):
                    txt = _message_body_text(msg, max_chars=self._block_max)
                    if txt.strip():
                        chunks.append(txt)
                    tc_txt = _tool_calls_as_text(msg, max_chars=min(self._block_max, 16_000))
                    if tc_txt:
                        chunks.append(tc_txt)
                elif msg is not None and isinstance(msg, BaseMessage):
                    chunks.append(_message_body_text(msg, max_chars=self._block_max))
                else:
                    t = getattr(g, "text", None)
                    if isinstance(t, str) and t.strip():
                        chunks.append(_trim(t, self._block_max))
        body = "\n\n".join(chunks)
        self._append_block(_format_block(agent=agent, kind="LLM 输出", body=body))

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        agent = _agent_from_tags(tags)
        body = f"{type(error).__name__}: {error!r}"
        self._append_block(_format_block(agent=agent, kind="LLM 错误", body=body))

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        agent = _agent_from_tags(tags)
        name = _tool_name(serialized)
        parts = [f"工具: {name}"]
        if input_str and input_str.strip():
            parts.append(_trim(input_str.strip(), self._block_max))
        if inputs:
            try:
                parts.append(json.dumps(inputs, ensure_ascii=False, separators=(",", ":")))
            except TypeError:
                parts.append(_trim(str(inputs), self._block_max))
        body = "\n".join(parts)
        kind = f"工具调用 ({name})"
        self._append_block(_format_block(agent=agent, kind=kind, body=body))

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        agent = _agent_from_tags(tags)
        if isinstance(output, str):
            body = _trim(output, self._block_max)
        elif isinstance(output, ToolMessage):
            body = _message_body_text(output, max_chars=self._block_max)
        else:
            try:
                body = _trim(json.dumps(output, ensure_ascii=False, default=str), self._block_max)
            except TypeError:
                body = _trim(repr(output), self._block_max)
        self._append_block(_format_block(agent=agent, kind="工具返回", body=body))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        agent = _agent_from_tags(tags)
        body = f"{type(error).__name__}: {error!r}"
        self._append_block(_format_block(agent=agent, kind="工具错误", body=body))
