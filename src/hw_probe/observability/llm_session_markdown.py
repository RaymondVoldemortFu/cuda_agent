from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from hw_probe.observability.llm_trace import _serialize_llm_result, _serialize_messages, _utc_iso
from hw_probe.observability.logging_setup import get_hw_probe_logger

_LOG = get_hw_probe_logger("llm_session_md")


def _trim(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    head = max_chars // 2
    tail = max_chars - head
    return s[:head] + "\n\n…(中间已截断，共省略约 " + str(len(s) - max_chars) + " 字符)…\n\n" + s[-tail:]


def _fenced_code(lang: str, body: str) -> str:
    body = body.replace("\r\n", "\n").rstrip("\n")
    fence = "```"
    while fence in body:
        fence += "`"
    lang = lang.strip() or ""
    prefix = f"{fence}{lang}\n" if lang else f"{fence}\n"
    return f"{prefix}{body}\n{fence}\n"


def _json_block(obj: Any, *, max_chars: int) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        text = repr(obj)
    return _fenced_code("json", _trim(text, max_chars))


def _serialized_label(serialized: dict[str, Any] | None) -> str:
    if not serialized:
        return "unknown"
    name = serialized.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    kid = serialized.get("id")
    if isinstance(kid, list) and kid:
        return ".".join(str(x) for x in kid[-4:])
    t = serialized.get("type")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return "unknown"


def _format_dumped_constructor(msg: dict[str, Any], *, max_chars: int) -> str:
    ids = msg.get("id")
    type_name = ids[-1] if isinstance(ids, list) and ids else "Message"
    kwargs = msg.get("kwargs") if isinstance(msg.get("kwargs"), dict) else {}
    parts: list[str] = [f"**{type_name}**"]

    content = kwargs.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(_fenced_code("", _trim(content, max_chars)))
    elif content is not None:
        parts.append(_json_block(content, max_chars=max_chars))

    tool_calls = kwargs.get("tool_calls")
    if tool_calls:
        parts.append("**tool_calls**")
        parts.append(_json_block(tool_calls, max_chars=max_chars))

    extra = {k: v for k, v in kwargs.items() if k not in ("content", "tool_calls")}
    if extra:
        parts.append("**其它 kwargs**")
        parts.append(_json_block(extra, max_chars=min(max_chars, 24_000)))

    return "\n\n".join(parts)


def _format_message_dump(msg: Any, *, max_chars: int) -> str:
    if isinstance(msg, dict) and msg.get("type") == "constructor":
        return _format_dumped_constructor(msg, max_chars=max_chars)
    if isinstance(msg, dict):
        return _json_block(msg, max_chars=max_chars)
    return _fenced_code("", _trim(str(msg), max_chars))


def _format_llm_response_payload(payload: dict[str, Any], *, max_chars: int) -> str:
    parts: list[str] = []
    gens = payload.get("generations") or []
    if not isinstance(gens, list):
        return _json_block(payload, max_chars=max_chars)
    for gi, row in enumerate(gens):
        if not isinstance(row, list):
            continue
        for hi, item in enumerate(row):
            if not isinstance(item, dict):
                parts.append(_fenced_code("", _trim(repr(item), max_chars)))
                continue
            if item.get("type") == "chat_generation":
                msg = (item.get("message") or {}) if isinstance(item.get("message"), dict) else {}
                parts.append(f"##### 生成块 {gi + 1}.{hi + 1}（chat）\n\n{_format_message_dump(msg, max_chars=max_chars)}")
            elif item.get("type") == "generation":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(f"##### 生成块 {gi + 1}.{hi + 1}（text）\n\n{_fenced_code('', _trim(text, max_chars))}")
                else:
                    parts.append(_json_block(item, max_chars=max_chars))
            else:
                parts.append(_json_block(item, max_chars=max_chars))
    llm_out = payload.get("llm_output")
    if llm_out:
        parts.append("**llm_output**")
        parts.append(_json_block(llm_out, max_chars=min(max_chars, 16_000)))
    return "\n\n".join(parts) if parts else _json_block(payload, max_chars=max_chars)


class MarkdownLlmSessionLog(BaseCallbackHandler):
    """将 LangChain 回调实时写入 Markdown，便于人类阅读（与 JSONL trace 并行）。"""

    def __init__(self, md_path: Path, *, session_id: str, block_max_chars: int) -> None:
        super().__init__()
        self._path = md_path
        self._session_id = session_id
        self._block_max = max(4_000, int(block_max_chars))
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._bootstrap_file()

    def _bootstrap_file(self) -> None:
        title = "# hw_probe · LLM 会话（可读版）\n\n"
        meta = (
            f"- **会话 ID**：`{self._session_id}`\n"
            f"- **文件创建（UTC）**：{_utc_iso()}\n\n"
            "> 以下为运行过程中**实时追加**的事件；刷新编辑器即可查看最新内容。\n\n"
            "---\n\n"
        )
        self._write_raw(title + meta, append=False)

    def emit_custom(self, event: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        self._seq += 1
        n = self._seq
        ts = _utc_iso()
        if event == "session_start":
            body = [
                f"## [{n}] 会话启动 — `{ts}`\n",
                f"- **工作区**：`{payload.get('workspace', '')}`\n",
                f"- **target_spec**：`{payload.get('target_spec', '')}`\n",
                "**argv**：",
                _json_block(payload.get("argv", []), max_chars=min(self._block_max, 16_000)),
            ]
            self._write("\n".join(body))
        elif event == "session_end":
            keys = payload.get("results_keys", [])
            self._write(
                f"\n## [{n}] 会话正常结束 — `{ts}`\n\n"
                f"- **results 键**：`{keys!r}`\n\n"
            )
        elif event == "session_interrupted":
            self._write(
                f"\n## [{n}] 会话中断 — `{ts}`\n\n"
                f"- **原因**：`{payload.get('reason', '')}`\n\n"
            )
        elif event == "session_error":
            tb = str(payload.get("traceback") or "")
            self._write(
                f"\n## [{n}] 会话异常 — `{ts}`\n\n"
                f"{_fenced_code('text', _trim(tb, self._block_max))}\n"
            )
        else:
            self._write(
                f"\n## [{n}] 自定义事件 `{event}` — `{ts}`\n\n"
                f"{_json_block(payload, max_chars=self._block_max)}\n"
            )
        _LOG.debug("llm_session_md 写入: %s", event)

    def _write_raw(self, text: str, *, append: bool) -> None:
        mode = "a" if append else "w"
        with self._lock:
            with self._path.open(mode, encoding="utf-8") as f:
                f.write(text)
                f.flush()

    def _write(self, text: str) -> None:
        self._write_raw(text, append=True)

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
        self._seq += 1
        n = self._seq
        model = _serialized_label(serialized)
        tag_s = ", ".join(tags or []) or "—"
        head = (
            f"\n## [{n}] LLM 请求（chat） — `{_utc_iso()}`\n\n"
            f"- **run_id**：`{run_id}`\n"
            f"- **parent_run_id**：`{parent_run_id}`\n"
            f"- **模型 / runnable**：`{model}`\n"
            f"- **tags**：{tag_s}\n\n"
        )
        dumped = _serialize_messages(messages)
        parts: list[str] = [head, "### 消息（输入）\n"]
        for turn_idx, turn in enumerate(dumped):
            parts.append(f"#### 轮次 {turn_idx + 1}\n")
            for mi, m in enumerate(turn):
                parts.append(f"##### 消息 {mi + 1}\n\n{_format_message_dump(m, max_chars=self._block_max)}")
        self._write("\n".join(parts) + "\n")

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
        self._seq += 1
        n = self._seq
        model = _serialized_label(serialized)
        tag_s = ", ".join(tags or []) or "—"
        blocks: list[str] = [
            f"\n## [{n}] LLM 请求（prompt） — `{_utc_iso()}`\n\n",
            f"- **run_id**：`{run_id}`\n",
            f"- **parent_run_id**：`{parent_run_id}`\n",
            f"- **runnable**：`{model}`\n",
            f"- **tags**：{tag_s}\n\n",
            "### prompts\n",
        ]
        for i, p in enumerate(prompts):
            blocks.append(f"#### prompt {i + 1}\n\n{_fenced_code('', _trim(p, self._block_max))}")
        self._write("\n".join(blocks) + "\n")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._seq += 1
        n = self._seq
        tag_s = ", ".join(tags or []) or "—"
        payload = _serialize_llm_result(response)
        body = _format_llm_response_payload(payload, max_chars=self._block_max)
        self._write(
            f"\n## [{n}] LLM 输出 — `{_utc_iso()}`\n\n"
            f"- **run_id**：`{run_id}`\n"
            f"- **parent_run_id**：`{parent_run_id}`\n"
            f"- **tags**：{tag_s}\n\n"
            f"### 模型返回\n\n{body}\n"
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._seq += 1
        n = self._seq
        tag_s = ", ".join(tags or []) or "—"
        self._write(
            f"\n## [{n}] LLM 错误 — `{_utc_iso()}`\n\n"
            f"- **run_id**：`{run_id}`\n"
            f"- **tags**：{tag_s}\n"
            f"- **类型**：`{type(error).__name__}`\n\n"
            f"{_fenced_code('text', _trim(repr(error), self._block_max))}\n"
        )

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
        self._seq += 1
        n = self._seq
        name = _serialized_label(serialized)
        tag_s = ", ".join(tags or []) or "—"
        parts = [
            f"\n## [{n}] 工具调用 — `{_utc_iso()}`\n\n",
            f"- **run_id**：`{run_id}`\n",
            f"- **parent_run_id**：`{parent_run_id}`\n",
            f"- **工具**：`{name}`\n",
            f"- **tags**：{tag_s}\n\n",
        ]
        if input_str and input_str.strip():
            parts.append("### 调用参数（input_str）\n\n")
            parts.append(_fenced_code("", _trim(input_str.strip(), self._block_max)))
        if inputs:
            parts.append("\n### 调用参数（inputs）\n\n")
            parts.append(_json_block(inputs, max_chars=self._block_max))
        self._write("".join(parts) + "\n")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._seq += 1
        n = self._seq
        tag_s = ", ".join(tags or []) or "—"
        if isinstance(output, str):
            out_text = _fenced_code("", _trim(output, self._block_max))
        else:
            out_text = _json_block(output, max_chars=self._block_max)
        self._write(
            f"\n## [{n}] 工具返回 — `{_utc_iso()}`\n\n"
            f"- **run_id**：`{run_id}`\n"
            f"- **parent_run_id**：`{parent_run_id}`\n"
            f"- **tags**：{tag_s}\n\n"
            f"### 输出\n\n{out_text}\n"
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._seq += 1
        n = self._seq
        tag_s = ", ".join(tags or []) or "—"
        self._write(
            f"\n## [{n}] 工具执行错误 — `{_utc_iso()}`\n\n"
            f"- **run_id**：`{run_id}`\n"
            f"- **tags**：{tag_s}\n"
            f"- **类型**：`{type(error).__name__}`\n\n"
            f"{_fenced_code('text', _trim(repr(error), self._block_max))}\n"
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._seq += 1
        n = self._seq
        label = _serialized_label(serialized)
        tag_s = ", ".join(tags or []) or "—"
        try:
            dumped_in = dumpd(inputs) if inputs is not None else None
        except Exception as exc:
            dumped_in = {"dump_error": type(exc).__name__, "detail": repr(exc), "repr": repr(inputs)[:50_000]}
        self._write(
            f"\n### [{n}] 链开始 — `{_utc_iso()}`\n\n"
            f"- **run_id**：`{run_id}` · **name**：`{label}` · **tags**：{tag_s}\n\n"
            f"{_json_block(dumped_in, max_chars=min(self._block_max, 32_000))}\n"
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._seq += 1
        n = self._seq
        tag_s = ", ".join(tags or []) or "—"
        try:
            dumped_out = dumpd(outputs) if outputs is not None else None
        except Exception as exc:
            dumped_out = {"dump_error": type(exc).__name__, "detail": repr(exc), "repr": repr(outputs)[:50_000]}
        self._write(
            f"\n### [{n}] 链结束 — `{_utc_iso()}`\n\n"
            f"- **run_id**：`{run_id}` · **tags**：{tag_s}\n\n"
            f"{_json_block(dumped_out, max_chars=min(self._block_max, 48_000))}\n"
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._seq += 1
        n = self._seq
        tag_s = ", ".join(tags or []) or "—"
        self._write(
            f"\n### [{n}] 链错误 — `{_utc_iso()}`\n\n"
            f"- **run_id**：`{run_id}` · **tags**：{tag_s}\n"
            f"- **类型**：`{type(error).__name__}`\n\n"
            f"{_fenced_code('text', _trim(repr(error), self._block_max))}\n"
        )
