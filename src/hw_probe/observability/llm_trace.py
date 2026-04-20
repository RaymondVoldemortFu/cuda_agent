from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from hw_probe.observability.logging_setup import get_hw_probe_logger

_LOG = get_hw_probe_logger("llm_trace")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_messages(batch: list[list[BaseMessage]]) -> list[list[dict[str, Any]]]:
    return [[dumpd(m) for m in turn] for turn in batch]


def _serialize_llm_result(response: LLMResult) -> dict[str, Any]:
    gens: list[list[dict[str, Any]]] = []
    for row in response.generations:
        row_out: list[dict[str, Any]] = []
        for g in row:
            if hasattr(g, "message") and g.message is not None:
                row_out.append({"type": "chat_generation", "message": dumpd(g.message)})
            elif hasattr(g, "text"):
                row_out.append({"type": "generation", "text": getattr(g, "text", "")})
            else:
                row_out.append({"type": type(g).__name__, "repr": repr(g)[:8000]})
        gens.append(row_out)
    return {
        "generations": gens,
        "llm_output": response.llm_output,
        "run": [r.model_dump() for r in response.run] if response.run else None,
    }


class JsonlLlmTraceHandler(BaseCallbackHandler):
    """将 LangChain 回调完整写入 JSONL（一行一个事件），便于离线审计与复现。"""

    def __init__(self, jsonl_path: Path, *, session_id: str) -> None:
        super().__init__()
        self._path = jsonl_path
        self._session_id = session_id
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit_custom(self, event: str, payload: dict[str, Any] | None = None) -> None:
        """写入自定义会话级事件（如 session_start / run_end）。"""
        rec: dict[str, Any] = {"event": event}
        if payload:
            rec.update(payload)
        self._append(rec)

    def _append(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", _utc_iso())
        record.setdefault("session_id", self._session_id)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        _LOG.debug("llm_trace 写入事件: %s", record.get("event"))

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
        self._append(
            {
                "event": "chat_model_start",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "tags": tags or [],
                "metadata": metadata or {},
                "serialized": serialized,
                "messages": _serialize_messages(messages),
            }
        )

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
        self._append(
            {
                "event": "llm_start",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "tags": tags or [],
                "metadata": metadata or {},
                "serialized": serialized,
                "prompts": prompts,
            }
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._append(
            {
                "event": "llm_end",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "tags": tags or [],
                "response": _serialize_llm_result(response),
            }
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
        self._append(
            {
                "event": "llm_error",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "tags": tags or [],
                "error_type": type(error).__name__,
                "error": repr(error),
            }
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
        self._append(
            {
                "event": "tool_start",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "tags": tags or [],
                "metadata": metadata or {},
                "serialized": serialized,
                "input_str": input_str,
                "inputs": inputs,
            }
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._append(
            {
                "event": "tool_end",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "tags": tags or [],
                "output": output,
            }
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
        self._append(
            {
                "event": "tool_error",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "tags": tags or [],
                "error_type": type(error).__name__,
                "error": repr(error),
            }
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
        try:
            dumped_inputs = dumpd(inputs) if inputs is not None else None
        except Exception as exc:
            dumped_inputs = {"dump_error": type(exc).__name__, "detail": repr(exc), "repr": repr(inputs)[:50_000]}
        self._append(
            {
                "event": "chain_start",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "tags": tags or [],
                "metadata": metadata or {},
                "serialized": serialized,
                "inputs": dumped_inputs,
            }
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
        try:
            dumped_outputs = dumpd(outputs) if outputs is not None else None
        except Exception as exc:
            dumped_outputs = {"dump_error": type(exc).__name__, "detail": repr(exc), "repr": repr(outputs)[:50_000]}
        self._append(
            {
                "event": "chain_end",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "tags": tags or [],
                "outputs": dumped_outputs,
            }
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
        self._append(
            {
                "event": "chain_error",
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "tags": tags or [],
                "error_type": type(error).__name__,
                "error": repr(error),
            }
        )
