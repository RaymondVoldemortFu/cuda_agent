from __future__ import annotations

import json
import traceback
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, create_react_agent

from hw_probe.agents.llm_util import extract_json_object
from hw_probe.agents.prompts import (
    PLANNER_SYSTEM,
    PROGRAMMER_SYSTEM,
    SUPERVISOR_SYSTEM,
    SYNTHESIZER_SYSTEM,
    planner_user_message,
    programmer_user_message,
    supervisor_user_message,
    synthesizer_user_message,
)
from hw_probe.agents.state import ProbeState
from hw_probe.config.settings import AppSettings
from hw_probe.observability.logging_setup import get_hw_probe_logger
from hw_probe.tools import make_cuda_tools, make_filesystem_tools, make_run_shell_tool

_LOG = get_hw_probe_logger("agents")


def _make_model(settings: AppSettings) -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "api_key": settings.api_key,
        "model": settings.model,
        "temperature": 0.15,
    }
    if settings.base_url and str(settings.base_url).strip():
        kwargs["base_url"] = str(settings.base_url).strip()
    return ChatOpenAI(**kwargs)


def _trace_invoke_config(
    trace_callbacks: list[Any] | None,
    *,
    tags: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # max_concurrency=1：工具串行，便于 Ctrl+C 尽快传到执行线程并配合可中断子进程
    cfg: dict[str, Any] = {"tags": tags, "max_concurrency": 1}
    if extra:
        cfg.update(extra)
    if trace_callbacks:
        cfg["callbacks"] = trace_callbacks
    return cfg


def _messages_to_evidence_snippet(messages: list[BaseMessage], max_chars: int) -> str:
    parts: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str) and c.strip():
                parts.append(f"[assistant]\n{c[:4000]}\n")
            if m.tool_calls:
                for tc in m.tool_calls:
                    name = tc.get("name", "")
                    parts.append(f"[tool_call] {name} {tc.get('args', {})}\n")
        elif isinstance(m, ToolMessage):
            body = str(m.content)
            if len(body) > 6000:
                body = body[:3000] + "\n...[truncated]...\n" + body[-3000:]
            parts.append(f"[tool_result:{m.name}]\n{body}\n")
    out = "\n".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars // 2] + "\n...[truncated]...\n" + out[-max_chars // 2 :]
    return out


def _all_tools(settings: AppSettings) -> list[Any]:
    return [
        *make_filesystem_tools(settings),
        make_run_shell_tool(settings),
        *make_cuda_tools(settings),
    ]


def build_planner_node(settings: AppSettings, trace_callbacks: list[Any] | None):
    model = _make_model(settings)

    def planner_node(state: ProbeState) -> dict[str, Any]:
        _LOG.debug("planner_node 开始")
        targets = state.get("targets") or []
        resp = model.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=planner_user_message(list(targets))),
            ],
            config=_trace_invoke_config(trace_callbacks, tags=["graph:planner"]),
        )
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        _LOG.debug("planner_node 结束, plan_len=%s", len(content))
        return {"plan": content}

    return planner_node


def build_programmer_node(settings: AppSettings, trace_callbacks: list[Any] | None):
    model = _make_model(settings)
    tools = _all_tools(settings)
    # 工具执行期异常（如 nvcc 失败）转为 ToolMessage，避免整图未捕获崩溃；仍保留完整错误文本供模型继续推理
    tool_node = ToolNode(tools, handle_tool_errors=True)
    agent = create_react_agent(model, tool_node)

    def programmer_node(state: ProbeState) -> dict[str, Any]:
        targets = list(state.get("targets") or [])
        plan = state.get("plan") or ""
        log = list(state.get("evidence_log") or [])
        rounds = int(state.get("programmer_rounds") or 0)
        next_round = rounds + 1
        tail = "\n\n---\n\n".join(log[-3:]) if log else ""

        _LOG.debug("programmer_node 开始 round=%s", next_round)
        user = programmer_user_message(
            targets=targets,
            plan=plan,
            evidence_so_far=tail,
            round_index=next_round,
            max_rounds=settings.supervisor_max_loops,
        )
        cfg = _trace_invoke_config(
            trace_callbacks,
            tags=["graph:programmer"],
            extra={"recursion_limit": max(32, settings.react_max_steps * 2)},
        )
        try:
            result = agent.invoke(
                {"messages": [SystemMessage(content=PROGRAMMER_SYSTEM), HumanMessage(content=user)]},
                config=cfg,
            )
        except Exception:
            tb = traceback.format_exc()
            _LOG.exception("programmer_node ReAct 子图异常（已捕获并写入 evidence，主流程继续）")
            log.append(f"=== programmer_round={next_round} SUBGRAPH_EXCEPTION ===\n{tb}")
            return {
                "programmer_rounds": next_round,
                "evidence_log": log,
            }

        msgs = list(result.get("messages") or [])
        snippet = _messages_to_evidence_snippet(msgs, settings.max_tool_output_chars)
        log.append(f"=== programmer_round={next_round} ===\n{snippet}")
        _LOG.debug("programmer_node 结束 round=%s messages=%s", next_round, len(msgs))
        return {
            "programmer_rounds": next_round,
            "evidence_log": log,
        }

    return programmer_node


def build_supervisor_node(settings: AppSettings, trace_callbacks: list[Any] | None):
    model = _make_model(settings)

    def supervisor_node(state: ProbeState) -> dict[str, Any]:
        _LOG.debug("supervisor_node 进入 state_keys=%s", list(state.keys()))
        if not (state.get("plan") or "").strip():
            _LOG.debug("supervisor -> planner（尚无计划）")
            return {"_route": "planner"}

        rounds = int(state.get("programmer_rounds") or 0)
        if rounds >= settings.supervisor_max_loops:
            _LOG.info("supervisor -> synthesize（已达 programmer 轮次上限 %s）", settings.supervisor_max_loops)
            return {"_route": "synthesize"}

        if rounds == 0:
            _LOG.debug("supervisor -> programmer（首轮执行）")
            return {"_route": "programmer"}

        log = state.get("evidence_log") or []
        tail = "\n\n---\n\n".join(log[-2:]) if log else ""
        user = supervisor_user_message(
            targets=list(state.get("targets") or []),
            plan=state.get("plan") or "",
            evidence_tail=tail,
            programmer_rounds=rounds,
            max_rounds=settings.supervisor_max_loops,
        )
        resp = model.invoke(
            [
                SystemMessage(content=SUPERVISOR_SYSTEM),
                HumanMessage(content=user),
            ],
            config=_trace_invoke_config(trace_callbacks, tags=["graph:supervisor"]),
        )
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        try:
            obj = extract_json_object(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            _LOG.warning("supervisor JSON 解析失败，默认继续 programmer。raw_prefix=%s", raw[:200])
            return {"_route": "programmer"}
        nxt = str(obj.get("next", "")).strip().lower()
        if nxt == "synthesize":
            _LOG.info("supervisor -> synthesize（模型决策）")
            return {"_route": "synthesize"}
        _LOG.debug("supervisor -> programmer（模型决策）")
        return {"_route": "programmer"}

    return supervisor_node


def build_synthesizer_node(settings: AppSettings, trace_callbacks: list[Any] | None):
    model = _make_model(settings)

    def synthesizer_node(state: ProbeState) -> dict[str, Any]:
        _LOG.debug("synthesizer_node 开始")
        targets = list(state.get("targets") or [])
        plan = state.get("plan") or ""
        log = state.get("evidence_log") or []
        evidence = "\n\n---\n\n".join(log)
        user = synthesizer_user_message(targets=targets, plan=plan, evidence=evidence)
        resp = model.invoke(
            [
                SystemMessage(content=SYNTHESIZER_SYSTEM),
                HumanMessage(content=user),
            ],
            config=_trace_invoke_config(trace_callbacks, tags=["graph:synthesize"]),
        )
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        parsed = extract_json_object(raw)
        results = {t: parsed.get(t) for t in targets}
        methodology = (
            "多智能体流程：Planner 产出计划；单一编程子智能体（ReAct）在工作区内完成"
            "探测、编码、编译、运行与 ncu；Synthesizer 从证据汇总数值。"
        )
        _LOG.debug("synthesizer_node 结束 keys=%s", list(results.keys()))
        return {"results": results, "methodology": methodology}

    return synthesizer_node
