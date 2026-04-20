from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from hw_probe.agents.nodes import (
    build_planner_node,
    build_programmer_node,
    build_supervisor_node,
    build_synthesizer_node,
)
from hw_probe.agents.state import ProbeState
from hw_probe.config.settings import AppSettings


def _route_from_supervisor(state: ProbeState) -> str:
    r = state.get("_route")
    if r not in ("planner", "programmer", "synthesize"):
        return "planner"
    return r


def build_probe_graph(
    settings: AppSettings,
    trace_callbacks: list[Any] | None = None,
) -> Any:
    cbs = trace_callbacks or []
    graph = StateGraph(ProbeState)
    graph.add_node("supervisor", build_supervisor_node(settings, cbs))
    graph.add_node("planner", build_planner_node(settings, cbs))
    graph.add_node("programmer", build_programmer_node(settings, cbs))
    graph.add_node("synthesize", build_synthesizer_node(settings, cbs))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "planner": "planner",
            "programmer": "programmer",
            "synthesize": "synthesize",
        },
    )
    graph.add_edge("planner", "supervisor")
    graph.add_edge("programmer", "supervisor")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_probe_graph(
    settings: AppSettings,
    *,
    targets: list[str],
    trace_callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    app = build_probe_graph(settings, trace_callbacks)
    cfg: dict[str, Any] = {"recursion_limit": 200, "max_concurrency": 1}
    if trace_callbacks:
        cfg["callbacks"] = trace_callbacks
    return dict(
        app.invoke(
            {"targets": targets, "programmer_rounds": 0, "evidence_log": []},
            config=cfg,
        )
    )
