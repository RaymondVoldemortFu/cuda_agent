from __future__ import annotations

from typing import Any, Literal, TypedDict


class ProbeState(TypedDict, total=False):
    """LangGraph 共享状态（编程与执行合并为单一路由键 programmer）。"""

    targets: list[str]
    plan: str
    programmer_rounds: int
    evidence_log: list[str]
    results: dict[str, Any]
    methodology: str
    _route: Literal["planner", "programmer", "synthesize"]
