from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_results_log(workspace: Path, log_name: str, line: str) -> None:
    path = workspace / log_name
    workspace.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def write_output_json(
    workspace: Path,
    filename: str,
    *,
    results: dict[str, Any],
    methodology: str,
    evidence: list[str],
) -> Path:
    """写入唯一交付物 output.json（内含 results，满足 README 单文件约束）。"""
    out = workspace / filename
    payload = {
        "results": results,
        "methodology": methodology,
        "evidence": evidence,
    }
    workspace.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
