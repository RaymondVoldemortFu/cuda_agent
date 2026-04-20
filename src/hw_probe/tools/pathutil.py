from __future__ import annotations

from pathlib import Path


def assert_under_workspace(workspace: Path, user_path: str | Path) -> Path:
    """将用户给定相对路径解析到工作区内；禁止跳出 workspace。"""
    root = workspace.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    raw = Path(user_path)
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"路径越界：{candidate} 不在工作区 {root} 内") from exc
    return candidate
