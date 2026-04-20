from __future__ import annotations

import json
from pathlib import Path


def load_target_spec(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 target 规格文件: {path}。"
            "评测环境请挂载 /target/target_spec.json；"
            "本地请将 HW_PROBE_ENVIRONMENT=development（或使用 --dev），"
            "并在仓库根提供 target/target_spec.json，或设置 HW_PROBE_TARGET_SPEC_PATH 为相对路径。"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    targets = data.get("targets")
    if not isinstance(targets, list) or not all(isinstance(t, str) and t.strip() for t in targets):
        raise ValueError(
            f"{path} 格式无效：需要 JSON 对象且包含非空字符串列表字段 \"targets\"（与课程 project_description 一致）。"
        )
    return [t.strip() for t in targets]
