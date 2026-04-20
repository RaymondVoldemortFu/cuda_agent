from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取 JSON 对象（支持 ```json 围栏）。"""
    s = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if m:
        s = m.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"未找到 JSON 对象: {text[:500]!r}")
    return json.loads(s[start : end + 1])
