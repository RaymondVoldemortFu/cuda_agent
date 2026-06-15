#!/usr/bin/env bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HW_PROBE_MAX_TOTAL_RUNTIME_MINUTES="${HW_PROBE_MAX_TOTAL_RUNTIME_MINUTES:-25}"
export HW_PROBE_TIME_REMINDER_INTERVAL_MINUTES="${HW_PROBE_TIME_REMINDER_INTERVAL_MINUTES:-5}"

# Stage2 needs the system PyTorch/CUDA installation from the evaluation image.
# Install only the agent-side Python dependencies into the system/user Python,
# then run with python3 so torch remains visible.
if ! python3 - <<'PY'
import langgraph  # noqa: F401
import langchain_openai  # noqa: F401
import pydantic_settings  # noqa: F401
import ninja  # noqa: F401
PY
then
  python3 -m pip install --user -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
fi
exec python3 -m hw_probe.main "$@"
