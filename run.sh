#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -z "${PYTHON:-}" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  uv python install 3.12
  PYTHON="$(uv python find 3.12)"
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HW_PROBE_MAX_TOTAL_RUNTIME_MINUTES="${HW_PROBE_MAX_TOTAL_RUNTIME_MINUTES:-25}"
export HW_PROBE_TIME_REMINDER_INTERVAL_MINUTES="${HW_PROBE_TIME_REMINDER_INTERVAL_MINUTES:-5}"
export HW_PROBE_OUTPUT_FILENAME="${HW_PROBE_OUTPUT_FILENAME:-output3.json}"

# Stage3 needs the system PyTorch/CUDA installation from the evaluation image.
# Install only agent-side Python dependencies into the user site when missing,
# then run with python3 so the environment's torch/CUDA remains visible.
if ! "$PYTHON" - <<'PY'
import langgraph  # noqa: F401
import langchain_openai  # noqa: F401
import pydantic_settings  # noqa: F401
PY
then
  "$PYTHON" -m pip install --user -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
fi
exec "$PYTHON" -m hw_probe.main "$@"
