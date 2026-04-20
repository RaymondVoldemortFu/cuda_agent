#!/usr/bin/env bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f uv.lock ]]; then
  uv sync --frozen
else
  uv sync
fi
exec uv run python -m hw_probe.main "$@"
