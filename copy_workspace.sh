#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="/workspace"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Error: workspace directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/output_workspace"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEST_DIR="$OUTPUT_ROOT/$TIMESTAMP"

SOURCE_REAL="$(cd "$SOURCE_DIR" && pwd -P)"
PROJECT_REAL="$(cd "$PROJECT_ROOT" && pwd -P)"

if [[ "$SOURCE_REAL" == "$PROJECT_REAL" || "$PROJECT_REAL"/output_workspace == "$SOURCE_REAL"/* ]]; then
  echo "Error: refusing to copy a source directory that would include output_workspace recursively: $SOURCE_REAL" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp -a "$SOURCE_REAL"/. "$DEST_DIR"/

echo "$DEST_DIR"
