#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  bash run.sh <base_dataset> <incremental_dataset> [extra args...]

Examples:
  bash run.sh shapenet co3d
  bash run.sh shapenet scanobjnn
  bash run.sh modelnet scanobjnn
  bash run.sh shapenet null
  bash run.sh modelnet null
  bash run.sh co3d null

Environment:
  PYTHON_BIN=python
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BASE_DATASET="shapenet"
INCREMENTAL_DATASET="co3d"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  BASE_DATASET="$1"
  shift
fi

if [[ $# -gt 0 && "${1}" != -* ]]; then
  INCREMENTAL_DATASET="$1"
  shift
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" main.py \
  --base_dataset "$BASE_DATASET" \
  --incremental_dataset "$INCREMENTAL_DATASET" \
  "$@"
