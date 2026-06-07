#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_ROOT="${ASSETS_ROOT:-$ROOT_DIR/assets}"

SHAPENET_ROOT="${SHAPENET_ROOT:-$ASSETS_ROOT/data/shapenet}"
CO3D_ROOT="${CO3D_ROOT:-$ASSETS_ROOT/data/co3d}"
MODELNET_ROOT="${MODELNET_ROOT:-$ASSETS_ROOT/data/modelnet}"
SCANOBJNN_ROOT="${SCANOBJNN_ROOT:-$ASSETS_ROOT/data/scanobjnn}"

CLIP_PRETRAINED_SRC="${CLIP_PRETRAINED_SRC:-$ASSETS_ROOT/uni3D/trainedModel/clip_model/open_clip_pytorch_model.bin}"
UNI3D_CKPT_SRC="${UNI3D_CKPT_SRC:-$ASSETS_ROOT/uni3D/trainedModel/checkpoints/model_b.pt}"

usage() {
  cat <<'EOF'
Usage:
  bash prepare_assets.sh

This script creates symbolic links inside the Point-UQ runtime layout.

Environment overrides:
  ASSETS_ROOT
  SHAPENET_ROOT
  CO3D_ROOT
  MODELNET_ROOT
  SCANOBJNN_ROOT
  CLIP_PRETRAINED_SRC
  UNI3D_CKPT_SRC
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ensure_source_exists() {
  local src="$1"
  if [[ ! -e "$src" ]]; then
    echo "Missing source path: $src" >&2
    exit 1
  fi
}

ensure_safe_target() {
  local dst="$1"
  case "$dst" in
    "$ROOT_DIR"/*) ;;
    *)
      echo "Refusing to modify target outside Point-UQ: $dst" >&2
      exit 1
      ;;
  esac
}

link_path() {
  local src="$1"
  local dst="$2"
  ensure_source_exists "$src"
  ensure_safe_target "$dst"
  mkdir -p "$(dirname "$dst")"
  rm -rf "$dst"
  ln -s "$src" "$dst"
}

mkdir -p \
  "$ROOT_DIR/base_train" \
  "$ROOT_DIR/exp_results" \
  "$ROOT_DIR/data" \
  "$ROOT_DIR/uni3D/data" \
  "$ROOT_DIR/uni3D/trainedModel/clip_model" \
  "$ROOT_DIR/uni3D/trainedModel/checkpoints"

link_path "$ASSETS_ROOT/uni3D/__init__.py" "$ROOT_DIR/uni3D/__init__.py"
link_path "$ASSETS_ROOT/uni3D/data/__init__.py" "$ROOT_DIR/uni3D/data/__init__.py"
link_path "$ASSETS_ROOT/uni3D/data/datasets.py" "$ROOT_DIR/uni3D/data/datasets.py"
link_path "$ASSETS_ROOT/uni3D/data/dataset_catalog.json" "$ROOT_DIR/uni3D/data/dataset_catalog.json"
link_path "$ASSETS_ROOT/uni3D/data/labels.json" "$ROOT_DIR/uni3D/data/labels.json"
link_path "$ASSETS_ROOT/uni3D/data/templates.json" "$ROOT_DIR/uni3D/data/templates.json"
link_path "$ASSETS_ROOT/uni3D/data/utils" "$ROOT_DIR/uni3D/data/utils"
link_path "$ASSETS_ROOT/uni3D/models" "$ROOT_DIR/uni3D/models"
link_path "$ASSETS_ROOT/uni3D/utils" "$ROOT_DIR/uni3D/utils"

link_path "$SHAPENET_ROOT" "$ROOT_DIR/data/shapenet"
link_path "$CO3D_ROOT" "$ROOT_DIR/data/co3d"
link_path "$MODELNET_ROOT" "$ROOT_DIR/data/modelnet"
link_path "$SCANOBJNN_ROOT" "$ROOT_DIR/data/scanobjnn"

link_path "$CLIP_PRETRAINED_SRC" "$ROOT_DIR/uni3D/trainedModel/clip_model/open_clip_pytorch_model.bin"
link_path "$UNI3D_CKPT_SRC" "$ROOT_DIR/uni3D/trainedModel/checkpoints/model_b.pt"

printf 'Point-UQ assets prepared from %s\n' "$ASSETS_ROOT"
printf 'ShapeNet root: %s\n' "$SHAPENET_ROOT"
printf 'CO3D root: %s\n' "$CO3D_ROOT"
printf 'ModelNet root: %s\n' "$MODELNET_ROOT"
printf 'ScanObjectNN root: %s\n' "$SCANOBJNN_ROOT"
