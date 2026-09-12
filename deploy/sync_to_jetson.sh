#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <user>@<jetson2-ip>" >&2
  exit 2
fi

TARGET="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-~/onnx-edge-ids-paper}"

rsync -az --delete \
  --exclude ".git/" \
  --exclude ".idea/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "manuscript/*.pdf" \
  "${ROOT_DIR}/" "${TARGET}:${DEST}/"

echo "[OK] Synced ONNX-EdgeIDS to ${TARGET}:${DEST}"
