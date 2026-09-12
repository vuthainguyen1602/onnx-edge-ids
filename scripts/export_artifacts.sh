#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

FEATURES_JSON="${FEATURES_JSON:-${ROOT_DIR}/artifacts/feature_columns.json}"
ONNX_MODEL="${ONNX_MODEL:-${ROOT_DIR}/artifacts/ids_rf.onnx}"
NUMPY_OUT="${NUMPY_OUT:-${ROOT_DIR}/artifacts/ids_rf_numpy.npz}"

mkdir -p "${ROOT_DIR}/artifacts"

echo "[INFO] ONNX model          : ${ONNX_MODEL}"
echo "[INFO] Feature columns     : ${FEATURES_JSON}"

python "${SCRIPT_DIR}/export_numpy.py" \
  --model "${ONNX_MODEL}" \
  --out "${NUMPY_OUT}" \
  --features "${FEATURES_JSON}"

echo "[OK] NumPy artifact: ${NUMPY_OUT}"
