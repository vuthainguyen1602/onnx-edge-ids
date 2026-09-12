#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/artifacts/ids_pipeline_model}"
FEATURES_JSON="${FEATURES_JSON:-${ROOT_DIR}/artifacts/feature_columns.json}"
ONNX_OUT="${ONNX_OUT:-${ROOT_DIR}/artifacts/ids_rf.onnx}"
NUMPY_OUT="${NUMPY_OUT:-${ROOT_DIR}/artifacts/ids_rf_numpy.npz}"

mkdir -p "${ROOT_DIR}/artifacts"

echo "[INFO] Spark PipelineModel : ${MODEL_DIR}"
echo "[INFO] Feature columns     : ${FEATURES_JSON}"

python "${SCRIPT_DIR}/export_onnx.py" \
  --model "${MODEL_DIR}" \
  --out "${ONNX_OUT}" \
  --features "${FEATURES_JSON}"

python "${SCRIPT_DIR}/export_numpy.py" \
  --model "${MODEL_DIR}" \
  --out "${NUMPY_OUT}" \
  --features "${FEATURES_JSON}"

echo "[OK] ONNX artifact : ${ONNX_OUT}"
echo "[OK] NumPy artifact: ${NUMPY_OUT}"
