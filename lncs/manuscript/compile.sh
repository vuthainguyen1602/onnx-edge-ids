#!/bin/bash
# Compile the ONNX-EdgeIDS paper (Springer LNCS + XeLaTeX), self-contained.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENDOR="$DIR/vendor"
OUT="$DIR/../output"

export TEXINPUTS="${VENDOR}//:${TEXINPUTS:-}"
export BSTINPUTS="${VENDOR}//:${BSTINPUTS:-}"

cd "$DIR"
mkdir -p "$OUT"

echo "Compiling main.tex with xelatex (vendored llncs.cls/splncs04.bst/fonts) ..."

xelatex -interaction=nonstopmode main.tex || true
bibtex main || true
xelatex -interaction=nonstopmode main.tex || true
xelatex -interaction=nonstopmode main.tex || true

cp main.pdf "$OUT/onnx-edge-ids-lncs.pdf"

echo ""
echo "PDF saved:"
echo "  $DIR/main.pdf"
echo "  $OUT/onnx-edge-ids-lncs.pdf"
