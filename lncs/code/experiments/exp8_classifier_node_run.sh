#!/usr/bin/env bash
# One classifier-node run with RSS and board-power sampling (run ON Jetson #2).
#
#   ENGINE=onnx|numpy BOOTSTRAP=<coordinator>:9092 PY=<venv python> \
#   ./lncs/code/experiments/exp8_classifier_node_run.sh <out-dir>
#
# Starts tegrastats and scripts/classifier_node.py (batch 20, linger 1 s, fresh
# consumer group, --metrics-csv), samples the node's VmRSS every 0.5 s with the
# number of batches served, prints "[ready]" once the node is up and stops
# IDLE_AFTER seconds after the last batch (or after MAX_WAIT seconds). Writes
# <out-dir>/DONE when finished.
set -euo pipefail

OUT=${1:?out dir}
ENGINE=${ENGINE:?onnx or numpy}
BOOTSTRAP=${BOOTSTRAP:?}
PY=${PY:-python}
IDLE_AFTER=${IDLE_AFTER:-10}
MAX_WAIT=${MAX_WAIT:-600}
case "$ENGINE" in
  onnx)  MODEL=artifacts/ids_rf.onnx ;;
  numpy) MODEL=artifacts/ids_rf_numpy.npz ;;
esac
cd "$(dirname "$0")/../../.."
mkdir -p "$OUT"

tegrastats --interval 200 > "$OUT/tegrastats.log" &
TS=$!
$PY -u scripts/classifier_node.py --bootstrap "$BOOTSTRAP" --topic ids-suspicious-flow \
    --group-id "exp8-cls-$ENGINE-$(date +%s%N)" --engine "$ENGINE" --model "$MODEL" \
    --batch-size 20 --linger-ms 1000 --metrics-csv "$OUT/metrics.csv" > "$OUT/classifier.log" 2>&1 &
NODE=$!
trap 'kill $NODE $TS 2>/dev/null || true; touch "$OUT/DONE"' EXIT

echo "epoch,vmrss_kb,vmhwm_kb,batches" > "$OUT/monitor.csv"
started=0; begun=$(date +%s); last_change=$begun; prev=0
while kill -0 $NODE 2>/dev/null; do
  rss=$(awk '/^VmRSS/{print $2}' /proc/$NODE/status)
  hwm=$(awk '/^VmHWM/{print $2}' /proc/$NODE/status)
  n=0; [ -f "$OUT/metrics.csv" ] && n=$(( $(wc -l < "$OUT/metrics.csv") - 1 ))
  [ "$n" -lt 0 ] && n=0
  now=$(date +%s.%N)
  echo "$now,$rss,$hwm,$n" >> "$OUT/monitor.csv"
  if [ "$started" = 0 ] && grep -q '^\[OK\] Topic' "$OUT/classifier.log"; then
    started=1; echo "[ready]" > "$OUT/READY"
  fi
  if [ "$n" != "$prev" ]; then prev=$n; last_change=${now%.*}; fi
  t=${now%.*}
  if [ "$n" -gt 0 ] && [ $((t - last_change)) -gt "$IDLE_AFTER" ]; then break; fi
  if [ $((t - begun)) -gt "$MAX_WAIT" ]; then break; fi
  sleep 0.5
done
