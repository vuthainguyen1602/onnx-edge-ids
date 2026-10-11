#!/usr/bin/env bash
# End-to-end (producer -> gate -> classifier verdict) latency over the deployed
# links. Run on the coordinator host, which also runs Kafka and the producer.
#
#   J1=bvdung@<jetson1> J2=bvdung@<jetson2> BOOTSTRAP=<coordinator-ip>:9092 \
#   RATE=100 LIMIT=10000 PASSES=1 ./lncs/code/experiments/exp7_e2e_run.sh <out-dir>
#
# Gate on Jetson #1 (its deployed checkout), ONNX classifier on Jetson #2 with
# --latency-csv, both on fresh consumer groups at their default batch/linger.
# Clock offsets of both boards against this host are taken before and after.
# exp7_e2e_summary.py turns <out-dir> into a JSON summary.
set -euo pipefail

OUT=${1:?out dir}
: "${J1:?}" "${J2:?}" "${BOOTSTRAP:?}"
RATE=${RATE:-100}; LIMIT=${LIMIT:-10000}; PASSES=${PASSES:-1}
CSV=${CSV:?set CSV=<replay csv>}
REMOTE_ROOT=${REMOTE_ROOT:-onnx-edge-ids-paper}
BENCH=${BENCH:-e2e-bench}
TAG=$(date +%Y%m%d_%H%M%S)
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=${PY:-$HERE/venv/bin/python}
mkdir -p "$OUT"
cd "$HERE"

echo "rate=$RATE limit=$LIMIT passes=$PASSES tag=$TAG" > "$OUT/run.txt"
$PY lncs/code/experiments/exp7_clock_offset.py "$J1" > "$OUT/offset_j1_before.json"
$PY lncs/code/experiments/exp7_clock_offset.py "$J2" > "$OUT/offset_j2_before.json"
ssh -o BatchMode=yes "$J1" uptime >> "$OUT/run.txt"
ssh -o BatchMode=yes "$J2" uptime >> "$OUT/run.txt"

# Classifier from a scratch copy of this checkout, so the board's own checkout is untouched.
ssh -o BatchMode=yes "$J2" "mkdir -p ~/$BENCH"
rsync -a --exclude __pycache__ scripts src artifacts "$J2:$BENCH/"
ssh -o BatchMode=yes "$J2" "cd ~/$BENCH && mkdir -p out/$TAG && \
  (setsid nohup ~/$REMOTE_ROOT/venv/bin/python -u scripts/classifier_node.py --bootstrap $BOOTSTRAP \
     --group-id e2e-cls-$TAG --engine onnx --batch-size 20 --linger-ms 1000 \
     --latency-csv out/$TAG/latency.csv > out/$TAG/classifier.log 2>&1 < /dev/null &)"
ssh -o BatchMode=yes "$J1" "cd ~/$REMOTE_ROOT && mkdir -p ~/$BENCH/out/$TAG && \
  (setsid nohup venv/bin/python -u scripts/gate_node.py --bootstrap $BOOTSTRAP \
     --group-id e2e-gate-$TAG --batch-size 100 --linger-ms 1000 \
     > ~/$BENCH/out/$TAG/gate.log 2>&1 < /dev/null &)"
sleep 12

for i in $(seq 1 "$PASSES"); do
  $PY scripts/csv_producer.py --csv "$CSV" --bootstrap "$BOOTSTRAP" --rate "$RATE" \
      --limit "$LIMIT" 2>&1 | grep -v -i warn | tail -1 | tee -a "$OUT/run.txt"
done
sleep 15

# [e]2e keeps pkill from matching the ssh shell that carries the pattern.
ssh -o BatchMode=yes "$J2" "pkill -f '[e]2e-cls-$TAG' || true"
ssh -o BatchMode=yes "$J1" "pkill -f '[e]2e-gate-$TAG' || true"
sleep 2
scp -q "$J2:$BENCH/out/$TAG/latency.csv" "$J2:$BENCH/out/$TAG/classifier.log" "$OUT/"
scp -q "$J1:$BENCH/out/$TAG/gate.log" "$OUT/"
$PY lncs/code/experiments/exp7_clock_offset.py "$J1" > "$OUT/offset_j1_after.json"
$PY lncs/code/experiments/exp7_clock_offset.py "$J2" > "$OUT/offset_j2_after.json"
echo "[done] $(($(wc -l < "$OUT/latency.csv") - 1)) verdicts in $OUT"
