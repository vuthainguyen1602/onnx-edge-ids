#!/usr/bin/env bash
# End-to-end (producer -> gate -> classifier verdict) latency over the deployed
# links. Run on the coordinator host, which also runs the producer (and Kafka).
#
#   J1=bvdung@<jetson1> J2=bvdung@<jetson2> BOOTSTRAP=<coordinator-ip>:9092 \
#   RATE=100 LIMIT=10000 PASSES=1 ./lncs/code/experiments/exp7_e2e_run.sh <out-dir>
#   TRANSPORT=zeromq J1=bvdung@<jetson1> J2=bvdung@<jetson2> \
#   RATE=100 LIMIT=10000 PASSES=1 ./lncs/code/experiments/exp7_e2e_run.sh <out-dir>
#
# Gate on Jetson #1, ONNX classifier on Jetson #2 with --latency-csv, both at
# their default batch/linger. Clock offsets of both boards against this host
# are taken before and after. exp7_e2e_summary.py turns <out-dir> into a JSON
# summary.
#
# Kafka (default): the gate runs from Jetson #1's own checkout, both nodes on
# fresh consumer groups. ZeroMQ: the gate runs from a scratch copy of this
# checkout (the board's own checkout predates the ZeroMQ transport), the
# producer pushes 100-row batches to <jetson1>:5555 and the gate forwards to
# <jetson2>:5556. Rows are stamped as they are read, so a row's latency still
# includes the wait for its batch to fill, as with the Kafka gate's linger.
set -euo pipefail

OUT=${1:?out dir}
TRANSPORT=${TRANSPORT:-kafka}
: "${J1:?}" "${J2:?}"
RATE=${RATE:-100}; LIMIT=${LIMIT:-10000}; PASSES=${PASSES:-1}
CSV=${CSV:?set CSV=<replay csv>}
REMOTE_ROOT=${REMOTE_ROOT:-onnx-edge-ids-paper}
BENCH=${BENCH:-e2e-bench}
TAG=$(date +%Y%m%d_%H%M%S)
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=${PY:-$HERE/venv/bin/python}
case "$TRANSPORT" in
  kafka)  : "${BOOTSTRAP:?}"
          GATE_DIR="~/$REMOTE_ROOT"
          GATE_ARGS="--bootstrap $BOOTSTRAP --group-id e2e-gate-$TAG --batch-size 100 --linger-ms 1000"
          CLS_ARGS="--bootstrap $BOOTSTRAP --group-id e2e-cls-$TAG"
          PRODUCER_ARGS=(--bootstrap "$BOOTSTRAP") ;;
  zeromq) GATE_DIR="~/$BENCH"
          GATE_ARGS="--transport zeromq --zmq-in 'tcp://*:5555' --zmq-out tcp://${J2#*@}:5556"
          CLS_ARGS="--transport zeromq --zmq-endpoint 'tcp://*:5556'"
          PRODUCER_ARGS=(--transport zeromq --zmq-endpoint "tcp://${J1#*@}:5555" --batch-size 100) ;;
  *) echo "TRANSPORT must be kafka or zeromq" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
cd "$HERE"

echo "rate=$RATE limit=$LIMIT passes=$PASSES tag=$TAG transport=$TRANSPORT" > "$OUT/run.txt"
$PY lncs/code/experiments/exp7_clock_offset.py "$J1" > "$OUT/offset_j1_before.json"
$PY lncs/code/experiments/exp7_clock_offset.py "$J2" > "$OUT/offset_j2_before.json"
ssh -o BatchMode=yes "$J1" uptime >> "$OUT/run.txt"
ssh -o BatchMode=yes "$J2" uptime >> "$OUT/run.txt"

# Classifier from a scratch copy of this checkout, so the board's own checkout is untouched.
ssh -o BatchMode=yes "$J2" "mkdir -p ~/$BENCH"
rsync -a --exclude __pycache__ scripts src artifacts "$J2:$BENCH/"
if [ "$TRANSPORT" = zeromq ]; then
  ssh -o BatchMode=yes "$J1" "mkdir -p ~/$BENCH"
  rsync -a --exclude __pycache__ scripts src artifacts "$J1:$BENCH/"
fi
# Each node's pid goes to a file: the ZeroMQ command lines carry no tag to pkill by.
ssh -o BatchMode=yes "$J2" "cd ~/$BENCH && mkdir -p out/$TAG && \
  (setsid nohup ~/$REMOTE_ROOT/venv/bin/python -u scripts/classifier_node.py $CLS_ARGS \
     --engine onnx --batch-size 20 --linger-ms 1000 \
     --latency-csv out/$TAG/latency.csv > out/$TAG/classifier.log 2>&1 < /dev/null & echo \$! > out/$TAG/classifier.pid)"
ssh -o BatchMode=yes "$J1" "cd $GATE_DIR && mkdir -p ~/$BENCH/out/$TAG && \
  (setsid nohup ~/$REMOTE_ROOT/venv/bin/python -u scripts/gate_node.py $GATE_ARGS \
     > ~/$BENCH/out/$TAG/gate.log 2>&1 < /dev/null & echo \$! > ~/$BENCH/out/$TAG/gate.pid)"
sleep 12

for i in $(seq 1 "$PASSES"); do
  $PY scripts/csv_producer.py --csv "$CSV" "${PRODUCER_ARGS[@]}" --rate "$RATE" \
      --limit "$LIMIT" 2>&1 | grep -v -i warn | tail -1 | tee -a "$OUT/run.txt"
done
sleep 15

ssh -o BatchMode=yes "$J2" "kill \$(cat ~/$BENCH/out/$TAG/classifier.pid) 2>/dev/null || true"
ssh -o BatchMode=yes "$J1" "kill \$(cat ~/$BENCH/out/$TAG/gate.pid) 2>/dev/null || true"
sleep 2
scp -q "$J2:$BENCH/out/$TAG/latency.csv" "$J2:$BENCH/out/$TAG/classifier.log" "$OUT/"
scp -q "$J1:$BENCH/out/$TAG/gate.log" "$OUT/"
$PY lncs/code/experiments/exp7_clock_offset.py "$J1" > "$OUT/offset_j1_after.json"
$PY lncs/code/experiments/exp7_clock_offset.py "$J2" > "$OUT/offset_j2_after.json"
echo "[done] $(($(wc -l < "$OUT/latency.csv") - 1)) verdicts in $OUT"
