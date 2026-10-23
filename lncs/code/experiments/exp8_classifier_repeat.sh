#!/usr/bin/env bash
# Repeated classifier-node runs per backend (run on the coordinator).
#
#   J1=bvdung@<jetson1> J2=bvdung@<jetson2> BOOTSTRAP=<coordinator-ip>:9092 \
#   CSV=<replay csv> REPS=5 ./lncs/code/experiments/exp8_classifier_repeat.sh <out-dir>
#   TRANSPORT=zeromq J1=bvdung@<jetson1> J2=bvdung@<jetson2> \
#   CSV=<replay csv> REPS=5 ./lncs/code/experiments/exp8_classifier_repeat.sh <out-dir>
#
# The gate runs once on Jetson #1 for the whole series. For each backend and
# repeat, exp8_classifier_node_run.sh starts a fresh classifier on Jetson #2,
# the producer replays PASSES x the CSV at --rate RATE, and the run directory is
# pulled back. exp8_summary.py aggregates <out-dir>.
#
# Kafka (default): the gate runs from Jetson #1's own checkout and producer,
# gate and classifier meet at BOOTSTRAP. ZeroMQ: the gate runs from a scratch
# copy of this checkout on Jetson #1 (the board's own checkout predates the
# ZeroMQ transport), the producer pushes to <jetson1>:5555, the gate to
# <jetson2>:5556; the producer batches 100 rows, standing in for the Kafka
# gate's batch of 100.
set -euo pipefail

OUT=${1:?out dir}
TRANSPORT=${TRANSPORT:-kafka}
: "${J1:?}" "${J2:?}" "${CSV:?}"
REPS=${REPS:-5}; RATE=${RATE:-1000}; PASSES=${PASSES:-3}
ENGINES=${ENGINES:-"onnx numpy"}
REMOTE_ROOT=${REMOTE_ROOT:-onnx-edge-ids-paper}
BENCH=${BENCH:-exp8-bench}
TAG=$(date +%Y%m%d_%H%M%S)
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=${PY:-$HERE/venv/bin/python}
case "$TRANSPORT" in
  kafka)  : "${BOOTSTRAP:?}"
          GATE_DIR="~/$REMOTE_ROOT"
          GATE_ARGS="--bootstrap $BOOTSTRAP --group-id exp8-gate-$TAG --batch-size 100 --linger-ms 1000"
          NODE_ENV="BOOTSTRAP=$BOOTSTRAP"
          PRODUCER_ARGS=(--bootstrap "$BOOTSTRAP") ;;
  zeromq) GATE_DIR="~/$BENCH"
          GATE_ARGS="--transport zeromq --zmq-in 'tcp://*:5555' --zmq-out tcp://${J2#*@}:5556"
          NODE_ENV="TRANSPORT=zeromq"
          PRODUCER_ARGS=(--transport zeromq --zmq-endpoint "tcp://${J1#*@}:5555" --batch-size 100) ;;
  *) echo "TRANSPORT must be kafka or zeromq" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
cd "$HERE"

ssh -o BatchMode=yes "$J2" "mkdir -p ~/$BENCH"
rsync -a --exclude __pycache__ scripts src artifacts "$J2:$BENCH/"
rsync -a lncs/code/experiments/exp8_classifier_node_run.sh "$J2:$BENCH/lncs/code/experiments/" 2>/dev/null || {
  ssh -o BatchMode=yes "$J2" "mkdir -p ~/$BENCH/lncs/code/experiments"
  rsync -a lncs/code/experiments/exp8_classifier_node_run.sh "$J2:$BENCH/lncs/code/experiments/"; }
if [ "$TRANSPORT" = zeromq ]; then
  ssh -o BatchMode=yes "$J1" "mkdir -p ~/$BENCH"
  rsync -a --exclude __pycache__ scripts src artifacts "$J1:$BENCH/"
fi

# The gate's pid goes to a file: the ZeroMQ command line carries no tag to pkill by.
ssh -o BatchMode=yes "$J1" "cd $GATE_DIR && mkdir -p ~/$BENCH && \
  (setsid nohup ~/$REMOTE_ROOT/venv/bin/python -u scripts/gate_node.py $GATE_ARGS \
     > ~/$BENCH/gate_$TAG.log 2>&1 < /dev/null & echo \$! > ~/$BENCH/gate_$TAG.pid)"
sleep 10
echo "tag=$TAG transport=$TRANSPORT reps=$REPS rate=$RATE passes=$PASSES engines=$ENGINES" > "$OUT/series.txt"

for engine in $ENGINES; do
  for rep in $(seq 1 "$REPS"); do
    run="${engine}_rep${rep}"
    ssh -o BatchMode=yes "$J2" "cd ~/$BENCH && rm -rf out/$run && mkdir -p out/$run && \
      (ENGINE=$engine $NODE_ENV PY=\$HOME/$REMOTE_ROOT/venv/bin/python \
       setsid nohup ./lncs/code/experiments/exp8_classifier_node_run.sh out/$run \
       > out/$run/runner.log 2>&1 < /dev/null &)"
    for _ in $(seq 1 60); do
      ssh -o BatchMode=yes "$J2" "test -f ~/$BENCH/out/$run/READY" && break; sleep 1
    done
    sleep 3
    for p in $(seq 1 "$PASSES"); do
      $PY scripts/csv_producer.py --csv "$CSV" "${PRODUCER_ARGS[@]}" --rate "$RATE" \
          2>&1 | grep -v -i warn | tail -1 | sed "s/^/$run pass$p: /" | tee -a "$OUT/series.txt"
    done
    for _ in $(seq 1 120); do
      ssh -o BatchMode=yes "$J2" "test -f ~/$BENCH/out/$run/DONE" && break; sleep 2
    done
    mkdir -p "$OUT/$run"
    scp -q "$J2:$BENCH/out/$run/*" "$OUT/$run/"
    echo "[$run] $(($(wc -l < "$OUT/$run/metrics.csv") - 1)) batches" | tee -a "$OUT/series.txt"
    sleep 10
  done
done

ssh -o BatchMode=yes "$J1" "kill \$(cat ~/$BENCH/gate_$TAG.pid) 2>/dev/null || true"
scp -q "$J1:$BENCH/gate_$TAG.log" "$OUT/gate.log"
echo "[done] $OUT"
