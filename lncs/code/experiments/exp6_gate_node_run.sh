#!/usr/bin/env bash
# Gate-node RSS and board power while it serves a replay (run ON the board).
#
#   BOOTSTRAP=<mac-ip>:9092 ./lncs/code/experiments/exp6_gate_node_run.sh <out-dir>
#   TRANSPORT=zeromq ZMQ_OUT=tcp://<mac-ip>:5556 ./lncs/code/experiments/exp6_gate_node_run.sh <out-dir>
#
# Starts tegrastats and scripts/gate_node.py, samples the gate's VmRSS every
# 0.5 s together with the number of batches it has logged, and waits. Start the
# producer from the coordinator once "[ready]" is printed; the script stops
# IDLE_AFTER seconds after the last batch. exp6_gate_node_summary.py turns the
# logs into exp6_gate_node.json.
#
# Kafka (default): batch 100, linger 1 s, fresh consumer group.
# ZeroMQ: the gate binds tcp://*:5555 and scores each batch as it arrives, so
# the producer's --batch-size 100 stands in for the gate's batch of 100. Its
# forwards go to ZMQ_OUT; run lncs/code/experiments/zmq_sink.py there first,
# since a PUSH socket with no peer blocks once its queue fills.
set -euo pipefail

OUT=${1:?out dir}
TRANSPORT=${TRANSPORT:-kafka}
PY=${PY:-python}
IDLE_AFTER=${IDLE_AFTER:-15}
case "$TRANSPORT" in
  kafka)  BOOTSTRAP=${BOOTSTRAP:?set BOOTSTRAP=<mac-ip>:9092}
          GATE_ARGS=(--bootstrap "$BOOTSTRAP" --group-id "exp6-gate-$(date +%s)" --batch-size 100 --linger-ms 1000) ;;
  zeromq) ZMQ_OUT=${ZMQ_OUT:?set ZMQ_OUT=tcp://<mac-ip>:5556}
          GATE_ARGS=(--transport zeromq --zmq-in "tcp://*:5555" --zmq-out "$ZMQ_OUT") ;;
  *) echo "TRANSPORT must be kafka or zeromq" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/../../.."
mkdir -p "$OUT"

tegrastats --interval 200 > "$OUT/tegrastats.log" &
TS=$!
$PY -u scripts/gate_node.py "${GATE_ARGS[@]}" > "$OUT/gate.log" 2>&1 &
GATE=$!
trap 'kill $GATE $TS 2>/dev/null || true' EXIT

echo "epoch,vmrss_kb,vmhwm_kb,batches" > "$OUT/monitor.csv"
started=0; last_change=$(date +%s.%N); prev=0
while kill -0 $GATE 2>/dev/null; do
  rss=$(awk '/^VmRSS/{print $2}' /proc/$GATE/status)
  hwm=$(awk '/^VmHWM/{print $2}' /proc/$GATE/status)
  n=$(grep -c '^\[[0-9,]*\] gate=' "$OUT/gate.log" || true)
  now=$(date +%s.%N)
  echo "$now,$rss,$hwm,$n" >> "$OUT/monitor.csv"
  # "[OK] Kafka Gate ..." or "[OK] ZeroMQ gate ..."
  if [ "$started" = 0 ] && grep -qi '^\[OK\] .*gate' "$OUT/gate.log"; then
    started=1; echo "[ready] gate pid $GATE"
  fi
  if [ "$n" != "$prev" ]; then prev=$n; last_change=$now; fi
  if [ "$n" -gt 0 ] && awk -v a="$now" -v b="$last_change" -v t="$IDLE_AFTER" 'BEGIN{exit !(a-b>t)}'; then
    break
  fi
  sleep 0.5
done
echo "[done] $(grep -c '^\[[0-9,]*\] gate=' "$OUT/gate.log" || true) batches"
