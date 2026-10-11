#!/usr/bin/env bash
# Gate-node RSS and board power while it serves a Kafka replay (run ON the board).
#
#   BOOTSTRAP=<mac-ip>:9092 ./lncs/code/experiments/exp6_gate_node_run.sh <out-dir>
#
# Starts tegrastats and scripts/gate_node.py (batch 100, linger 1 s, fresh
# consumer group), samples the gate's VmRSS every 0.5 s together with the number
# of batches it has logged, and waits. Start the producer from the coordinator
# once "[ready]" is printed; the script stops IDLE_AFTER seconds after the last
# batch. exp6_gate_node_summary.py turns the logs into exp6_gate_node.json.
set -euo pipefail

OUT=${1:?out dir}
BOOTSTRAP=${BOOTSTRAP:?set BOOTSTRAP=<mac-ip>:9092}
PY=${PY:-python}
IDLE_AFTER=${IDLE_AFTER:-15}
cd "$(dirname "$0")/../../.."
mkdir -p "$OUT"

tegrastats --interval 200 > "$OUT/tegrastats.log" &
TS=$!
$PY -u scripts/gate_node.py --bootstrap "$BOOTSTRAP" --group-id "exp6-gate-$(date +%s)" \
    --batch-size 100 --linger-ms 1000 > "$OUT/gate.log" 2>&1 &
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
  if [ "$started" = 0 ] && grep -q '^\[OK\] Gate' "$OUT/gate.log"; then
    started=1; echo "[ready] gate pid $GATE"
  fi
  if [ "$n" != "$prev" ]; then prev=$n; last_change=$now; fi
  if [ "$n" -gt 0 ] && awk -v a="$now" -v b="$last_change" -v t="$IDLE_AFTER" 'BEGIN{exit !(a-b>t)}'; then
    break
  fi
  sleep 0.5
done
echo "[done] $(grep -c '^\[[0-9,]*\] gate=' "$OUT/gate.log" || true) batches"
