#!/usr/bin/env bash
# Repeat the board measurements the paper reports from a single run (run on the coordinator).
#
#   J1=bvdung@<jetson1> J2=bvdung@<jetson2> COORD=<coordinator-ip> BOOTSTRAP=<coordinator-ip>:9092 \
#   CSV=/path/to/replay_cicids2017.csv REPS=5 ./lncs/code/experiments/exp18_board_repeats.sh <out-dir>
#
# Two parts, chosen with STEPS (default "e2e gate"):
#
#   e2e   exp7_e2e_run.sh REPS times for each configuration in E2E_CONFIGS
#         (transport:rate:limit:passes; rate 0 = the producer's maximum). The
#         defaults are the four rows of the end-to-end table, with the
#         parameters of the original runs: 6,000 records at 64 rows/s (ZeroMQ)
#         or 100 rows/s requested (Kafka, which reached 64), and 2 x 10,000
#         records at the maximum rate.
#   gate  exp6_gate_node_run.sh REPS times per transport on GATE_HOST (default
#         J2, as in the paper: Jetson #1 was busy), from a scratch copy of this
#         checkout. Kafka: the producer sends 2 x 10,000 records at its maximum
#         rate (about 950 rows/s through the broker). ZeroMQ: the producer sends
#         the same records at GATE_ZMQ_RATE (default 975 rows/s, the Kafka
#         rate), and zmq_sink.py drains the gate's output on this host.
#
#   energy  exp20_classifier_energy.py on ENERGY_HOST (default J2): idle, then
#         each classifier backend at saturation, then idle again, ENERGY_REPEATS
#         times, all in one session, for idle-subtracted energy per inference.
#         Run it when nothing else is using that board; it is not in the
#         default STEPS because it needs about 10 minutes of a quiet board.
#
# Then exp18_repeat_summary.py writes mean, SD, median and range per
# configuration. Kafka must be running on the coordinator for the Kafka runs.
# Every run directory is kept under <out-dir>, so a failed repeat can be redone
# alone (STEPS=gate GATE_TRANSPORTS=zeromq REPS=1 ...) without touching the others.
set -euo pipefail

OUT=${1:?out dir}
: "${J1:?}" "${J2:?}" "${CSV:?set CSV=<replay csv>}"
REPS=${REPS:-5}
STEPS=${STEPS:-"e2e gate"}
E2E_CONFIGS=${E2E_CONFIGS:-"zeromq:64:6000:1 zeromq:0:10000:2 kafka:100:6000:1 kafka:0:10000:2"}
GATE_TRANSPORTS=${GATE_TRANSPORTS:-"kafka zeromq"}
GATE_HOST=${GATE_HOST:-$J2}
GATE_ZMQ_RATE=${GATE_ZMQ_RATE:-975}
GATE_PASSES=${GATE_PASSES:-2}
PAUSE=${PAUSE:-20}
REMOTE_ROOT=${REMOTE_ROOT:-onnx-edge-ids-paper}
BENCH=${BENCH:-exp18-bench}
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=${PY:-$HERE/venv/bin/python}
EXP=lncs/code/experiments
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$HERE"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/series.txt"; }

if [[ " $STEPS " == *" e2e "* ]]; then
  dirs=()
  for cfg in $E2E_CONFIGS; do
    IFS=: read -r transport rate limit passes <<< "$cfg"
    name="${transport}_rate$([ "$rate" = 0 ] && echo max || echo "$rate")"
    for rep in $(seq 1 "$REPS"); do
      d="$OUT/e2e/${name}_rep${rep}"
      log "e2e $name rep $rep -> $d"
      if [ "$transport" = kafka ]; then : "${BOOTSTRAP:?set BOOTSTRAP for the Kafka runs}"; fi
      TRANSPORT=$transport RATE=$rate LIMIT=$limit PASSES=$passes CSV=$CSV J1=$J1 J2=$J2 \
        BOOTSTRAP=${BOOTSTRAP:-} REMOTE_ROOT=$REMOTE_ROOT PY=$PY \
        "$EXP/exp7_e2e_run.sh" "$d" > "$OUT/e2e_${name}_rep${rep}.log" 2>&1 \
        && log "$(tail -1 "$OUT/e2e_${name}_rep${rep}.log")" \
        || log "e2e $name rep $rep FAILED, see $OUT/e2e_${name}_rep${rep}.log"
      [ -s "$d/latency.csv" ] && dirs+=("$d")
      sleep "$PAUSE"
    done
  done
  if [ "${#dirs[@]}" -gt 0 ]; then
    $PY "$EXP/exp7_e2e_summary.py" "$OUT/exp18_e2e.json" "${dirs[@]}" > /dev/null
    log "e2e summary of ${#dirs[@]} runs -> $OUT/exp18_e2e.json"
  fi
fi

if [[ " $STEPS " == *" gate "* ]]; then
  : "${COORD:?set COORD=<coordinator-ip> (the ZeroMQ gate forwards to it, the Kafka gate reads from it)}"
  ssh -o BatchMode=yes "$GATE_HOST" "mkdir -p ~/$BENCH/$EXP"
  rsync -a --exclude __pycache__ scripts src artifacts "$GATE_HOST:$BENCH/"
  rsync -a "$EXP/exp6_gate_node_run.sh" "$GATE_HOST:$BENCH/$EXP/"
  for transport in $GATE_TRANSPORTS; do
    for rep in $(seq 1 "$REPS"); do
      run="gate_${transport}_rep${rep}"
      log "gate $transport rep $rep on $GATE_HOST"
      sink=""
      if [ "$transport" = zeromq ]; then
        $PY "$EXP/zmq_sink.py" --bind "tcp://*:5556" > "$OUT/${run}_sink.log" 2>&1 &
        sink=$!
        node_env="TRANSPORT=zeromq ZMQ_OUT=tcp://$COORD:5556"
        producer=(--transport zeromq --zmq-endpoint "tcp://${GATE_HOST#*@}:5555" --batch-size 100 --rate "$GATE_ZMQ_RATE")
      else
        node_env="BOOTSTRAP=${BOOTSTRAP:-$COORD:9092}"
        producer=(--bootstrap "${BOOTSTRAP:-$COORD:9092}" --rate 0)
      fi
      ssh -o BatchMode=yes "$GATE_HOST" "cd ~/$BENCH && rm -rf out/$run && mkdir -p out/$run && \
        ($node_env PY=\$HOME/$REMOTE_ROOT/venv/bin/python \
         setsid nohup ./$EXP/exp6_gate_node_run.sh out/$run > out/$run.runner.log 2>&1 < /dev/null &)"
      for _ in $(seq 1 90); do
        ssh -o BatchMode=yes "$GATE_HOST" "grep -q '^\[ready\]' ~/$BENCH/out/$run.runner.log" && break; sleep 1
      done
      sleep 3
      for p in $(seq 1 "$GATE_PASSES"); do
        $PY scripts/csv_producer.py --csv "$CSV" "${producer[@]}" 2>&1 | grep -v -i warn | tail -1 \
          | sed "s/^/$run pass$p: /" | tee -a "$OUT/series.txt"
      done
      for _ in $(seq 1 180); do
        ssh -o BatchMode=yes "$GATE_HOST" "grep -q '^\[done\]' ~/$BENCH/out/$run.runner.log" && break; sleep 2
      done
      [ -n "$sink" ] && { kill "$sink" 2>/dev/null || true; wait "$sink" 2>/dev/null || true; }
      mkdir -p "$OUT/gate/$run"
      scp -q "$GATE_HOST:$BENCH/out/$run/*" "$GATE_HOST:$BENCH/out/$run.runner.log" "$OUT/gate/$run/"
      $PY "$EXP/exp6_gate_node_summary.py" "$OUT/gate/$run" "$OUT/gate/$run.json" > /dev/null \
        && log "$run: $(grep -o '"records": [0-9]*' "$OUT/gate/$run.json")" \
        || log "$run: summary FAILED, see $OUT/gate/$run"
      sleep "$PAUSE"
    done
  done
fi

if [[ " $STEPS " == *" energy "* ]]; then
  EH=${ENERGY_HOST:-$J2}
  log "energy on $EH (batch ${ENERGY_BATCH:-20}, ${ENERGY_SECONDS:-60} s per window, ${ENERGY_REPEATS:-3} repeats)"
  ssh -o BatchMode=yes "$EH" "mkdir -p ~/$BENCH/$EXP"
  rsync -a --exclude __pycache__ scripts src artifacts "$EH:$BENCH/"
  rsync -a "$EXP/exp20_classifier_energy.py" "$EH:$BENCH/$EXP/"
  rsync -a "$CSV" "$EH:$BENCH/replay.csv"
  ssh -o BatchMode=yes "$EH" "cd ~/$BENCH && rm -rf out/energy && mkdir -p out && \
    ~/$REMOTE_ROOT/venv/bin/python $EXP/exp20_classifier_energy.py --csv replay.csv \
      --batch ${ENERGY_BATCH:-20} --seconds ${ENERGY_SECONDS:-60} --idle ${ENERGY_IDLE:-60} \
      --repeats ${ENERGY_REPEATS:-3} --workdir out/energy --out out/exp20_classifier_energy.json" \
    2>&1 | grep -v -i warn | tee -a "$OUT/series.txt"
  mkdir -p "$OUT/energy"
  scp -q -r "$EH:$BENCH/out/exp20_classifier_energy.json" "$EH:$BENCH/out/energy" "$OUT/energy/"
fi

$PY "$EXP/exp18_repeat_summary.py" "$OUT"
log "[done] $OUT"
