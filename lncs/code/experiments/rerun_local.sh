#!/bin/bash
# Re-run every dataset experiment that needs no board, from the repository root:
#
#   DATA=/path/to/cicids2017/test_data.parquet \
#   SHIFTED=/path/to/cse_cic_ids2018/test_data.parquet \
#   TRAIN=/path/to/cicids2017/train_data.parquet \
#   REPLAY_CSV=/path/to/replay_cicids2017.csv \
#   ./lncs/code/experiments/rerun_local.sh
#
# Only DATA is required; steps whose input is unset are skipped. Results are
# written over lncs/code/results/*.json, so `git diff lncs/code/results` shows
# what moved. exp3/exp4/exp9 need the gate artifacts (artifacts/anomaly_*).
set -uo pipefail

: "${DATA:?set DATA to the CICIDS2017 test_data.parquet directory}"
EXP=lncs/code/experiments
LOG=lncs/code/results/rerun_local.log
mkdir -p lncs/code/results results
: > "$LOG"

run() {
    echo "=== $*" | tee -a "$LOG"
    if "$@" >> "$LOG" 2>&1; then echo "    ok" | tee -a "$LOG"; else echo "    FAILED (see $LOG)" | tee -a "$LOG"; fi
}

run python -m pytest -q tests lncs/code/tests
if [ -n "${REPLAY_CSV:-}" ]; then
    run python scripts/validate_parity.py --csv "$REPLAY_CSV" --rows 10000 --report-json results/parity_report.json
else
    run python scripts/validate_parity.py --rows 2000 --report-json results/parity_report.json
fi

run python $EXP/exp1_lockstep_benchmark.py --data "$DATA"
run python $EXP/exp2_boundary_parity.py --data "$DATA"
run python $EXP/exp2b_probe_ablation.py --data "$DATA"
run python $EXP/exp2c_confidence_gate.py --data "$DATA"

if ls artifacts/anomaly_* > /dev/null 2>&1; then
    run python $EXP/exp3_exact_early_exit.py --data "$DATA"
    if [ -n "${SHIFTED:-}" ]; then
        run python $EXP/exp4_drift.py --in-dist "$DATA" --shifted "$SHIFTED"
        run python $EXP/exp9_drift_alarm.py --in-dist "$DATA" --shifted "$SHIFTED"
    fi
else
    echo "=== skip exp3/exp4/exp9: no artifacts/anomaly_*" | tee -a "$LOG"
fi

if [ -n "${TRAIN:-}" ]; then
    run python $EXP/exp5_generalization.py --train "$TRAIN" --test "$DATA"
    run python $EXP/exp5b_ort_label_regression.py --model lncs/code/results/exp5_models/sklearn_rf.onnx --data "$DATA"
fi

if command -v xelatex > /dev/null; then
    run ./lncs/manuscript/compile.sh
fi

echo
git --no-pager diff --stat -- lncs/code/results
