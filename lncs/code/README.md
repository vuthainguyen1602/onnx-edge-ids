# Experiment code

Code for the measurements in the LNCS manuscript. The per-tree baseline, the float64 variant, the early-exit rule and the boundary prober are not part of the served package, so they live here; a test checks that this traversal and the packaged `NumpyInferenceEngine` return the same labels.

```
lncs/code/
  edgeids/forest.py      per-tree loop, block-lockstep traversal, exact early exit
  edgeids/onnx_forest.py generic traversal of any ONNX Scaler + TreeEnsembleClassifier graph (all split modes), boundary prober
  edgeids/data.py        labelled test split loader, deployed anomaly gate
  experiments/           one script per experiment, each writes results/<name>.json
  results/               JSON written by the last run (the numbers quoted in the paper)
  tests/                 run on the shipped artifact alone, no dataset needed
```

## Setup

For the full walk-through (data layout, gate artifacts, what each result field means, what depends on the machine) see [`../REPRODUCE.md`](../REPRODUCE.md).

```bash
python -m venv venv && source venv/bin/activate
pip install -e . && pip install -r lncs/code/requirements.txt
python -m pytest lncs/code/tests -q
```

## Artifacts

The served forests and feature order are read from `artifacts/` at the repository root. The gate artifacts (`artifacts/anomaly_*`) are not redistributed; copy them from the upstream thesis repository as described in `docs/deploy_2jetson.md`. `exp3` and `exp4` need them, `exp1`, `exp2` and the tests do not.

## Data

The experiments read the labelled CICIDS2017 **test split** as a Parquet directory with the 29 deployed feature columns (`artifacts/feature_columns.json`) plus `label` and `label_binary`. The dataset is not redistributed here; it comes from the Canadian Institute for Cybersecurity (Sharafaldin et al., ICISSP 2018). The split used for the paper has 447,275 rows (15.0% attacks). `exp4` also needs the CSE-CIC-IDS2018 test split prepared the same way (477,771 rows, 12.7% attacks, same 29 columns).

## Experiments

```bash
DATA=/path/to/test_data.parquet
python lncs/code/experiments/exp1_lockstep_benchmark.py --data $DATA   # Table: per-tree vs lockstep
python lncs/code/experiments/exp2_boundary_parity.py    --data $DATA   # Table: boundary-directed parity
python lncs/code/experiments/exp3_exact_early_exit.py   --data $DATA   # Table: exact early exit
python lncs/code/experiments/exp4_drift.py --in-dist $DATA --shifted /path/to/cse_cic_ids2018/test_data.parquet   # Table: distribution shift
python lncs/code/experiments/exp5_generalization.py --train /path/to/train_data.parquet --test $DATA   # Table: other forests, other converters
python lncs/code/experiments/exp5b_ort_label_regression.py --model lncs/code/results/exp5_models/sklearn_rf.onnx --data $DATA   # ONNX Runtime version sweep
```

`exp5` trains four more forests (scikit-learn RandomForest and HistGradientBoosting, LightGBM, XGBoost) on a 600,000-row sample of the training split, exports them with skl2onnx/onnxmltools and runs the boundary probe on each; it needs `lightgbm xgboost skl2onnx onnxmltools` (in `requirements.txt`) and writes the exported models to `results/exp5_models/` (not committed). `exp5b` installs each requested ONNX Runtime release into a throwaway venv and replays the test split through one exported model.

| Script | Question | Headline result (`results/*.json`) |
|--------|----------|-----------------------------------|
| `exp1` | Does advancing trees in lockstep beat one Python loop per tree, and where does it stop winning? | Jetson #2: 80x at batch 1, 20x at 20, 2.2x at 500, 1.2x at 2000 (Apple M3: 71x ... 0.9x) |
| `exp2` | Does replaying random rows find a runtime that disagrees with ONNX Runtime at split thresholds? | float64 traversal: 328 label flips on 287,223 boundary probes, 15 on 447,275 replayed rows; float32: 0 and 0 |
| `exp2b` | Does the path-satisfying construction earn its keep over writing boundary values into unrouted random rows? | unrouted probes at the same budget: 133 flips vs 328 (2.5x fewer), 27% of internal nodes exercised vs 99.9% reached; per exercising probe the densities agree |
| `exp3` | How many of the 200 trees does traffic need if labels must stay identical? | 104.1 on gate-forwarded flows, 1.9x on Jetson #2 (2.05x on M3), 50,548/50,548 labels identical |
| `exp4` | With model and gate frozen, what breaks on CSE-CIC-IDS2018 traffic and what does not? | F1 0.98 to 0.24, gate attack recall 0.72 to 0.01; float32 parity 0 flips and early exit exact on 477,771 shifted flows; undecided-at-first-check 4.8% to 28.2% |
| `exp5` | Do the probes generalize to other trainers and to converter parity (framework vs ONNX Runtime)? | float32 traversal 0 flips on every forest; probes 24 to 169x denser than replay where divergence is rare (sklearn RF, LightGBM, deployed RF), 3 to 7x where replay already sees 0.2 to 0.3% flips (HGB, XGBoost); trees/pair 72 (deployed RF, 32 bins) vs 4 (255 bins) vs 2 (exact splits) |
| `exp5b` | Does the ONNX Runtime release change the labels of a stock skl2onnx RandomForest export? | 1.26.0 and 1.27.0 label attack whenever P(attack) > 0 instead of > 0.5: 380,842 of 447,275 flows relabelled; 1.19.2 to 1.25.1 and 1.28.0 to 1.30.0 agree with scikit-learn |
| `exp6` | What does the gate node cost on the board while it serves a Kafka replay? | Jetson #2, batch 100: 21,191 rows/s inference, p95 0.0569 ms/row, 123.6 MB RSS, 5.25 W board power (4.75 W idle) |
| `exp7` | How long does a forwarded flow take from the producer to its verdict over Wi-Fi, and which stage sets it? | 64 rows/s: e2e p50 1.58 s, p95 2.11 s (classifier linger); max rate: p50 3.55 s, p95 6.44 s (queueing ahead of the gate, which drains about 965 rows/s) |
| `exp8` | What do the classifier node and cold start cost on the board over five runs, per traversal? | batch 20, Jetson #2, 25 W: ONNX 15,174 +/- 179 rows/s, per-tree NumPy (v0.2.0) 127 +/- 1, block-lockstep NumPy 2,752 +/- 17; cold start to first verdict ONNX 670 ms, per-tree 482 ms, lockstep 357 ms |

`results/` holds the Apple M3 (ARM64) run of every experiment. `exp1` and `exp3` were re-run on Jetson #2 (25 W mode, NumPy 2.2.6, ONNX Runtime 1.23.2) into `results_jetson/`, which the paper's traversal and early-exit tables quote; their label counts match the M3 run exactly. Label-agreement and disagreement counts do not depend on the machine, so none of these experiments needs Jetson hardware: any x86 or ARM64 machine with the dataset and artifacts reproduces the counts.

`anomaly_*.pkl` were pickled with an older scikit-learn; loading them under a newer one prints a version warning.
