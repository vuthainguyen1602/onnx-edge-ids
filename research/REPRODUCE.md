# Reproducing the paper on your own machine

Everything in this guide runs on an ordinary laptop or server (x86-64 or ARM64, Linux or macOS). No Jetson board is needed: the disagreement and label-identity counts in the paper's tables are machine-independent, and throughput is labelled by machine wherever it appears. The one exception is the power column of the on-board table, which comes from `tegrastats` on the boards.

The steps go from a bare clone to regenerated result files and a compiled PDF. Steps 1 and 2 need no dataset. Steps 3 onward need the CICIDS2017 test split (and, for `exp4`, the CSE-CIC-IDS2018 test split).

## What you need

| Item | Notes |
|------|-------|
| Python 3.9 or newer | The paper's runs used 3.12.4 |
| A C compiler is **not** needed | Every dependency ships wheels for x86-64 and ARM64 |
| Disk | Your Parquet splits, about 11 MB for the `exp5` exported models, and one throwaway virtualenv per ONNX Runtime release for `exp5b` |
| Time (Apple M3) | `exp2`: about 2 minutes; `exp5`: training the four forests takes about a minute, probing them longer; `exp5b`: dominated by seven `pip install onnxruntime==...` downloads; the rest: minutes |
| XeLaTeX | Only for Step 7 (the PDF); TeX Live or MacTeX |

Library versions the paper's result files were produced with (`research/code/results/exp5_generalization.json` records them):

| numpy | onnxruntime | scikit-learn | lightgbm | xgboost | skl2onnx | onnxmltools | pyarrow |
|-------|-------------|--------------|----------|---------|----------|-------------|---------|
| 2.5.3 | 1.30.0 | 1.9.1 | 4.7.0 | 3.4.1 | 1.20.0 | 1.16.0 | 25.0.1 |

Newer versions should work. ONNX Runtime is the one to pin if you want the counts against ONNX Runtime to match exactly: a different release is a different runtime, and `exp5b` exists precisely because two releases label differently.

## Step 1: clone and install

```bash
git clone https://github.com/vuthainguyen1602/onnx-edge-ids.git
cd onnx-edge-ids
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -e ".[kafka,zeromq]"             # the served package (numpy, kafka-python, pyzmq)
pip install -e research/code                  # the experiments' library (edgeids)
pip install -r research/code/requirements.txt    # experiments: onnxruntime, pyarrow, scikit-learn, lightgbm, xgboost, skl2onnx, onnxmltools, pytest
```

The served forest is already in the repository: `artifacts/ids_rf.onnx` (ONNX), `artifacts/ids_rf_numpy.npz` (the NumPy export of the same graph) and `artifacts/feature_columns.json` (the 29 input columns in order).

## Step 2: checks that need no dataset

Run the test suite. It loads the shipped artifact, checks that the per-tree, block-lockstep and early-exit traversals agree, that the generic ONNX traversal reproduces ONNX Runtime, and that the experiment traversal matches the packaged `NumpyInferenceEngine`:

```bash
python -m pytest research/code/tests -q
```

Then exercise both serving backends on random rows and run the replay parity gate the same way (omit `--csv` and it probes with random flows):

```bash
python scripts/smoke_predict.py --engine onnx  --model artifacts/ids_rf.onnx
python scripts/smoke_predict.py --engine numpy --model artifacts/ids_rf_numpy.npz
python scripts/validate_parity.py --rows 2000 --report-json results/parity_report.json
```

`validate_parity.py` exits non-zero if label agreement drops below 100% or the confidence delta exceeds 1e-5. You can also re-export the NumPy artifact from the ONNX file and confirm it is byte-for-byte what is committed:

```bash
NUMPY_OUT=/tmp/ids_rf_numpy.npz ./scripts/export_artifacts.sh
python - <<'PY'
import numpy as np
a, b = np.load("artifacts/ids_rf_numpy.npz"), np.load("/tmp/ids_rf_numpy.npz")
assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files); print("identical")
PY
```

If everything above passes, the software side is reproduced. The rest is about the data.

## Step 3: prepare the data

The datasets are not redistributed. Get them from the Canadian Institute for Cybersecurity and cite the source papers (see `docs/data.md`):

- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html (Sharafaldin et al., ICISSP 2018)
- CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html (only for `exp4`)

The experiments read each split as a **directory of Parquet part files** (`<dir>/*.parquet`) with these columns:

- the 29 feature columns named exactly as in `artifacts/feature_columns.json` (CICFlowMeter headers lower-cased, spaces and slashes replaced by underscores: `flow_duration`, `total_fwd_packets`, `flow_bytes_s`, ...), numeric;
- `label`: the attack name string (`BENIGN`, `DoS Hulk`, ...);
- `label_binary`: integer, 0 for benign and 1 for attack.

Extra columns are ignored. NaN and Inf in the features are set to 0 by the loader, the same policy as the serving path. The splits behind the paper's numbers:

| Split | Rows | Attack share | Used by |
|-------|------|--------------|---------|
| CICIDS2017 test | 447,275 | 15.0% | `exp1` to `exp5b` |
| CICIDS2017 train | the rest of the same dataset | | `exp5` samples 600,000 rows from it |
| CSE-CIC-IDS2018 test | 477,771 | 12.7% | `exp4` |

If you build the split yourself, the pandas skeleton below produces the expected layout from any DataFrame with those columns:

```python
import json, pandas as pd
cols = json.load(open("artifacts/feature_columns.json"))
df = ...                                        # your merged, cleaned CICIDS2017 frame
df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]
df["label_binary"] = (df["label"].str.upper() != "BENIGN").astype("int32")
test = df.sample(frac=0.2, random_state=42)
train = df.drop(test.index)
for name, part in [("train_data.parquet", train), ("test_data.parquet", test)]:
    part[cols + ["label", "label_binary"]].to_parquet(f"data/{name}/part-0.parquet", index=False)
```

A different split reproduces every mechanism the paper reports (0 flips for the float32 traversal, probes denser than replay, the ONNX Runtime 1.26/1.27 relabelling) but not the exact row counts, which belong to the paper's split. The boundary probes of `exp2` do not depend on the split at all beyond the row count: they are derived from the forest.

For Step 6 (the Kafka pipeline) and the replay gate on real flows you also want a CSV with the 29 feature columns as header:

```python
pd.read_parquet("data/test_data.parquet")[cols].head(20000).to_csv("replay_cicids2017.csv", index=False)
```

This CSV exercises the pipeline but does not reproduce the board figures of the paper. The board runs (`exp6`, `exp7`, `exp8` and their `*_kafkareplay` ZeroMQ counterparts in `code/results_jetson/`) replayed `jetson/sender/replay_cicids2017.csv` from the upstream thesis repository (10,000 rows rebuilt from that repository's own test split, 77 columns of which the producer sends the 29 above, sha1 `b4ec528bf3cb…`). The gate forwards 1,181 of its rows (11.81%), against 1,560 of the first 10,000 rows above (15.6%) and 1,694 of the first 20,000 (8.47%), and forwarded flows set the classifier's per-row cost: on the same board and transport, the first 10,000 rows above give ONNX Runtime 1.8x the throughput of the thesis replay (`exp8_summary_zeromq_head10k.json` vs `exp8_summary_zeromq_kafkareplay.json`). Use the thesis CSV to compare against the paper's board numbers.

## Step 4: gate artifacts (only for `exp3`, `exp4` and the pipeline)

The anomaly gate (autoencoder, scaler, threshold) is not redistributed. Copy `anomaly_autoencoder.pkl`, `anomaly_scaler.pkl` and `anomaly_threshold.json` into `artifacts/` from the upstream thesis repository as described in `docs/deploy_2jetson.md`. `exp1`, `exp2`, `exp2b`, `exp5`, `exp5b` and the tests do not need them. The pickles were written with an older scikit-learn and print a version warning on load; that is expected.

## Step 5: run the experiments

Each script writes one JSON under `research/code/results/`, overwriting the committed file of the same name, so `git diff research/code/results/` afterwards shows exactly what changed against the paper. Run them from the repository root:

```bash
DATA=/path/to/data/test_data.parquet
TRAIN=/path/to/data/train_data.parquet
DATA2018=/path/to/data_2018/test_data.parquet

python research/code/experiments/exp1_lockstep_benchmark.py --data $DATA
python research/code/experiments/exp1b_whole_forest_lockstep.py --data $DATA
python research/code/experiments/exp2_boundary_parity.py    --data $DATA
python research/code/experiments/exp2b_probe_ablation.py    --data $DATA        # reads exp2's JSON for the density ratios
python research/code/experiments/exp3_exact_early_exit.py   --data $DATA        # needs Step 4
python research/code/experiments/exp4_drift.py --in-dist $DATA --shifted $DATA2018   # needs Step 4
python research/code/experiments/exp5_generalization.py --train $TRAIN --test $DATA
python research/code/experiments/exp5b_ort_label_regression.py \
    --model research/code/results/exp5_models/sklearn_rf.onnx --data $DATA
python research/code/experiments/exp11_probe_seeds_and_faults.py --data $DATA   # ten base-row draws, data-free probes, six injected faults (~15 min)
for s in 1 2 3 4; do                                                        # four more training seeds (~9 min each)
  python research/code/experiments/exp5_generalization.py --train $TRAIN --test $DATA --seed $s \
      --models sklearn_rf,sklearn_hgb,lightgbm,xgboost_hist \
      --model-dir /tmp/exp5_models_seed$s --out research/code/results/exp5_seeds/seed$s.json
done
python research/code/experiments/exp5_seeds_summary.py                          # ranges over seeds 0 to 4
python research/code/experiments/exp12_hgb_threshold_rounding.py                # no dataset; rerun after installing skl2onnx main to see the fix
python research/code/experiments/exp12b_cicids_hgb_frontend.py --train $TRAIN --test $DATA
python research/code/experiments/exp14_fleet_divergence.py --data $DATA         # fleet divergence on real attack flows
python research/code/experiments/exp15_converter_sweep.py --train $TRAIN --test $DATA --out research/code/results/exp15_converter_sweep_released.json
python research/code/experiments/exp11b_fault_confidence_gate.py --data $DATA   # the six faults against a gate that also checks confidences

UNSW=/path/to/unsw_nb15     # UNSW_NB15_training-set.csv (175,341 rows) and UNSW_NB15_testing-set.csv (82,332 rows)
python research/code/experiments/exp10_cross_dataset.py --name UNSW-NB15 \
    --train $UNSW/UNSW_NB15_training-set.csv --test $UNSW/UNSW_NB15_testing-set.csv --label-col label
TON=/path/to/ton_iot/Train_Test_Network_dataset   # train_test_network.csv (211,043 rows); no test file, so a stratified 70/30 split
python research/code/experiments/exp10_cross_dataset.py --name ToN-IoT --train $TON/train_test_network.csv --label-col label
python research/code/experiments/exp10_table.py                                 # LaTeX rows from exp10's JSON
for s in 1 2 3 4; do                                                        # four more training seeds per dataset
  python research/code/experiments/exp10_cross_dataset.py --name UNSW-NB15 --train $UNSW/UNSW_NB15_training-set.csv \
      --test $UNSW/UNSW_NB15_testing-set.csv --label-col label --seed $s \
      --model-dir /tmp/exp10_models_seed$s --out research/code/results/exp10_seeds/unsw_nb15_seed$s.json
  python research/code/experiments/exp10_cross_dataset.py --name ToN-IoT --train $TON/train_test_network.csv --label-col label \
      --seed $s --model-dir /tmp/exp10_models_seed$s --out research/code/results/exp10_seeds/ton_iot_seed$s.json
done
python research/code/experiments/exp10_seeds_summary.py                         # ranges over seeds 0 to 4
python research/code/experiments/seeds_table.py                                # LaTeX rows of the five-seed appendix table
python research/code/experiments/exp5c_float32_ties.py --unsw $UNSW --run 2:/tmp/exp10_models_seed2/unsw_nb15 \
    --out research/code/results/exp10c_float32_ties_unsw.json                   # the 3 float32 flips of seed 2 are ties
# replay with the 1e-5 confidence check on all 13 models (runtime parity), and converter parity
# with the same check (retrains the 12 models with seed 0; the label-flip counts must match exp5/exp10)
python research/code/experiments/exp16_confidence_replay.py --cicids $DATA --unsw $UNSW --ton-iot $TON/train_test_network.csv
python research/code/experiments/exp17_converter_confidence.py --cicids-train $TRAIN --cicids-test $DATA \
    --unsw $UNSW --ton-iot $TON/train_test_network.csv
# the release gate's data-free probes (scaler-offset base rows, all nodes) on all 13 models, with timings;
# --converter retrains the 12 models as exp17 does and also tests framework vs export (~5 min on an M3)
python research/code/experiments/exp19_datafree_probes.py --converter --cicids-train $TRAIN --cicids-test $DATA \
    --unsw $UNSW --ton-iot $TON/train_test_network.csv
# Leafparity 0.1.2 on the 12 converter pairs; run from a separate venv with leafparity and the exp5 library
# versions (scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, ONNX Runtime 1.30), e.g. .lpvenv/
.lpvenv/bin/python research/code/experiments/exp21_leafparity.py --cicids-train $TRAIN --cicids-test $DATA \
    --unsw $UNSW --ton-iot $TON/train_test_network.csv
# Treelite 4.8 (GTIL) as a third runtime, rebuilt from each ONNX graph's node table; needs Python >= 3.11 and
# treelite in its own venv, e.g. .tlvenv/ (plus the numpy/onnxruntime/scikit-learn versions above)
.tlvenv/bin/python research/code/experiments/exp22_treelite_runtime.py --cicids $DATA --unsw $UNSW --ton-iot $TON/train_test_network.csv
# lockstep block-budget ablation; --export-npy writes the 20,000 rows so a board without pandas can run the same rows
python research/code/experiments/exp23_lockstep_block_ablation.py --data $DATA
python research/code/experiments/exp23_lockstep_block_ablation.py --data $DATA --export-npy rows20k.npy   # then on the board: --npy rows20k.npy
# do the three probes straddle the threshold after float32 scaling? plus a two-probe set on the exact float32
# crossing (bisection), both through the data-free gate on all 13 models; needs no data (~6 min on an M3)
python research/code/experiments/exp24_probe_straddle.py
# exporter faults injected into the .npz of two models, against every gate (~20 min on an M3)
python research/code/experiments/exp25_exporter_mutations.py --data $DATA
python research/code/experiments/exp25_table.py                               # LaTeX rows of the fault table
# exact early exit on the twelve trained models
python research/code/experiments/exp26_early_exit_all_models.py --cicids $DATA --unsw $UNSW --ton-iot $TON/train_test_network.csv
# shipped exporter, engine and gate on all thirteen models (structure, probes, reach)
python research/code/experiments/exp28_shipped_toolkit_all_models.py
# fault localization: the gate's two levels on the fourteen exporter faults and the six arithmetic faults
python research/code/experiments/exp29_fault_localization.py
python research/code/experiments/exp29_table.py                               # LaTeX rows of the localization table
# probe completeness: exact per-node disagreement sets of six arithmetic faults on the 13 models (seconds; no data)
python research/code/experiments/exp31_probe_completeness.py
python research/code/experiments/exp31_table.py                               # LaTeX rows of the completeness table
# probes vs. replay, uniform, boundary and coverage-guided fuzzing on the same faults and budgets (~4 h on an M3)
python research/code/experiments/exp30_testing_baselines.py --data $DATA
python research/code/experiments/exp30_table.py                               # LaTeX rows of the baselines table
# cost of the shipped gate's probe check on the 13 models, crossing pair vs. three-probe set (~10 min)
python research/code/experiments/exp32_gate_timing.py
# ranked localization (top-1, top-3, MRR) over exporter and single-node faults, crossing pair (~5 min)
python research/code/experiments/exp33_localization_ranking.py
# gradual shift: windows mixing CICIDS2017 and CSE-CIC-IDS2018 in steps from 0 to 100% (~5 min)
python research/code/experiments/exp34_gradual_drift.py --in-dist $DATA --shifted $DATA2018
# Treelite compiled to C (TL2cgen, needs a C compiler) against ONNX Runtime on the data-free probes; in .tlvenv/ plus `pip install tl2cgen`
.tlvenv/bin/python research/code/experiments/exp35_treelite_compiled.py
# replay gate: 1,000 repeated 2,000-row draws (uniform, stratified proportional and balanced) and a
# confidence-tolerance sweep 1e-7..1e-2 on all 13 models, runtime parity (~5 min)
python research/code/experiments/exp36_replay_sampling.py --cicids $DATA --unsw $UNSW --ton-iot $TON/train_test_network.csv
# ablation: node, branch and crossing coverage, float64 flips and single-node faults of the six exp30 generators (~5 min)
python research/code/experiments/exp37_generator_coverage.py --data $DATA
# cross-architecture ties: write probe sets, run both runtimes on each machine, compare.
# the two tie-prone models are the CICIDS2017 RF of seed 4 (exp5 --seed 4 --models sklearn_rf) and the UNSW-NB15 RF of seed 2
python research/code/experiments/exp27_make_probe_sets.py --out $EXP27 --cicids $DATA --unsw $UNSW \
    --ton-iot $TON/train_test_network.csv --extra CICIDS2017:4:/tmp/exp5_models_seed4/sklearn_rf.onnx \
    --extra UNSW-NB15:2:/tmp/exp10_models_seed2/unsw_nb15/sklearn_rf.onnx
for f in research/code/experiments/exp27_run_ort.py research/code/experiments/exp27_run_numpy.py; do
  python $f --dir $EXP27 --tag m3                                              # repeat with --tag x86 in an x86-64
done                                                                           # container and --tag jetson on the board
python research/code/experiments/exp27_compare.py --dir $EXP27 --pair m3:x86 --pair m3:jetson \
    --pair m3_numpy:x86_numpy --pair m3_numpy:jetson_numpy --pair m3:m3_numpy --pair x86:x86_numpy --pair jetson:jetson_numpy
# exp13 (GPU providers) runs on Jetson #2 with a GPU build of ONNX Runtime; see results_jetson/exp13_ort_providers.json
```

What to expect, and which numbers depend on your machine:

| Script | Paper table | Machine-independent | Machine-dependent |
|--------|-------------|---------------------|-------------------|
| `exp1` | per-tree vs lockstep | label agreement on 5,000 rows | all rows/s and speedups (71x at batch 1 ... 0.9x at 2,000 on Apple M3; 80x ... 1.2x on Jetson #2) |
| `exp2` | boundary parity | 95,741 internal nodes, 815 distinct (feature, threshold) pairs, 287,223 probes, float64 flips 328 on probes and 15 on replay, float32 flips 0 and 0 | the `seconds` field |
| `exp2b` | probe ablation (unrouted row of the boundary table) | 133 float64 flips on 286,065 unrouted probes (117 rows per pair, seed 0), 38% of probes exercising their pair, 27% of internal nodes exercised, float32 flips 0 | the `seconds` field |
| `exp3` | exact early exit | 104.1 of 200 trees on gate-forwarded rows, labels identical on every row, share of rows decided at the first check | the `*_seconds` and `speedup` fields (2.05x on Apple M3, 1.94x on Jetson #2) |
| `exp4` | distribution shift | F1 and gate recall on both splits, 0 parity flips, early exit exact, undecided share 4.8% vs 28.2% | none |
| `exp5` | other forests | float32 flips 0 on every forest; probe-vs-replay density ratios; trees per shared threshold (72 / 4 / 2) | `train_seconds` |
| `exp10` | UNSW-NB15 and ToN-IoT (`tab:datasets`) | 39 / 14 features; float32 flips 0 on every model; float64 / framework flips on probes and replay; framework fed the graph's float32 scaling: 0-5 replay flips, except 1,014 for UNSW-NB15 HGB | `train_seconds`, `seconds` |
| `exp5b` | release sweep | 1.26.0 and 1.27.0 label attack whenever P > 0 (negative first probability column); the other releases agree with scikit-learn | none |

Notes:

- `exp5` trains scikit-learn RandomForest and HistGradientBoosting, LightGBM and XGBoost on a 600,000-row sample (`--train-rows`, seed 0), writes the exported models to `research/code/results/exp5_models/` and probes up to 100,000 internal nodes per model (`--max-nodes`). The counts for the four trained forests depend on the training sample; the row for the deployed forest (`deployed_rf`) does not. Use `--models` to run a subset.
- `exp5b` creates one virtualenv per release under `$TMPDIR/exp5b_ort_venvs` (`--venv-dir`) and installs `onnxruntime==<version>` into each, so it needs network access and a Python for which wheels of every listed release exist (the paper's run used Python 3.12; trim `--versions` if a release has no wheel for yours). The committed result was taken on this same export (`exp5` with the default seed and 600,000 training rows), so a re-run with the defaults reproduces the count exactly; a different training sample reproduces the relabelling pattern (0 flips on agreeing releases, hundreds of thousands on 1.26.0 and 1.27.0), not the count.
- `exp10` uses the whole UNSW-NB15 training and test files (both are under the `--train-rows`/`--test-rows` caps), so with the versions above a re-run reproduces the committed counts. In five full runs one count moved once: LightGBM's framework-vs-ORT flips on the on-threshold probes read 89 instead of 90 (261 instead of 262 in total); the trained model and every other count were identical. A run replaces the whole entry for `--name`, so run a `--models` subset with its own `--out` or it drops the other models from the committed file.
- `exp1` timings are noisy on a busy machine. Close other work and, on laptops, keep the machine on mains power.

## Step 6 (optional): the two-node pipeline on two ordinary hosts

The on-board table needs the pipeline, not the boards. Any two Linux/macOS hosts plus a third (or one of the two) as producer will do, with a Kafka broker on it for the Kafka transport. ZeroMQ needs no broker (`pip install -e ".[zeromq]"`):

```bash
# classifier host (binds :5556; serves exactly --batch-size rows unless the linger expires)
python scripts/classifier_node.py --transport zeromq --zmq-endpoint 'tcp://*:5556' --engine onnx --batch-size 20 --metrics-csv results/onnx_b20.csv

# gate host (binds :5555, scores each producer batch as it arrives)
python scripts/gate_node.py --transport zeromq --zmq-in 'tcp://*:5555' --zmq-out tcp://<classifier-ip>:5556

# producer host: 100-row batches, standing in for the Kafka gate's batch of 100
python scripts/csv_producer.py --csv replay_cicids2017.csv --transport zeromq --zmq-endpoint tcp://<gate-ip>:5555 --batch-size 100 --rate 0
```

The board scripts take `TRANSPORT=zeromq` for the same runs: `exp6_gate_node_run.sh` (with `zmq_sink.py` draining the gate's output on the coordinator), `exp7_e2e_run.sh` and `exp8_classifier_repeat.sh`. The paper's ZeroMQ figures are the `*_kafkareplay` results; the gate needs scikit-learn and joblib in the venv it runs from.

**Repeating the single-run board figures.** `exp18_board_repeats.sh` (run on the coordinator) repeats, `REPS` times each, the four end-to-end rows and the gate-node rows over Kafka and ZeroMQ, with the parameters of the original runs, and `exp18_repeat_summary.py` reports mean, SD, median and range per configuration:

```bash
J1=bvdung@<jetson1> J2=bvdung@<jetson2> COORD=<coordinator-ip> BOOTSTRAP=<coordinator-ip>:9092 \
CSV=/path/to/replay_cicids2017.csv REPS=5 ./research/code/experiments/exp18_board_repeats.sh results_jetson_repeats/
```

`STEPS=energy` (not in the default) measures idle-subtracted inference energy per backend with `exp20_classifier_energy.py` on a quiet Jetson #2: idle, each backend at saturation, idle again, three times, in one session (about 10 minutes). `STEPS=e2e` or `STEPS=gate` runs one part; `E2E_CONFIGS` and `GATE_TRANSPORTS` narrow it further, e.g. to redo one failed repeat. Kafka must be running on the coordinator for the Kafka runs, and both boards need the venv the earlier board runs used, at `~/onnx-edge-ids-paper/venv` (with scikit-learn and joblib for the gate).

For Kafka, follow `docs/deploy_2jetson.md` and read "Jetson #1" as the gate host and "Jetson #2" as the classifier host. The only board-specific parts are `tegrastats` for the power column and `requirements_jetson_onnx.txt` for the ARM64 ONNX Runtime wheel; on a laptop or server, `pip install -e .[onnx]` installs the runtime instead.

```bash
# Kafka host (bring up any Kafka broker on :9092, then)
python scripts/init_topics.py --bootstrap localhost:9092 --partitions 2

# gate host
python scripts/gate_node.py --bootstrap <kafka-ip>:9092

# classifier host, once per backend and batch size
python scripts/classifier_node.py --bootstrap <kafka-ip>:9092 --engine onnx  --batch-size 20 --metrics-csv results/onnx_b20.csv
python scripts/classifier_node.py --bootstrap <kafka-ip>:9092 --engine numpy --batch-size 20 --metrics-csv results/numpy_b20.csv

# Kafka host: replay flows
python scripts/csv_producer.py --csv replay_cicids2017.csv --rate 0
```

Throughput, p95 latency, RSS and cold start will differ from the Jetson figures and should be reported as measured on your hosts. Label agreement between the two backends and the forward rate of the gate should not differ.

## Step 7 (optional): compare with the paper

The manuscript sources are not distributed in this repository. Numbers in the paper are typed in, not generated, so after re-running an experiment compare `git diff research/code/results/` with the paper's tables by hand.

## If something does not match

- **Counts differ against ONNX Runtime but the float32 traversal still shows 0 flips:** check `python -c "import onnxruntime; print(onnxruntime.__version__)"`. The paper's counts are against 1.30.0 with the CPU provider.
- **`exp2` flips differ by a handful on replay:** the replay rows are the test split, so a different split gives a different count; the probe counts should match exactly since they come from the forest. `exp2b`'s probes embed test-split rows too (base rows drawn with seed 0), so all of its counts move with the split; the routed-vs-unrouted density ratio should stay near 2.5x.
- **`exp3`/`exp4` fail to load `anomaly_*.pkl`:** Step 4 was skipped, or a very old scikit-learn is installed; 1.3 or newer loads them with a warning.
- **`exp5b` fails to install a release:** that release has no wheel for your Python or platform. Drop it from `--versions`; the regression is visible with `1.25.1,1.26.0,1.28.0` alone.

## Where each paper number comes from

Paths are relative to `research/code/`.

| Table or figure | Result files |
|---|---|
| Boundary parity (`tab:boundary`) | `results/exp2_boundary_parity.json`, `results/exp2b_probe_ablation.json` |
| Base-row draws, data-free probes, injected faults (`tab:faults`) | `results/exp11_probe_seeds_and_faults.json` |
| Fleet divergence on real attack flows (boundary section) | `results/exp14_fleet_divergence.json` |
| Other forests (`tab:generalize`) | `results/exp5_generalization.json` (seed 0), `results/exp5_seeds/seed1.json`..`seed4.json`, `results/exp5_seeds_summary.json` (ranges over five seeds) |
| CICIDS2017 HGB threshold rounding = 963 of 1,362 framework flips | `results/exp12b_cicids_hgb_frontend_skl2onnx_1.20.0.json`, `results/exp12b_cicids_hgb_frontend_skl2onnx_main.json` |
| skl2onnx 1.20.0 HGB threshold rounding (toy, datasets paragraph) | `results/exp12_hgb_threshold_rounding_skl2onnx_1.20.0.json`, `results/exp12_hgb_threshold_rounding_skl2onnx_main.json` |
| Converter/runtime sweep, seven families, faithful float32 always matches ORT | `results/exp15_converter_sweep_released.json`, `results/exp15_converter_sweep_skl2onnx_main.json` |
| GPU providers leave the forest on the CPU kernel (threats) | `results_jetson/exp13_ort_providers.json` |
| Other datasets (`tab:datasets`) | `results/exp10_cross_dataset.json` |
| Replay gate with the confidence check: six faults, label-only vs. labels + confidence | `results/exp11b_fault_confidence_gate.json` (and `results/exp2c_confidence_gate.json` for the float64 fault alone) |
| Replay gate with the confidence check on all 13 models, runtime parity | `results/exp16_confidence_replay.json` |
| Converter parity (framework vs. export) with the confidence check | `results/exp17_converter_confidence.json` |
| Data-free probes on all 13 models: gate verdicts and cost (`tab:datafree`) | `results/exp19_datafree_probes.json` |
| Leafparity on the converter pairs (`tab:leafparity-run`) | `results/exp21_leafparity.json` |
| Treelite as a third runtime of the same graphs (`tab:treelite`) | `results/exp22_treelite_runtime.json` |
| Lockstep block-budget ablation (`tab:blockbudget`) | `results/exp23_lockstep_block_ablation.json` (Apple M3), `results_jetson/exp23_lockstep_block_ablation.json` (Jetson #2) |
| Five training seeds on UNSW-NB15 and ToN-IoT (sensitivity, threats) | `results/exp10_seeds_summary.json` (from `results/exp10_seeds/`), `results/exp10c_float32_ties_unsw.json` |
| Shipped gate's own run: probes in 16 s, 2,000-row replay 0.14 s (per-row means x 2,000, both runtimes) | `validation/parity_boundary_m3_20261013.json` (see `validation/README.md`) |
| Five-seed appendix table, all twelve trained models (`tab:seeds`) | `results/exp5_seeds_summary.json`, `results/exp10_seeds_summary.json` (rows printed by `experiments/seeds_table.py`) |
| Probe straddle after float32 scaling, and exact-crossing probes (Section 4, probe construction) | `results/exp24_probe_straddle.json` |
| Exporter faults vs. structure check, replay and probe sets (`tab:mutants`) | `results/exp25_exporter_mutations.json` |
| Exact early exit on the twelve trained models | `results/exp26_early_exit_all_models.json` |
| Cross-architecture ties: both runtimes on M3, x86-64 and Jetson | `results/exp27_cross_arch_ties.json` |
| Shipped exporter and engine on all thirteen models; reach and pair coverage | `results/exp28_shipped_toolkit_all_models.json` |
| Fault localization from failing probes (probe targets; first divergence artifact vs. graph; `tab:localize`) | `results/exp29_fault_localization.json` (rows printed by `experiments/exp29_table.py`) |
| Probe completeness: per-node disagreement sets, three-probe set vs. crossing pair (`tab:completeness`) | `results/exp31_probe_completeness.json` (rows printed by `experiments/exp31_table.py`) |
| Probes against testing baselines, including engine-code mutants and single-node faults (`tab:baselines`) | `results/exp30_testing_baselines.json` (rows printed by `experiments/exp30_table.py`) |
| Shipped gate's probe-check cost on the 13 models, crossing pair vs. three-probe set (operational recommendation) | `results/exp32_gate_timing.json` |
| Ranked localization: top-1, top-3 and MRR over 48 single-element faults | `results/exp33_localization_ranking.json` |
| Gradual shift: detection, serving parity and undecided-rate alarm on mixtures of the two splits | `results/exp34_gradual_drift.json` |
| Compiled Treelite (TL2cgen) as a further runtime, on the data-free probes | `results/exp35_treelite_compiled.json` |
| Repeated and stratified replay draws, confidence-tolerance sweep (RQ1) | `results/exp36_replay_sampling.json` |
| Generator ablation: coverage, float64 flips, single-node faults (`tab:baselines`, EASE version) | `results/exp37_generator_coverage.json` |
| Backend cost (`tab:engines`) | `results_jetson/exp8_summary.json`, `results_jetson/exp8_summary_zeromq_kafkareplay.json`, `results_jetson/exp6_gate_node.json`, `results_jetson/exp6_gate_node_zeromq_kafkareplay.json` |
| Cold start (`tab:coldstart`) | `results_jetson/exp8_summary.json` (ONNX Runtime and per-tree rows, pooled), `results_jetson/exp8_cold_start_j2_head.json` (lockstep row) |
| End-to-end latency (`tab:e2e`) | `results_jetson/exp7_e2e.json`, `results_jetson/exp7_e2e_zeromq_kafkareplay.json` |
| Gate drain over ZeroMQ, 100,000 records (Discussion) | `results_jetson/exp6_gate_node_zeromq_kafkareplay_max.json` |
| Five-run repeats: end-to-end table, gate rows, energy per row (`tab:e2e`, `tab:engines`, Section 7.1) | `results_jetson/exp18_repeats/exp18_summary.json` (aggregates), `exp18_e2e.json` (per run), `gate/gate_*_rep*.json`, `energy/exp20_classifier_energy.json` |
| Lockstep throughput (`fig:lockstep`) | `results_jetson/exp1_lockstep_benchmark.json`, `results/exp1_lockstep_benchmark.json` |
| Whole-forest lockstep slower than the per-tree loop at batch 2,000 (lockstep section, Discussion) | `results/exp1b_whole_forest_lockstep.json` |
| Exact early exit (`tab:earlyexit`) | `results_jetson/exp3_exact_early_exit.json` |
| Distribution shift (`tab:drift`) | `results/exp4_drift.json` |
| Drift alarm (`tab:alarm`) | `results/exp9_drift_alarm.json` |
