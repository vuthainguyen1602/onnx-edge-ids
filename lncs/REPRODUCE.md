# Reproducing the LNCS paper on your own machine

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

Library versions the paper's result files were produced with (`lncs/code/results/exp5_generalization.json` records them):

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
pip install -e ".[kafka]"                    # the served package (numpy, kafka-python)
pip install -r lncs/code/requirements.txt    # experiments: onnxruntime, pyarrow, scikit-learn, lightgbm, xgboost, skl2onnx, onnxmltools, pytest
```

The served forest is already in the repository: `artifacts/ids_rf.onnx` (ONNX), `artifacts/ids_rf_numpy.npz` (the NumPy export of the same graph) and `artifacts/feature_columns.json` (the 29 input columns in order).

## Step 2: checks that need no dataset

Run the test suite. It loads the shipped artifact, checks that the per-tree, block-lockstep and early-exit traversals agree, that the generic ONNX traversal reproduces ONNX Runtime, and that the experiment traversal matches the packaged `NumpyInferenceEngine`:

```bash
python -m pytest lncs/code/tests -q
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

## Step 4: gate artifacts (only for `exp3`, `exp4` and the pipeline)

The anomaly gate (autoencoder, scaler, threshold) is not redistributed. Copy `anomaly_autoencoder.pkl`, `anomaly_scaler.pkl` and `anomaly_threshold.json` into `artifacts/` from the upstream thesis repository as described in `docs/deploy_2jetson.md`. `exp1`, `exp2`, `exp2b`, `exp5`, `exp5b` and the tests do not need them. The pickles were written with an older scikit-learn and print a version warning on load; that is expected.

## Step 5: run the experiments

Each script writes one JSON under `lncs/code/results/`, overwriting the committed file of the same name, so `git diff lncs/code/results/` afterwards shows exactly what changed against the paper. Run them from the repository root:

```bash
DATA=/path/to/data/test_data.parquet
TRAIN=/path/to/data/train_data.parquet
DATA2018=/path/to/data_2018/test_data.parquet

python lncs/code/experiments/exp1_lockstep_benchmark.py --data $DATA
python lncs/code/experiments/exp2_boundary_parity.py    --data $DATA
python lncs/code/experiments/exp2b_probe_ablation.py    --data $DATA        # reads exp2's JSON for the density ratios
python lncs/code/experiments/exp3_exact_early_exit.py   --data $DATA        # needs Step 4
python lncs/code/experiments/exp4_drift.py --in-dist $DATA --shifted $DATA2018   # needs Step 4
python lncs/code/experiments/exp5_generalization.py --train $TRAIN --test $DATA
python lncs/code/experiments/exp5b_ort_label_regression.py \
    --model lncs/code/results/exp5_models/sklearn_rf.onnx --data $DATA
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
| `exp5b` | release sweep | 1.26.0 and 1.27.0 label attack whenever P > 0 (negative first probability column); the other releases agree with scikit-learn | none |

Notes:

- `exp5` trains scikit-learn RandomForest and HistGradientBoosting, LightGBM and XGBoost on a 600,000-row sample (`--train-rows`, seed 0), writes the exported models to `lncs/code/results/exp5_models/` and probes up to 100,000 internal nodes per model (`--max-nodes`). The counts for the four trained forests depend on the training sample; the row for the deployed forest (`deployed_rf`) does not. Use `--models` to run a subset.
- `exp5b` creates one virtualenv per release under `$TMPDIR/exp5b_ort_venvs` (`--venv-dir`) and installs `onnxruntime==<version>` into each, so it needs network access and a Python for which wheels of every listed release exist (the paper's run used Python 3.12; trim `--versions` if a release has no wheel for yours). The committed result was taken on this same export (`exp5` with the default seed and 600,000 training rows), so a re-run with the defaults reproduces the count exactly; a different training sample reproduces the relabelling pattern (0 flips on agreeing releases, hundreds of thousands on 1.26.0 and 1.27.0), not the count.
- `exp1` timings are noisy on a busy machine. Close other work and, on laptops, keep the machine on mains power.

## Step 6 (optional): the two-node Kafka pipeline on two ordinary hosts

The on-board table needs the pipeline, not the boards. Any two Linux/macOS hosts plus a third (or one of the two) running Kafka will do; follow `docs/deploy_2jetson.md` and read "Jetson #1" as the gate host and "Jetson #2" as the classifier host. The only board-specific parts are `tegrastats` for the power column and `requirements_jetson_onnx.txt` for the ARM64 ONNX Runtime wheel; on a laptop or server, `pip install -e .[onnx]` installs the runtime instead.

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

## Step 7 (optional): compile the manuscript

The tables quote `lncs/code/results/*.json`; the LaTeX source is self-contained (class, bibliography style and fonts are vendored):

```bash
cd lncs/manuscript && ./compile.sh
```

This writes `lncs/manuscript/main.pdf` and `lncs/output/onnx-edge-ids-lncs.pdf`. Numbers in the tables are typed into `main.tex`, not generated, so after re-running an experiment compare `git diff lncs/code/results/` with the table by hand.

## If something does not match

- **Counts differ against ONNX Runtime but the float32 traversal still shows 0 flips:** check `python -c "import onnxruntime; print(onnxruntime.__version__)"`. The paper's counts are against 1.30.0 with the CPU provider.
- **`exp2` flips differ by a handful on replay:** the replay rows are the test split, so a different split gives a different count; the probe counts should match exactly since they come from the forest. `exp2b`'s probes embed test-split rows too (base rows drawn with seed 0), so all of its counts move with the split; the routed-vs-unrouted density ratio should stay near 2.5x.
- **`exp3`/`exp4` fail to load `anomaly_*.pkl`:** Step 4 was skipped, or a very old scikit-learn is installed; 1.3 or newer loads them with a warning.
- **`exp5b` fails to install a release:** that release has no wheel for your Python or platform. Drop it from `--versions`; the regression is visible with `1.25.1,1.26.0,1.28.0` alone.
