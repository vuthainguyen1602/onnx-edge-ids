# ONNX-EdgeIDS Software Impacts Submission

This repository is the manuscript workspace for a **Software Impacts** article,
not a long SOICT-style systems paper.

Software Impacts publishes short software publications. The submission should
make the software artifact citable and reusable, with an impact overview and
metadata about the current code version. The journal page and indexed examples
describe the format as a short descriptive paper of about three pages with an
Impact Overview statement and code/repository metadata.

## Proposed Software Name

**ONNX-EdgeIDS**

## Proposed Title

ONNX-EdgeIDS: A Lightweight Two-Stage Edge Serving Toolkit for Network Intrusion Detection

## Software Pitch

ONNX-EdgeIDS serves portable ONNX or transparent NumPy IDS artifacts on Jetson-class edge devices. The package focuses on the serving side of the problem: Kafka flow replay, a first-stage gate, a second-stage classifier node, and scripts for measuring throughput and latency without carrying a heavyweight training runtime into the inference path.

## Why This Fits Software Impacts

The core contribution is reusable software:

- artifact preparation: ONNX artifact to lightweight NumPy fallback;
- runtime: ONNX/NumPy classifier interface for Kafka-streamed network flows;
- validation: ONNX-versus-NumPy prediction checks;
- benchmark scripts: edge inference cost, latency, memory, and energy.

The SOICT paper can be cited as the research context that exposed the software need: the IDS model can be accurate while the serving stack is still too heavy for edge hardware. This article should focus on the resulting software artifact and its impact, not repeat the full distributed IDS evaluation.

## Files

- `src/onnx_edge_ids/`: reusable ONNX/NumPy inference package.
- `artifacts/`: the served ONNX and NumPy artifacts plus the feature-column metadata.
- `scripts/export_numpy.py`: convert an ONNX tree-ensemble artifact to the NumPy format.
- `scripts/export_artifacts.sh`: create the NumPy artifact from an ONNX artifact.
- `scripts/validate_parity.py`: check that the NumPy artifact reproduces the ONNX artifact; exits non-zero outside tolerance.
- `scripts/init_topics.py`: create Kafka input/suspicious topics.
- `scripts/csv_producer.py`: replay CICIDS2017-derived CSV flow rows to Kafka at a target rate.
- `scripts/gate_node.py`: Jetson #1 anomaly gate with Kafka forwarding.
- `scripts/classifier_node.py`: minimal Jetson #2 Kafka classifier loop.
- `scripts/smoke_predict.py`: local smoke test for exported artifacts.
- `docs/data.md`: CICIDS2017 dataset note and citation requirement.
- `docs/deploy_2jetson.md`: two-Jetson test plan using thesis gate + ONNX classifier.
- `deploy/sync_to_jetson.sh`: rsync this repo to a Jetson node.
- `experiments/plan.md`: validation and benchmark plan.
- `notes/paper_pitch.md`: idea notes and contribution boundaries.

## Dataset

The deployment replay uses a local CICIDS2017-derived flow CSV. The raw IDS 2017/CICIDS2017 dataset is not committed; see `docs/data.md` for the official source and citation.

## Quick Local Test

Optional NumPy artifact export from the ONNX model:

```bash
ONNX_MODEL=artifacts/ids_rf.onnx \
FEATURES_JSON=artifacts/feature_columns.json \
./scripts/export_artifacts.sh

python scripts/smoke_predict.py --engine onnx --model artifacts/ids_rf.onnx
python scripts/smoke_predict.py --engine numpy --model artifacts/ids_rf_numpy.npz
```

Validate that the two backends agree before deploying. This is the check the
manuscript reports, and it fails the command rather than only printing a warning:

```bash
python scripts/validate_parity.py \
  --csv /path/to/replay_cicids2017.csv --rows 2000 \
  --report-json results/parity_report.json
```

## Jetson #2 Minimal Classifier Test

Copy this repo to Jetson #2, copy `artifacts/ids_rf.onnx` and
`artifacts/feature_columns.json`, install the optional runtime, then run:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install kafka-python
pip install -r requirements_jetson_onnx.txt

python scripts/classifier_node.py \
  --bootstrap <mac-ip>:9092 \
  --topic ids-suspicious-flow \
  --engine onnx \
  --model artifacts/ids_rf.onnx \
  --features artifacts/feature_columns.json
```

If ONNX Runtime is not available on the Jetson image, use:

```bash
python scripts/classifier_node.py \
  --bootstrap <mac-ip>:9092 \
  --topic ids-suspicious-flow \
  --engine numpy \
  --model artifacts/ids_rf_numpy.npz \
  --features artifacts/feature_columns.json
```

## Before Submission

- Archive the software release and fill in the Zenodo/Software Heritage DOI.
- Run export validation on replay flows and fill metadata tables.
- Add at least one figure showing the train/export/serve workflow.

## Citation And License

This repository is versioned as `v0.1.0` and released under the MIT license.
After the first public release is archived, update `CITATION.cff` and the
manuscript with the archive DOI.
