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

ONNX-EdgeIDS: A Lightweight ONNX Serving Layer for Spark-Trained Intrusion
Detection on Jetson Edge Devices

## Software Pitch

ONNX-EdgeIDS bridges distributed IDS model training and lightweight edge
deployment. It exports a Spark-trained intrusion-detection pipeline into a
portable ONNX or NumPy serving artifact, validates the exported classifier
against the original Spark `PipelineModel`, and serves real-time network-flow
messages on Jetson-class edge devices without running PySpark in the inference
process.

## Why This Fits Software Impacts

The core contribution is reusable software:

- exporter: Spark `PipelineModel` to lightweight serving artifact;
- runtime: ONNX/NumPy classifier interface for Kafka-streamed network flows;
- validation: Spark-versus-exported-artifact prediction checks;
- benchmark scripts: edge inference cost, latency, memory, and energy.

The SOICT paper can be cited as the research context that exposed the software
need: Spark serving works but is costly on Jetson. This article should focus on
the resulting software artifact and its impact, not repeat the full distributed
IDS evaluation.

## Files

- `manuscript/main.tex`: Software Impacts draft.
- `manuscript/references.bib`: bibliography.
- `src/onnx_edge_ids/`: reusable ONNX/NumPy inference package.
- `scripts/export_artifacts.sh`: Spark `PipelineModel` to ONNX + NumPy artifacts.
- `scripts/init_topics.py`: create Kafka input/suspicious topics.
- `scripts/csv_producer.py`: replay CSV rows to Kafka at a target rate.
- `scripts/gate_node.py`: Jetson #1 anomaly gate with Kafka forwarding.
- `scripts/classifier_node.py`: minimal Jetson #2 Kafka classifier loop.
- `scripts/smoke_predict.py`: local smoke test for exported artifacts.
- `docs/deploy_2jetson.md`: two-Jetson test plan using thesis gate + ONNX classifier.
- `deploy/sync_to_jetson.sh`: rsync this repo to a Jetson node.
- `experiments/plan.md`: validation and benchmark plan.
- `notes/paper_pitch.md`: idea notes and contribution boundaries.

## Quick Local Test

From this repository:

```bash
MODEL_DIR=/Users/thainguyenvu/Desktop/Thesis_IDS/jetson/model/ids_pipeline_model \
FEATURES_JSON=/Users/thainguyenvu/Desktop/Thesis_IDS/jetson/model/feature_columns.json \
./scripts/export_artifacts.sh

python scripts/smoke_predict.py --engine onnx --model artifacts/ids_rf.onnx
python scripts/smoke_predict.py --engine numpy --model artifacts/ids_rf_numpy.npz
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
