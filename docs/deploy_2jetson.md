# Two-Jetson Deployment

This repo can run the lightweight two-node Kafka path by itself. The thesis repo
is still useful for the full PostgreSQL/Grafana/alerting stack, but the scripts
below are enough to stress-test the serving layer.

Use this split for the ONNX-EdgeIDS software test:

- **Mac:** Kafka + CSV producer
- **Jetson #1:** this repo's anomaly gate, forwarding suspicious flows
- **Jetson #2:** this repo's ONNX classifier, consuming `ids-suspicious-flow`

## 1. Prepare Artifacts On The Mac

From this repo:

```bash
cp /Users/thainguyenvu/Desktop/Thesis_IDS/jetson/model/feature_columns.json artifacts/feature_columns.json

MODEL_DIR=/Users/thainguyenvu/Desktop/Thesis_IDS/jetson/model/ids_pipeline_model \
FEATURES_JSON=artifacts/feature_columns.json \
./scripts/export_artifacts.sh

python scripts/smoke_predict.py --engine onnx --model artifacts/ids_rf.onnx --features artifacts/feature_columns.json
```

Expected artifacts:

```text
artifacts/feature_columns.json
artifacts/ids_rf.onnx
artifacts/ids_rf_numpy.npz
```

Copy anomaly artifacts from the thesis repo:

```bash
cp /Users/thainguyenvu/Desktop/Thesis_IDS/jetson/model/anomaly_autoencoder.pkl artifacts/
cp /Users/thainguyenvu/Desktop/Thesis_IDS/jetson/model/anomaly_scaler.pkl artifacts/
cp /Users/thainguyenvu/Desktop/Thesis_IDS/jetson/model/anomaly_threshold.json artifacts/
```

Start Kafka on the Mac using the thesis Docker Compose:

```bash
cd /Users/thainguyenvu/Desktop/Thesis_IDS/jetson
docker compose up -d
cd /Users/thainguyenvu/Desktop/onnx-edge-ids-paper
python scripts/init_topics.py --bootstrap localhost:9092 --partitions 2
```

## 2. Deploy Gate To Jetson #1

From this repo on the Mac:

```bash
./deploy/sync_to_jetson.sh <user>@<jetson1-ip>
```

On Jetson #1:

```bash
cd ~/onnx-edge-ids-paper
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .[gate]
pip install kafka-python
```

Run:

```bash
python scripts/gate_node.py \
  --bootstrap <mac-ip>:9092 \
  --input-topic ids-network-flow \
  --output-topic ids-suspicious-flow \
  --features artifacts/feature_columns.json \
  --model artifacts/anomaly_autoencoder.pkl \
  --scaler artifacts/anomaly_scaler.pkl \
  --threshold artifacts/anomaly_threshold.json \
  --batch-size 100 \
  --linger-ms 1000
```

## 3. Deploy ONNX Classifier To Jetson #2

From this repo on the Mac:

```bash
./deploy/sync_to_jetson.sh <user>@<jetson2-ip>
```

On Jetson #2:

```bash
cd ~/onnx-edge-ids-paper
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install kafka-python
pip install -r requirements_jetson_onnx.txt
```

Run ONNX classifier:

```bash
python scripts/classifier_node.py \
  --bootstrap <mac-ip>:9092 \
  --topic ids-suspicious-flow \
  --group-id onnx-edge-ids-classifier \
  --engine onnx \
  --model artifacts/ids_rf.onnx \
  --features artifacts/feature_columns.json \
  --batch-size 20 \
  --linger-ms 1000 \
  --metrics-csv results/classifier_onnx.csv
```

If ONNX Runtime cannot install on the Jetson image, use NumPy fallback:

```bash
python scripts/classifier_node.py \
  --bootstrap <mac-ip>:9092 \
  --topic ids-suspicious-flow \
  --group-id onnx-edge-ids-classifier \
  --engine numpy \
  --model artifacts/ids_rf_numpy.npz \
  --features artifacts/feature_columns.json \
  --batch-size 20 \
  --linger-ms 1000
```

## 4. Start Sender On Mac

From this repo:

```bash
python scripts/csv_producer.py \
  --bootstrap localhost:9092 \
  --topic ids-network-flow \
  --csv /Users/thainguyenvu/Desktop/Thesis_IDS/jetson/sender/replay_cicids2017.csv \
  --rate 1000
```

Use `--rate 0` for maximum load.

## 5. What To Watch

Jetson #1 should print forwarded suspicious-flow counts.

Jetson #2 should print lines like:

```text
[20] engine=1.234ms wall=2.345ms attacks=0/20
```

This minimal ONNX classifier does not yet write verdicts to PostgreSQL. It is
for validating the serving layer and measuring classifier latency under Kafka
load. The thesis repo path still has the full DB/alert/Grafana pipeline.
