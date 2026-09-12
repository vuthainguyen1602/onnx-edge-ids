# Two-Jetson Deployment

This repo can run the lightweight two-node Kafka path by itself. The thesis repo
is still useful for the full PostgreSQL/Grafana/alerting stack, but the scripts
below are enough to stress-test the serving layer.

Use this split for the ONNX-EdgeIDS software test with a CICIDS2017-derived replay CSV:

- **Mac:** Kafka + CSV producer
- **Jetson #1:** this repo's anomaly gate, forwarding suspicious flows
- **Jetson #2:** this repo's ONNX classifier, consuming `ids-suspicious-flow`

Kafka runs on the Mac in this split, not on a board. The three addresses below
are written as placeholders on purpose: the boards take their addresses over
DHCP, so they change across sessions. Read the current one on each host with
`hostname -I`, or reserve them on the router before a measurement run. A board
that loses its wired link falls back to Wi-Fi and reappears on a different
address, which is the usual reason a documented IP stops answering.

## 1. Prepare Artifacts On The Mac

From this repo:

```bash
cp /path/to/ids_rf.onnx artifacts/ids_rf.onnx
cp /path/to/feature_columns.json artifacts/feature_columns.json

ONNX_MODEL=artifacts/ids_rf.onnx \
FEATURES_JSON=artifacts/feature_columns.json \
./scripts/export_artifacts.sh

# Load check for a single backend
python scripts/smoke_predict.py --engine onnx --model artifacts/ids_rf.onnx --features artifacts/feature_columns.json

# Validation gate: the NumPy artifact must reproduce the ONNX artifact.
# Exits non-zero if label agreement or the confidence delta leaves tolerance.
python scripts/validate_parity.py \
  --onnx artifacts/ids_rf.onnx \
  --numpy artifacts/ids_rf_numpy.npz \
  --features artifacts/feature_columns.json \
  --csv /path/to/replay_cicids2017.csv --rows 2000 \
  --report-json results/parity_report.json
```

Expected artifacts:

```text
artifacts/feature_columns.json
artifacts/ids_rf.onnx
artifacts/ids_rf_numpy.npz
```

Copy anomaly artifacts from the thesis repo:

```bash
cp <thesis-repo>/jetson/model/anomaly_autoencoder.pkl artifacts/
cp <thesis-repo>/jetson/model/anomaly_scaler.pkl artifacts/
cp <thesis-repo>/jetson/model/anomaly_threshold.json artifacts/
```

Start Kafka on the Mac using the thesis Docker Compose:

```bash
cd <thesis-repo>/jetson
docker compose up -d
cd <this-repo>
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
  --csv <your-path>/replay_cicids2017.csv \
  --rate 1000
```

Use `--rate 0` for maximum load.

## 5. Batching And Flush Behaviour

`--batch-size` is the number of records an inference call receives, and
`--linger-ms` is how long a partial batch waits before it is served anyway. A
batch is flushed when it fills *or* when the linger window expires, so under a
thin stream both nodes serve small batches by design; raise `--linger-ms` to
trade added delay for fuller batches.

Batch size dominates the NumPy backend. On an Orin Nano it serves roughly
9 rows/s at `--batch-size 1`, 150 rows/s at 20, and 1,650 rows/s at 500, because
its tree traversal is vectorised across the batch. The ONNX backend is far less
sensitive. Size the batch for the backend actually in use.

The gate does not block on `producer.flush()` per batch by default. Measured on
an Orin Nano against a broker one hop away, a synchronous flush every batch costs
about 10% of end-to-end gate throughput (915 vs 831 rows/s over 20,000 records at
a 28.6% forward rate); in isolation the produce path alone loses closer to 30%,
but the gate forwards only a fraction of what it consumes, so the cost is spread
thin. Pass `--flush-every 1` if you would rather narrow the window in which an
unclean exit drops records whose input offsets were already auto-committed.

Neither board's CPU nor the network is the limit at that point. On the same
board `kafka-python` consumes about 2,500 msg/s and produces about 2,200 msg/s,
while the ONNX backend serves over 20,000 rows/s; the identical code on a laptop
reaches 6,600 msg/s consuming and 9,900 msg/s producing. The serving pipeline is
bound by the pure-Python Kafka client on the edge board, not by inference. A
C-backed client such as `confluent-kafka` is the lever that moves this ceiling.

## 6. What To Watch

Jetson #1 should print forwarded suspicious-flow counts.

Jetson #2 should print lines like:

```text
[20] engine=1.234ms wall=2.345ms attacks=0/20
```

This minimal ONNX classifier does not yet write verdicts to PostgreSQL. It is
for validating the serving layer and measuring classifier latency under Kafka
load. The thesis repo path still has the full DB/alert/Grafana pipeline.
