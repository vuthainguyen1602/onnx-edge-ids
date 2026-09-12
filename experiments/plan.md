# Experiment Plan

## Models

Primary model:

- RandomForest on SHAP Top-30 features, same selected model as SOICT deployment.

Optional models:

- DecisionTree as a tiny sanity baseline.
- GBT only if export is reliable; do not let this block the paper.

## Engines

1. PySpark `PipelineModel` serving: reproduced baseline.
2. NumPy artifact serving: dependency-minimal fallback.
3. ONNX Runtime CPU provider.
4. ONNX Runtime CUDA/TensorRT provider if Jetson setup supports it.

## Measurements

Engine-only:

- throughput flows/s
- latency p50/p95/p99
- CPU percent
- RAM percent
- active energy per inference
- model artifact size
- cold-start load time

End-to-end streaming:

- Kafka producer timestamp to verdict timestamp
- classifier-only inference latency
- verdict throughput from PostgreSQL inserts
- attack F1 on replay labels
- dropped/late message count if measurable
- power via `tegrastats`

## Validation

For each exported artifact:

- Check feature order against `feature_columns.json`.
- Compare predictions against Spark `PipelineModel` on at least 5,000 replay rows.
- Report label agreement.
- Report max and mean probability delta.
- If probability delta is non-zero, store Spark-compatible confidence as a
  secondary issue and base correctness on labels plus attack F1.

## Minimum Table Set

Table 1: Export validation.

| Artifact | Rows | Label agreement | Max probability delta | Size MB |
|----------|------|-----------------|-----------------------|---------|

Table 2: Engine-only cost.

| Engine | Throughput | p95 | CPU | RAM | Energy/inf |
|--------|------------|-----|-----|-----|------------|

Table 3: End-to-end streaming cost.

| Classifier runtime | Verdict throughput | E2E p95 | F1 | RAM | Energy/verdict |
|--------------------|--------------------|---------|----|-----|----------------|

## Minimum Figure Set

- Architecture figure: Spark training path versus ONNX serving path.
- Bar chart: throughput and p95 by runtime.
- Optional: latency CDF for PySpark vs ONNX.

## Acceptance Criteria For A Real Paper Draft

- ONNX runtime integrated into classifier role, not only microbenchmark.
- At least three repeated runs per runtime.
- Same replay rate, batch size, feature set, and board state.
- Explicit statement that SOICT is prior system paper; this is the serving
  optimization paper.
