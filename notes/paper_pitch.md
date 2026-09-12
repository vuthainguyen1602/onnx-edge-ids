# Software Impacts Pitch

## Software name

ONNX-EdgeIDS

## One-sentence pitch

ONNX-EdgeIDS helps researchers prepare lightweight ONNX or NumPy artifacts for network intrusion detection, validate prediction parity against a reference offline model, and serve real-time traffic on Jetson-class edge devices through a two-stage Kafka pipeline.

## What the paper is about

This is a software publication. The paper should describe:

- what the software does;
- why it is useful;
- how it can be reused;
- where the code is archived;
- what impact it has on edge IDS research and deployment.

It should not read like a full experimental SOICT follow-up. The benchmarks are
supporting evidence for impact, not the center of the article.

## Impact overview angle

The SOICT work exposed a limitation: the IDS model can be accurate while the serving stack remains too heavy for an 8 GB Jetson board. ONNX-EdgeIDS turns that limitation into reusable software by separating offline model development from online serving.

## Claims we can safely make now

- The software targets ONNX-compatible binary IDS classifiers and includes a transparent NumPy fallback for tree-ensemble artifacts.
- It preserves feature-order metadata and validates exported artifacts against
  reference-model predictions.
- It enables lightweight edge serving with ONNX Runtime or a transparent NumPy
  fallback.
- It supports benchmarking NumPy and ONNX runtimes on the same trained model artifact and feature set.

## Claims to make only after measurement

- Exact throughput gain over the previous heavyweight serving baseline.
- Exact p95 latency reduction.
- Exact energy saving.
- TensorRT/CUDA acceleration benefit.
- Probability parity with the reference confidence output.

## Required submission fill-ins

- Public repository URL.
- Version tag.
- Software license.
- Archive DOI or Software Heritage identifier.
- Contact email.
- Validation rows, label agreement, probability delta.
- One workflow figure.
