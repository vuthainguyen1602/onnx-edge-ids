#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_edge_ids import FeatureMatrixBuilder


@dataclass(frozen=True)
class GateResult:
    scores: np.ndarray
    is_anomaly: np.ndarray
    inference_time_ms: float


class AnomalyGate:
    def __init__(self, features, model_path, scaler_path, threshold_path):
        self.builder = FeatureMatrixBuilder(features, dtype=np.float32)
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(threshold_path, "r") as f:
            payload = json.load(f)
        self.threshold = float(payload["threshold"])

    def score_batch(self, messages):
        start = time.perf_counter()
        x = self.builder.build(messages)
        x_scaled = self.scaler.transform(x)
        x_hat = self.model.predict(x_scaled)
        if x_hat.ndim == 1:
            x_hat = x_hat.reshape(-1, x_scaled.shape[1])
        mse = np.mean((x_scaled - x_hat) ** 2, axis=1)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return GateResult(
            scores=mse.astype(np.float32),
            is_anomaly=(mse >= self.threshold),
            inference_time_ms=elapsed_ms,
        )


def main():
    parser = argparse.ArgumentParser(description="ONNX-EdgeIDS anomaly gate node")
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--input-topic", default="ids-network-flow")
    parser.add_argument("--output-topic", default="ids-suspicious-flow")
    parser.add_argument("--group-id", default="onnx-edge-ids-gate")
    parser.add_argument("--features", default=str(ROOT / "artifacts" / "feature_columns.json"))
    parser.add_argument("--model", default=str(ROOT / "artifacts" / "anomaly_autoencoder.pkl"))
    parser.add_argument("--scaler", default=str(ROOT / "artifacts" / "anomaly_scaler.pkl"))
    parser.add_argument("--threshold", default=str(ROOT / "artifacts" / "anomaly_threshold.json"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--linger-ms", type=float, default=1000.0)
    parser.add_argument("--max-messages", type=int, default=0)
    args = parser.parse_args()

    gate = AnomalyGate(args.features, args.model, args.scaler, args.threshold)
    consumer = KafkaConsumer(
        args.input_topic,
        bootstrap_servers=args.bootstrap,
        group_id=args.group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks=1,
        retries=3,
        linger_ms=5,
    )

    print(
        f"[OK] Gate input={args.input_topic} output={args.output_topic} "
        f"batch={args.batch_size} threshold={gate.threshold:.6f}"
    )
    total = 0
    forwarded = 0
    batch = []

    def flush_batch():
        nonlocal total, forwarded, batch
        if not batch:
            return
        result = gate.score_batch(batch)
        flags = result.is_anomaly.tolist()
        for i, (message, is_anomaly) in enumerate(zip(batch, flags)):
            if not is_anomaly:
                continue
            payload = dict(message)
            payload["_forwarded_at"] = time.time()
            payload["_anomaly_score"] = float(result.scores[i])
            producer.send(args.output_topic, key=str(total + i), value=payload)
            forwarded += 1
        producer.flush()
        total += len(batch)
        print(
            f"[{total:,}] gate={result.inference_time_ms:.3f}ms "
            f"forwarded={forwarded:,}/{total:,} ({forwarded / total:.2%})"
        )
        batch = []

    try:
        while True:
            records = consumer.poll(timeout_ms=int(args.linger_ms), max_records=args.batch_size)
            if not records:
                flush_batch()
                continue
            for topic_records in records.values():
                for record in topic_records:
                    batch.append(record.value)
                    if len(batch) >= args.batch_size:
                        flush_batch()
                    if args.max_messages and total >= args.max_messages:
                        flush_batch()
                        return
            flush_batch()
    finally:
        flush_batch()
        consumer.close()
        producer.close()


if __name__ == "__main__":
    main()
