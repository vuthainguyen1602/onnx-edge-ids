#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal Jetson #2 classifier loop for ONNX-EdgeIDS.

This is intentionally smaller than the thesis pipeline: it consumes suspicious
flow dictionaries from Kafka, builds the feature matrix, and runs ONNX/NumPy
inference. Database writes and alerting can be wired from the thesis repo after
the serving backend is validated on the boards.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kafka import KafkaConsumer

from onnx_edge_ids import FeatureMatrixBuilder, create_inference_engine


def main():
    parser = argparse.ArgumentParser(description="ONNX-EdgeIDS Jetson classifier node")
    parser.add_argument("--bootstrap", required=True, help="Kafka bootstrap server, e.g. 192.168.1.165:9092")
    parser.add_argument("--topic", default="ids-suspicious-flow")
    parser.add_argument("--group-id", default="onnx-edge-ids-classifier")
    parser.add_argument("--engine", choices=["onnx", "numpy"], default="onnx")
    parser.add_argument("--model", default=str(ROOT / "artifacts" / "ids_rf.onnx"))
    parser.add_argument("--features", default=str(ROOT / "artifacts" / "feature_columns.json"))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--linger-ms", type=float, default=1000.0,
                        help="Flush a partial batch after this many ms.")
    parser.add_argument("--max-messages", type=int, default=0,
                        help="Stop after this many messages; 0 means run forever.")
    parser.add_argument("--metrics-csv", default="",
                        help="Optional CSV path for per-batch latency metrics.")
    args = parser.parse_args()

    builder = FeatureMatrixBuilder(args.features)
    engine = create_inference_engine(args.engine, args.model)
    metrics_file = None
    metrics_writer = None
    if args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("w", newline="")
        metrics_writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "timestamp",
                "total_messages",
                "batch_size",
                "engine_ms",
                "wall_ms",
                "avg_ms",
                "attacks_found",
                "throughput_since_start",
            ],
        )
        metrics_writer.writeheader()
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        group_id=args.group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    print(
        f"[OK] Topic: {args.topic} | engine: {args.engine} | "
        f"batch: {args.batch_size} | linger: {args.linger_ms:.0f}ms"
    )
    total = 0
    batch = []
    started_all = time.time()

    def flush_batch():
        nonlocal total, batch
        if not batch:
            return
        started = time.time()
        result = engine.predict_batch(builder.build(batch))
        total += result.stats["batch_size"]
        wall_ms = (time.time() - started) * 1000.0
        throughput = total / max(time.time() - started_all, 1e-9)
        print(
            f"[{total:,}] engine={result.stats['inference_time_ms']:.3f}ms "
            f"wall={wall_ms:.3f}ms rate={throughput:.1f}/s "
            f"attacks={result.stats['attacks_found']}/{result.stats['batch_size']}"
        )
        if metrics_writer:
            metrics_writer.writerow({
                "timestamp": time.time(),
                "total_messages": total,
                "batch_size": result.stats["batch_size"],
                "engine_ms": result.stats["inference_time_ms"],
                "wall_ms": round(wall_ms, 3),
                "avg_ms": result.stats["avg_time_ms"],
                "attacks_found": result.stats["attacks_found"],
                "throughput_since_start": round(throughput, 3),
            })
            metrics_file.flush()
        batch = []

    try:
        while True:
            records = consumer.poll(timeout_ms=int(args.linger_ms), max_records=args.batch_size)
            if not records:
                flush_batch()
                continue

            for topic_records in records.values():
                for message in topic_records:
                    batch.append(message.value)
                    if len(batch) >= args.batch_size:
                        flush_batch()

                    if args.max_messages and total >= args.max_messages:
                        flush_batch()
                        return

            flush_batch()
    finally:
        flush_batch()
        consumer.close()
        if metrics_file:
            metrics_file.close()


if __name__ == "__main__":
    main()
