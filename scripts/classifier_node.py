#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal Jetson #2 classifier loop for ONNX-EdgeIDS.

This is intentionally smaller than the thesis pipeline: it consumes suspicious
flows from the gate, either as JSON records from Kafka or as float32 batches
over ZeroMQ (onnx_edge_ids.flow_frames), and runs ONNX/NumPy inference. Database writes and alerting can be wired from the thesis repo after
the serving backend is validated on the boards.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_edge_ids import FeatureMatrixBuilder, create_inference_engine, flow_frames


def _stamp(value):
    """Format a timestamp for latency.csv; missing values stay empty."""
    if value is None:
        return ""
    value = float(value)
    return "" if np.isnan(value) else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(description="ONNX-EdgeIDS Jetson classifier node")
    parser.add_argument("--transport", choices=["zeromq", "kafka"], default=None,
                        help="Default: kafka when --bootstrap is given, zeromq otherwise.")
    parser.add_argument("--zmq-endpoint", default="tcp://*:5556",
                        help="ZeroMQ endpoint the classifier binds for the gate's batches.")
    parser.add_argument("--bootstrap", default="", help="Kafka bootstrap server, e.g. <mac-ip>:9092")
    parser.add_argument("--topic", default="ids-suspicious-flow")
    parser.add_argument("--group-id", default="onnx-edge-ids-classifier")
    parser.add_argument("--engine", choices=["onnx", "numpy"], default="onnx")
    parser.add_argument("--model", default="")
    parser.add_argument("--features", default=str(ROOT / "artifacts" / "feature_columns.json"))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--linger-ms", type=float, default=1000.0,
                        help="Flush a partial batch after this many ms.")
    parser.add_argument("--max-messages", type=int, default=0,
                        help="Stop after this many flow records; 0 means run forever.")
    parser.add_argument("--metrics-csv", default="",
                        help="Optional CSV path for per-batch latency metrics.")
    parser.add_argument("--latency-csv", default="",
                        help="Optional CSV path: per record, the producer's send time, the "
                             "gate's forward time and the verdict time (each host's own clock).")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    transport = args.transport or ("kafka" if args.bootstrap else "zeromq")
    if transport == "kafka" and not args.bootstrap:
        parser.error("--bootstrap is required when --transport kafka")

    builder = FeatureMatrixBuilder(args.features)
    model_path = args.model
    if not model_path:
        if args.engine == "numpy":
            model_path = str(ROOT / "artifacts" / "ids_rf_numpy.npz")
        else:
            model_path = str(ROOT / "artifacts" / "ids_rf.onnx")

    engine = create_inference_engine(args.engine, model_path)
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
    latency_file = None
    latency_writer = None
    if args.latency_csv:
        latency_path = Path(args.latency_csv)
        latency_path.parent.mkdir(parents=True, exist_ok=True)
        latency_file = latency_path.open("w", newline="")
        latency_writer = csv.writer(latency_file)
        latency_writer.writerow(["sent_at", "forwarded_at", "verdict_at", "prediction"])

    total = 0
    started_all = time.time()

    def serve(x, sent_at, forwarded_at):
        """Classify one batch and record its metrics and per-record latency rows."""
        nonlocal total
        started = time.time()
        result = engine.predict_batch(x)
        verdict_at = time.time()
        total += result.stats["batch_size"]
        if latency_writer:
            for s_at, f_at, pred in zip(sent_at, forwarded_at, result.predictions.tolist()):
                latency_writer.writerow([_stamp(s_at), _stamp(f_at), f"{verdict_at:.6f}", int(pred)])
            latency_file.flush()
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

    try:
        if transport == "zeromq":
            consume_zeromq(args, builder.n_features, serve, lambda: total)
        else:
            consume_kafka(args, builder, serve, lambda: total)
    finally:
        if metrics_file:
            metrics_file.close()
        if latency_file:
            latency_file.close()


def consume_zeromq(args, n_features, serve, served):
    """Gather the gate's batches and serve --batch-size rows at a time.

    Gate batches are split, so a served batch holds exactly --batch-size rows
    unless the linger window, counted from its oldest row's arrival, expired
    first. Leftover rows wait for the next batch, as with the Kafka path.
    """
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind(args.zmq_endpoint)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    print(
        f"[OK] ZeroMQ classifier on {args.zmq_endpoint} | engine: {args.engine} | "
        f"batch: {args.batch_size} | linger: {args.linger_ms:.0f}ms"
    )
    linger_s = args.linger_ms / 1000.0
    pending = []  # (batch, arrival) in arrival order
    rows = 0

    def flush(limit=None):
        """Serve the oldest min(limit, rows) pending rows; keep the rest pending."""
        nonlocal pending, rows
        if not pending:
            return
        batch = flow_frames.concat([b for b, _ in pending])
        n = len(batch["x"]) if limit is None else min(limit, len(batch["x"]))
        missing = np.full(n, np.nan)
        head = {k: v[:n] for k, v in batch.items()}
        serve(head["x"], head.get("sent_at", missing), head.get("forwarded_at", missing))
        # The leftover keeps the arrival time of the gate batch its first row came in.
        seen, rest = 0, []
        for b, arrival in pending:
            size = len(b["x"])
            if seen + size > n:
                rest.append(({k: v[max(0, n - seen):] for k, v in b.items()}, arrival))
            seen += size
        pending = rest
        rows = sum(len(b["x"]) for b, _ in pending)

    try:
        while True:
            if not pending:
                timeout_ms = int(args.linger_ms)
            else:
                timeout_ms = max(0, int((pending[0][1] + linger_s - time.monotonic()) * 1000.0))
            if poller.poll(timeout_ms):
                while rows < args.batch_size:
                    try:
                        frames = sock.recv_multipart(zmq.NOBLOCK, copy=False)
                    except zmq.Again:
                        break
                    try:
                        batch = flow_frames.unpack(frames, cols=n_features)
                    except ValueError as exc:
                        print(f"[WARN] dropped malformed batch: {exc}")
                        continue
                    pending.append((batch, time.monotonic()))
                    rows += len(batch["x"])

            while rows >= args.batch_size:
                flush(args.batch_size)
            if pending and time.monotonic() >= pending[0][1] + linger_s:
                flush()

            if args.max_messages and served() >= args.max_messages:
                return
    finally:
        flush()
        sock.close(linger=0)
        ctx.term()


def consume_kafka(args, builder, serve, served):
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        group_id=args.group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    print(
        f"[OK] Kafka topic: {args.topic} | engine: {args.engine} | "
        f"batch: {args.batch_size} | linger: {args.linger_ms:.0f}ms"
    )
    batch = []

    def flush():
        nonlocal batch
        if batch:
            serve(builder.build(batch), [m.get("_timestamp") for m in batch],
                  [m.get("_forwarded_at") for m in batch])
        batch = []

    deadline = None

    try:
        while True:
            # Wait the full linger window for the first record of a batch, then
            # only the time still left before that batch is due.
            if deadline is None:
                timeout_ms = int(args.linger_ms)
            else:
                timeout_ms = max(0, int((deadline - time.monotonic()) * 1000.0))

            records = consumer.poll(
                timeout_ms=timeout_ms, max_records=args.batch_size - len(batch)
            )

            for topic_records in records.values():
                for message in topic_records:
                    if not batch:
                        deadline = time.monotonic() + args.linger_ms / 1000.0
                    batch.append(message.value)

            if len(batch) >= args.batch_size or (
                deadline is not None and time.monotonic() >= deadline
            ):
                flush()
                deadline = None

            if args.max_messages and served() >= args.max_messages:
                flush()
                return
    finally:
        flush()
        consumer.close()

if __name__ == "__main__":
    main()
