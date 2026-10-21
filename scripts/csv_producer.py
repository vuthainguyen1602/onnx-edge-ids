#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Replay CSV flow rows to the gate, as JSON records on Kafka or float32 batches over ZeroMQ."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clean_column_name(name):
    result = name.strip().lower()
    for ch in [" ", ".", "-", "/", "(", ")"]:
        result = result.replace(ch, "_")
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def parse_value(value):
    try:
        if value is None or value == "":
            return 0.0
        return float(value) if "." in value or "e" in value.lower() else int(value)
    except (ValueError, TypeError):
        return value


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = {name: clean_column_name(name) for name in reader.fieldnames}
        for row in reader:
            yield {fieldnames[name]: parse_value(value) for name, value in row.items()}


def pace(delay, started, total):
    """Sleep until row ``total`` is due at the target rate."""
    if delay > 0:
        wait = started + total * delay - time.time()
        if wait > 0:
            time.sleep(wait)


def send_kafka(args, delay):
    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks=1,
        retries=3,
        batch_size=65536,
        linger_ms=10,
    )
    total = 0
    started = time.time()
    try:
        for payload in read_rows(args.csv):
            payload["_timestamp"] = time.time()
            producer.send(args.topic, key=str(total), value=payload)
            total += 1

            if args.flush_every and total % args.flush_every == 0:
                producer.flush()
                elapsed = time.time() - started
                print(f"[{total:,}] rate={total / elapsed:.1f} rows/s")

            if args.limit and total >= args.limit:
                break
            if delay > 0:
                time.sleep(delay)
    finally:
        producer.flush()
        producer.close()
    return total, started


def send_zeromq(args, delay):
    """Send rows in --batch-size float32 batches; each row keeps its own send time.

    Rows are stamped as they are read at the target rate, so a row's latency
    includes the time it waits for its batch to fill, as on the Kafka path
    where the gate lingers.
    """
    import numpy as np
    import zmq

    sys.path.insert(0, str(ROOT / "src"))
    from onnx_edge_ids import FeatureMatrixBuilder, flow_frames

    builder = FeatureMatrixBuilder(args.features)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(args.zmq_endpoint)
    total = 0
    batch, stamps = [], []
    started = time.time()

    def flush():
        nonlocal batch, stamps
        if batch:
            sock.send_multipart(
                flow_frames.pack(builder.build(batch), sent_at=np.asarray(stamps)), copy=False
            )
        batch, stamps = [], []

    try:
        for payload in read_rows(args.csv):
            pace(delay, started, total)
            batch.append(payload)
            stamps.append(time.time())
            total += 1
            if len(batch) >= args.batch_size:
                flush()
            if args.flush_every and total % args.flush_every == 0:
                elapsed = time.time() - started
                print(f"[{total:,}] rate={total / elapsed:.1f} rows/s")
            if args.limit and total >= args.limit:
                break
    finally:
        flush()
        # Block until queued batches leave (up to 10 s) before the context closes.
        sock.close(linger=10000)
        ctx.term()
    return total, started


def main():
    parser = argparse.ArgumentParser(description="Replay CSV rows to the gate over Kafka or ZeroMQ")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--transport", choices=["kafka", "zeromq"], default="kafka")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="ids-network-flow")
    parser.add_argument("--zmq-endpoint", default="tcp://localhost:5555",
                        help="Gate endpoint to connect to when --transport zeromq.")
    parser.add_argument("--features", default=str(ROOT / "artifacts" / "feature_columns.json"),
                        help="Feature columns of the float32 batches sent over ZeroMQ.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Rows per ZeroMQ batch (the Kafka path sends one record per row).")
    parser.add_argument("--rate", type=float, default=100.0, help="Rows/s; <=0 sends as fast as possible")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    delay = 1.0 / args.rate if args.rate > 0 else 0.0
    send = send_zeromq if args.transport == "zeromq" else send_kafka
    total, started = send(args, delay)

    elapsed = time.time() - started
    print(f"[OK] Sent {total:,} rows in {elapsed:.1f}s ({total / elapsed:.1f} rows/s)")

if __name__ == "__main__":
    main()
