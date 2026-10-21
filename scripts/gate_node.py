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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_edge_ids import FeatureMatrixBuilder, flow_frames


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
        return self.score_matrix(self.builder.build(messages))

    def score_matrix(self, x):
        start = time.perf_counter()
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


def serve_zeromq(gate, args):
    """Score each incoming flow batch and forward its suspicious rows.

    The producer already batches rows (csv_producer.py --batch-size), so every
    received batch is scored as it arrives; --batch-size and --linger-ms apply
    to the Kafka path only. Forwarded batches keep the producer's per-row
    sent_at and add forwarded_at and the anomaly score.
    """
    import zmq

    ctx = zmq.Context()
    sock_in = ctx.socket(zmq.PULL)
    sock_in.bind(args.zmq_in)
    sock_out = ctx.socket(zmq.PUSH)
    sock_out.connect(args.zmq_out)
    n_features = gate.builder.n_features
    print(
        f"[OK] ZeroMQ gate in={args.zmq_in} out={args.zmq_out} "
        f"threshold={gate.threshold:.6f}"
    )
    total = 0
    forwarded = 0
    try:
        while True:
            try:
                batch = flow_frames.unpack(sock_in.recv_multipart(copy=False), cols=n_features)
            except ValueError as exc:
                print(f"[WARN] dropped malformed batch: {exc}")
                continue
            result = gate.score_matrix(batch["x"])
            keep = result.is_anomaly
            if keep.any():
                sent_at = batch.get("sent_at")
                sock_out.send_multipart(
                    flow_frames.pack(
                        batch["x"][keep],
                        sent_at=None if sent_at is None else sent_at[keep],
                        forwarded_at=np.full(int(keep.sum()), time.time()),
                        score=result.scores[keep],
                    ),
                    copy=False,
                )
                forwarded += int(keep.sum())
            total += len(keep)
            print(
                f"[{total:,}] gate={result.inference_time_ms:.3f}ms "
                f"forwarded={forwarded:,}/{total:,} ({forwarded / total:.2%})"
            )
            if args.max_messages and total >= args.max_messages:
                return
    finally:
        sock_in.close(linger=0)
        # Give queued forwards a moment to leave before the context closes.
        sock_out.close(linger=2000)
        ctx.term()


def main():
    parser = argparse.ArgumentParser(description="ONNX-EdgeIDS anomaly gate node")
    parser.add_argument("--transport", choices=["zeromq", "kafka"], default=None,
                        help="Default: kafka when --bootstrap is given, zeromq otherwise.")
    parser.add_argument("--zmq-in", default="tcp://*:5555",
                        help="ZeroMQ endpoint the gate binds for incoming flow batches.")
    parser.add_argument("--zmq-out", default="tcp://localhost:5556",
                        help="ZeroMQ endpoint of the classifier the gate forwards to.")
    parser.add_argument("--bootstrap", default="", help="Kafka bootstrap server, e.g. <mac-ip>:9092")
    parser.add_argument("--input-topic", default="ids-network-flow")
    parser.add_argument("--output-topic", default="ids-suspicious-flow")
    parser.add_argument("--group-id", default="onnx-edge-ids-gate")
    parser.add_argument("--features", default=str(ROOT / "artifacts" / "feature_columns.json"))
    parser.add_argument("--model", default=str(ROOT / "artifacts" / "anomaly_autoencoder.pkl"))
    parser.add_argument("--scaler", default=str(ROOT / "artifacts" / "anomaly_scaler.pkl"))
    parser.add_argument("--threshold", default=str(ROOT / "artifacts" / "anomaly_threshold.json"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--linger-ms", type=float, default=1000.0)
    parser.add_argument("--flush-every", type=int, default=0,
                        help="Block on producer.flush() every N gate batches. "
                             "0 (default) flushes only at shutdown and lets the "
                             "producer batch freely; a synchronous flush per batch "
                             "costs ~30-50%% throughput from an edge board. Set 1 to "
                             "narrow the window in which an unclean exit can drop "
                             "already-committed input offsets.")
    parser.add_argument("--max-messages", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    transport = args.transport or ("kafka" if args.bootstrap else "zeromq")
    if transport == "kafka" and not args.bootstrap:
        parser.error("--bootstrap is required when --transport kafka")

    gate = AnomalyGate(args.features, args.model, args.scaler, args.threshold)

    if transport == "zeromq":
        serve_zeromq(gate, args)
        return

    from kafka import KafkaConsumer, KafkaProducer
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
        f"[OK] Kafka Gate input={args.input_topic} output={args.output_topic} "
        f"batch={args.batch_size} threshold={gate.threshold:.6f}"
    )
    total = 0
    forwarded = 0
    batches_done = 0
    batch = []

    def flush_batch():
        nonlocal total, forwarded, batches_done, batch
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
        batches_done += 1
        if args.flush_every and batches_done % args.flush_every == 0:
            producer.flush()
        total += len(batch)
        print(
            f"[{total:,}] gate={result.inference_time_ms:.3f}ms "
            f"forwarded={forwarded:,}/{total:,} ({forwarded / total:.2%})"
        )
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
                for record in topic_records:
                    if not batch:
                        deadline = time.monotonic() + args.linger_ms / 1000.0
                    batch.append(record.value)

            if len(batch) >= args.batch_size or (
                deadline is not None and time.monotonic() >= deadline
            ):
                flush_batch()
                deadline = None

            if args.max_messages and total >= args.max_messages:
                flush_batch()
                return
    finally:
        flush_batch()
        producer.flush()
        consumer.close()
        producer.close()


if __name__ == "__main__":
    main()
