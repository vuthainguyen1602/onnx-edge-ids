#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZeroMQ Zero-Copy Streaming Benchmark for ONNX-EdgeIDS.

Evaluates high-speed inter-node flow transmission between Gate and Classifier:
- Direct transmission of contiguous float32 NumPy memory buffers.
- Compares naive copy vs zero-copy (copy=False).
- End-to-end integration with ONNX Runtime and Lockstep NumPy inference engines.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_edge_ids import create_inference_engine


def run_receiver(endpoint, engine_type, model_path, num_batches, batch_size, num_features, ready_event, result_queue):
    """Classifier receiver process: receives raw float32 buffer and runs inference."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind(endpoint)

    engine = None
    if engine_type:
        engine = create_inference_engine(engine_type, model_path)

    ready_event.set()

    total_flows = 0
    inference_times = []
    start_time = None

    for _ in range(num_batches):
        msg = sock.recv(copy=False)
        if start_time is None:
            start_time = time.perf_counter()

        # Zero-copy view directly into incoming socket frame memory
        x_batch = np.frombuffer(msg.buffer, dtype=np.float32).reshape(-1, num_features)
        total_flows += x_batch.shape[0]

        if engine:
            t0 = time.perf_counter()
            engine.predict_batch(x_batch)
            inference_times.append((time.perf_counter() - t0) * 1000.0)

    elapsed = time.perf_counter() - start_time
    sock.close()
    ctx.term()

    result_queue.put({
        "total_flows": total_flows,
        "elapsed_sec": elapsed,
        "throughput": total_flows / elapsed,
        "mean_inf_ms": np.mean(inference_times) if inference_times else 0.0,
        "p95_inf_ms": np.percentile(inference_times, 95) if inference_times else 0.0,
    })


def run_sender(endpoint, num_batches, batch_size, num_features, zero_copy, ready_event):
    """Gate sender process: transmits float32 contiguous arrays."""
    ready_event.wait()
    time.sleep(0.05)  # small stabilization window

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(endpoint)

    # Pre-generate synthetic flow feature batches (float32 contiguous memory)
    rng = np.random.default_rng(42)
    sample_batch = rng.standard_normal((batch_size, num_features)).astype(np.float32)
    sample_bytes = sample_batch.tobytes()

    for _ in range(num_batches):
        if zero_copy:
            sock.send(sample_batch, copy=False)
        else:
            sock.send(sample_bytes, copy=True)

    sock.close()
    ctx.term()


def benchmark_configuration(endpoint, engine_type, model_path, num_batches, batch_size, num_features, zero_copy):
    ready_event = mp.Event()
    result_queue = mp.Queue()

    recv_proc = mp.Process(
        target=run_receiver,
        args=(endpoint, engine_type, model_path, num_batches, batch_size, num_features, ready_event, result_queue),
    )
    send_proc = mp.Process(
        target=run_sender,
        args=(endpoint, num_batches, batch_size, num_features, zero_copy, ready_event),
    )

    recv_proc.start()
    send_proc.start()

    send_proc.join()
    recv_proc.join()

    return result_queue.get()


def main():
    parser = argparse.ArgumentParser(description="ZeroMQ Zero-Copy Streaming Benchmark")
    parser.add_argument("--endpoint", default="ipc:///tmp/zmq_ids_bench.ipc")
    parser.add_argument("--tcp-endpoint", default="tcp://127.0.0.1:5599")
    parser.add_argument("--model", default=str(ROOT / "artifacts" / "ids_rf.onnx"))
    parser.add_argument("--num-batches", type=int, default=1000)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[20, 100, 500])
    args = parser.parse_args()

    num_features = 29
    print("=" * 75)
    print("ZeroMQ Edge-Native Zero-Copy Streaming Benchmark (ONNX-EdgeIDS)")
    print(f"Features: {num_features} (float32) | Batches per run: {args.num_batches:,}")
    print("=" * 75)

    # 1. Pure Transport Benchmark (ZeroMQ Wire Throughput)
    print("\n--- 1. Pure Network Transport Throughput (No Inference) ---")
    for b in args.batch_sizes:
        # IPC Zero-copy
        res_ipc = benchmark_configuration(
            args.endpoint, None, args.model, args.num_batches, b, num_features, zero_copy=True
        )
        # TCP Zero-copy (127.0.0.1 loopback)
        res_tcp = benchmark_configuration(
            args.tcp_endpoint, None, args.model, args.num_batches, b, num_features, zero_copy=True
        )
        print(
            f"Batch {b:4d} | IPC Zero-Copy: {res_ipc['throughput']:>10,.0f} flows/s "
            f"({res_ipc['throughput'] * num_features * 4 / 1024 / 1024:>6.1f} MB/s) | "
            f"TCP Zero-Copy: {res_tcp['throughput']:>10,.0f} flows/s "
            f"({res_tcp['throughput'] * num_features * 4 / 1024 / 1024:>6.1f} MB/s)"
        )

    # 2. End-to-End Streaming Benchmark (Transport + In-Process Serving Engine)
    print("\n--- 2. End-to-End Streaming (Transport + ONNX Runtime Serving) ---")
    for b in args.batch_sizes:
        res_onnx = benchmark_configuration(
            args.tcp_endpoint, "onnx", args.model, args.num_batches, b, num_features, zero_copy=True
        )
        print(
            f"Batch {b:4d} | End-to-End ONNX Serving: {res_onnx['throughput']:>10,.0f} flows/s | "
            f"Inference Latency: mean={res_onnx['mean_inf_ms']:.3f}ms, p95={res_onnx['p95_inf_ms']:.3f}ms"
        )

    # 3. End-to-End Streaming Benchmark (Transport + NumPy Lockstep Engine)
    print("\n--- 3. End-to-End Streaming (Transport + NumPy Lockstep Serving) ---")
    np_model_path = str(ROOT / "artifacts" / "ids_rf_numpy.npz")
    for b in args.batch_sizes:
        res_np = benchmark_configuration(
            args.tcp_endpoint, "numpy", np_model_path, args.num_batches, b, num_features, zero_copy=True
        )
        print(
            f"Batch {b:4d} | End-to-End NumPy Serving: {res_np['throughput']:>10,.0f} flows/s | "
            f"Inference Latency: mean={res_np['mean_inf_ms']:.3f}ms, p95={res_np['p95_inf_ms']:.3f}ms"
        )

    print("\n" + "=" * 75)
    print("[SUCCESS] ZeroMQ zero-copy benchmark completed!")
    print("=" * 75)


if __name__ == "__main__":
    main()
