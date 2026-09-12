#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate that the NumPy serving artifact reproduces the ONNX artifact.

Runs both backends over the same flow records and reports label agreement and
the largest confidence difference, so a deployment mismatch is caught before the
artifacts are copied to the edge boards. Exits non-zero if either tolerance is
exceeded, which makes it usable as a release gate.

Usage:
    python scripts/validate_parity.py \
        --onnx  artifacts/ids_rf.onnx \
        --numpy artifacts/ids_rf_numpy.npz \
        --features artifacts/feature_columns.json \
        --csv data/replay_cicids2017.csv --rows 2000
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

from onnx_edge_ids import FeatureMatrixBuilder, create_inference_engine


def load_rows(csv_path, columns, rows, seed):
    """Replayed flow records when a CSV is given, otherwise a random probe."""
    if csv_path:
        out = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for i, record in enumerate(csv.DictReader(f)):
                if i >= rows:
                    break
                out.append(record)
        if not out:
            raise SystemExit(f"[ERR] No rows read from {csv_path}")
        return out, f"{len(out)} rows from {csv_path}"

    rng = np.random.default_rng(seed)
    out = [
        {c: float(v) for c, v in zip(columns, rng.lognormal(3.0, 3.0, len(columns)))}
        for _ in range(rows)
    ]
    return out, f"{rows} synthetic rows (lognormal, seed={seed})"


def percentile(values, p):
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def run_engine(name, model, matrix, batch):
    engine = create_inference_engine(name, str(model))
    preds, confs, per_row_ms = [], [], []
    for i in range(0, len(matrix), batch):
        chunk = matrix[i:i + batch]
        result = engine.predict_batch(chunk)
        preds.append(result.predictions)
        confs.append(result.confidences)
        per_row_ms.append(result.stats["inference_time_ms"] / len(chunk))
    return np.concatenate(preds), np.concatenate(confs), per_row_ms


def main():
    ap = argparse.ArgumentParser(description="ONNX vs NumPy serving parity check")
    ap.add_argument("--onnx", default=str(ROOT / "artifacts" / "ids_rf.onnx"))
    ap.add_argument("--numpy", default=str(ROOT / "artifacts" / "ids_rf_numpy.npz"))
    ap.add_argument("--features", default=str(ROOT / "artifacts" / "feature_columns.json"))
    ap.add_argument("--csv", default=None, help="Replay CSV; omit to probe with random flows")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-confidence-delta", type=float, default=1e-5,
                    help="Fail if any confidence differs by more than this.")
    ap.add_argument("--min-agreement", type=float, default=1.0,
                    help="Fail below this label-agreement fraction (1.0 = exact).")
    ap.add_argument("--report-json", default="", help="Optional path for a machine-readable report")
    args = ap.parse_args()

    columns = json.load(open(args.features))
    rows, source = load_rows(args.csv, columns, args.rows, args.seed)
    matrix = FeatureMatrixBuilder(args.features).build(rows)

    print(f"[INFO] input     : {source}")
    print(f"[INFO] matrix    : {matrix.shape} dtype={matrix.dtype}")
    print(f"[INFO] batch size: {args.batch}")

    onnx_pred, onnx_conf, onnx_ms = run_engine("onnx", args.onnx, matrix, args.batch)
    numpy_pred, numpy_conf, numpy_ms = run_engine("numpy", args.numpy, matrix, args.batch)

    total = int(onnx_pred.shape[0])
    agree = int(np.count_nonzero(onnx_pred == numpy_pred))
    agreement = agree / total
    max_delta = float(np.abs(onnx_conf - numpy_conf).max())

    for label, preds, per_row in (("onnx", onnx_pred, onnx_ms),
                                  ("numpy", numpy_pred, numpy_ms)):
        print(f"\n[{label}] attacks={int(np.count_nonzero(preds == 1))}/{total}")
        print(f"  per-row ms: mean={np.mean(per_row):.4f} "
              f"p50={percentile(per_row, 50):.4f} p95={percentile(per_row, 95):.4f}")
        print(f"  capacity  : {1000.0 / np.mean(per_row):,.0f} rows/s")

    print(f"\n[PARITY] label agreement    : {agree}/{total} ({100.0 * agreement:.4f}%)")
    print(f"[PARITY] max confidence delta: {max_delta:.3e}")

    failures = []
    if agreement < args.min_agreement:
        failures.append(f"label agreement {agreement:.6f} < {args.min_agreement}")
        for row in np.flatnonzero(onnx_pred != numpy_pred)[:10]:
            print(f"  row {row}: onnx={onnx_pred[row]} ({onnx_conf[row]:.6f}) "
                  f"numpy={numpy_pred[row]} ({numpy_conf[row]:.6f})")
    if max_delta > args.max_confidence_delta:
        failures.append(f"confidence delta {max_delta:.3e} > {args.max_confidence_delta:.3e}")

    if args.report_json:
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": source,
            "rows": total,
            "batch_size": args.batch,
            "label_agreement": agreement,
            "max_confidence_delta": max_delta,
            "onnx_per_row_ms_mean": float(np.mean(onnx_ms)),
            "onnx_per_row_ms_p95": percentile(onnx_ms, 95),
            "numpy_per_row_ms_mean": float(np.mean(numpy_ms)),
            "numpy_per_row_ms_p95": percentile(numpy_ms, 95),
            "passed": not failures,
        }
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[OK] Wrote {args.report_json}")

    if failures:
        print("\n[FAIL] " + "; ".join(failures))
        return 1
    print("\n[OK] NumPy artifact reproduces the ONNX artifact within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
