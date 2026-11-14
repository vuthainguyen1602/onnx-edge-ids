"""The six injected faults of exp11 against the gate as shipped: labels AND confidences.

exp11 counts label flips only, and so does the 2,000-row pass probability it
reports. The shipped gate (scripts/validate_parity.py) also fails when any
row's confidence differs from ONNX Runtime's by more than 1e-5. exp2c showed
that for the float64 fault this confidence check alone makes a 2,000-row replay
fail almost surely. This repeats that measurement for every fault, on

  replay   the whole test split,
  probes   the seed-0 boundary probes of exp2/exp11 (base rows from the test split),
  offset   the data-free probes the release gate builds (base rows at the scaler offset),

and reports per fault: label flips, rows whose confidence is off by more than
the tolerance, rows that trip either check, the largest confidence deviation,
and the probability that a 2,000-row replay trips neither check.

Confidence is the probability of the row's own predicted class, as in the gate
and in exp2c.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.forest import _sum_over_trees, load_forest
from dataclasses import replace

from exp11_probe_seeds_and_faults import build_probes


def scores_with(forest, scaled, batch=2000):
    trees = np.arange(forest.n_trees)
    return np.concatenate([
        _sum_over_trees(forest, scaled[i:i + batch], trees, forest.leaf_scores)
        for i in range(0, len(scaled), batch)
    ]).astype(np.float64)


def faults(f32, f64):
    """name -> function(x float32) returning class scores. Mirrors exp11.faults line for line."""
    o32, s32 = f32.offset, f32.scale
    o64, s64 = f64.offset, f64.scale
    internal = ~f32.is_leaf

    strict = replace(f32, threshold=np.where(internal, np.nextafter(f32.threshold, np.float32(-np.inf)), f32.threshold))
    decimal = replace(f32, threshold=np.array([float(f"{t:.6g}") for t in f32.threshold], dtype=np.float32))
    raw_thr = f32.threshold.copy()
    raw_thr[internal] = f32.threshold[internal] / s32[f32.feature[internal]] + o32[f32.feature[internal]]
    raw = replace(f32, threshold=raw_thr.astype(np.float32))

    return {
        "float32 (deployed)": lambda x: scores_with(f32, (x - o32) * s32),
        "float64": lambda x: scores_with(f64, (x.astype(np.float64) - o64) * s64),
        "float64_scaler": lambda x: scores_with(f32, ((x.astype(np.float64) - o64) * s64).astype(np.float32)),
        "scaler_reordered": lambda x: scores_with(f32, x * s32 - o32 * s32),
        "raw_thresholds": lambda x: scores_with(raw, np.ascontiguousarray(x, dtype=np.float32)),
        "strict_less": lambda x: scores_with(strict, (x - o32) * s32),
        "decimal_thresholds": lambda x: scores_with(decimal, (x - o32) * s32),
    }


def onnx_ref(session, x, batch=20000):
    name = session.get_inputs()[0].name
    labels, probs = [], []
    for i in range(0, len(x), batch):
        lab, prob = session.run(None, {name: x[i:i + batch]})
        labels.append(np.asarray(lab).reshape(-1))
        probs.append(np.asarray(prob, dtype=np.float64).reshape(len(labels[-1]), -1))
    labels = np.concatenate(labels).astype(np.int64)
    probs = np.concatenate(probs)
    return labels, probs[np.arange(len(labels)), labels]


def compare(scores, ref_labels, ref_conf, tolerance, replay_rows):
    labels = np.argmax(scores, axis=1)
    conf = scores[np.arange(len(labels)), labels]
    delta = np.abs(conf - ref_conf)
    flips = labels != ref_labels
    over = delta > tolerance
    trips = flips | over
    n = len(labels)
    return {
        "rows": int(n),
        "label_flips": int(flips.sum()),
        "rows_over_tolerance": int(over.sum()),
        "rows_tripping_gate": int(trips.sum()),
        "max_confidence_delta": float(delta.max()),
        "prob_replay_passes_labels_only": float((1.0 - flips.mean()) ** replay_rows),
        "prob_replay_passes_labels_and_confidence": float((1.0 - trips.mean()) ** replay_rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test_data.parquet directory")
    ap.add_argument("--tolerance", type=float, default=1e-5, help="the gate's --max-confidence-delta")
    ap.add_argument("--replay-rows", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp11b_fault_confidence_gate.json"))
    args = ap.parse_args()

    import onnxruntime as ort

    started = time.perf_counter()
    x, _, _ = load_test_split(args.data)
    npz = os.path.join(ARTIFACTS, "ids_rf_numpy.npz")
    f32, f64 = load_forest(npz, "float32"), load_forest(npz, "float64")
    session = ort.InferenceSession(os.path.join(ARTIFACTS, "ids_rf.onnx"), providers=["CPUExecutionProvider"])
    internal = np.flatnonzero(~f32.is_leaf)
    feat = f32.feature[internal]

    rng = np.random.default_rng(0)  # same draw as exp11 seed 0
    probes = build_probes(f64, internal, feat, x[rng.choice(len(x), size=len(internal), replace=True)])
    offset = build_probes(f64, internal, feat, np.repeat(f32.offset.reshape(1, -1), len(internal), axis=0))

    sets = {"replay": x, "probes": probes, "offset_probes": offset}
    refs = {k: onnx_ref(session, v) for k, v in sets.items()}

    report = {"numpy": np.__version__, "onnxruntime": ort.__version__, "tolerance": args.tolerance,
              "replay_rows": args.replay_rows, "faults": {}}
    for name, fn in faults(f32, f64).items():
        report["faults"][name] = {k: compare(fn(v), *refs[k], args.tolerance, args.replay_rows) for k, v in sets.items()}
        print(name, json.dumps(report["faults"][name]), flush=True)

    report["seconds"] = round(time.perf_counter() - started, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
