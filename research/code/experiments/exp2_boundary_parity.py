"""Boundary-directed parity test: probe every split threshold instead of replaying random rows.

For each internal node (feature f, threshold t) build a row that satisfies the
root-to-node path, then set feature f to the raw value that lands on t after
scaling, and to its float32 neighbours. Random replay almost never produces such
rows, so it cannot see two runtimes disagreeing about which side of a split a
value is on.
"""

import argparse
import json
import os
import time

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.forest import load_forest
from edgeids.probes import artifact_path_rows, artifact_reaches
from edgeids.runtime import lockstep_labels, session_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test_data.parquet directory")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp2_boundary_parity.json"))
    args = ap.parse_args()

    import onnxruntime as ort

    x, _, _ = load_test_split(args.data)
    npz = os.path.join(ARTIFACTS, "ids_rf_numpy.npz")
    f32, f64 = load_forest(npz, "float32"), load_forest(npz, "float64")
    session = ort.InferenceSession(os.path.join(ARTIFACTS, "ids_rf.onnx"), providers=["CPUExecutionProvider"])

    internal = np.flatnonzero(~f32.is_leaf)
    feat = f32.feature[internal]
    pairs = np.unique(np.stack([feat, f32.threshold[internal].view(np.int32)]), axis=1).shape[1]

    rng = np.random.default_rng(0)
    base = x[rng.choice(len(x), size=len(internal), replace=True)]
    started = time.perf_counter()
    rows = artifact_path_rows(f64, internal, base)
    raw = (f64.threshold[internal] / f64.scale[feat] + f64.offset[feat]).astype(np.float32)
    variants = {
        "on_threshold": raw,
        "one_ulp_below": np.nextafter(raw, np.float32(-np.inf)),
        "one_ulp_above": np.nextafter(raw, np.float32(np.inf)),
    }

    report = {
        "internal_nodes": int(len(internal)),
        "distinct_feature_threshold_pairs": int(pairs),
        "variants": {},
    }
    print(f"internal nodes {len(internal):,} | distinct (feature, threshold) pairs {pairs:,}")

    total = {"probes": 0, "f32": 0, "f64": 0}
    for name, values in variants.items():
        probes = rows.copy()
        probes[np.arange(len(internal)), feat] = values
        on_path = artifact_reaches(f32, probes, internal)
        lab_onnx = session_labels(session, probes)
        d32 = int((lockstep_labels(f32, probes) != lab_onnx).sum())
        d64 = int((lockstep_labels(f64, probes) != lab_onnx).sum())
        report["variants"][name] = {
            "probes": int(len(probes)),
            "probes_reaching_their_node": int(on_path.sum()),
            "numpy_float32_vs_onnx_label_disagreements": d32,
            "numpy_float64_vs_onnx_label_disagreements": d64,
        }
        total["probes"] += len(probes); total["f32"] += d32; total["f64"] += d64
        print(name, report["variants"][name])

    real_onnx = session_labels(session, x)
    r32 = int((lockstep_labels(f32, x) != real_onnx).sum())
    r64 = int((lockstep_labels(f64, x) != real_onnx).sum())
    rate = r64 / len(x)
    report["boundary_total"] = {
        "probes": total["probes"],
        "numpy_float32_vs_onnx_label_disagreements": total["f32"],
        "numpy_float64_vs_onnx_label_disagreements": total["f64"],
    }
    report["random_replay"] = {
        "rows": int(len(x)),
        "numpy_float32_vs_onnx_label_disagreements": r32,
        "numpy_float64_vs_onnx_label_disagreements": r64,
        "float64_disagreement_rate": rate,
        "prob_2000_row_replay_sees_no_disagreement": float((1.0 - rate) ** 2000),
    }
    print("random_replay", report["random_replay"])
    report["seconds"] = round(time.perf_counter() - started, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
