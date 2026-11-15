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
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.forest import load_forest, predict_lockstep, scale_input


def parents(forest):
    parent = np.full(len(forest.left), -1, dtype=np.int64)
    went_left = np.zeros(len(forest.left), dtype=bool)
    internal = np.flatnonzero(~forest.is_leaf)
    parent[forest.left[internal]] = internal
    went_left[forest.left[internal]] = True
    parent[forest.right[internal]] = internal
    return parent, went_left


def path_rows(f64, targets, base):
    """One row per target node whose features satisfy every split on the way to it."""
    parent, went_left = parents(f64)
    rows = base.astype(np.float64).copy()
    for i, node in enumerate(targets):
        lo, hi = {}, {}
        cur = node
        while parent[cur] >= 0:
            p = parent[cur]
            g, t = int(f64.feature[p]), float(f64.threshold[p])
            if went_left[cur]:
                hi[g] = min(hi.get(g, np.inf), t)
            else:
                lo[g] = max(lo.get(g, -np.inf), t)
            cur = p
        for g in set(lo) | set(hi):
            a, b = lo.get(g, -np.inf), hi.get(g, np.inf)
            if np.isfinite(a) and np.isfinite(b):
                s = (a + b) / 2.0
            elif np.isfinite(b):
                s = b - 1.0
            else:
                s = a + 1.0
            rows[i, g] = s / f64.scale[g] + f64.offset[g]
    return rows.astype(np.float32)


def reaches(forest, rows, targets):
    """Whether row i passes through node targets[i] under this forest's arithmetic."""
    scaled = scale_input(forest, rows)
    tree = np.searchsorted(forest.tree_offsets, targets, side="right") - 1
    cur = forest.tree_offsets[tree].copy()
    idx = np.arange(len(targets))
    hit = cur == targets
    active = ~forest.is_leaf[cur] & ~hit
    while active.any():
        a = np.flatnonzero(active)
        c = cur[a]
        go_left = scaled[idx[a], forest.feature[c]] <= forest.threshold[c]
        cur[a] = np.where(go_left, forest.left[c], forest.right[c])
        hit[a] = cur[a] == targets[a]
        active[a] = ~forest.is_leaf[cur[a]] & ~hit[a]
    return hit


def onnx_labels(session, x, batch=20000):
    name = session.get_inputs()[0].name
    return np.concatenate([
        np.asarray(session.run(None, {name: x[i:i + batch]})[0]).reshape(-1)
        for i in range(0, len(x), batch)
    ]).astype(np.int64)


def numpy_labels(forest, x, batch=2000):
    return np.concatenate([predict_lockstep(forest, x[i:i + batch])[0] for i in range(0, len(x), batch)])


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
    rows = path_rows(f64, internal, base)
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
        on_path = reaches(f32, probes, internal)
        lab_onnx = onnx_labels(session, probes)
        d32 = int((numpy_labels(f32, probes) != lab_onnx).sum())
        d64 = int((numpy_labels(f64, probes) != lab_onnx).sum())
        report["variants"][name] = {
            "probes": int(len(probes)),
            "probes_reaching_their_node": int(on_path.sum()),
            "numpy_float32_vs_onnx_label_disagreements": d32,
            "numpy_float64_vs_onnx_label_disagreements": d64,
        }
        total["probes"] += len(probes); total["f32"] += d32; total["f64"] += d64
        print(name, report["variants"][name])

    real_onnx = onnx_labels(session, x)
    r32 = int((numpy_labels(f32, x) != real_onnx).sum())
    r64 = int((numpy_labels(f64, x) != real_onnx).sum())
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
