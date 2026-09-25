"""Ablation: does the path-satisfying probe construction of exp2 earn its keep?

exp2 builds one probe per internal node and routes it to that node by
construction (walk to the root, satisfy every ancestor split). A cheaper gate
skips the routing: for each of the forest's distinct (feature, threshold)
pairs, write the boundary value (on threshold and one float32 step either
side) into random replay rows and let the row's own path cross a node
carrying that pair, or not. This script runs that unrouted baseline at the
same probe budget and measures what routing buys: label-flip density against
ONNX Runtime and the share of internal nodes actually exercised.

A probe "exercises" a node when its traversal visits that node and the node
carries the probed (feature, threshold) pair, i.e. the comparison the probe
was built to stress is actually evaluated on the probe's value.
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


def onnx_labels(session, x, batch=20000):
    name = session.get_inputs()[0].name
    return np.concatenate([
        np.asarray(session.run(None, {name: x[i:i + batch]})[0]).reshape(-1)
        for i in range(0, len(x), batch)
    ]).astype(np.int64)


def numpy_labels(forest, x, batch=2000):
    return np.concatenate([predict_lockstep(forest, x[i:i + batch])[0] for i in range(0, len(x), batch)])


def pair_table(f32):
    """Distinct (feature, threshold) pairs and a node -> pair-id map (-1 for leaves)."""
    internal = np.flatnonzero(~f32.is_leaf)
    bits = f32.threshold[internal].view(np.int32).astype(np.int64) & 0xFFFFFFFF
    codes, inv = np.unique(f32.feature[internal] * (1 << 32) + bits, return_inverse=True)
    node_pair = np.full(len(f32.left), -1, dtype=np.int64)
    node_pair[internal] = inv
    pair_feat = (codes >> 32).astype(np.int64)
    pair_thr = (codes & 0xFFFFFFFF).astype(np.uint32).view(np.float32)
    return pair_feat, pair_thr, node_pair, len(internal)


def walk_marking_pair(f32, rows, target_pair, node_pair, exercised):
    """Walk every tree on every row; mark visited nodes carrying target_pair.

    Returns per-row whether any such node was visited, and sets
    exercised[node] = True for each one.
    """
    scaled = scale_input(f32, rows)
    n = scaled.shape[0]
    node = np.repeat(f32.tree_offsets[:-1].reshape(-1, 1), n, axis=1)
    hit = np.zeros(n, dtype=bool)
    active = ~f32.is_leaf[node]
    while active.any():
        t_idx, r_idx = np.nonzero(active)
        cur = node[t_idx, r_idx]
        match = node_pair[cur] == target_pair
        if match.any():
            hit[r_idx[match]] = True
            exercised[np.unique(cur[match])] = True
        go_left = scaled[r_idx, f32.feature[cur]] <= f32.threshold[cur]
        nxt = np.where(go_left, f32.left[cur], f32.right[cur])
        node[t_idx, r_idx] = nxt
        active[t_idx, r_idx] = ~f32.is_leaf[nxt]
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test_data.parquet directory")
    ap.add_argument("--per-pair", type=int, default=117,
                    help="base rows per (feature, threshold) pair; 117 matches exp2's 95,741 probes per variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp2b_probe_ablation.json"))
    args = ap.parse_args()

    import onnxruntime as ort

    x, _, _ = load_test_split(args.data)
    npz = os.path.join(ARTIFACTS, "ids_rf_numpy.npz")
    f32, f64 = load_forest(npz, "float32"), load_forest(npz, "float64")
    session = ort.InferenceSession(os.path.join(ARTIFACTS, "ids_rf.onnx"), providers=["CPUExecutionProvider"])

    pair_feat, pair_thr, node_pair, n_internal = pair_table(f32)
    n_pairs = len(pair_feat)
    raw = (pair_thr.astype(np.float64) / f64.scale[pair_feat] + f64.offset[pair_feat]).astype(np.float32)
    print(f"internal nodes {n_internal:,} | distinct (feature, threshold) pairs {n_pairs:,} | "
          f"{args.per_pair} base rows per pair")

    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    exercised = np.zeros(len(f32.left), dtype=bool)
    variant_names = ["on_threshold", "one_ulp_below", "one_ulp_above"]
    probes = {name: np.empty((n_pairs * args.per_pair, x.shape[1]), dtype=np.float32) for name in variant_names}
    hits = {name: 0 for name in variant_names}

    for j in range(n_pairs):
        base = x[rng.choice(len(x), size=args.per_pair, replace=False)]
        values = {
            "on_threshold": raw[j],
            "one_ulp_below": np.nextafter(raw[j], np.float32(-np.inf)),
            "one_ulp_above": np.nextafter(raw[j], np.float32(np.inf)),
        }
        lo = j * args.per_pair
        stacked = []
        for name in variant_names:
            p = base.copy()
            p[:, pair_feat[j]] = values[name]
            probes[name][lo:lo + args.per_pair] = p
            stacked.append(p)
        hit = walk_marking_pair(f32, np.concatenate(stacked), j, node_pair, exercised)
        for k, name in enumerate(variant_names):
            hits[name] += int(hit[k * args.per_pair:(k + 1) * args.per_pair].sum())

    report = {
        "internal_nodes": n_internal,
        "distinct_feature_threshold_pairs": n_pairs,
        "base_rows_per_pair": args.per_pair,
        "seed": args.seed,
        "variants": {},
    }
    total = {"probes": 0, "hits": 0, "f32": 0, "f64": 0}
    for name in variant_names:
        p = probes[name]
        lab_onnx = onnx_labels(session, p)
        d32 = int((numpy_labels(f32, p) != lab_onnx).sum())
        d64 = int((numpy_labels(f64, p) != lab_onnx).sum())
        report["variants"][name] = {
            "probes": int(len(p)),
            "probes_exercising_their_pair": hits[name],
            "numpy_float32_vs_onnx_label_disagreements": d32,
            "numpy_float64_vs_onnx_label_disagreements": d64,
        }
        total["probes"] += len(p); total["hits"] += hits[name]; total["f32"] += d32; total["f64"] += d64
        print(name, report["variants"][name])

    density = total["f64"] / total["probes"]
    report["unrouted_total"] = {
        "probes": total["probes"],
        "probes_exercising_their_pair": total["hits"],
        "share_exercising_their_pair": round(total["hits"] / total["probes"], 4),
        "internal_nodes_exercised": int(exercised.sum()),
        "share_of_internal_nodes_exercised": round(float(exercised.sum()) / n_internal, 4),
        "numpy_float32_vs_onnx_label_disagreements": total["f32"],
        "numpy_float64_vs_onnx_label_disagreements": total["f64"],
        "float64_flip_density": density,
    }
    print("unrouted_total", report["unrouted_total"])

    exp2_path = os.path.join(RESULTS, "exp2_boundary_parity.json")
    if os.path.exists(exp2_path):
        with open(exp2_path) as f:
            exp2 = json.load(f)
        routed = exp2["boundary_total"]
        routed_density = routed["numpy_float64_vs_onnx_label_disagreements"] / routed["probes"]
        replay_rate = exp2["random_replay"]["float64_disagreement_rate"]
        report["routed_reference_exp2"] = {
            "probes": routed["probes"],
            "numpy_float64_vs_onnx_label_disagreements": routed["numpy_float64_vs_onnx_label_disagreements"],
            "float64_flip_density": routed_density,
            "internal_nodes_reached": exp2["variants"]["on_threshold"]["probes_reaching_their_node"],
        }
        report["density_ratios"] = {
            "unrouted_vs_replay": round(density / replay_rate, 1),
            "routed_vs_replay": round(routed_density / replay_rate, 1),
            "routed_vs_unrouted": round(routed_density / density, 2),
        }
        print("density_ratios", report["density_ratios"])

    report["seconds"] = round(time.perf_counter() - started, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
