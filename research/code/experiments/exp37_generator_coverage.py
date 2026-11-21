"""Coverage and yield of the input generators of exp30, at the same budget.

exp30 counts which faults each generator detects. This measures, for the same
six generators (replay, uniform random, unrouted boundary fuzzing,
coverage-guided fuzzing, the three-probe set and the crossing pair), what they
exercise and what they find on the two CICIDS2017 models of exp30, at the
model's three-probe count (the crossing pair has two probes per node and uses
two thirds of it):

  * node coverage: share of internal nodes whose comparison some input evaluates
    (float32 scaling and the node's own comparison, as ONNX Runtime does);
  * branch coverage: share of internal nodes at which inputs take both branches;
  * crossing coverage: share of internal nodes evaluated on both values of the
    node's exact float32 crossing (the last raw float32 sent to the true branch
    and its successor), the inputs that decide any single-switch fault there;
  * float64 traversal vs ONNX Runtime: label flips and inputs whose probability
    deviates by more than 1e-5; float32 traversal: largest deviation;
  * generation time, and single-node faults detected (copied from exp30).

    python exp37_generator_coverage.py --data $DATA
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

from edgeids.data import RESULTS, load_test_split
from edgeids.onnx_forest import load_onnx_forest
from edgeids.runtime import label, numpy_probs, ort_probs

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp30", os.path.join(HERE, "exp30_testing_baselines.py"))
exp30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp30)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, build_probes, crossing_values, load_forest_arrays  # noqa: E402

TOL = 1e-5


def exercise(forest, x, cross_lo, cross_hi):
    """Per internal node: evaluated, went true, went false, hit c_A, hit c_A+."""
    n_nodes = len(forest.feature)
    seen = np.zeros(n_nodes, bool)
    went_t = np.zeros(n_nodes, bool)
    went_f = np.zeros(n_nodes, bool)
    hit_lo = np.zeros(n_nodes, bool)
    hit_hi = np.zeros(n_nodes, bool)
    off = forest.offset.astype(np.float32)
    sc = forest.scale.astype(np.float32)
    z = ((x - off) * sc).astype(np.float32)
    thr = forest.threshold.astype(np.float32)
    idx = np.arange(len(x))
    for t in range(len(forest.tree_offsets) - 1):
        cur = np.full(len(x), forest.tree_offsets[t], dtype=np.int64)
        active = ~forest.is_leaf[cur]
        while active.any():
            a = np.flatnonzero(active)
            c = cur[a]
            g = forest.feature[c]
            v = z[idx[a], g]
            r = x[idx[a], g]
            go = np.where(forest.strict[c], v < thr[c], v <= thr[c])
            seen[c] = True
            went_t[c[go]] = True
            went_f[c[~go]] = True
            hit_lo[c[r == cross_lo[c]]] = True
            hit_hi[c[r == cross_hi[c]]] = True
            cur[a] = np.where(go, forest.left[c], forest.right[c])
            active[a] = ~forest.is_leaf[cur[a]]
    return seen, went_t, went_f, hit_lo, hit_hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CICIDS2017 test_data.parquet directory")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp37_generator_coverage.json"))
    args = ap.parse_args()
    x, _, _ = load_test_split(args.data)
    x = np.ascontiguousarray(x, dtype=np.float32)
    base = json.load(open(os.path.join(RESULTS, "exp30_testing_baselines.json")))
    report = {"tolerance": TOL, "seed": exp30.SEED, "models": {}}
    for model, onnx_path in exp30.MODELS.items():
        forest = load_forest_arrays(onnx_path, np.float64)
        internal = np.flatnonzero(~forest.is_leaf)
        start = (forest.threshold[internal] / forest.scale[forest.feature[internal]]
                 + forest.offset[forest.feature[internal]]).astype(np.float32)
        ok, last_true = crossing_values(forest, internal, start)
        cross_lo = np.full(len(forest.feature), np.nan, np.float32)
        cross_hi = np.full(len(forest.feature), np.nan, np.float32)
        cross_lo[internal[ok]] = last_true[ok]
        cross_hi[internal[ok]] = np.nextafter(last_true[ok], np.float32(np.inf))
        probes = build_probes(onnx_path).rows
        ctx = {"x": x, "lo": x.min(axis=0), "hi": x.max(axis=0), "forest": forest, "internal": internal,
               "probes": probes, "crossing": build_probes(onnx_path, CROSSING_VARIANTS).rows}
        budget = len(probes)
        onnx_forest = load_onnx_forest(onnx_path)
        faults = base["models"][model]["faults"]
        single = [k for k, v in faults.items() if v["kind"] == "single_node"]
        entry = {"internal_nodes": int(len(internal)), "crossing_found": int(ok.sum()), "budget": budget,
                 "generators": {}}
        for method, gen in exp30.GENERATORS.items():
            rng = np.random.default_rng(exp30.SEED)
            t0 = time.perf_counter()
            xin = np.ascontiguousarray(gen(rng, budget, ctx), dtype=np.float32)
            gen_s = time.perf_counter() - t0
            seen, wt, wf, hl, hh = exercise(forest, xin, cross_lo, cross_hi)
            ref = ort_probs(onnx_path, xin)
            p32 = numpy_probs(onnx_forest, xin, np.float32)
            p64 = numpy_probs(onnx_forest, xin, np.float64)
            d64 = np.abs(p64 - ref).max(axis=1)
            f64 = label(p64) != label(ref)
            n_int = len(internal)
            row = {"inputs": int(len(xin)), "generation_seconds": round(gen_s, 2),
                   "node_coverage": float(seen[internal].sum() / n_int),
                   "branch_coverage": float((wt & wf)[internal].sum() / n_int),
                   "crossing_coverage": float((hl & hh)[internal].sum() / n_int),
                   "float64_label_flips": int(f64.sum()),
                   "float64_over_tolerance": int((f64 | (d64 > TOL)).sum()),
                   "float32_max_delta": float(np.abs(p32 - ref).max()),
                   "single_node_detected_probabilities": int(sum(faults[k].get(f"{method}@{budget}", False) for k in single)),
                   "single_node_detected_margins": int(sum(faults[k].get(f"margin:{method}@{budget}", False) for k in single)),
                   "single_node_faults": len(single)}
            entry["generators"][method] = row
            print(model, method, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        report["models"][model] = entry
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
