"""Three levels of what the crossing pair guarantees for a single-node fault: routing, output, label.

Theorem 1 is about routing: for a fault that is single-switch at node v, some crossing-pair
probe built for v is decided differently at v. Whether the gate then fails depends on two
further steps, and this measures each on the same faults and the same denominator:

  reached   a probe built for v reaches v (both traversals agree above v, so this is the
            reference graph's routing);
  routing   on some probe, the tree holding v ends in a different leaf;
  output    some probe's probability moves by more than 1e-5 (the gate's check), or, for
            logistic models, its margin by more than 1e-4;
  label     some probe's label changes.

A routed difference does not reach the output when both subtrees of v end, for that probe's
other features, in leaves of equal weight; a changed output does not reach the probability
when a logistic output saturates; and it changes the label only when it crosses one half.

Faults: twenty single-node threshold faults per model, each moving one node's decision by one
float32 input just past ONNX Runtime's crossing (graph_faults.past_crossing), on the deployed
forest and the CICIDS2017 XGBoost model of exp5 (both with a Scaler) and on the two models of
exp43 (no Scaler). Both sides are the float32 traversal of the graph (onnx_forest), the
reference graph against the faulty one, on the gate's data-free crossing-pair probes built
from the reference graph.

    python research/code/experiments/exp48_guarantee_levels.py
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS
from edgeids.graph_faults import Graph, graph_mutants
from edgeids.onnx_forest import _go_true, _leaf_of_tree, load_onnx_forest, raw_scores, scale_input
from edgeids.runtime import probabilities

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, build_probes  # noqa: E402

TOL = 1e-5
MARGIN_TOL = 1e-4
MODELS = {
    "deployed_rf": os.path.join(ARTIFACTS, "ids_rf.onnx"),
    "xgboost_hist (Scaler)": os.path.join(RESULTS, "exp5_models", "xgboost_hist.onnx"),
    "sklearn_rf (no Scaler)": os.path.join(RESULTS, "exp43_models", "sklearn_rf.onnx"),
    "xgboost_hist (no Scaler)": os.path.join(RESULTS, "exp43_models", "xgboost_hist.onnx"),
}


def tree_of(forest, node):
    return int(np.searchsorted(forest.tree_offsets, node, side="right") - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp48_guarantee_levels.json"))
    args = ap.parse_args()
    tmp = tempfile.mkdtemp()
    report = {"tolerance": TOL, "margin_tolerance": MARGIN_TOL, "probe_set": "crossing", "models": {}, "summary": {}}
    totals = {"faults": 0, "reached": 0, "routing": 0, "output_probability": 0, "label": 0,
              "logistic_faults": 0, "output_margin": 0}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print("missing", path, "(run exp5 / exp43 first)"); continue
        g = Graph(path)
        ref = g.forest
        logistic = ref.post_transform == "LOGISTIC"
        pr = build_probes(path, CROSSING_VARIANTS)
        x = pr.rows
        s_ref = raw_scores(ref, x, np.float32)
        p_ref = probabilities(ref, s_ref)
        sc_ref = scale_input(ref, x, np.float32)
        muts = {k: v for k, v in graph_mutants(g, single=args.count, coupled=0, multi_sizes=()).items()
                if k.startswith("single_node/")}
        entry = {"logistic": logistic, "probes": int(len(x)), "faults": {}}
        for key, (kw, faulty, _) in muts.items():
            v = faulty[0]
            bad = load_onnx_forest(g.write(os.path.join(tmp, "f.onnx"), **kw))
            own = pr.node == v                                 # probes built for the faulty node
            t = tree_of(ref, v)
            leaf_ref = _leaf_of_tree(ref, sc_ref[own], ref.threshold.astype(np.float32), t)
            # reaching v: v lies on the probe's path in the reference tree
            on_path = np.zeros(int(own.sum()), dtype=bool)
            node = np.full(int(own.sum()), ref.tree_offsets[t], dtype=np.int64)
            for _ in range(64):
                on_path |= node == v
                live = ~ref.is_leaf[node]
                if not live.any():
                    break
                idx = np.flatnonzero(live)
                cur = node[idx]
                go = _go_true(ref.mode[cur], sc_ref[own][idx, ref.feature[cur]], ref.threshold[cur].astype(np.float32))
                node[idx] = np.where(go, ref.left[cur], ref.right[cur])
            reached = bool(on_path.any())
            leaf_bad = _leaf_of_tree(bad, scale_input(bad, x[own], np.float32), bad.threshold.astype(np.float32), t)
            routing = bool(np.any(leaf_ref != leaf_bad))
            s_bad = raw_scores(bad, x, np.float32)
            p_bad = probabilities(bad, s_bad)
            out_p = bool(np.any(np.abs(p_ref - p_bad) > TOL))
            label = bool(np.any((p_ref[:, 1] > 0.5) != (p_bad[:, 1] > 0.5)))
            row = {"node": int(v), "reached": reached, "routing": routing, "output_probability": out_p, "label": label,
                   "max_probability_change": float(np.abs(p_ref - p_bad).max())}
            if logistic:
                out_m = bool(np.any(np.abs(s_ref[:, -1] - s_bad[:, -1]) > MARGIN_TOL))
                row["output_margin"] = out_m
                row["max_margin_change"] = float(np.abs(s_ref[:, -1] - s_bad[:, -1]).max())
            entry["faults"][key] = row
            totals["faults"] += 1
            for k in ("reached", "routing", "output_probability", "label"):
                totals[k] += row[k]
            if logistic:
                totals["logistic_faults"] += 1; totals["output_margin"] += row["output_margin"]
        n = len(entry["faults"])
        entry["summary"] = {k: sum(f[k] for f in entry["faults"].values()) for k in ("reached", "routing", "output_probability", "label")}
        if logistic:
            entry["summary"]["output_margin"] = sum(f["output_margin"] for f in entry["faults"].values())
        entry["summary"]["faults"] = n
        report["models"][name] = entry
        print(name, entry["summary"], flush=True)
    report["summary"] = totals
    print("total", totals)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
