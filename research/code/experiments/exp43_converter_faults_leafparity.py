"""Leafparity and the data-free probes on the same converter faults, same models, same graphs.

exp21 ran Leafparity on the converter pairs as trained, where every pair already
differs (a float64 StandardScaler against the graph's float32 Scaler), so it could
not ask which injected fault each check finds. This removes the gap: XGBoost (hist)
and the scikit-learn random forest are trained on CICIDS2017 without a scaler, and
Leafparity proves each graph EQUIVALENT to its model for float32 inputs. Faults are
then written into the graph itself, as a faulty converter would (``graph_faults.py``):

  * whole-graph: every threshold one float32 step up, two split features swapped,
    tree 0's leaf weights swapped, every split mode flipped (LEQ <-> LT), the base
    value dropped (XGBoost);
  * single-switch, one node: one threshold one step up, and twenty single-node
    faults that move one node's decision by one float32 input (as exp30);
  * coupled one-element: ten single leaves' weights swapped, ten single nodes'
    split feature changed (as exp40);
  * multi-node: ten draws each of 2, 5 and 10 simultaneous single-element faults of
    random kinds (threshold past the crossing, split feature, leaf weights).

For every fault, against the training model:

  * Leafparity 0.1.2 (input_dtype float32, no missing values): verdict, whether a
    class change was found, time, regions examined, and which faulty elements its
    findings name (tree and node, or tree and leaf, of the converted graph);
  * the gate's data-free crossing pair built from the faulty graph: framework vs
    ONNX Runtime on the probes, failing on a label flip or a probability deviation
    over 1e-5 (and, for XGBoost, on margins over 1e-4), probes, time, and the rank
    of the faulty elements among the failing probes' target nodes;
  * a 2,000-row replay of the test split: probability that it passes.

The gate's runtime check (NumPy engine vs ONNX Runtime on one graph) passes every
fault here by construction: both runtimes read the same faulty graph.

    .lpvenv/bin/python research/code/experiments/exp43_converter_faults_leafparity.py \\
        --cicids-train $TRAIN --cicids-test $DATA
"""

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter

import numpy as np

from edgeids.data import RESULTS
from edgeids.datasets import load_cicids
from edgeids.graph_faults import Graph, graph_mutants
from edgeids.models import build_models
from edgeids.runtime import ort_probs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, build_probes  # noqa: E402

TOL = 1e-5
MARGIN_TOL = 1e-4
REPLAY_ROWS = 2000
MODELS = ("xgboost_hist", "sklearn_rf")


def export(key, model, n_features, out):
    if key == "xgboost_hist":
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        onx = onnxmltools.convert_xgboost(model, initial_types=[("input", FloatTensorType([None, n_features]))],
                                          target_opset=15)
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onx = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, n_features]))],
                              options={id(model): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3})
    with open(out, "wb") as f:
        f.write(onx.SerializeToString())
    return out


def margin_graph(path, out):
    """Copy of the graph with post_transform NONE: ONNX Runtime then returns [-s, s] for one score column."""
    import onnx
    m = onnx.load(path)
    for n in m.graph.node:
        if n.op_type == "TreeEnsembleClassifier":
            for a in n.attribute:
                if a.name == "post_transform":
                    a.s = b"NONE"
    onnx.save_model(m, out)
    return out


def element_of(g, node):
    """(tree, node id in tree, is_leaf) in the converted graph for a global node id."""
    i = g.order[node]
    return int(g.tree_ids[i]), int(g.node_ids[i]), bool(g.forest.is_leaf[node])


def tree_of(g, node):
    return int(np.searchsorted(g.forest.tree_offsets, node, side="right") - 1)


def probe_localization(g, faulty, fail_targets):
    """Localization from the failing probes' target nodes, ranked by failure count.

    Element level: an internal faulty node is found when it is a failing target; a probe
    targets internal nodes only, so it cannot name a faulty leaf. Tree level: the trees of
    the failing targets, ranked by failure count, against the trees holding a faulty element."""
    counts = Counter(int(t) for t in fail_targets)
    if not counts or not faulty:
        return None
    ranked = [n for n, _ in counts.most_common()]
    nodes = {n for n in faulty if not g.forest.is_leaf[n]}
    trees = {tree_of(g, n) for n in faulty}
    tree_counts = Counter()
    for n, c in counts.items():
        tree_counts[tree_of(g, n)] += c
    ranked_trees = [t for t, _ in tree_counts.most_common()]
    return {"top1_element": bool(ranked[0] in nodes), "elements_named": sum(1 for n in nodes if n in counts),
            "internal_elements": len(nodes), "leaf_elements": len(faulty) - len(nodes),
            "top1_tree": bool(ranked_trees[0] in trees), "trees_named": sum(1 for t in trees if t in tree_counts),
            "trees": len(trees),
            "first_tree_hit_rank": next((i + 1 for i, t in enumerate(ranked_trees) if t in trees), None)}


def leafparity_localization(g, faulty, findings):
    """Localization from Leafparity's findings, in the order it reports them.

    Element level: a faulty internal node or leaf named as (tree, node) or (tree, leaf) of the
    converted graph. Tree level: the trees its findings name."""
    if not faulty:
        return None
    want = {element_of(g, n)[:2] + (("leaf" if g.forest.is_leaf[n] else "node"),) for n in faulty}
    trees = {tree_of(g, n) for n in faulty}

    def named(f):
        s, ts = set(), set()
        for o in f.get("occurrences", []):
            t = o.get("tree_converted")
            ts.add(t)
            if o.get("node_converted") is not None:
                s.add((t, o["node_converted"], "node"))
            if o.get("leaf_converted") is not None:
                s.add((t, o["leaf_converted"], "leaf"))
        return s, ts
    per = [named(f) for f in findings]
    if not per:
        return None
    all_el = set().union(*[s for s, _ in per]); all_tr = set().union(*[ts for _, ts in per])
    return {"top1_element": bool(per[0][0] & want), "elements_named": len(all_el & want),
            "elements": len(want), "top1_tree": bool(per[0][1] & trees), "trees_named": len(all_tr & trees),
            "trees": len(trees)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids-train", required=True)
    ap.add_argument("--cicids-test", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--worst-case-seconds", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0, help="first N faults per model (smoke test)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp43_converter_faults_leafparity.json"))
    args = ap.parse_args()

    import leafparity
    import onnxruntime
    import sklearn
    import xgboost
    from leafparity import analyze

    x_train, y_train, x_test = load_cicids(args.cicids_train, args.cicids_test)
    x_train = x_train.astype(np.float32)
    x_test = np.ascontiguousarray(x_test, dtype=np.float32)
    n_features = x_train.shape[1]
    model_dir = os.path.join(RESULTS, "exp43_models")
    os.makedirs(model_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()
    report = {"versions": {"leafparity": "0.1.2", "numpy": np.__version__, "sklearn": sklearn.__version__,
                           "xgboost": xgboost.__version__, "onnxruntime": onnxruntime.__version__},
              "settings": {"input_dtype": "float32", "allow_missing": False, "tolerance": TOL,
                           "margin_tolerance": MARGIN_TOL, "replay_rows": REPLAY_ROWS, "probe_set": "crossing",
                           "worst_case_seconds": args.worst_case_seconds, "scaler": "none"},
              "models": {}}

    for key in args.models.split(","):
        model = build_models(0)[key][1]
        model.fit(x_train, y_train)
        path = export(key, model, n_features, os.path.join(model_dir, f"{key}.onnx"))
        g = Graph(path)
        is_xgb = key == "xgboost_hist"

        def frame_probs(x):
            return model.predict_proba(x).astype(np.float64)

        def frame_margin(x):
            return model.predict(x, output_margin=True).astype(np.float64)

        ref_replay = frame_probs(x_test)
        ref_replay_m = frame_margin(x_test) if is_xgb else None

        def run_fault(name, gpath, faulty, cls):
            row = {"class": cls, "faulty_elements": len(faulty)}
            t0 = time.perf_counter()
            try:
                r = analyze(model, gpath, input_dtype="float32", allow_missing=False,
                            worst_case_seconds=args.worst_case_seconds, max_findings=10000)
                v = r.verdict
                row["leafparity"] = {
                    "verdict": v["status"], "detected": v["status"] != "EQUIVALENT",
                    "class_change_found": bool(v.get("label_flip_found")),
                    "distinct_problems": v.get("distinct_problems"), "problem_kinds": v.get("problem_kinds"),
                    "max_raw_difference_found": v.get("max_raw_difference_found"),
                    "max_raw_difference_guaranteed": v.get("max_raw_difference_guaranteed"),
                    "rounding_allowance": v.get("rounding_allowance"),
                    "regions_examined": r.to_summary_dict().get("joint_regions_examined"),
                    "localization": leafparity_localization(g, faulty, r.findings)}
            except Exception as e:
                row["leafparity"] = {"error": f"{type(e).__name__}: {e}", "detected": None}
            row["leafparity"]["seconds"] = round(time.perf_counter() - t0, 2)

            t0 = time.perf_counter()
            pr = build_probes(gpath, CROSSING_VARIANTS)
            ref = frame_probs(pr.rows)
            got = ort_probs(gpath, pr.rows)
            flips = (ref[:, 1] > 0.5) != (got[:, 1] > 0.5)
            fail = flips | np.any(np.abs(ref - got) > TOL, axis=1)
            probe = {"probes": len(pr), "label_flips": int(flips.sum()), "failing": int(fail.sum()),
                     "max_deviation": float(np.abs(ref - got).max()),
                     "detected": bool(fail.any())}
            if is_xgb:
                mg = ort_probs(margin_graph(gpath, os.path.join(tmp, "margin.onnx")), pr.rows)[:, -1]
                mfail = np.abs(frame_margin(pr.rows) - mg) > MARGIN_TOL
                probe["margin_failing"] = int(mfail.sum())
                probe["margin_detected"] = bool(mfail.any())
                fail_for_loc = mfail
            else:
                fail_for_loc = fail
            probe["seconds"] = round(time.perf_counter() - t0, 2)
            probe["localization"] = probe_localization(g, faulty, pr.node[fail]) if len(faulty) <= 10 else None
            if is_xgb:
                probe["margin_localization"] = (probe_localization(g, faulty, pr.node[fail_for_loc])
                                                if len(faulty) <= 10 else None)
            row["probes"] = probe

            got_r = ort_probs(gpath, x_test)
            rf = ((ref_replay[:, 1] > 0.5) != (got_r[:, 1] > 0.5)) | np.any(np.abs(ref_replay - got_r) > TOL, axis=1)
            row["replay"] = {"rows_failing": int(rf.sum()), "label_flips": int(((ref_replay[:, 1] > 0.5) != (got_r[:, 1] > 0.5)).sum()),
                             "max_deviation": float(np.abs(ref_replay - got_r).max()),
                             "prob_passes": float((1 - rf.mean()) ** REPLAY_ROWS)}
            if is_xgb:
                mg_r = ort_probs(margin_graph(gpath, os.path.join(tmp, "margin.onnx")), x_test)[:, -1]
                mrf = np.abs(ref_replay_m - mg_r) > MARGIN_TOL
                row["replay"]["margin_prob_passes"] = float((1 - mrf.mean()) ** REPLAY_ROWS)
            lp, pb = row["leafparity"], row["probes"]
            print(f"{key:12s} {name:28s} LP={lp.get('verdict', lp.get('error'))[:14]:14s} {lp['seconds']:6.1f}s "
                  f"probes fail={pb['failing']}{'/m' + str(pb.get('margin_failing')) if is_xgb else ''} "
                  f"{pb['seconds']:5.1f}s replay P(pass)={row['replay']['prob_passes']:.3f}", flush=True)
            return row

        entry = {"control/unmutated": run_fault("control/unmutated", path, [], "control")}
        muts = graph_mutants(g)
        for i, (name, (kw, faulty, cls)) in enumerate(muts.items()):
            if args.limit and i >= args.limit:
                break
            gpath = g.write(os.path.join(tmp, "fault.onnx"), **kw)
            entry[name] = run_fault(name, gpath, faulty, cls)
            report["models"][key] = {"internal_nodes": int(len(g.internal)), "faults": entry}
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1, default=str)
        report["models"][key] = {"internal_nodes": int(len(g.internal)), "faults": entry}

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
