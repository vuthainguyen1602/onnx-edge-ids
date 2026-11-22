"""Localization of complex faults, ranked: the data-free probes and Leafparity on the same faults.

exp33 ranked faults confined to one element; exp43 reported only whether the first finding
lay in a faulty tree. This ranks every fault of exp43 that touches at most ten elements
(twenty single-node threshold faults, twenty coupled one-element faults and thirty
multi-node faults of 2, 5 and 10 elements of mixed kinds, per model) and scores, for each
check, the list it hands an engineer:

  probes      internal nodes ordered by how many failing probes target them (ties broken by
              node id), and trees ordered by their nodes' failing probes;
  Leafparity  the (tree, node) and (tree, leaf) elements its findings name, in the order it
              reports them, and their trees.

For each fault and list: the rank of the first faulty element (top-1, top-3, reciprocal
rank), the number of listed elements inspected before it (the whole list if none is
faulty), and, for multi-element faults, the share of faulty elements within the first ten.
Element level: probes target internal nodes only, so a faulty leaf can be found only at the
tree level; Leafparity names leaves. Tree level: the same for both.

Probes fail on margins for XGBoost (logistic) and on probabilities for the random forest,
as in exp43, against the training model. Models and faults are those of exp43.

    .lpvenv/bin/python research/code/experiments/exp49_localization_ranks.py --cicids-train $TRAIN --cicids-test $DATA
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

from edgeids.data import RESULTS
from edgeids.datasets import load_cicids
from edgeids.graph_faults import Graph, graph_mutants
from edgeids.models import build_models
from edgeids.runtime import ort_probs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, build_probes  # noqa: E402
from exp43_converter_faults_leafparity import export, margin_graph  # noqa: E402

TOL = 1e-5
MARGIN_TOL = 1e-4


def tree_of(g, node):
    return int(np.searchsorted(g.forest.tree_offsets, node, side="right") - 1)


def score(ranked, targets):
    """Metrics of one ranked list against the set of faulty items."""
    first = next((i + 1 for i, e in enumerate(ranked) if e in targets), None)
    return {"listed": len(ranked), "first_rank": first, "top1": first == 1, "top3": first is not None and first <= 3,
            "rr": 1.0 / first if first else 0.0, "inspected_before": (first - 1) if first else len(ranked),
            "found_in_top10": sum(1 for e in ranked[:10] if e in targets) / max(1, len(targets))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids-train", required=True)
    ap.add_argument("--cicids-test", required=True)
    ap.add_argument("--models", default="xgboost_hist,sklearn_rf")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp49_localization_ranks.json"))
    args = ap.parse_args()
    from leafparity import analyze

    x_train, y_train, _ = load_cicids(args.cicids_train, args.cicids_test)
    x_train = x_train.astype(np.float32)
    tmp = tempfile.mkdtemp()
    model_dir = os.path.join(RESULTS, "exp43_models")
    os.makedirs(model_dir, exist_ok=True)
    report = {"tolerance": TOL, "margin_tolerance": MARGIN_TOL, "models": {}}
    for key in args.models.split(","):
        model = build_models(0)[key][1].fit(x_train, y_train)
        path = export(key, model, x_train.shape[1], os.path.join(model_dir, f"{key}.onnx"))
        g = Graph(path)
        xgb = key.startswith("xgboost")
        muts = graph_mutants(g)
        entry = {}
        for name, (kw, faulty, cls) in muts.items():
            if "/" not in name or len(faulty) > 10:
                continue                                  # whole-graph faults: no element to localize
            gp = g.write(os.path.join(tmp, "fault.onnx"), **kw)
            nodes = {n for n in faulty if not g.forest.is_leaf[n]}
            trees = {tree_of(g, n) for n in faulty}
            el_all = {(int(g.tree_ids[g.order[n]]), int(g.node_ids[g.order[n]]),
                       "leaf" if g.forest.is_leaf[n] else "node") for n in faulty}
            row = {"class": cls, "elements": len(faulty), "internal_elements": len(nodes), "trees": len(trees)}
            # probes
            pr = build_probes(gp, CROSSING_VARIANTS)
            if xgb:
                fail = np.abs(model.predict(pr.rows, output_margin=True).astype(np.float64)
                              - ort_probs(margin_graph(gp, os.path.join(tmp, "m.onnx")), pr.rows)[:, -1]) > MARGIN_TOL
            else:
                ref, got = model.predict_proba(pr.rows).astype(np.float64), ort_probs(gp, pr.rows)
                fail = ((ref[:, 1] > 0.5) != (got[:, 1] > 0.5)) | np.any(np.abs(ref - got) > TOL, axis=1)
            tgt = pr.node[fail]
            if len(tgt):
                ids, cnt = np.unique(tgt, return_counts=True)
                order = np.lexsort((ids, -cnt))
                ranked_nodes = [int(i) for i in ids[order]]
                tcount = {}
                for i, c in zip(ids, cnt):
                    tcount[tree_of(g, int(i))] = tcount.get(tree_of(g, int(i)), 0) + int(c)
                ranked_trees = [t for t, _ in sorted(tcount.items(), key=lambda kv: (-kv[1], kv[0]))]
                row["probes"] = {"detected": True,
                                 "element": score(ranked_nodes, nodes) if nodes else None,
                                 "tree": score(ranked_trees, trees)}
            else:
                row["probes"] = {"detected": False}
            # Leafparity
            t0 = time.perf_counter()
            try:
                r = analyze(model, gp, input_dtype="float32", allow_missing=False, worst_case_seconds=30, max_findings=10000)
                ranked_el, ranked_tr = [], []
                for f in r.findings:
                    for o in f.get("occurrences", []):
                        t = o.get("tree_converted")
                        for kind, k in (("node", "node_converted"), ("leaf", "leaf_converted")):
                            if o.get(k) is not None and (t, o[k], kind) not in ranked_el:
                                ranked_el.append((t, o[k], kind))
                        if t not in ranked_tr:
                            ranked_tr.append(t)
                det = r.verdict["status"] != "EQUIVALENT"
                row["leafparity"] = {"detected": det,
                                     "element": score(ranked_el, el_all) if det else None,
                                     "tree": score(ranked_tr, trees) if det else None}
            except Exception as e:
                row["leafparity"] = {"detected": False, "error": f"{type(e).__name__}: {e}"}
            row["leafparity"]["seconds"] = round(time.perf_counter() - t0, 1)
            entry[name] = row
            pe, le = row["probes"].get("element") or {}, row["leafparity"].get("element") or {}
            print(f"{key:12s} {name:26s} probes el-rank={pe.get('first_rank')} tree-rank={(row['probes'].get('tree') or {}).get('first_rank')} "
                  f"| LP el-rank={le.get('first_rank')} tree-rank={(row['leafparity'].get('tree') or {}).get('first_rank')}", flush=True)
            report["models"][key] = entry
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1, default=str)

    # summary by fault group, tool and level
    summary = {}
    for key, entry in report["models"].items():
        for name, row in entry.items():
            grp = name.split("/")[0].replace("multi_2", "multi").replace("multi_5", "multi").replace("multi_10", "multi")
            if grp == "coupled":
                grp = "coupled leaf" if "leaf_" in name else "coupled feature"
            for tool in ("probes", "leafparity"):
                for level in ("element", "tree"):
                    s = summary.setdefault(f"{key}|{grp}|{tool}|{level}", {"faults": 0, "detected": 0, "scored": 0,
                                                                           "top1": 0, "top3": 0, "rr": 0.0,
                                                                           "inspected_before": [], "found_in_top10": []})
                    s["faults"] += 1
                    s["detected"] += bool(row[tool].get("detected"))
                    m = row[tool].get(level)
                    if m:
                        s["scored"] += 1; s["top1"] += m["top1"]; s["top3"] += m["top3"]; s["rr"] += m["rr"]
                        s["inspected_before"].append(m["inspected_before"]); s["found_in_top10"].append(m["found_in_top10"])
    for s in summary.values():
        n = s["scored"]
        s["mrr"] = round(s.pop("rr") / n, 3) if n else None
        ib = s.pop("inspected_before"); ft = s.pop("found_in_top10")
        s["inspected_before_median"] = float(np.median(ib)) if ib else None
        s["found_in_top10_mean"] = round(float(np.mean(ft)), 3) if ft else None
    report["summary"] = summary
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    for k, s in summary.items():
        print(k, s)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
