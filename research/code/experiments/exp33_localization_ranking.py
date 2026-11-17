"""Ranked fault localization: top-1, top-3 and mean reciprocal rank over 53 faults.

exp29 scored the gate's two localization levels on the fourteen exporter faults.
This widens the fault set with the twenty single-node faults per model of
exp30 (one node's threshold moved just past ONNX Runtime's float32 crossing, so
the faulty element is known exactly) and scores both levels as rankings, using
the gate's default crossing pair and its own failing rule (a label change or a
probability moved by more than 1e-5):

  L2 (first divergence, artifact vs. graph): nodes ranked by how many failing
      probes first diverge there; correct when the node (or, for scaler and
      feature faults, the node's feature; for leaf faults, its tree) is one the
      fault changed;
  L1 (probe targets, any two runtimes): (feature, threshold) pairs ranked by
      failing probes for single-node faults, features ranked for feature faults.

Per fault: the rank of the first correct suspect; aggregated as top-1, top-3 and
MRR over the faults the gate detects. Global faults (every threshold, every
scale) have no single suspect and are left out of the ranking.

    python exp33_localization_ranking.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS
from edgeids.faults import exporter_mutants, single_node_mutants

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from onnx_edge_ids import create_inference_engine  # noqa: E402
from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, build_probes, load_forest_arrays  # noqa: E402
from onnx_edge_ids.localize import first_divergences, summarize_failures  # noqa: E402

TOL = 1e-5
MODELS = {"deployed_rf": os.path.join(ARTIFACTS, "ids_rf.onnx"),
          "xgboost_hist": os.path.join(RESULTS, "exp5_models", "xgboost_hist.onnx")}
LOCALIZED = {"thr_one_up", "feature_swap", "offset_one_up", "leaf_tree0"}   # exporter faults with a single suspect


def outputs(engine, x, batch=50000):
    p, c = [], []
    for i in range(0, len(x), batch):
        r = engine.predict_batch(x[i:i + batch])
        p.append(r.predictions); c.append(r.confidences)
    return np.concatenate(p), np.concatenate(c)


def truth(base, mut, forest):
    f32 = lambda k: np.asarray(base[k]).astype(np.float32) != np.asarray(mut[k]).astype(np.float32)
    thr_nodes = set(np.flatnonzero(f32("threshold") & ~base["is_leaf"]).tolist())
    feat_nodes = np.flatnonzero(base["feature"] != mut["feature"])
    features = set(base["feature"][feat_nodes].tolist()) | set(mut["feature"][feat_nodes].tolist())
    features |= set(np.flatnonzero(f32("offset") | f32("scale")).tolist())
    leaf_rows = np.flatnonzero(np.any(f32("leaf_scores"), axis=1))
    trees = set((np.searchsorted(base["tree_offsets"], leaf_rows, side="right") - 1).tolist())
    pairs = {(int(forest.feature[n]), float(np.float32(forest.threshold[n]))) for n in thr_nodes}
    return {"nodes": thr_nodes, "features": features, "trees": trees, "pairs": pairs}


def first_rank(items, ok):
    for i, it in enumerate(items, 1):
        if ok(it):
            return i
    return None


def l2_ok(t):
    def ok(d):
        if d["kind"] == "threshold":
            return d["node"] in t["nodes"]
        if d["kind"] in ("scaler", "split feature"):
            return d["feature"] in t["features"]
        if d["kind"] == "leaf weight":
            return d["tree"] in t["trees"]
        return False
    return ok


def summarize(rows, key):
    ranks = [r[key] for r in rows if r.get("detected")]
    n = len(ranks)
    if not n:
        return None
    return {"faults": n, "top1": sum(1 for r in ranks if r == 1) / n, "top3": sum(1 for r in ranks if r and r <= 3) / n,
            "mrr": sum((1 / r) if r else 0 for r in ranks) / n}


def main():
    tmp = tempfile.mkdtemp()
    report = {"tolerance": TOL, "probe_set": "crossing", "faults": {}, "summary": {}}
    for model, onnx_path in MODELS.items():
        npz = os.path.join(tmp, f"{model}.npz")
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_numpy.py"), "--model", onnx_path,
                        "--out", npz], check=True, capture_output=True)
        base = dict(np.load(npz, allow_pickle=False))
        forest = load_forest_arrays(onnx_path, np.float64)
        probes = build_probes(onnx_path, CROSSING_VARIANTS)
        ref = outputs(create_inference_engine("onnx", onnx_path), probes.rows)
        faults = {f"exporter/{k}": v for k, v in exporter_mutants(base).items() if k in LOCALIZED}
        faults.update({f"single_node/{k}": v for k, v in single_node_mutants(base, onnx_path, 0).items()})
        for name, arrays in faults.items():
            path = os.path.join(tmp, f"{model}_{name.replace('/', '_')}.npz")
            np.savez(path, **arrays)
            got = outputs(create_inference_engine("numpy", path), probes.rows)
            failing = (ref[0] != got[0]) | (np.abs(ref[1] - got[1]) > TOL)
            t = truth(base, arrays, forest)
            row = {"family": name.split("/")[0], "detected": bool(failing.any()), "failing_probes": int(failing.sum())}
            if failing.any():
                l1 = summarize_failures(onnx_path, probes, failing, top=10 ** 6)
                l2 = first_divergences(onnx_path, path, probes.rows[np.flatnonzero(failing)[:5000]], top=10 ** 6)
                row["l2_rank"] = first_rank(l2["top_nodes"], l2_ok(t))
                if t["pairs"] and not t["features"]:
                    row["l1_rank"] = first_rank(l1["top_pairs"], lambda p: (p["feature"], p["threshold"]) in t["pairs"])
                elif t["features"]:
                    row["l1_rank"] = first_rank(l1["top_features"], lambda p: p["feature"] in t["features"])
                else:
                    row["l1_rank"] = None   # a tree fault: probe targets cannot name a tree
                row["l1_scope"] = l1["scope"]
            report["faults"][f"{model}/{name}"] = row
            print(model, name, json.dumps(row), flush=True)
    rows = list(report["faults"].values())
    for fam in ("exporter", "single_node", "all"):
        sel = [r for r in rows if fam == "all" or r["family"] == fam]
        report["summary"][fam] = {"faults": len(sel), "detected": sum(r["detected"] for r in sel),
                                  "L2": summarize(sel, "l2_rank"),
                                  "L1": summarize([r for r in sel if "l1_rank" in r and r["l1_rank"] is not None
                                                   or (r.get("detected") and r.get("l1_rank") is None
                                                       and "l1_scope" in r and r["family"] == "single_node")],
                                                  "l1_rank")}
    print(json.dumps(report["summary"], indent=1))
    out = os.path.join(RESULTS, "exp33_localization_ranking.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
