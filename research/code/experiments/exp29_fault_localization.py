"""From detection to localization: does a failing gate point at the fault?

The gate's localization (onnx_edge_ids.localize) has two levels:
  L1, from the probes' own target nodes (any two runtimes): a scope (threshold,
      feature or global) and suspects;
  L2, artifact vs. graph walked tree by tree on the failing probes: the first
      divergent node of each probe and its kind (threshold, scaler, split
      feature, split mode, leaf weight, base value).

Faults, with ground truth:
  * the fourteen exporter faults of exp25 (deployed forest and CICIDS2017
    XGBoost), truth = the array elements that differ from the exported artifact;
  * the six arithmetic faults of exp11 in the float32 traversal of the deployed
    forest (truth: global, artifact unchanged), detected against the deployed
    float32 traversal, which equals ONNX Runtime on every probe.

Per fault: does L1 name the right scope and suspects (and, for feature
faults, do the faulty features rank first in its list), are all of L2's
divergences consistent with the truth (precision), and is L2's top node one of
the faulty elements.

    python exp29_fault_localization.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS
from edgeids.faults import exporter_mutants, fault_scorers
from edgeids.forest import load_forest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from onnx_edge_ids import create_inference_engine  # noqa: E402
from onnx_edge_ids.boundary_probe import build_probes  # noqa: E402
from onnx_edge_ids.localize import first_divergences, summarize_failures  # noqa: E402

TOL = 1e-5
MODELS = {"deployed_rf": os.path.join(ARTIFACTS, "ids_rf.onnx"),
          "xgboost_hist": os.path.join(RESULTS, "exp5_models", "xgboost_hist.onnx")}


def truth(base, mut):
    """What the fault changed, and the scope a localizer should report."""
    f32 = lambda k: (np.asarray(base[k]).astype(np.float32) != np.asarray(mut[k]).astype(np.float32))
    thr_nodes = set(np.flatnonzero(f32("threshold") & ~base["is_leaf"]).tolist())
    feat_nodes = np.flatnonzero(base["feature"] != mut["feature"])
    features = set(base["feature"][feat_nodes].tolist()) | set(mut["feature"][feat_nodes].tolist())
    scaler_feats = set(np.flatnonzero(f32("offset") | f32("scale")).tolist())
    mode_nodes = set(np.flatnonzero(base["strict"] != mut["strict"]).tolist())
    leaf_rows = np.flatnonzero(np.any(f32("leaf_scores"), axis=1))
    trees = set((np.searchsorted(base["tree_offsets"], leaf_rows, side="right") - 1).tolist())
    base_changed = bool(np.any(f32("base_values")))
    if len(thr_nodes) == 1 and not (features or scaler_feats or mode_nodes or trees or base_changed):
        scope = "threshold"
    elif (features or scaler_feats) and len(features | scaler_feats) <= 2 and not (thr_nodes or mode_nodes or trees):
        scope = "feature"
    elif trees and len(trees) <= 2 and not (thr_nodes or features or scaler_feats or mode_nodes or base_changed):
        scope = "tree"
    else:
        scope = "global"
    return {"scope": scope, "threshold_nodes": thr_nodes, "features": features | scaler_feats,
            "scaler_features": scaler_feats, "mode_nodes": mode_nodes, "trees": trees, "base": base_changed}


def consistent(d, t):
    k = d["kind"]
    if k == "threshold":
        return d["node"] in t["threshold_nodes"]
    if k == "scaler":
        return d["feature"] in t["scaler_features"]
    if k == "split feature":
        return d["feature"] in t["features"]
    if k == "split mode":
        return d["node"] in t["mode_nodes"]
    if k == "leaf weight":
        return d["tree"] in t["trees"]
    return False


def l1_correct(l1, t):
    if t["scope"] in ("global", "tree"):
        return l1["scope"] == "global"           # L1 cannot name a tree; global is the honest answer
    if l1["scope"] != t["scope"]:
        return False
    if t["scope"] == "feature":
        return {s["feature"] for s in l1["suspects"]} <= t["features"]
    return True                                   # threshold: the single changed pair


def outputs(engine, x, batch=50000):
    p, c = [], []
    for i in range(0, len(x), batch):
        r = engine.predict_batch(x[i:i + batch])
        p.append(r.predictions); c.append(r.confidences)
    return np.concatenate(p), np.concatenate(c)


def exporter_faults(report, tmp):
    for name, onnx_path in MODELS.items():
        npz = os.path.join(tmp, f"{name}.npz")
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_numpy.py"), "--model", onnx_path,
                        "--out", npz], check=True, capture_output=True)
        base = dict(np.load(npz, allow_pickle=False))
        base.setdefault("strict", np.zeros(len(base["left"]), dtype=bool))
        base.setdefault("base_values", np.zeros(base["leaf_scores"].shape[1]))
        probes = build_probes(onnx_path)
        ref = outputs(create_inference_engine("onnx", onnx_path), probes.rows)
        for fault, arrays in exporter_mutants(base).items():
            path = os.path.join(tmp, f"{name}_{fault}.npz")
            np.savez(path, **arrays)
            got = outputs(create_inference_engine("numpy", path), probes.rows)
            failing = (ref[0] != got[0]) | (np.abs(ref[1] - got[1]) > TOL)
            t = truth(base, arrays)
            row = {"truth_scope": t["scope"], "failing_probes": int(failing.sum())}
            if failing.any():
                l1 = summarize_failures(onnx_path, probes, failing)
                sample = np.flatnonzero(failing)[:5000]
                l2 = first_divergences(onnx_path, path, probes.rows[sample], top=1000000)
                rows_total = sum(d["rows"] for d in l2["top_nodes"])
                rows_ok = sum(d["rows"] for d in l2["top_nodes"] if consistent(d, t))
                row.update({
                    "l1_scope": l1["scope"], "l1_suspects": l1["suspects"], "l1_correct": l1_correct(l1, t),
                    "l1_top_features": [d["feature"] for d in l1["top_features"][:3]],
                    "l1_truth_features_in_top_k": (set(d["feature"] for d in l1["top_features"][:len(t["features"])])
                                                   == t["features"]) if t["scope"] == "feature" else None,
                    "l2_divergent_nodes": l2["divergent_nodes"], "l2_by_kind": l2["by_kind"],
                    "l2_base_value_differs": l2["base_value_differs"],
                    "l2_precision": (rows_ok / rows_total) if rows_total else None,
                    "l2_top_node_correct": consistent(l2["top_nodes"][0], t) if l2["top_nodes"] else None,
                    "l2_base_correct": (l2["base_value_differs"] == t["base"]),
                })
            report["exporter_faults"][f"{name}/{fault}"] = row
            print(name, fault, json.dumps({k: v for k, v in row.items() if k != "l1_suspects"}), flush=True)


def arithmetic_faults(report):
    onnx_path = MODELS["deployed_rf"]
    probes = build_probes(onnx_path)
    f32 = load_forest(os.path.join(ARTIFACTS, "ids_rf_numpy.npz"), np.float32)
    f64 = load_forest(os.path.join(ARTIFACTS, "ids_rf_numpy.npz"), np.float64)
    scorers = fault_scorers(f32, f64)
    ref = scorers["float32 (deployed)"](probes.rows)
    l2 = first_divergences(onnx_path, os.path.join(ARTIFACTS, "ids_rf_numpy.npz"), probes.rows[:5000])
    for name, fn in scorers.items():
        if name == "float32 (deployed)":
            continue
        got = fn(probes.rows)
        failing = (np.argmax(ref, 1) != np.argmax(got, 1)) | (np.abs(ref - got).max(axis=1) > TOL)
        l1 = summarize_failures(onnx_path, probes, failing)
        row = {"truth_scope": "global", "failing_probes": int(failing.sum()), "l1_scope": l1["scope"],
               "l1_correct": l1["scope"] == "global", "features_implicated": l1["features_implicated"],
               "features_probed": l1["features_probed"],
               "l2_says_artifact_matches_graph": (not l2["by_kind"]) and not l2["base_value_differs"]}
        report["arithmetic_faults"][name] = row
        print("arithmetic", name, json.dumps(row), flush=True)


def main():
    report = {"tolerance": TOL, "exporter_faults": {}, "arithmetic_faults": {}}
    exporter_faults(report, tempfile.mkdtemp())
    arithmetic_faults(report)
    out = os.path.join(RESULTS, "exp29_fault_localization.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print("wrote", out)


if __name__ == "__main__":
    main()
