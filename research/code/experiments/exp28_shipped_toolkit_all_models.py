"""The shipped exporter, engine and gate on all thirteen models.

Up to v0.4 the shipped exporter accepted BRANCH_LEQ graphs with two score
columns only, so the twelve trained models of Section 5 were run through the
experiment code. This exports each of the thirteen graphs with the shipped
``scripts/export_numpy.py`` and, on the gate's data-free probes built from the
graph, reports:

  * the structure check of the gate (NumPy artifact vs. graph at float32);
  * label flips and the largest confidence difference of the shipped NumPy
    engine against ONNX Runtime;
  * which probes reach their node, and for nodes whose probes do not, whether
    their (feature, threshold) pair is still exercised at another node that a
    probe does reach.

    python exp28_shipped_toolkit_all_models.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS
from edgeids.models import MODEL_KEYS

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from onnx_edge_ids import create_inference_engine  # noqa: E402
from onnx_edge_ids.boundary_probe import build_probes, load_forest_arrays  # noqa: E402
from onnx_edge_ids.onnx_artifact import check_npz_matches_onnx  # noqa: E402


def reached(forest, probes):
    """Per probe: does it pass through its target node under float32 arithmetic."""
    x = probes.rows.astype(np.float32)
    scaled = (x - forest.offset.astype(np.float32)) * forest.scale.astype(np.float32)
    thr = forest.threshold.astype(np.float32)
    tree = np.searchsorted(forest.tree_offsets, probes.node, side="right") - 1
    cur = forest.tree_offsets[tree].copy()
    idx = np.arange(len(cur))
    hit = cur == probes.node
    active = ~forest.is_leaf[cur] & ~hit
    while active.any():
        a = np.flatnonzero(active)
        c = cur[a]
        v = scaled[idx[a], forest.feature[c]]
        go = np.where(forest.strict[c], v < thr[c], v <= thr[c])
        cur[a] = np.where(go, forest.left[c], forest.right[c])
        hit[a] = cur[a] == probes.node[a]
        active[a] = ~forest.is_leaf[cur[a]] & ~hit[a]
    return hit


def main():
    jobs = [("CICIDS2017", "deployed_rf", os.path.join(ARTIFACTS, "ids_rf.onnx"))]
    for ds, d in (("CICIDS2017", "exp5_models"), ("UNSW-NB15", os.path.join("exp10_models", "unsw_nb15")),
                  ("ToN-IoT", os.path.join("exp10_models", "ton_iot"))):
        jobs += [(ds, k, os.path.join(RESULTS, d, f"{k}.onnx")) for k in MODEL_KEYS]
    tmp = tempfile.mkdtemp()
    report = {"models": {}}
    for ds, key, onnx_path in jobs:
        npz = os.path.join(tmp, f"{ds}_{key}.npz")
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_numpy.py"), "--model", onnx_path,
                        "--out", npz], check=True, capture_output=True)
        forest = load_forest_arrays(onnx_path, np.float64)
        probes = build_probes(onnx_path)
        ort, eng = create_inference_engine("onnx", onnx_path), create_inference_engine("numpy", npz)
        flips, delta = 0, 0.0
        for i in range(0, len(probes), 50000):
            a, b = ort.predict_batch(probes.rows[i:i + 50000]), eng.predict_batch(probes.rows[i:i + 50000])
            flips += int((a.predictions != b.predictions).sum())
            delta = max(delta, float(np.abs(a.confidences - b.confidences).max()))
        hit = reached(forest, probes)
        node_hit = np.zeros(len(forest.left), dtype=bool)
        np.logical_or.at(node_hit, probes.node, hit)
        internal = np.flatnonzero(~forest.is_leaf)
        pair = lambda n: (int(forest.feature[n]), np.float32(forest.threshold[n]).view(np.int32).item())
        covered = {pair(n) for n in internal if node_hit[n]}
        missed = [n for n in internal if not node_hit[n]]
        rep = {"split_modes": "BRANCH_LT" if forest.strict[internal].all() else "BRANCH_LEQ",
               "post_transform": eng.post_transform, "score_columns": eng.n_targets,
               "structure_mismatches": check_npz_matches_onnx(npz, onnx_path),
               "probes": int(len(probes)), "label_flips": flips, "max_confidence_delta": delta,
               "probes_reaching_node": float(hit.mean()),
               "nodes_no_probe_reaches": len(missed),
               "of_those_pair_exercised_elsewhere": int(sum(pair(n) in covered for n in missed))}
        report["models"][f"{ds}/{key}"] = rep
        print(ds, key, rep, flush=True)
    out = os.path.join(RESULTS, "exp28_shipped_toolkit_all_models.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
