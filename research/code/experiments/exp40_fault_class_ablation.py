"""Fault-class ablation: single-switch vs coupled faults, confined to one node or spread over many.

The crossing pair is complete for single-switch faults (a node's decision stays one threshold switch on
its own feature). exp30 covers three cells of the 2x2 grid that this separates:

                     one node                          many nodes
  single-switch      single-node threshold (exp30)     six arithmetic faults (exp30)
  coupled            -- this experiment --             exporter faults: feature swap, leaf weights (exp30)

This fills the missing cell with two coupled faults confined to one element, injected into the NumPy
artifact served by the shipped engine, twenty of each per model: one leaf's scores swapped (negated for
a single score column), and one internal node's split feature replaced by another feature. Each is run
through the six input generators of exp30 at the same two budgets, with the same detection rule
(a changed label or a probability moved by more than 1e-5; margins at 1e-4 where applicable), on the
deployed forest and the CICIDS2017 XGBoost model. It also records, per fault, whether the gate's
structure check (artifact vs graph) sees it.

    python exp40_fault_class_ablation.py --data $DATA
"""

import argparse
import importlib.util
import json
import os
import tempfile
import time

import numpy as np

from edgeids.data import RESULTS, load_test_split
from edgeids.faults import coupled_single_node_mutants

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp30", os.path.join(HERE, "exp30_testing_baselines.py"))
exp30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp30)


def structure_differs(base, arrays):
    return any(not np.array_equal(np.asarray(base[k]).astype(np.float32) if np.asarray(base[k]).dtype.kind == "f" else base[k],
                                  np.asarray(arrays[k]).astype(np.float32) if np.asarray(arrays[k]).dtype.kind == "f" else arrays[k])
               for k in ("threshold", "feature", "leaf_scores", "offset", "scale"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CICIDS2017 test_data.parquet directory")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp40_fault_class_ablation.json"))
    args = ap.parse_args()
    x, _, _ = load_test_split(args.data)
    x = np.ascontiguousarray(x, dtype=np.float32)
    tmp = tempfile.mkdtemp()
    report = {"tolerance": exp30.TOL, "margin_tolerance": exp30.MARGIN_TOL, "seed": exp30.SEED, "models": {}}
    import subprocess
    import sys
    for model, onnx_path in exp30.MODELS.items():
        forest = exp30.load_forest_arrays(onnx_path, np.float64)
        probes = exp30.build_probes(onnx_path).rows
        ctx = {"x": x, "lo": x.min(axis=0), "hi": x.max(axis=0), "forest": forest,
               "internal": np.flatnonzero(~forest.is_leaf), "probes": probes,
               "crossing": exp30.build_probes(onnx_path, exp30.CROSSING_VARIANTS).rows}
        budgets = [2000, len(probes)]
        inputs = {}
        for b in budgets:
            for method, gen in exp30.GENERATORS.items():
                inputs[(method, b)] = gen(np.random.default_rng(exp30.SEED), b, ctx)
        npz = os.path.join(tmp, f"{model}.npz")
        subprocess.run([sys.executable, os.path.join(exp30.REPO, "scripts", "export_numpy.py"), "--model", onnx_path,
                        "--out", npz], check=True, capture_output=True)
        base = dict(np.load(npz, allow_pickle=False))
        ort_eng = exp30.create_inference_engine("onnx", onnx_path)
        ort_none = exp30.create_inference_engine("onnx", exp30.none_graph(onnx_path, os.path.join(tmp, f"{model}_none.onnx")))

        def ort_ref(xx, batch=50000):
            labels, probs = [], []
            for i in range(0, len(xx), batch):
                r = ort_eng._infer(np.ascontiguousarray(xx[i:i + batch], dtype=np.float32))
                labels.append(r[0]); probs.append(np.asarray(r[1], dtype=np.float64))
            return np.concatenate(labels), np.concatenate(probs)

        def margin_ref(xx, batch=50000):
            return np.concatenate([np.asarray(ort_none._infer(np.ascontiguousarray(xx[i:i + batch], dtype=np.float32))[1],
                                              dtype=np.float64) for i in range(0, len(xx), batch)])
        refs = {(method, b): ort_ref(xin) for (method, b), xin in inputs.items()}
        mrefs = {(method, b): margin_ref(xin) for (method, b), xin in inputs.items()}
        entry = {"probes": len(probes), "budgets": budgets, "faults": {}}
        for name, arrays in coupled_single_node_mutants(base, exp30.SEED).items():
            t0 = time.perf_counter()
            path = os.path.join(tmp, f"{model}_{name}.npz")
            np.savez(path, **arrays)
            eng = exp30.create_inference_engine("numpy", path)
            meng = exp30.create_inference_engine("numpy", exp30.none_artifact(path, path[:-4] + "_none.npz"))
            row = {"kind": "coupled_leaf" if name.startswith("leaf") else "coupled_feature",
                   "structure": bool(structure_differs(base, arrays))}
            for (method, b), xin in inputs.items():
                row[f"{method}@{b}"] = exp30.detected(refs[(method, b)], exp30.probs_of(eng, xin))
                mg = exp30.probs_of(meng, xin)[1]
                row[f"margin:{method}@{b}"] = bool(np.any(np.abs(mrefs[(method, b)][:, -1] - mg[:, -1]) > exp30.MARGIN_TOL))
            row["seconds"] = round(time.perf_counter() - t0, 1)
            entry["faults"][f"{row['kind']}/{name}"] = row
            print(model, name, {k: v for k, v in row.items() if k.endswith(f"@{budgets[1]}") or k == "structure"}, flush=True)
            with open(args.out, "w") as f:
                json.dump(report | {"models": report["models"] | {model: entry}}, f, indent=1)
        report["models"][model] = entry
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
