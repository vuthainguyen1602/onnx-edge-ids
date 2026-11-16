"""Which gate catches an exporter that mis-reads the ONNX graph?

The shipped gate compares ONNX Runtime with the NumPy engine serving the
exported ``.npz``. Up to v0.4 it built its boundary probes from that same
``.npz``, so an exporter bug that moves a threshold moves the probe with it.
This injects exporter mutants into the ``.npz`` of two CICIDS2017 models (the
deployed forest, BRANCH_LEQ with two score columns, and XGBoost, BRANCH_LT with
one logistic column and a base value) and runs every gate on each:

  * structure: the gate's float32 comparison of the ``.npz`` against the graph;
  * replay: probability that a 2,000-row replay of the test split passes,
    comparing labels only or also confidences at 1e-5;
  * probes built from the ``.npz`` (the old gate) and from the ONNX graph (the
    new gate), and the two exact-crossing probes per node built from the graph
    (the gate's --crossing-probes): label flips and probes over 1e-5 against
    ONNX Runtime.

Mutants (one exporter defect each):
  thr_all_up        every threshold one float32 step up (a rounding-direction bug)
  thr_one_up        a single threshold one float32 step up
  feature_swap      two split-feature ids exchanged (a column-order bug)
  offset_one_up     one scaler offset one float32 step up
  scale_fp16        scaler scale rounded through float16 (a precision bug)
  leaf_tree0        tree 0's leaf weights negated / class columns swapped
  mode_dropped      BRANCH_LT written as BRANCH_LEQ (XGBoost only)
  base_dropped      base value dropped (XGBoost only)

    python exp25_exporter_mutations.py --data $DATA
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, load_test_split

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from onnx_edge_ids import create_inference_engine  # noqa: E402
from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, build_probes, crossing_values, load_forest_arrays  # noqa: E402
from onnx_edge_ids.onnx_artifact import check_npz_matches_onnx  # noqa: E402

TOL = 1e-5
REPLAY_ROWS = 2000
MODELS = {"deployed_rf": os.path.join(ARTIFACTS, "ids_rf.onnx"),
          "xgboost_hist": os.path.join(RESULTS, "exp5_models", "xgboost_hist.onnx")}


def up(a):
    return np.nextafter(a.astype(np.float32), np.float32(np.inf)).astype(np.float64)


def mutants(d):
    internal = np.flatnonzero(~d["is_leaf"])
    out = {}
    m = dict(d); m["threshold"] = np.where(d["is_leaf"], d["threshold"], up(d["threshold"])); out["thr_all_up"] = m
    m = dict(d); t = d["threshold"].copy(); t[internal[0]] = up(t[internal[0]:internal[0] + 1])[0]
    m["threshold"] = t; out["thr_one_up"] = m
    used = np.bincount(d["feature"][internal]).argsort()[::-1]
    fa, fb = int(used[0]), int(used[1])
    f = d["feature"].copy(); f[d["feature"] == fa], f[d["feature"] == fb] = fb, fa
    m = dict(d); m["feature"] = f; out["feature_swap"] = m
    m = dict(d); o = d["offset"].copy(); o[fa] = up(o[fa:fa + 1])[0]; m["offset"] = o; out["offset_one_up"] = m
    m = dict(d); m["scale"] = d["scale"].astype(np.float16).astype(np.float64); out["scale_fp16"] = m
    m = dict(d); ls = d["leaf_scores"].copy(); t0 = slice(int(d["tree_offsets"][0]), int(d["tree_offsets"][1]))
    ls[t0] = ls[t0, ::-1] if ls.shape[1] == 2 else -ls[t0]
    m["leaf_scores"] = ls; out["leaf_tree0"] = m
    if d["strict"].any():
        m = dict(d); m["strict"] = np.zeros_like(d["strict"]); out["mode_dropped"] = m
    if np.any(d["base_values"] != 0):
        m = dict(d); m["base_values"] = np.zeros_like(d["base_values"]); out["base_dropped"] = m
    return out


def one_step_visible(onnx_path, node):
    """Can any float32 input tell threshold t from t + one float32 step at ``node``?

    Only if some raw float32 value scales to exactly the side the shift moves: for
    BRANCH_LEQ a scaled value in (t, t+ulp], for BRANCH_LT one equal to t. The
    first raw value past the float32 crossing is the smallest candidate.
    """
    f = load_forest_arrays(onnx_path, np.float64)
    n = np.array([node])
    g = f.feature[n]
    t = np.float32(f.threshold[node])
    start = (f.threshold[n] / f.scale[g] + f.offset[g]).astype(np.float32)
    _, last_true = crossing_values(f, n, start)
    first_false = np.nextafter(last_true, np.float32(np.inf))
    z = ((first_false - np.float32(f.offset[g][0])) * np.float32(f.scale[g][0])).astype(np.float32)[0]
    return bool(z == t) if f.strict[node] else bool(z <= np.nextafter(t, np.float32(np.inf)))


def outputs(engine, x, batch=50000):
    p, c = [], []
    for i in range(0, len(x), batch):
        r = engine.predict_batch(x[i:i + batch])
        p.append(r.predictions); c.append(r.confidences)
    return np.concatenate(p), np.concatenate(c)


def compare(ref, got):
    flips = ref[0] != got[0]
    over = np.abs(ref[1] - got[1]) > TOL
    return flips, flips | over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CICIDS2017 test_data.parquet directory")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp25_exporter_mutations.json"))
    args = ap.parse_args()
    x, _, _ = load_test_split(args.data)
    x = np.ascontiguousarray(x, dtype=np.float32)
    report = {"replay_rows_scored": int(len(x)), "replay_gate_rows": REPLAY_ROWS, "tolerance": TOL, "models": {}}
    tmp = tempfile.mkdtemp()
    for name, onnx_path in MODELS.items():
        npz = os.path.join(tmp, f"{name}.npz")
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_numpy.py"), "--model", onnx_path,
                        "--out", npz], check=True, capture_output=True)
        base = dict(np.load(npz, allow_pickle=False))
        ort = create_inference_engine("onnx", onnx_path)
        ref_replay = outputs(ort, x)
        graph_probes = build_probes(onnx_path).rows
        ref_graph = outputs(ort, graph_probes)
        cross_probes = build_probes(onnx_path, CROSSING_VARIANTS).rows
        ref_cross = outputs(ort, cross_probes)
        entry = {}
        for mname, arrays in [("none", base)] + list(mutants(base).items()):
            path = os.path.join(tmp, f"{name}_{mname}.npz")
            np.savez(path, **arrays)
            eng = create_inference_engine("numpy", path)
            structure = check_npz_matches_onnx(path, onnx_path)
            f_l, f_c = compare(ref_replay, outputs(eng, x))
            npz_probes = build_probes(path).rows
            n_l, n_c = compare(outputs(ort, npz_probes), outputs(eng, npz_probes))
            g_l, g_c = compare(ref_graph, outputs(eng, graph_probes))
            c_l, c_c = compare(ref_cross, outputs(eng, cross_probes))
            row = {
                "structure_mismatches": structure,
                "replay": {"label_flips": int(f_l.sum()), "rows_tripping_conf": int(f_c.sum()),
                           "prob_passes_labels_only": float((1 - f_l.mean()) ** REPLAY_ROWS),
                           "prob_passes_with_conf": float((1 - f_c.mean()) ** REPLAY_ROWS)},
                "probes_from_npz": {"probes": int(len(npz_probes)), "label_flips": int(n_l.sum()),
                                    "tripping_conf": int(n_c.sum())},
                "probes_from_onnx": {"probes": int(len(graph_probes)), "label_flips": int(g_l.sum()),
                                     "tripping_conf": int(g_c.sum())},
                "crossing_probes_from_onnx": {"probes": int(len(cross_probes)), "label_flips": int(c_l.sum()),
                                              "tripping_conf": int(c_c.sum())},
            }
            if mname == "thr_one_up":
                row["shift_visible_to_some_float32_input"] = one_step_visible(
                    onnx_path, int(np.flatnonzero(~base["is_leaf"])[0]))
            entry[mname] = row
            print(f"{name:12s} {mname:14s} struct={'FAIL' if structure else 'pass'} "
                  f"replay flips={row['replay']['label_flips']} conf={row['replay']['rows_tripping_conf']} "
                  f"P(pass)={row['replay']['prob_passes_labels_only']:.3f}/{row['replay']['prob_passes_with_conf']:.3g} "
                  f"npz-probes {n_l.sum()}/{n_c.sum()} onnx-probes {g_l.sum()}/{g_c.sum()} "
                  f"crossing {c_l.sum()}/{c_c.sum()}", flush=True)
        report["models"][name] = entry
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
