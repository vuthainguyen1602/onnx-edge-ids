"""What survives distribution shift and what does not.

The model, gate and threshold are frozen as deployed (trained on CICIDS2017).
They are run unchanged on the CICIDS2017 test split (in-distribution) and on the
CSE-CIC-IDS2018 test split (shifted: different network, year and attack tools,
same 29 features). Detection quality is expected to move; the serving
guarantees (runtime parity, exact early exit) should not.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from edgeids.data import ARTIFACTS, RESULTS, Gate, load_test_split
from edgeids.forest import load_forest, margin_bounds, predict_early_exit, predict_lockstep
from exp2_boundary_parity import numpy_labels, onnx_labels, path_rows

BATCH = 2000


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def early_exit(forest, bounds, schedule, x):
    full = np.concatenate([predict_lockstep(forest, x[i:i + BATCH])[0] for i in range(0, len(x), BATCH)])
    out = [predict_early_exit(forest, x[i:i + BATCH], schedule, bounds=bounds) for i in range(0, len(x), BATCH)]
    labels = np.concatenate([o[0] for o in out])
    used = np.concatenate([o[1] for o in out])
    return {
        "rows": int(len(x)),
        "labels_identical_to_full_forest": int((labels == full).sum()),
        "mean_trees_used": round(float(used.mean()), 2),
        "rows_decided_at_first_check": round(float((used == schedule[0]).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dist", required=True, help="CICIDS2017 test_data.parquet")
    ap.add_argument("--shifted", required=True, help="CSE-CIC-IDS2018 test_data.parquet")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp4_drift.json"))
    args = ap.parse_args()

    import onnxruntime as ort

    npz = os.path.join(ARTIFACTS, "ids_rf_numpy.npz")
    f32, f64 = load_forest(npz, "float32"), load_forest(npz, "float64")
    session = ort.InferenceSession(os.path.join(ARTIFACTS, "ids_rf.onnx"), providers=["CPUExecutionProvider"])
    gate = Gate()
    bounds = margin_bounds(f32)
    first = f32.n_trees // 2 + 4
    schedule = [first] + [8] * ((f32.n_trees - first + 7) // 8)
    internal = np.flatnonzero(~f32.is_leaf)
    feat = f32.feature[internal]
    on_threshold = (f64.threshold[internal] / f64.scale[feat] + f64.offset[feat]).astype(np.float32)

    report = {}
    for name, path in (("cicids2017_in_distribution", args.in_dist), ("cse_cic_ids2018_shifted", args.shifted)):
        x, y, attack = load_test_split(path)
        lab_onnx = onnx_labels(session, x)
        lab32 = numpy_labels(f32, x)
        lab64 = numpy_labels(f64, x)
        forwarded = gate.score(x) >= gate.threshold

        per_attack = {}
        for a in np.unique(attack[y == 1]):
            m = attack == a
            per_attack[str(a)] = {"rows": int(m.sum()),
                                  "classifier_recall": round(float(lab_onnx[m].mean()), 4),
                                  "gate_recall": round(float(forwarded[m].mean()), 4)}

        # Same split table, base rows drawn from this dataset: does the base distribution matter?
        rng = np.random.default_rng(0)
        probes = path_rows(f64, internal, x[rng.choice(len(x), size=len(internal), replace=True)])
        probes[np.arange(len(internal)), feat] = on_threshold
        probe_onnx = onnx_labels(session, probes)

        report[name] = {
            "rows": int(len(x)),
            "attack_rate": round(float(y.mean()), 4),
            "detector": prf(y, lab_onnx),
            "gate": {
                "forward_rate": round(float(forwarded.mean()), 4),
                "attack_recall": round(float(forwarded[y == 1].mean()), 4),
                "benign_forward_rate": round(float(forwarded[y == 0].mean()), 4),
            },
            "two_stage_end_to_end": prf(y, (forwarded & (lab_onnx == 1)).astype(np.int64)),
            "replay_parity_vs_onnx": {
                "numpy_float32_label_disagreements": int((lab32 != lab_onnx).sum()),
                "numpy_float64_label_disagreements": int((lab64 != lab_onnx).sum()),
            },
            "on_threshold_probes_vs_onnx": {
                "probes": int(len(probes)),
                "numpy_float32_label_disagreements": int((numpy_labels(f32, probes) != probe_onnx).sum()),
                "numpy_float64_label_disagreements": int((numpy_labels(f64, probes) != probe_onnx).sum()),
            },
            "exact_early_exit_all_traffic": early_exit(f32, bounds, schedule, x),
            "exact_early_exit_gate_forwarded": early_exit(f32, bounds, schedule, x[forwarded]),
            "per_attack": per_attack,
        }
        print(name)
        for k, v in report[name].items():
            if k != "per_attack":
                print("  ", k, v)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
