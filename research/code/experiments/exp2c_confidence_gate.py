"""How often does a 2,000-row replay gate pass a float64 traversal once it also checks confidences?

exp2 counts label flips only. The shipped gate (scripts/validate_parity.py) also
fails when any confidence differs from ONNX Runtime by more than 1e-5, and one
tree changing leaf already moves a 200-tree confidence by about 0.005. This
replays the whole test split through ONNX Runtime and both NumPy traversals and
reports, per traversal, the rows that would trip either check and the
probability that a 2,000-row replay sees none of them.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.forest import load_forest, predict_lockstep


def onnx_predict(session, x, batch=20000):
    name = session.get_inputs()[0].name
    labels, probs = [], []
    for i in range(0, len(x), batch):
        lab, prob = session.run(None, {name: x[i:i + batch]})
        labels.append(np.asarray(lab).reshape(-1))
        probs.append(np.asarray(prob, dtype=np.float64).reshape(len(labels[-1]), -1))
    return np.concatenate(labels).astype(np.int64), np.concatenate(probs)


def numpy_predict(forest, x, batch=2000):
    labels, scores = zip(*(predict_lockstep(forest, x[i:i + batch]) for i in range(0, len(x), batch)))
    return np.concatenate(labels), np.concatenate(scores).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test_data.parquet directory")
    ap.add_argument("--tolerance", type=float, default=1e-5, help="the gate's --max-confidence-delta")
    ap.add_argument("--replay-rows", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp2c_confidence_gate.json"))
    args = ap.parse_args()

    import onnxruntime as ort

    x, _, _ = load_test_split(args.data)
    npz = os.path.join(ARTIFACTS, "ids_rf_numpy.npz")
    session = ort.InferenceSession(os.path.join(ARTIFACTS, "ids_rf.onnx"), providers=["CPUExecutionProvider"])
    ref_labels, ref_probs = onnx_predict(session, x)
    ref_conf = ref_probs[np.arange(len(x)), ref_labels]

    report = {"rows": int(len(x)), "tolerance": args.tolerance, "replay_rows": args.replay_rows}
    for dtype in ("float32", "float64"):
        labels, scores = numpy_predict(load_forest(npz, dtype), x)
        # the gate compares each backend's confidence in its own predicted class
        delta = np.abs(scores[np.arange(len(x)), labels] - ref_conf)
        flips = labels != ref_labels
        over = delta > args.tolerance
        trips = flips | over
        rate = float(trips.mean())
        report[dtype] = {
            "label_flips": int(flips.sum()),
            "rows_over_tolerance": int(over.sum()),
            "rows_tripping_gate": int(trips.sum()),
            "max_confidence_delta": float(delta.max()),
            "prob_replay_passes_labels_only": float((1.0 - flips.mean()) ** args.replay_rows),
            "prob_replay_passes_labels_and_confidence": float((1.0 - rate) ** args.replay_rows),
        }
        print(dtype, report[dtype])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
