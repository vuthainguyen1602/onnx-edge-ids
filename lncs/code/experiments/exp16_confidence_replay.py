"""Replay with the gate's confidence check, on every forest of exp5 and exp10.

exp5 and exp10 count label flips of a float64 traversal against ONNX Runtime on
the replayed test split, and turn that into the chance a 2,000-row replay gate
passes it. The shipped gate (scripts/validate_parity.py) also fails when any
probability differs by more than a tolerance (1e-5). exp11b showed that on the
deployed forest this confidence check alone catches every injected fault by
replay. This asks the same of the twelve trained forests and of the deployed one:

  * float32 traversal: the largest probability deviation from ONNX Runtime on
    replay, i.e. whether a 1e-5 tolerance is usable at all for that model
    (a correct runtime must stay below it);
  * float64 traversal: label flips, rows beyond the tolerance, and the chance
    that a 2,000-row replay trips neither check, at 1e-5 and at looser
    tolerances a deployment might need.

Deviation is the largest absolute difference over the probability row, which
for a binary model equals the difference in the predicted class's confidence
whenever the labels agree. Models are read from their saved ONNX files; the
test splits are rebuilt exactly as exp5/exp10 build them. The training
frameworks are not involved (their pipelines are not saved), so this covers
runtime parity, not converter parity.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.onnx_forest import load_onnx_forest, raw_scores
from exp10_cross_dataset import binary_labels, choose_features, matrix, read_table

MODELS = ("sklearn_rf", "sklearn_hgb", "lightgbm", "xgboost_hist")
TOLERANCES = (1e-5, 1e-4, 1e-3, 1e-2)


def probabilities(forest, scores):
    """The probability output ONNX Runtime returns for these scores (binary models)."""
    s = scores
    if forest.n_targets == 1:
        p1 = 1.0 / (1.0 + np.exp(-s[:, 0])) if forest.post_transform == "LOGISTIC" else s[:, 0]
        return np.stack([1.0 - p1, p1], axis=1)
    if forest.post_transform == "SOFTMAX":
        z = np.exp(s - s.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)
    if forest.post_transform == "LOGISTIC":
        return 1.0 / (1.0 + np.exp(-s))
    return s


def numpy_probs(forest, x, dtype, batch=50000):
    return np.concatenate([probabilities(forest, raw_scores(forest, x[i:i + batch], dtype))
                           for i in range(0, len(x), batch)])


def ort_probs(path, x, batch=50000):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return np.concatenate([np.asarray(s.run(None, {name: x[i:i + batch]})[1], dtype=np.float64)
                           for i in range(0, len(x), batch)])


def label(p):
    # class 1 iff P(class 1) > 0.5, as exp5 derives ONNX Runtime's label
    return (p[:, 1] > 0.5).astype(np.int64)


def compare(p, ref, replay_rows):
    delta = np.abs(p - ref).max(axis=1)
    flips = label(p) != label(ref)
    out = {"label_flips": int(flips.sum()), "max_delta": float(delta.max()),
           "prob_replay_passes_labels_only": float((1.0 - flips.mean()) ** replay_rows), "tolerances": {}}
    for tol in TOLERANCES:
        trips = flips | (delta > tol)
        out["tolerances"][f"{tol:g}"] = {"rows_tripping_gate": int(trips.sum()),
                                         "prob_replay_passes": float((1.0 - trips.mean()) ** replay_rows)}
    return out


def run_model(path, x, replay_rows):
    forest = load_onnx_forest(path)
    ref = ort_probs(path, x)
    rep = {"post_transform": forest.post_transform, "n_targets": int(forest.n_targets), "rows": int(len(x))}
    for dtype in (np.float32, np.float64):
        rep[np.dtype(dtype).name] = compare(numpy_probs(forest, x, dtype), ref, replay_rows)
    return rep


def unsw(path_dir):
    train = read_table(os.path.join(path_dir, "UNSW_NB15_training-set.csv"))
    test = read_table(os.path.join(path_dir, "UNSW_NB15_testing-set.csv"))
    cols, _ = choose_features(train, "label", [], [])
    return matrix(test, cols), cols


def ton_iot(csv):
    from sklearn.model_selection import train_test_split
    df = read_table(csv)
    cols, _ = choose_features(df, "label", [], [])
    x, y = matrix(df, cols), binary_labels(df["label"], ["benign", "normal", "0"])
    _, x_test, _, _ = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)
    return x_test, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids", required=True, help="CICIDS2017 test_data.parquet directory")
    ap.add_argument("--unsw", default="", help="directory with UNSW_NB15_{training,testing}-set.csv")
    ap.add_argument("--ton-iot", default="", help="ToN-IoT train_test_network.csv")
    ap.add_argument("--replay-rows", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp16_confidence_replay.json"))
    args = ap.parse_args()

    import onnxruntime
    started = time.perf_counter()
    report = {"numpy": np.__version__, "onnxruntime": onnxruntime.__version__, "replay_rows": args.replay_rows,
              "datasets": {}}
    jobs = []
    x, _, _ = load_test_split(args.cicids)
    jobs.append(("CICIDS2017", x, [("deployed_rf", os.path.join(ARTIFACTS, "ids_rf.onnx"))] +
                 [(m, os.path.join(RESULTS, "exp5_models", f"{m}.onnx")) for m in MODELS]))
    if args.unsw:
        x, cols = unsw(args.unsw)
        assert len(cols) == 39 and len(x) == 82332, (len(cols), len(x))
        jobs.append(("UNSW-NB15", x, [(m, os.path.join(RESULTS, "exp10_models", "unsw_nb15", f"{m}.onnx")) for m in MODELS]))
    if args.ton_iot:
        x, cols = ton_iot(args.ton_iot)
        assert len(cols) == 14 and len(x) == 63313, (len(cols), len(x))
        jobs.append(("ToN-IoT", x, [(m, os.path.join(RESULTS, "exp10_models", "ton_iot", f"{m}.onnx")) for m in MODELS]))

    for name, x, models in jobs:
        report["datasets"][name] = {"rows": int(len(x)), "models": {}}
        for key, path in models:
            t0 = time.perf_counter()
            rep = run_model(path, x, args.replay_rows)
            rep["seconds"] = round(time.perf_counter() - t0, 1)
            report["datasets"][name]["models"][key] = rep
            print(name, key, json.dumps({d: {k: rep[d][k] for k in ("label_flips", "max_delta")} |
                                         {"pass@1e-5": rep[d]["tolerances"]["1e-05"]["prob_replay_passes"]}
                                         for d in ("float32", "float64")}), flush=True)
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)

    report["seconds"] = round(time.perf_counter() - started, 1)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
