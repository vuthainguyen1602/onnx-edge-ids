"""Converter parity (training framework vs ONNX Runtime) with the gate's confidence check.

exp16 showed that once a replay gate also compares probabilities at 1e-5, it
catches a float64 runtime traversal on 12 of 13 forests. Converter parity is
the other comparison of exp5/exp10: the framework's own prediction against
ONNX Runtime on the exported graph. Here the framework's front end is the
scikit-learn StandardScaler in float64 while the graph scales in float32, so
its probabilities may differ from the export's on most rows even when nothing
is wrong in the trees. This measures, for the twelve trained forests:

  * framework (float64 scaler) vs ONNX Runtime, as Tables 4/5 compare them:
    label flips, largest probability deviation, and the chance a 2,000-row
    replay passes at several tolerances;
  * the same with the framework fed the graph's float32 Scaler output, which
    removes the front-end difference and leaves the trees.

The models are retrained exactly as exp5 (CICIDS2017) and exp10 (UNSW-NB15,
ToN-IoT) train them, with seed 0 and the same library versions, and scored
against the ONNX files those runs saved. The label-flip counts must reproduce
the "training framework / replay" column of Tables 4 and 5.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgeids.data import RESULTS, load_test_split
from edgeids.onnx_forest import load_onnx_forest
from exp5_generalization import build_models
from exp10_cross_dataset import binary_labels, choose_features, matrix, read_table
from exp16_confidence_replay import MODELS, compare, ort_probs


def cicids(train_dir, test_dir, train_rows=600000, seed=0):
    x_test, _, _ = load_test_split(test_dir)
    x_train, y_train, _ = load_test_split(train_dir)
    rng = np.random.default_rng(seed)
    if len(x_train) > train_rows:
        idx = np.sort(rng.choice(len(x_train), size=train_rows, replace=False))
        x_train, y_train = x_train[idx], y_train[idx]
    return x_train, y_train, x_test


def unsw(path_dir):
    train = read_table(os.path.join(path_dir, "UNSW_NB15_training-set.csv"))
    test = read_table(os.path.join(path_dir, "UNSW_NB15_testing-set.csv"))
    cols, _ = choose_features(train, "label", [], [])
    benign = ["benign", "normal", "0"]
    return matrix(train, cols), binary_labels(train["label"], benign), matrix(test, cols)


def ton_iot(csv):
    from sklearn.model_selection import train_test_split
    df = read_table(csv)
    cols, _ = choose_features(df, "label", [], [])
    x, y = matrix(df, cols), binary_labels(df["label"], ["benign", "normal", "0"])
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)
    return x_train, y_train, x_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids-train", required=True)
    ap.add_argument("--cicids-test", required=True)
    ap.add_argument("--unsw", default="")
    ap.add_argument("--ton-iot", default="")
    ap.add_argument("--replay-rows", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp17_converter_confidence.json"))
    args = ap.parse_args()

    import lightgbm, onnxruntime, sklearn, xgboost
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    started = time.perf_counter()
    report = {"versions": {"numpy": np.__version__, "onnxruntime": onnxruntime.__version__, "sklearn": sklearn.__version__,
                           "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__},
              "replay_rows": args.replay_rows, "datasets": {}}
    jobs = [("CICIDS2017", lambda: cicids(args.cicids_train, args.cicids_test), os.path.join(RESULTS, "exp5_models"))]
    if args.unsw:
        jobs.append(("UNSW-NB15", lambda: unsw(args.unsw), os.path.join(RESULTS, "exp10_models", "unsw_nb15")))
    if args.ton_iot:
        jobs.append(("ToN-IoT", lambda: ton_iot(args.ton_iot), os.path.join(RESULTS, "exp10_models", "ton_iot")))

    for name, load, model_dir in jobs:
        x_train, y_train, x_test = load()
        report["datasets"][name] = {"train_rows": int(len(x_train)), "test_rows": int(len(x_test)), "models": {}}
        for key in MODELS:
            t0 = time.perf_counter()
            _, est = build_models(0)[key]
            pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
            pipe.fit(x_train.astype(np.float64), y_train)
            path = os.path.join(model_dir, f"{key}.onnx")
            ref = ort_probs(path, x_test)
            forest = load_onnx_forest(path)
            z32 = ((x_test - forest.offset.astype(np.float32)) * forest.scale.astype(np.float32)).astype(np.float64)
            rep = {
                "framework_predict_vs_ort_label_flips": int((np.asarray(pipe.predict(x_test.astype(np.float64))).astype(np.int64)
                                                             != (ref[:, 1] > 0.5)).sum()),
                "float64_scaler": compare(pipe.predict_proba(x_test.astype(np.float64)), ref, args.replay_rows),
                "graph_float32_scaler": compare(pipe.named_steps["model"].predict_proba(z32), ref, args.replay_rows),
                "seconds": round(time.perf_counter() - t0, 1),
            }
            report["datasets"][name]["models"][key] = rep
            print(name, key, rep["framework_predict_vs_ort_label_flips"],
                  json.dumps({k: {"flips": rep[k]["label_flips"], "max_delta": rep[k]["max_delta"],
                                  "pass@1e-5": rep[k]["tolerances"]["1e-05"]["prob_replay_passes"],
                                  "pass@1e-2": rep[k]["tolerances"]["0.01"]["prob_replay_passes"]}
                              for k in ("float64_scaler", "graph_float32_scaler")}), flush=True)
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)

    report["seconds"] = round(time.perf_counter() - started, 1)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
