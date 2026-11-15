"""Leafparity on the twelve converter-parity pairs (training framework vs its ONNX export).

Leafparity (github.com/Leafparity/leafparity) checks whether an exported ONNX
tree ensemble is equivalent to the model it came from, by walking both over the
whole input space with each runtime's arithmetic, and returns a proof or witness
inputs. It answers the converter question of exp17/exp19, not the runtime
question (two serving runtimes of one graph), so this compares it with replay
and the boundary probes on the converter pairs only.

The twelve pipelines (StandardScaler + model) are retrained exactly as exp17
retrains them (seed 0, same library versions) and analysed against the ONNX
files exp5/exp10 saved. Inputs are what the pipeline serves: float32 features
without NaN (the feature builder maps non-finite values to 0), so the analysis
runs with input_dtype=float32 and allow_missing=False. A test-split sample is
passed as background so witnesses look like real flows.

Run it from a venv that has leafparity and the training libraries at the versions
of exp5 (scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, ONNX Runtime 1.30):

    python research/code/experiments/exp21_leafparity.py --cicids-train $TRAIN --cicids-test $DATA \\
        --unsw $UNSW --ton-iot $TON/train_test_network.csv
"""

import argparse
import json
import os
import time

import numpy as np

from edgeids.data import RESULTS
from edgeids.datasets import load_cicids, load_ton_iot, load_unsw
from edgeids.models import MODEL_KEYS, build_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids-train", required=True)
    ap.add_argument("--cicids-test", required=True)
    ap.add_argument("--unsw", default="")
    ap.add_argument("--ton-iot", default="")
    ap.add_argument("--models", default=",".join(MODEL_KEYS))
    ap.add_argument("--background-rows", type=int, default=2000)
    ap.add_argument("--worst-case-seconds", type=float, default=30.0)
    ap.add_argument("--node-limit", type=int, default=200000, help="leafparity's default")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp21_leafparity.json"))
    args = ap.parse_args()

    import leafparity
    import lightgbm
    import onnxruntime
    import sklearn
    import xgboost
    from leafparity import analyze
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    report = {"versions": {"leafparity": getattr(leafparity, "__version__", "0.1.2"), "numpy": np.__version__,
                           "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                           "xgboost": xgboost.__version__, "onnxruntime": onnxruntime.__version__},
              "settings": {"input_dtype": "float32", "allow_missing": False, "background_rows": args.background_rows,
                           "worst_case_seconds": args.worst_case_seconds, "node_limit": args.node_limit},
              "datasets": {}}
    jobs = [("CICIDS2017", lambda: load_cicids(args.cicids_train, args.cicids_test), os.path.join(RESULTS, "exp5_models"))]
    if args.unsw:
        jobs.append(("UNSW-NB15", lambda: load_unsw(args.unsw), os.path.join(RESULTS, "exp10_models", "unsw_nb15")))
    if args.ton_iot:
        jobs.append(("ToN-IoT", lambda: load_ton_iot(args.ton_iot), os.path.join(RESULTS, "exp10_models", "ton_iot")))

    for name, load, model_dir in jobs:
        x_train, y_train, x_test = load()
        rng = np.random.default_rng(0)
        background = x_test[rng.choice(len(x_test), size=min(args.background_rows, len(x_test)), replace=False)]
        report["datasets"][name] = {}
        for key in args.models.split(","):
            pipe = Pipeline([("scaler", StandardScaler()), ("model", build_models(0)[key][1])])
            pipe.fit(x_train.astype(np.float64), y_train)
            path = os.path.join(model_dir, f"{key}.onnx")
            t0 = time.perf_counter()
            try:
                r = analyze(pipe, path, input_dtype="float32", allow_missing=False,
                            background=background.astype(np.float64), worst_case_seconds=args.worst_case_seconds,
                            node_limit=args.node_limit)
                entry = {"verdict": r.verdict, "summary": r.to_summary_dict()}
            except Exception as e:  # recorded, not fatal: the comparison reports what the tool could not do
                entry = {"error": f"{type(e).__name__}: {e}"}
            entry["seconds"] = round(time.perf_counter() - t0, 1)
            report["datasets"][name][key] = entry
            v = entry.get("verdict", {})
            print(name, key, v.get("status", entry.get("error")), "problems", v.get("distinct_problems"),
                  "class_change", v.get("label_flip_found"), "s", entry["seconds"], flush=True)
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2, default=str)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
