"""Converter parity (training framework vs ONNX Runtime) with the gate's confidence check.

exp16 showed that once a replay gate also compares probabilities at 1e-5, it
catches a float64 runtime traversal on 12 of 13 forests. Converter parity is
the other comparison of exp5/exp10: the framework's own prediction against
ONNX Runtime on the exported graph. Here the framework's front end is the
scikit-learn StandardScaler in float64 while the graph scales in float32, so
its probabilities may differ from the export's on most rows even when nothing
is wrong in the trees. This measures, for the twelve trained forests:

  * framework (float64 scaler) vs ONNX Runtime, as Tables 4/5 replay_gate them:
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
import time

import numpy as np

from edgeids.data import RESULTS
from edgeids.datasets import load_cicids, load_ton_iot, load_unsw
from edgeids.gates import replay_gate
from edgeids.models import MODEL_KEYS, build_models
from edgeids.onnx_forest import load_onnx_forest
from edgeids.runtime import ort_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids-train", required=True)
    ap.add_argument("--cicids-test", required=True)
    ap.add_argument("--unsw", default="")
    ap.add_argument("--ton-iot", default="")
    ap.add_argument("--replay-rows", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp17_converter_confidence.json"))
    args = ap.parse_args()

    import lightgbm
    import onnxruntime
    import sklearn
    import xgboost
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    started = time.perf_counter()
    report = {"versions": {"numpy": np.__version__, "onnxruntime": onnxruntime.__version__, "sklearn": sklearn.__version__,
                           "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__},
              "replay_rows": args.replay_rows, "datasets": {}}
    jobs = [("CICIDS2017", lambda: load_cicids(args.cicids_train, args.cicids_test), os.path.join(RESULTS, "exp5_models"))]
    if args.unsw:
        jobs.append(("UNSW-NB15", lambda: load_unsw(args.unsw), os.path.join(RESULTS, "exp10_models", "unsw_nb15")))
    if args.ton_iot:
        jobs.append(("ToN-IoT", lambda: load_ton_iot(args.ton_iot), os.path.join(RESULTS, "exp10_models", "ton_iot")))

    for name, load, model_dir in jobs:
        x_train, y_train, x_test = load()
        report["datasets"][name] = {"train_rows": int(len(x_train)), "test_rows": int(len(x_test)), "models": {}}
        for key in MODEL_KEYS:
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
                "float64_scaler": replay_gate(pipe.predict_proba(x_test.astype(np.float64)), ref, args.replay_rows),
                "graph_float32_scaler": replay_gate(pipe.named_steps["model"].predict_proba(z32), ref, args.replay_rows),
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
