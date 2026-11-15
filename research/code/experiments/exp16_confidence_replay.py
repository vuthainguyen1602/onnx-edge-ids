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
import time

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.datasets import ton_iot_test, unsw_test
from edgeids.gates import replay_gate
from edgeids.models import MODEL_KEYS
from edgeids.onnx_forest import load_onnx_forest
from edgeids.runtime import numpy_probs, ort_probs


def run_model(path, x, replay_rows):
    forest = load_onnx_forest(path)
    ref = ort_probs(path, x)
    rep = {"post_transform": forest.post_transform, "n_targets": int(forest.n_targets), "rows": int(len(x))}
    for dtype in (np.float32, np.float64):
        rep[np.dtype(dtype).name] = replay_gate(numpy_probs(forest, x, dtype), ref, replay_rows)
    return rep


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
                 [(m, os.path.join(RESULTS, "exp5_models", f"{m}.onnx")) for m in MODEL_KEYS]))
    if args.unsw:
        x, cols = unsw_test(args.unsw)
        assert len(cols) == 39 and len(x) == 82332, (len(cols), len(x))
        jobs.append(("UNSW-NB15", x, [(m, os.path.join(RESULTS, "exp10_models", "unsw_nb15", f"{m}.onnx")) for m in MODEL_KEYS]))
    if args.ton_iot:
        x, cols = ton_iot_test(args.ton_iot)
        assert len(cols) == 14 and len(x) == 63313, (len(cols), len(x))
        jobs.append(("ToN-IoT", x, [(m, os.path.join(RESULTS, "exp10_models", "ton_iot", f"{m}.onnx")) for m in MODEL_KEYS]))

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
