"""The release gate against the ONNX Runtime releases whose decision rule changed (1.26.0, 1.27.0).

exp5b found that ONNX Runtime 1.26.0 and 1.27.0 label single-column binary tree graphs as attack whenever
the score is positive instead of above one half, relabelling 85% of the replayed flows of the scikit-learn
random forest. That sweep ran replay only. The gate compares the NumPy engine's float32 traversal with
whatever ONNX Runtime is installed, so the question here is whether the data-free probes see the change too.

For all thirteen models, against the installed ONNX Runtime: the float32 traversal on the data-free
three-probe set and crossing pair (label flips, probes over 1e-5, verdict) and on replay of the whole test
split (label flips, chance a 2,000-row replay passes). Run once per release, each in its own venv:

    PYTHONPATH=research/code <venv>/bin/python research/code/experiments/exp41_ort_release_gate.py \
        --cicids $DATA --unsw $UNSW --ton-iot $TON/train_test_network.csv
"""

import argparse
import importlib.util
import json
import os

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.datasets import ton_iot_test, unsw_test
from edgeids.gates import probe_gate, replay_gate
from edgeids.models import MODEL_KEYS
from edgeids.onnx_forest import load_onnx_forest
from edgeids.probes import datafree_probes
from edgeids.runtime import numpy_probs, ort_probs

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp19", os.path.join(HERE, "exp19_datafree_probes.py"))
exp19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp19)


def main():
    import onnxruntime
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids", required=True)
    ap.add_argument("--unsw", default="")
    ap.add_argument("--ton-iot", default="")
    ap.add_argument("--replay-rows", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or os.path.join(RESULTS, f"exp41_ort_release_gate_{onnxruntime.__version__}.json")
    report = {"onnxruntime": onnxruntime.__version__, "numpy": np.__version__, "datasets": {}}
    jobs = []
    x, _, _ = load_test_split(args.cicids)
    jobs.append(("CICIDS2017", x, [("deployed_rf", os.path.join(ARTIFACTS, "ids_rf.onnx"))] +
                 [(m, os.path.join(RESULTS, "exp5_models", f"{m}.onnx")) for m in MODEL_KEYS]))
    if args.unsw:
        jobs.append(("UNSW-NB15", unsw_test(args.unsw)[0],
                     [(m, os.path.join(RESULTS, "exp10_models", "unsw_nb15", f"{m}.onnx")) for m in MODEL_KEYS]))
    if args.ton_iot:
        jobs.append(("ToN-IoT", ton_iot_test(args.ton_iot)[0],
                     [(m, os.path.join(RESULTS, "exp10_models", "ton_iot", f"{m}.onnx")) for m in MODEL_KEYS]))
    for name, x, models in jobs:
        x = np.ascontiguousarray(x, dtype=np.float32)
        report["datasets"][name] = {}
        for key, path in models:
            forest = load_onnx_forest(path)
            rep = {"n_targets": int(forest.n_targets), "post_transform": forest.post_transform}
            for probe_set, build in (("three", datafree_probes), ("crossing", exp19.crossing_probes)):
                _, probes = build(forest)
                rep[probe_set] = probe_gate(numpy_probs(forest, probes, np.float32), ort_probs(path, probes))
            g = replay_gate(numpy_probs(forest, x, np.float32), ort_probs(path, x), args.replay_rows)
            rep["replay"] = {"label_flips": g["label_flips"], "max_delta": g["max_delta"],
                             "pass_labels": g["prob_replay_passes_labels_only"],
                             "pass_conf": g["tolerances"]["1e-05"]["prob_replay_passes"]}
            report["datasets"][name][key] = rep
            print(onnxruntime.__version__, name, key, rep["n_targets"],
                  {k: (rep[k]["gate_fails"], rep[k]["label_flips"]) for k in ("three", "crossing")},
                  "replay flips", rep["replay"]["label_flips"], flush=True)
            with open(out, "w") as f:
                json.dump(report, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
