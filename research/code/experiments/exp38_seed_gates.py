"""The gate measurements of exp16, exp17 and exp19 over training seeds 1-4 (seed 0 is in those files).

exp5/exp10 already give five-seed ranges for probe and replay label flips. The
confidence-checking replay (exp16 runtime, exp17 converter) and the data-free
gate (exp19, three-probe set and crossing pair) were single training runs. This
retrains the twelve trained models per seed exactly as exp5 (CICIDS2017) and
exp10 (UNSW-NB15, ToN-IoT) train them, exports each to ONNX, and records:

  * runtime parity (float64 traversal vs ONNX Runtime) and converter parity
    (framework vs ONNX Runtime) on the replayed test split: label flips and the
    chance that a 2,000-row replay passes, comparing labels only or also
    confidences at 1e-5;
  * float32 traversal on replay: label flips and largest probability deviation;
  * the data-free gate with the three-probe set and with the crossing pair:
    float32, float64 and framework verdicts (label flips, probes over 1e-5).

Seeds change the CICIDS2017 training sample, the ToN-IoT 70/30 split and every
model's own randomness, as in exp5/exp10; UNSW-NB15's training file is smaller
than the 600,000-row sample, so there only the model's randomness changes.

    python exp38_seed_gates.py --cicids-train $TRAIN --cicids-test $DATA --unsw $UNSW \
        --ton-iot $TON/train_test_network.csv --seeds 1,2,3,4
"""

import argparse
import importlib.util
import json
import os
import tempfile
import time

import numpy as np

from edgeids.data import RESULTS
from edgeids.datasets import binary_labels, choose_features, load_cicids, load_unsw, matrix, read_table
from edgeids.gates import replay_gate
from edgeids.models import MODEL_KEYS, build_models, export_pipeline
from edgeids.onnx_forest import load_onnx_forest
from edgeids.runtime import numpy_probs, ort_probs

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp19", os.path.join(HERE, "exp19_datafree_probes.py"))
exp19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp19)


def load_ton_iot_seed(csv, seed):
    """ToN-IoT as exp10 builds it for one seed: stratified 70/30 split with random_state=seed."""
    from sklearn.model_selection import train_test_split
    df = read_table(csv)
    cols, _ = choose_features(df, "label", [], [])
    x, y = matrix(df, cols), binary_labels(df["label"], ["benign", "normal", "0"])
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
    return x_train, y_train, x_test


def summary(gate):
    t = gate["tolerances"]["1e-05"]
    return {"label_flips": gate["label_flips"], "max_delta": gate["max_delta"],
            "pass_labels": gate["prob_replay_passes_labels_only"], "pass_conf": t["prob_replay_passes"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids-train", required=True)
    ap.add_argument("--cicids-test", required=True)
    ap.add_argument("--unsw", default="")
    ap.add_argument("--ton-iot", default="")
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--only", default="", help="comma-separated dataset names to run (default: all given)")
    ap.add_argument("--replay-rows", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp38_seed_gates.json"))
    args = ap.parse_args()

    import onnxruntime
    import sklearn
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    seeds = [int(s) for s in args.seeds.split(",")]
    report = {"versions": {"numpy": np.__version__, "onnxruntime": onnxruntime.__version__, "sklearn": sklearn.__version__},
              "seeds": seeds, "replay_rows": args.replay_rows, "datasets": {}}
    if os.path.exists(args.out):
        report = json.load(open(args.out))
    tmp = tempfile.mkdtemp()
    started = time.perf_counter()
    jobs = [("CICIDS2017", lambda s: load_cicids(args.cicids_train, args.cicids_test, seed=s))]
    if args.unsw:
        jobs.append(("UNSW-NB15", lambda s: load_unsw(args.unsw)))
    if args.ton_iot:
        jobs.append(("ToN-IoT", lambda s: load_ton_iot_seed(args.ton_iot, s)))

    if args.only:
        jobs = [j for j in jobs if j[0] in args.only.split(",")]
    for name, load in jobs:
        ds = report["datasets"].setdefault(name, {})
        for seed in seeds:
            if str(seed) in ds and len(ds[str(seed)]) == len(MODEL_KEYS):
                continue
            x_train, y_train, x_test = load(seed)
            per = ds.setdefault(str(seed), {})
            for key in MODEL_KEYS:
                if key in per:
                    continue
                t0 = time.perf_counter()
                _, est = build_models(seed)[key]
                pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
                pipe.fit(x_train.astype(np.float64), y_train)
                path = os.path.join(tmp, f"{name}_{seed}_{key}.onnx")
                export_pipeline(pipe, x_train.shape[1], path)
                forest = load_onnx_forest(path)
                ref = ort_probs(path, x_test)
                rep = {
                    "replay_runtime_float64": summary(replay_gate(numpy_probs(forest, x_test, np.float64), ref, args.replay_rows)),
                    "replay_float32": summary(replay_gate(numpy_probs(forest, x_test, np.float32), ref, args.replay_rows)),
                    "replay_converter": summary(replay_gate(pipe.predict_proba(x_test.astype(np.float64)), ref, args.replay_rows)),
                }
                for probe_set in ("three", "crossing"):
                    r = exp19.run_model(path, pipe, probe_set)
                    rep[f"datafree_{probe_set}"] = {k: r[k] for k in ("probes", "reaching_their_node", "float32", "float64", "framework")}
                rep["seconds"] = round(time.perf_counter() - t0, 1)
                per[key] = rep
                print(name, seed, key, {k: (v["label_flips"], round(v["pass_conf"], 4)) for k, v in rep.items() if k.startswith("replay")},
                      {p: tuple(rep[f"datafree_{p}"][c]["gate_fails"] for c in ("float32", "float64", "framework"))
                       for p in ("three", "crossing")}, rep["seconds"], flush=True)
                with open(args.out, "w") as f:
                    json.dump(report, f, indent=2)
    report["seconds_last_run"] = round(time.perf_counter() - started, 1)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
