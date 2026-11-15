"""Data-free boundary probes, with the gate's confidence check, on every model of exp16/exp17.

exp11b showed that on the deployed forest the release gate's data-free probe
set (three probes per internal node, every unconstrained feature at the scaler
offset) catches every injected fault, on confidence rather than on labels.
Tables 4/5 build their probes from test-split rows instead. This builds the
data-free set for all thirteen models, over ALL internal nodes (no sampling, as
the gate does), and reports per model:

  * cost: internal nodes, probes, the share reaching their node, and the time
    to build the probes and to run them through ONNX Runtime and the float32
    traversal (the gate's two backends) on this machine;
  * runtime parity: float32 and float64 traversal vs ONNX Runtime;
  * converter parity (with --converter): the training framework, retrained as
    exp17 retrains it, vs ONNX Runtime on the same probes;

each as label flips, rows whose probability deviates by more than 1e-5, and
whether the gate (labels + confidences) fails.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgeids.data import ARTIFACTS, RESULTS
from edgeids.onnx_forest import load_onnx_forest, path_rows, reaches
from exp16_confidence_replay import MODELS, label, numpy_probs, ort_probs

TOL = 1e-5


def build_probes(forest):
    internal = np.flatnonzero(~forest.is_leaf)
    feat = forest.feature[internal]
    base = np.repeat(forest.offset.astype(np.float32).reshape(1, -1), len(internal), axis=0)
    rows = path_rows(forest, internal, base)
    scale, offset = forest.scale.astype(np.float64), forest.offset.astype(np.float64)
    safe = np.where(scale[feat] != 0, scale[feat], 1.0)
    raw = np.where(scale[feat] != 0, forest.threshold[internal].astype(np.float64) / safe + offset[feat],
                   offset[feat]).astype(np.float32)
    out = []
    for values in (raw, np.nextafter(raw, np.float32(-np.inf)), np.nextafter(raw, np.float32(np.inf))):
        p = rows.copy()
        p[np.arange(len(internal)), feat] = values
        out.append(p)
    return internal, np.concatenate(out)


def gate(p, ref):
    delta = np.abs(p - ref).max(axis=1)
    flips = label(p) != label(ref)
    over = delta > TOL
    return {"label_flips": int(flips.sum()), "rows_over_tolerance": int(over.sum()),
            "max_delta": float(delta.max()), "gate_fails": bool((flips | over).any())}


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t0, 2)


def run_model(path, pipe=None):
    forest = load_onnx_forest(path)
    (internal, probes), t_build = timed(lambda: build_probes(forest))
    reach = reaches(forest, probes, np.tile(internal, 3))
    ref, t_ort = timed(lambda: ort_probs(path, probes))
    p32, t_32 = timed(lambda: numpy_probs(forest, probes, np.float32))
    rep = {"internal_nodes": int(len(internal)), "probes": int(len(probes)),
           "reaching_their_node": float(reach.mean()),
           "seconds": {"build": t_build, "onnxruntime": t_ort, "numpy_float32": t_32},
           "float32": gate(p32, ref), "float64": gate(numpy_probs(forest, probes, np.float64), ref)}
    if pipe is not None:
        rep["framework"] = gate(pipe.predict_proba(probes.astype(np.float64)), ref)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--converter", action="store_true", help="retrain the 12 models (as exp17) and test framework vs ORT")
    ap.add_argument("--cicids-train", default="")
    ap.add_argument("--cicids-test", default="")
    ap.add_argument("--unsw", default="")
    ap.add_argument("--ton-iot", default="")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp19_datafree_probes.json"))
    args = ap.parse_args()

    import onnxruntime
    started = time.perf_counter()
    report = {"numpy": np.__version__, "onnxruntime": onnxruntime.__version__, "tolerance": TOL, "datasets": {}}

    jobs = [("CICIDS2017", os.path.join(RESULTS, "exp5_models"), ["deployed_rf"] + list(MODELS))]
    loaders = {}
    if args.converter:
        from exp17_converter_confidence import cicids, ton_iot, unsw
        loaders["CICIDS2017"] = lambda: cicids(args.cicids_train, args.cicids_test)
        if args.unsw:
            loaders["UNSW-NB15"] = lambda: unsw(args.unsw)
        if args.ton_iot:
            loaders["ToN-IoT"] = lambda: ton_iot(args.ton_iot)
    jobs += [("UNSW-NB15", os.path.join(RESULTS, "exp10_models", "unsw_nb15"), list(MODELS)),
             ("ToN-IoT", os.path.join(RESULTS, "exp10_models", "ton_iot"), list(MODELS))]

    for name, model_dir, keys in jobs:
        report["datasets"][name] = {}
        train = loaders[name]() if name in loaders else None
        for key in keys:
            path = os.path.join(ARTIFACTS, "ids_rf.onnx") if key == "deployed_rf" else os.path.join(model_dir, f"{key}.onnx")
            pipe = None
            if train is not None and key != "deployed_rf":
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from exp5_generalization import build_models
                x_train, y_train, _ = train
                pipe = Pipeline([("scaler", StandardScaler()), ("model", build_models(0)[key][1])])
                pipe.fit(x_train.astype(np.float64), y_train)
            rep = run_model(path, pipe)
            report["datasets"][name][key] = rep
            line = {k: (rep[k]["label_flips"], rep[k]["rows_over_tolerance"], rep[k]["gate_fails"])
                    for k in ("float32", "float64", "framework") if k in rep}
            print(name, key, rep["probes"], f"reach={rep['reaching_their_node']:.4f}", rep["seconds"], line, flush=True)
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)

    report["seconds"] = round(time.perf_counter() - started, 1)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
