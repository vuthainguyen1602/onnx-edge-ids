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

--probe-set crossing builds the gate's default set instead: two probes per node
on the exact float32 crossing (the construction of exp24), on the same
data-free rows; results go to exp19_datafree_probes_crossing.json by default.
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS
from edgeids.datasets import load_cicids, load_ton_iot, load_unsw
from edgeids.gates import TOL, probe_gate
from edgeids.models import MODEL_KEYS, build_models
from edgeids.onnx_forest import load_onnx_forest, reaches
from edgeids.probes import datafree_probes
from edgeids.runtime import numpy_probs, ort_probs


def crossing_probes(forest):
    """The data-free crossing pair of exp24: rows routed to each node, g(v) on c_A and its successor."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("exp24", os.path.join(here, "exp24_probe_straddle.py"))
    exp24 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp24)
    from edgeids.onnx_forest import path_rows
    internal, three = datafree_probes(forest)
    n = len(internal)
    feat = forest.feature[internal]
    r0 = three[:n][np.arange(n), feat]
    ok, a, _ = exp24.crossing(forest, internal, r0)
    base = np.repeat(forest.offset.astype(np.float32).reshape(1, -1), n, axis=0)
    rows = path_rows(forest, internal, base)
    out = []
    for k in (a, a + 1):
        p = rows.copy()
        p[np.arange(n), feat] = np.where(ok, exp24.unordered(k), r0)
        out.append(p)
    return internal, np.concatenate(out)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t0, 2)


def run_model(path, pipe=None, probe_set="three"):
    forest = load_onnx_forest(path)
    build = crossing_probes if probe_set == "crossing" else datafree_probes
    (internal, probes), t_build = timed(lambda: build(forest))
    reach = reaches(forest, probes, np.tile(internal, len(probes) // len(internal)))
    ref, t_ort = timed(lambda: ort_probs(path, probes))
    p32, t_32 = timed(lambda: numpy_probs(forest, probes, np.float32))
    rep = {"internal_nodes": int(len(internal)), "probes": int(len(probes)),
           "reaching_their_node": float(reach.mean()),
           "seconds": {"build": t_build, "onnxruntime": t_ort, "numpy_float32": t_32},
           "float32": probe_gate(p32, ref), "float64": probe_gate(numpy_probs(forest, probes, np.float64), ref)}
    if pipe is not None:
        rep["framework"] = probe_gate(pipe.predict_proba(probes.astype(np.float64)), ref)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--converter", action="store_true", help="retrain the 12 models (as exp17) and test framework vs ORT")
    ap.add_argument("--cicids-train", default="")
    ap.add_argument("--cicids-test", default="")
    ap.add_argument("--unsw", default="")
    ap.add_argument("--ton-iot", default="")
    ap.add_argument("--probe-set", choices=("three", "crossing"), default="three")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.out:
        suffix = "_crossing" if args.probe_set == "crossing" else ""
        args.out = os.path.join(RESULTS, f"exp19_datafree_probes{suffix}.json")

    import onnxruntime
    started = time.perf_counter()
    report = {"numpy": np.__version__, "onnxruntime": onnxruntime.__version__, "tolerance": TOL,
              "probe_set": args.probe_set, "datasets": {}}

    jobs = [("CICIDS2017", os.path.join(RESULTS, "exp5_models"), ["deployed_rf"] + list(MODEL_KEYS))]
    loaders = {}
    if args.converter:
        loaders["CICIDS2017"] = lambda: load_cicids(args.cicids_train, args.cicids_test)
        if args.unsw:
            loaders["UNSW-NB15"] = lambda: load_unsw(args.unsw)
        if args.ton_iot:
            loaders["ToN-IoT"] = lambda: load_ton_iot(args.ton_iot)
    jobs += [("UNSW-NB15", os.path.join(RESULTS, "exp10_models", "unsw_nb15"), list(MODEL_KEYS)),
             ("ToN-IoT", os.path.join(RESULTS, "exp10_models", "ton_iot"), list(MODEL_KEYS))]

    for name, model_dir, keys in jobs:
        report["datasets"][name] = {}
        train = loaders[name]() if name in loaders else None
        for key in keys:
            path = os.path.join(ARTIFACTS, "ids_rf.onnx") if key == "deployed_rf" else os.path.join(model_dir, f"{key}.onnx")
            pipe = None
            if train is not None and key != "deployed_rf":
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                x_train, y_train, _ = train
                pipe = Pipeline([("scaler", StandardScaler()), ("model", build_models(0)[key][1])])
                pipe.fit(x_train.astype(np.float64), y_train)
            rep = run_model(path, pipe, args.probe_set)
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
