"""Hummingbird as a runtime of a different family: trees compiled to tensor programs, on CPU and GPU.

ONNX Runtime and the NumPy engine both walk trees node by node; Treelite (exp22,
exp35) does too. Hummingbird compiles a TreeEnsembleClassifier graph into tensor
operations (GEMM, or vectorised tree traversal) run by PyTorch, on the CPU or on
a GPU. This converts each of the thirteen ONNX graphs with Hummingbird and runs:

  * float32, CPU: tree_implementation gemm, tree_trav and perf_tree_trav;
  * float32, GPU: gemm on the Apple GPU through PyTorch's MPS device;
  * float64, CPU, gemm (the converted module cast to double): a divergent
    runtime the gate should fail, the analogue of the float64 traversal;

each against ONNX Runtime on the gate's data-free probes (three-probe set and
crossing pair: label flips, probes over 1e-5, verdict) and on replay of the
whole test split (label flips, largest deviation, chance a 2,000-row replay
passes comparing labels only or also confidences at 1e-5).

With --base-values-check, for each graph whose TreeEnsembleClassifier has non-zero base_values, it
also compares Hummingbird (float32, CPU gemm) with ONNX Runtime on a copy of the graph with base_values
set to zero, to separate a dropped base value from any other difference (stored as "base_values_zeroed").

Runs in its own venv (.hbvenv: torch, hummingbird-ml, onnxruntime 1.30):
    PYTHONPATH=research/code .hbvenv/bin/python research/code/experiments/exp39_hummingbird.py \
        --cicids $DATA --unsw $UNSW --ton-iot $TON/train_test_network.csv
"""

import argparse
import importlib.util
import json
import os
import time
import warnings

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.datasets import ton_iot_test, unsw_test
from edgeids.gates import probe_gate, replay_gate
from edgeids.models import MODEL_KEYS
from edgeids.onnx_forest import load_onnx_forest
from edgeids.probes import datafree_probes
from edgeids.runtime import ort_probs

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp19", os.path.join(HERE, "exp19_datafree_probes.py"))
exp19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp19)

VARIANTS = [  # name, device, tree_implementation, float64
    ("cpu_gemm", "cpu", "gemm", False),
    ("cpu_tree_trav", "cpu", "tree_trav", False),
    ("cpu_perf_tree_trav", "cpu", "perf_tree_trav", False),
    ("gpu_mps_gemm", "mps", "gemm", False),
    ("cpu_gemm_float64", "cpu", "gemm", True),
]


def predict(model, x, f64, batch=20000):
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        out.append(model.predict_proba(xb.astype(np.float64) if f64 else xb))
    return np.concatenate(out).astype(np.float64)


def run_model(path, x_test, replay_rows):
    import hummingbird.ml as hb
    import onnx
    import torch
    forest = load_onnx_forest(path)
    _, three = datafree_probes(forest)
    _, cross = exp19.crossing_probes(forest)
    sets = {"three": three, "crossing": cross}
    refs = {k: ort_probs(path, v) for k, v in sets.items()}
    ref_replay = ort_probs(path, x_test)
    graph = onnx.load(path)
    rep = {"probes": {k: int(len(v)) for k, v in sets.items()}, "variants": {}}
    for name, device, impl, f64 in VARIANTS:
        t0 = time.perf_counter()
        try:
            model = hb.convert(graph, "torch", three[:100], device=device, extra_config={"tree_implementation": impl})
            if f64:
                model.model.double()
            actual = str(next(model.model.parameters()).device) if any(True for _ in model.model.parameters()) else device
            v = {"device": actual}
            for k, x in sets.items():
                v[k] = probe_gate(predict(model, x, f64), refs[k])
            g = replay_gate(predict(model, x_test, f64), ref_replay, replay_rows)
            v["replay"] = {"label_flips": g["label_flips"], "max_delta": g["max_delta"],
                           "pass_labels": g["prob_replay_passes_labels_only"],
                           "pass_conf": g["tolerances"]["1e-05"]["prob_replay_passes"]}
        except Exception as e:  # out of memory, unsupported op or device: record, do not hide
            v = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        v["seconds"] = round(time.perf_counter() - t0, 1)
        rep["variants"][name] = v
        if device == "mps":
            torch.mps.empty_cache()
    return rep


def zero_base_values(path, out):
    import onnx
    m = onnx.load(path)
    node = [n for n in m.graph.node if n.op_type == "TreeEnsembleClassifier"][0]
    attrs = [a for a in node.attribute if a.name == "base_values"]
    if not attrs or not any(attrs[0].floats):
        return None
    attrs[0].floats[:] = [0.0] * len(attrs[0].floats)
    onnx.save(m, out)
    return out


def base_values_check(path, x_test, replay_rows, tmp):
    import hummingbird.ml as hb
    import onnx
    zero = zero_base_values(path, os.path.join(tmp, os.path.basename(path)))
    if zero is None:
        return None
    forest = load_onnx_forest(path)
    _, three = datafree_probes(forest)
    _, cross = exp19.crossing_probes(forest)
    model = hb.convert(onnx.load(path), "torch", three[:100], device="cpu", extra_config={"tree_implementation": "gemm"})
    out = {}
    for k, x in {"three": three, "crossing": cross}.items():
        out[k] = probe_gate(predict(model, x, False), ort_probs(zero, x))
    g = replay_gate(predict(model, x_test, False), ort_probs(zero, x_test), replay_rows)
    out["replay"] = {"label_flips": g["label_flips"], "max_delta": g["max_delta"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids", required=True)
    ap.add_argument("--unsw", default="")
    ap.add_argument("--ton-iot", default="")
    ap.add_argument("--replay-rows", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp39_hummingbird.json"))
    ap.add_argument("--base-values-check", action="store_true")
    args = ap.parse_args()

    import hummingbird
    import onnxruntime
    import torch
    report = {"versions": {"hummingbird": hummingbird.__version__, "torch": torch.__version__,
                           "onnxruntime": onnxruntime.__version__, "numpy": np.__version__},
              "replay_rows": args.replay_rows, "datasets": {}}
    if os.path.exists(args.out):
        report = json.load(open(args.out))
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
    import tempfile
    tmp = tempfile.mkdtemp()
    for name, x, models in jobs:
        ds = report["datasets"].setdefault(name, {})
        for key, path in models:
            if args.base_values_check and key in ds:
                chk = base_values_check(path, np.ascontiguousarray(x, dtype=np.float32), args.replay_rows, tmp)
                if chk is not None:
                    ds[key]["base_values_zeroed"] = chk
                    print(name, key, "base_values zeroed:", {k: (v.get("gate_fails"), v["label_flips"], f'{v["max_delta"]:.1e}')
                                                             for k, v in chk.items()}, flush=True)
                    with open(args.out, "w") as f:
                        json.dump(report, f, indent=2)
                continue
            if key in ds:
                continue
            rep = run_model(path, np.ascontiguousarray(x, dtype=np.float32), args.replay_rows)
            ds[key] = rep
            print(name, key, {n: (v.get("error", "")[:40] or
                                  (v["three"]["gate_fails"], v["crossing"]["gate_fails"], v["replay"]["label_flips"],
                                   f'{max(v["three"]["max_delta"], v["crossing"]["max_delta"], v["replay"]["max_delta"]):.1e}'),
                                  v["seconds"]) for n, v in rep["variants"].items()}, flush=True)
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
