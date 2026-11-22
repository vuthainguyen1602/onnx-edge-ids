"""Hummingbird on a CUDA GPU (Jetson Orin): tree tensor programs under full float32, TF32 and bf16 matmul.

exp39 ran Hummingbird on CPUs and the Apple GPU. This runs the same conversions on the Jetson's Ampere
GPU, where PyTorch can compute float32 matrix products in reduced precision: TF32 (10-bit mantissa) under
torch.set_float32_matmul_precision("high"), and bf16 under "medium". The GEMM implementation compares
features with thresholds through matrix products, so reduced precision can move a value across a split:
a GPU-specific divergence of the kind a release gate on a GPU board must catch.

Variants (float32 inputs): cuda gemm at "highest" (full float32), "high" (TF32) and "medium" (bf16);
cuda tree_trav and perf_tree_trav at "highest"; cpu gemm as a control. Each against ONNX Runtime 1.30
(CPU) on the data-free three-probe set and crossing pair and on replay of the whole test split. Hummingbird
0.4.12 drops a graph's base values (exp39), so for the six graphs that have them the reference is ONNX
Runtime on a copy with base_values zeroed, isolating GPU arithmetic from that known defect.

    PYTHONPATH=research/code ~/hbvenv/bin/python research/code/experiments/exp42_hummingbird_cuda.py \
        --cicids $DATA --unsw $UNSW --ton-iot $TON/train_test_network.csv
"""

import argparse
import importlib.util
import json
import os
import platform
import tempfile
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


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp19 = load("exp19", "exp19_datafree_probes.py")
exp39 = load("exp39", "exp39_hummingbird.py")

VARIANTS = [  # name, device, tree_implementation, matmul precision
    ("cuda_gemm_fp32", "cuda", "gemm", "highest"),
    ("cuda_gemm_tf32", "cuda", "gemm", "high"),
    ("cuda_gemm_bf16", "cuda", "gemm", "medium"),
    ("cuda_tree_trav", "cuda", "tree_trav", "highest"),
    ("cuda_perf_tree_trav", "cuda", "perf_tree_trav", "highest"),
    ("cpu_gemm", "cpu", "gemm", "highest"),
]


def run_model(path, x_test, replay_rows, tmp):
    import hummingbird.ml as hb
    import onnx
    import torch
    forest = load_onnx_forest(path)
    _, three = datafree_probes(forest)
    _, cross = exp19.crossing_probes(forest)
    sets = {"three": three, "crossing": cross}
    zero = exp39.zero_base_values(path, os.path.join(tmp, os.path.basename(path)))
    ref_path = zero or path
    refs = {k: ort_probs(ref_path, v) for k, v in sets.items()}
    ref_replay = ort_probs(ref_path, x_test)
    graph = onnx.load(path)
    rep = {"reference": "base_values zeroed" if zero else "graph", "probes": {k: int(len(v)) for k, v in sets.items()},
           "variants": {}}
    for name, device, impl, precision in VARIANTS:
        t0 = time.perf_counter()
        try:
            torch.set_float32_matmul_precision(precision)
            model = hb.convert(graph, "torch", three[:100], device=device, extra_config={"tree_implementation": impl})
            v = {"device": device, "matmul_precision": precision}
            for k, x in sets.items():
                v[k] = probe_gate(exp39.predict(model, x, False), refs[k])
            g = replay_gate(exp39.predict(model, x_test, False), ref_replay, replay_rows)
            v["replay"] = {"label_flips": g["label_flips"], "max_delta": g["max_delta"],
                           "pass_labels": g["prob_replay_passes_labels_only"],
                           "pass_conf": g["tolerances"]["1e-05"]["prob_replay_passes"]}
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # out of memory or unsupported: record, do not hide
            v = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        v["seconds"] = round(time.perf_counter() - t0, 1)
        rep["variants"][name] = v
    torch.set_float32_matmul_precision("highest")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids", required=True)
    ap.add_argument("--unsw", default="")
    ap.add_argument("--ton-iot", default="")
    ap.add_argument("--replay-rows", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp42_hummingbird_cuda.json"))
    args = ap.parse_args()
    import hummingbird
    import onnxruntime
    import torch
    report = {"versions": {"hummingbird": hummingbird.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
                           "onnxruntime": onnxruntime.__version__, "numpy": np.__version__},
              "machine": f"{platform.machine()} {platform.node()}",
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
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
    tmp = tempfile.mkdtemp()
    for name, x, models in jobs:
        ds = report["datasets"].setdefault(name, {})
        for key, path in models:
            if key in ds:
                continue
            rep = run_model(path, np.ascontiguousarray(x, dtype=np.float32), args.replay_rows, tmp)
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
