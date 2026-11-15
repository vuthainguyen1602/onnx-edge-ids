"""Do ONNX Runtime's GPU execution providers change the forest's verdicts on the board?

Runs one tree-ensemble graph under the CPU, CUDA and TensorRT execution
providers of a GPU build of ONNX Runtime, on the boundary probes and the
replayed test split, and reports (a) which provider each graph node was
assigned to, read from ONNX Runtime's verbose log at session creation and
from its profile when the run completes, and (b) label
flips and the largest probability difference against the CPU provider, and
against reference labels computed elsewhere (the development machine).

Needs only numpy and onnxruntime, so it runs in a bare environment on the
board. Inputs are .npy files written on the development machine:

    python exp13_ort_providers.py --model ids_rf.onnx --probes probes.npy --replay replay.npy \
        --ref-probe-labels m3_probe_labels.npy --ref-replay-labels m3_replay_labels.npy --out exp13.json
"""

import argparse
import collections
import json
import os
import platform
import re
import subprocess
import sys

import numpy as np
import onnxruntime as ort

CONFIGS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}


def run(model, providers, sets, batch=20000):
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.log_severity_level = 3
    session = ort.InferenceSession(model, sess_options=so, providers=providers)
    name = session.get_inputs()[0].name
    out = {}
    for key, x in sets.items():
        labs, probs = [], []
        for i in range(0, len(x), batch):
            lab, prob = session.run(None, {name: x[i:i + batch]})
            labs.append(np.asarray(lab).reshape(-1))
            probs.append(np.asarray(prob))
        out[key] = (np.concatenate(labs).astype(np.int64), np.concatenate(probs))
    profile = session.end_profiling()
    assigned = collections.defaultdict(set)
    with open(profile) as f:
        for event in json.load(f):
            args = event.get("args", {})
            if event.get("cat") == "Node" and "provider" in args:
                assigned[args.get("op_name", event.get("name", "?"))].add(args["provider"])
    os.remove(profile)
    return session.get_providers(), {k: sorted(v) for k, v in assigned.items()}, out


def placement(model, providers):
    """Node -> provider as ONNX Runtime logs it at session creation, plus any 'Cannot find plugin' lines.

    The verbose log is written by the native library, so it is read from a child process's stderr.
    This works even when a later run() fails, e.g. a GPU kernel without its libraries.
    """
    code = ("import onnxruntime as ort, sys\n"
            "so = ort.SessionOptions(); so.log_severity_level = 0\n"
            f"ort.InferenceSession({model!r}, sess_options=so, providers={providers!r})\n")
    err = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stderr
    placed, current = {}, None
    for line in err.splitlines():
        m = re.search(r"Node\(s\) placed on \[(\w+)\]", line)
        if m:
            current = m.group(1)
            continue
        m = re.search(r"VerifyEachNodeIsAssignedToAnEp\]\s+(\w+) \(", line)
        if m and current:
            placed.setdefault(current, []).append(m.group(1))
    missing = sorted(set(re.findall(r"Cannot find plugin: (\w+)", err)))
    return placed, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--ref-probe-labels", default="")
    ap.add_argument("--ref-replay-labels", default="")
    ap.add_argument("--ref-name", default="development machine")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sets = {"probes": np.load(args.probes), "replay": np.load(args.replay)}
    refs = {k: np.load(p).astype(np.int64) for k, p in (("probes", args.ref_probe_labels), ("replay", args.ref_replay_labels)) if p}
    report = {"machine": f"{platform.machine()} {platform.system()}", "numpy": np.__version__, "onnxruntime": ort.__version__,
              "available_providers": ort.get_available_providers(), "model": os.path.basename(args.model),
              "rows": {k: int(len(v)) for k, v in sets.items()}, "reference": args.ref_name if refs else None, "configs": {}}
    cpu = None
    for key, providers in CONFIGS.items():
        entry = {"requested": providers}
        entry["node_placement_at_session_creation"], plugins = placement(args.model, providers)
        if plugins:
            entry["tensorrt_cannot_find_plugin"] = plugins
        try:
            got, assigned, out = run(args.model, providers, sets)
        except Exception as exc:  # a provider that cannot load is a result, not a crash
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            report["configs"][key] = entry
            print(key, entry, flush=True)
            continue
        entry["session_providers"] = got
        entry["node_assignment"] = assigned
        if key == "cpu":
            cpu = out
        for name, (lab, prob) in out.items():
            e = {}
            if cpu is not None and key != "cpu":
                e["label_flips_vs_cpu_provider"] = int((lab != cpu[name][0]).sum())
                e["max_probability_delta_vs_cpu_provider"] = float(np.abs(prob - cpu[name][1]).max())
            if name in refs:
                e["label_flips_vs_reference"] = int((lab != refs[name]).sum())
            entry[name] = e
        report["configs"][key] = entry
        print(key, json.dumps(entry), flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
