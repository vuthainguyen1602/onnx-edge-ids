"""Run ONNX Runtime over every probe set of an exp27 directory and save labels and probabilities.

Needs only numpy and onnxruntime, so the same file runs on the development
machine, in an x86-64 container and on a Jetson board:

    python exp27_run_ort.py --dir DIR --tag m3
    python exp27_run_ort.py --dir DIR --tag m3_1t --threads 1   # one intra-op thread

ONNX Runtime may split a tree ensemble's sum over trees across intra-op
threads, so the summation order, and hence the label of an exact tie, can
depend on the thread count; --threads fixes it.
"""

import argparse
import glob
import json
import os
import platform

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=0, help="intra-op threads (0: ONNX Runtime's default)")
    args = ap.parse_args()
    import onnxruntime as ort

    out = os.path.join(args.dir, "outputs", args.tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "env.json"), "w") as f:
        flags = []
        if os.path.exists("/proc/cpuinfo"):
            words = set(open("/proc/cpuinfo").read().split())
            flags = sorted(w for w in ("sse4_2", "avx", "avx2", "fma", "avx512f", "asimd", "fphp") if w in words)
        json.dump({"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
                   "numpy": np.__version__, "onnxruntime": ort.__version__, "cpu_flags": flags,
                   "cpu_count": os.cpu_count(), "intra_op_threads": args.threads or "default"}, f, indent=1)
    for probe_file in sorted(glob.glob(os.path.join(args.dir, "probes", "*.npy"))):
        stem = os.path.basename(probe_file)[:-4]
        model = os.path.join(args.dir, "models", stem.rsplit("__", 1)[0] + ".onnx")
        opts = ort.SessionOptions()
        if args.threads:
            opts.intra_op_num_threads = args.threads
        s = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        x = np.load(probe_file)
        labels, probs = [], []
        for i in range(0, len(x), 50000):
            lab, pr = s.run(None, {name: x[i:i + 50000]})
            labels.append(np.asarray(lab).reshape(-1)); probs.append(np.asarray(pr, dtype=np.float32))
        np.savez_compressed(os.path.join(out, stem + ".npz"), labels=np.concatenate(labels).astype(np.int8),
                            probs=np.concatenate(probs))
        print(args.tag, stem, len(x), flush=True)


if __name__ == "__main__":
    main()
