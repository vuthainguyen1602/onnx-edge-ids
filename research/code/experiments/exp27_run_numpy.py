"""Run the shipped NumPy engine over every probe set of an exp27 directory (labels and probabilities).

Needs numpy and a copy of the shipped package in DIR/src (exp27 copies it);
the .npz artifacts in DIR/npz are exported on the development machine with the
shipped exporter, so every machine serves byte-identical artifacts. Batches of
2,000 rows, the gate's probe batch, fix the lockstep block size and hence the
summation order within a machine.

    python exp27_run_numpy.py --dir DIR --tag m3
"""

import argparse
import glob
import json
import os
import platform
import sys

import numpy as np

BATCH = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.dir, "src"))
    from onnx_edge_ids.numpy_engine import NumpyInferenceEngine

    out = os.path.join(args.dir, "outputs", args.tag + "_numpy")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "env.json"), "w") as f:
        json.dump({"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
                   "numpy": np.__version__, "engine": "shipped NumPy engine, batch 2000"}, f, indent=1)
    for probe_file in sorted(glob.glob(os.path.join(args.dir, "probes", "*.npy"))):
        stem = os.path.basename(probe_file)[:-4]
        eng = NumpyInferenceEngine(os.path.join(args.dir, "npz", stem.rsplit("__", 1)[0] + ".npz"))
        x = np.load(probe_file)
        labels, probs = [], []
        for i in range(0, len(x), BATCH):
            p, pr = eng._infer(x[i:i + BATCH])
            labels.append(p); probs.append(np.asarray(pr, dtype=np.float32))
        np.savez_compressed(os.path.join(out, stem + ".npz"), labels=np.concatenate(labels).astype(np.int8),
                            probs=np.concatenate(probs))
        print(args.tag, stem, len(x), flush=True)


if __name__ == "__main__":
    main()
