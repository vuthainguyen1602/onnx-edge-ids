"""Write the probe sets and models of the cross-architecture tie check (exp27) to one directory.

Routing is fixed by IEEE 754 on any machine; a label is a sum of leaf weights,
and a build that sums in another order can flip a row whose class scores tie.
exp27 runs ONNX Runtime on the same inputs on three machines and compares. This
script writes, for each model, its ONNX graph and its probe sets as .npy:

  datafree   the release gate's probes (all nodes, scaler-offset base rows)
  testrows   the probes of exp5/exp10 (test-split base rows, the run's seed)

for the thirteen seed-0 models, plus the two models where exp5c found float32
ties: the CICIDS2017 scikit-learn random forest of seed 4 (retrain it with
exp5_generalization.py --seed 4 --models sklearn_rf --model-dir DIR) and the
UNSW-NB15 one of seed 2 (exp10_cross_dataset.py --seed 2).

    python exp27_make_probe_sets.py --out DIR --cicids $DATA --unsw $UNSW --ton-iot $TON/train_test_network.csv \
        --extra CICIDS2017:4:/tmp/exp5_models_seed4/sklearn_rf.onnx \
        --extra UNSW-NB15:2:/tmp/exp10_models_seed2/unsw_nb15/sklearn_rf.onnx
"""

import argparse
import os
import shutil

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.datasets import ton_iot_test, unsw_test
from edgeids.models import MODEL_KEYS
from edgeids.onnx_forest import load_onnx_forest, path_rows
from edgeids.probes import datafree_probes

MAX_NODES = 100000


def testrow_probes(forest, x, seed):
    """The exact probe draw of edgeids.probes.probe_model (exp5/exp10) for this seed."""
    rng = np.random.default_rng(seed)
    internal = np.flatnonzero(~forest.is_leaf)
    if len(internal) > MAX_NODES:
        internal = np.sort(rng.choice(internal, size=MAX_NODES, replace=False))
    feat = forest.feature[internal]
    rows = path_rows(forest, internal, x[rng.choice(len(x), size=len(internal), replace=True)])
    scale, offset = forest.scale.astype(np.float64), forest.offset.astype(np.float64)
    raw = np.where(scale[feat] != 0, forest.threshold[internal].astype(np.float64)
                   / np.where(scale[feat] != 0, scale[feat], 1) + offset[feat], offset[feat]).astype(np.float32)
    out = []
    for values in (raw, np.nextafter(raw, np.float32(-np.inf)), np.nextafter(raw, np.float32(np.inf))):
        p = rows.copy()
        p[np.arange(len(internal)), feat] = values
        out.append(p)
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cicids", required=True)
    ap.add_argument("--unsw", required=True)
    ap.add_argument("--ton-iot", required=True)
    ap.add_argument("--extra", action="append", default=[], help="DATASET:SEED:path.onnx")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "probes"), exist_ok=True)
    tests = {"CICIDS2017": lambda: load_test_split(args.cicids)[0], "UNSW-NB15": lambda: unsw_test(args.unsw)[0],
             "ToN-IoT": lambda: ton_iot_test(args.ton_iot)[0]}
    dirs = {"CICIDS2017": os.path.join(RESULTS, "exp5_models"),
            "UNSW-NB15": os.path.join(RESULTS, "exp10_models", "unsw_nb15"),
            "ToN-IoT": os.path.join(RESULTS, "exp10_models", "ton_iot")}
    jobs = [("CICIDS2017", 0, "deployed_rf", os.path.join(ARTIFACTS, "ids_rf.onnx"))]
    jobs += [(ds, 0, k, os.path.join(d, f"{k}.onnx")) for ds, d in dirs.items() for k in MODEL_KEYS]
    for spec in args.extra:
        ds, seed, path = spec.split(":", 2)
        jobs.append((ds, int(seed), "sklearn_rf", path))
    cache = {}
    for ds, seed, key, path in jobs:
        name = f"{ds}__{key}__seed{seed}"
        shutil.copy(path, os.path.join(args.out, "models", name + ".onnx"))
        forest = load_onnx_forest(path)
        if ds not in cache:
            cache[ds] = np.ascontiguousarray(tests[ds](), dtype=np.float32)
        np.save(os.path.join(args.out, "probes", name + "__datafree.npy"), datafree_probes(forest)[1].astype(np.float32))
        if key != "deployed_rf":  # the deployed forest's test-row probes are exp2's, not drawn by probe_model
            np.save(os.path.join(args.out, "probes", name + "__testrows.npy"), testrow_probes(forest, cache[ds], seed))
        print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
