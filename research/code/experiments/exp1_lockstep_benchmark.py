"""Per-tree loop vs lockstep traversal on real flows: throughput by batch size, plus label check."""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.forest import load_forest, predict_lockstep, predict_per_tree

PLAN = [(1, 500), (20, 3000), (100, 10000), (500, 10000), (2000, 20000)]


def throughput(fn, forest, x, batch, reps=5):
    best = np.inf
    for _ in range(reps):
        start = time.perf_counter()
        for i in range(0, len(x), batch):
            fn(forest, x[i:i + batch])
        best = min(best, time.perf_counter() - start)
    return len(x) / best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp1_lockstep_benchmark.json"))
    args = ap.parse_args()

    x, _, _ = load_test_split(args.data, limit=20000)
    forest = load_forest(os.path.join(ARTIFACTS, "ids_rf_numpy.npz"))

    a, _ = predict_per_tree(forest, x[:5000])
    b, _ = predict_lockstep(forest, x[:5000])
    report = {
        "machine": f"{platform.machine()} {platform.system()} {platform.processor()}",
        "numpy": np.__version__,
        "trees": forest.n_trees,
        "label_agreement_5000_rows": int((a == b).sum()),
        "batches": [],
    }
    print("label agreement per-tree vs lockstep:", report["label_agreement_5000_rows"], "/ 5000")

    for batch, n in PLAN:
        slow = throughput(predict_per_tree, forest, x[:n], batch)
        fast = throughput(predict_lockstep, forest, x[:n], batch)
        row = {"batch": batch, "rows": n, "per_tree_rows_s": round(slow, 1),
               "lockstep_rows_s": round(fast, 1), "speedup": round(fast / slow, 1)}
        report["batches"].append(row)
        print(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
