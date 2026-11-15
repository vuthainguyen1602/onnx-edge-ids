"""Whole-forest lockstep (commit 59f2723) vs per-tree loop and block-lockstep on real flows."""

import argparse
import glob
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.forest import load_forest, predict_lockstep, predict_per_tree, scale_input

PLAN = [(1, 500), (20, 3000), (100, 10000), (500, 10000), (2000, 20000)]


def predict_whole_forest(forest, matrix):
    """The traversal of 59f2723: one (trees, rows) node matrix and a scatter-add of leaf scores."""
    scaled = scale_input(forest, matrix)
    n_rows = scaled.shape[0]
    node = np.repeat(forest.tree_offsets[:-1].reshape(-1, 1), n_rows, axis=1)
    active = np.ones((forest.n_trees, n_rows), dtype=bool)
    scores = np.zeros((n_rows, forest.n_classes), dtype=forest.dtype)

    while np.any(active):
        leaf = forest.is_leaf[node] & active
        if np.any(leaf):
            t_idx, r_idx = np.nonzero(leaf)
            np.add.at(scores, r_idx, forest.leaf_scores[node[t_idx, r_idx]])
            active[t_idx, r_idx] = False

        if np.any(active):
            t_idx, r_idx = np.nonzero(active)
            cur = node[t_idx, r_idx]
            go_left = scaled[r_idx, forest.feature[cur]] <= forest.threshold[cur]
            node[t_idx, r_idx] = np.where(go_left, forest.left[cur], forest.right[cur])

    return np.argmax(scores, axis=1).astype(np.int64), scores


METHODS = {
    "per_tree": predict_per_tree,
    "whole_forest": predict_whole_forest,
    "block_lockstep": predict_lockstep,
}


def run_once(fn, forest, x, batch):
    start = time.perf_counter()
    for i in range(0, len(x), batch):
        fn(forest, x[i:i + batch])
    return time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp1b_whole_forest_lockstep.json"))
    args = ap.parse_args()

    x, _, _ = load_test_split(args.data, limit=20000)
    forest = load_forest(os.path.join(ARTIFACTS, "ids_rf_numpy.npz"))

    base, _ = predict_per_tree(forest, x[:5000])
    agreement = {name: int((fn(forest, x[:5000])[0] == base).sum()) for name, fn in METHODS.items()}
    print("label agreement with per-tree on 5000 rows:", agreement)

    report = {
        "machine": f"{platform.machine()} {platform.system()} {platform.processor()}",
        "numpy": np.__version__,
        "trees": forest.n_trees,
        "dtype": forest.dtype.__name__,
        "reps": args.reps,
        "data_parts": len(glob.glob(os.path.join(args.data, "*.parquet"))),
        "load_average_start": list(os.getloadavg()),
        "label_agreement_5000_rows": agreement,
        "batches": [],
    }

    for batch, n in PLAN:
        times = {name: [] for name in METHODS}
        # Rotate through the methods within each repetition so that load drift hits all of them.
        for _ in range(args.reps):
            for name, fn in METHODS.items():
                times[name].append(run_once(fn, forest, x[:n], batch))
        row = {"batch": batch, "rows": n}
        for name in METHODS:
            row[f"{name}_rows_s"] = round(n / min(times[name]), 1)
            row[f"{name}_median_rows_s"] = round(n / float(np.median(times[name])), 1)
        row["whole_forest_vs_per_tree"] = round(row["whole_forest_rows_s"] / row["per_tree_rows_s"], 3)
        row["block_vs_whole_forest"] = round(row["block_lockstep_rows_s"] / row["whole_forest_rows_s"], 3)
        report["batches"].append(row)
        print(row, flush=True)

    report["load_average_end"] = list(os.getloadavg())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
