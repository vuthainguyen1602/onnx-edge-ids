"""Exact early exit on the twelve trained models, not only the deployed forest.

exp3 checks on the deployed forest that the early-exit rule of Section 7.6
(stop once the trees not yet evaluated can no longer change the sign of the
margin) returns the full forest's label on every row. The proof is over the
reals; here each of the twelve exp5/exp10 models (seed 0) is checked on its own
test split, against two references: the float32 full traversal of the same
graph and ONNX Runtime.

The margin is generalised from the deployed forest's two-column vote:
  two score columns           margin = s_attack - s_benign, Attack iff > 0
  one column, LOGISTIC        margin = score + base value,  Attack iff > 0
  one column, NONE (P(attack)) margin = score + base - 0.5, Attack iff > 0
Per-tree bounds [l_t, u_t] are the min and max leaf contribution of tree t.
Schedule: random forests (equal weights) are first checked after T/2 + 4 trees
as in exp3, boosted models after 8 trees; then every 8 trees.

    python exp26_early_exit_all_models.py --cicids $DATA --unsw $UNSW --ton-iot $TON/train_test_network.csv
"""

import argparse
import json
import os
import time

import numpy as np

from edgeids.data import RESULTS, load_test_split
from edgeids.datasets import ton_iot_test, unsw_test
from edgeids.models import MODEL_KEYS
from edgeids.onnx_forest import _go_true, load_onnx_forest, predict, scale_input
from edgeids.runtime import ort_labels


def margin_leaves(forest):
    """Per-node margin contribution (0 for internal nodes) and the label cut-off."""
    s = forest.leaf_scores.astype(np.float64)
    if forest.n_targets == 2:
        return s[:, 1] - s[:, 0], 0.0
    cut = 0.5 if forest.post_transform == "NONE" else 0.0
    return s[:, 0], float(cut - forest.base_values[0])


def bounds(forest, contrib):
    lo, hi = np.empty(forest.n_trees), np.empty(forest.n_trees)
    for t in range(forest.n_trees):
        a, b = forest.tree_offsets[t], forest.tree_offsets[t + 1]
        d = contrib[a:b][forest.is_leaf[a:b]]
        lo[t], hi[t] = d.min(), d.max()
    return lo, hi


def leaves(forest, scaled, tree):
    thr = forest.threshold.astype(np.float32)
    node = np.full(len(scaled), forest.tree_offsets[tree], dtype=np.int64)
    active = ~forest.is_leaf[node]
    while active.any():
        rows = np.flatnonzero(active)
        cur = node[rows]
        go = _go_true(forest.mode[cur], scaled[rows, forest.feature[cur]], thr[cur])
        nxt = np.where(go, forest.left[cur], forest.right[cur])
        node[rows] = nxt
        active[rows] = ~forest.is_leaf[nxt]
    return node


def early_exit(forest, x, schedule, contrib, cut, lo, hi):
    scaled = scale_input(forest, x, np.float32)
    n = len(x)
    margin = np.zeros(n)
    labels = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=np.int64)
    alive = np.arange(n)
    rem_lo, rem_hi = lo.sum(), hi.sum()
    t = 0
    for size in schedule:
        if alive.size == 0 or t >= forest.n_trees:
            break
        chunk = range(t, min(t + size, forest.n_trees))
        for tree in chunk:
            margin[alive] += contrib[leaves(forest, scaled[alive], tree)]
            rem_lo -= lo[tree]
            rem_hi -= hi[tree]
        used[alive] += len(chunk)
        t += len(chunk)
        m = margin[alive] - cut
        sure_attack = m + rem_lo > 0
        sure_benign = m + rem_hi <= 0
        labels[alive[sure_attack]] = 1
        alive = alive[~(sure_attack | sure_benign)]
    if alive.size:
        labels[alive] = (margin[alive] - cut > 0).astype(np.int64)
    return labels, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids", required=True)
    ap.add_argument("--unsw", required=True)
    ap.add_argument("--ton-iot", required=True)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp26_early_exit_all_models.json"))
    args = ap.parse_args()
    data = {"CICIDS2017": (lambda: load_test_split(args.cicids)[0], os.path.join(RESULTS, "exp5_models")),
            "UNSW-NB15": (lambda: unsw_test(args.unsw)[0], os.path.join(RESULTS, "exp10_models", "unsw_nb15")),
            "ToN-IoT": (lambda: ton_iot_test(args.ton_iot)[0], os.path.join(RESULTS, "exp10_models", "ton_iot"))}
    report = {"datasets": {}}
    for name, (load, model_dir) in data.items():
        x = np.ascontiguousarray(load(), dtype=np.float32)
        report["datasets"][name] = {"rows": int(len(x)), "models": {}}
        for key in MODEL_KEYS:
            path = os.path.join(model_dir, f"{key}.onnx")
            forest = load_onnx_forest(path)
            contrib, cut = margin_leaves(forest)
            lo, hi = bounds(forest, contrib)
            first = forest.n_trees // 2 + 4 if key == "sklearn_rf" else 8
            schedule = [first] + [8] * ((forest.n_trees - first + 7) // 8)
            t0 = time.perf_counter()
            ee, used = early_exit(forest, x, schedule, contrib, cut, lo, hi)
            t_ee = time.perf_counter() - t0
            t0 = time.perf_counter()
            full = predict(forest, x, np.float32)
            t_full = time.perf_counter() - t0
            ort = ort_labels(path, x)
            rep = {"trees": forest.n_trees, "first_check_after": first,
                   "labels_identical_to_float32_full_forest": int((ee == full).sum()),
                   "labels_identical_to_onnxruntime": int((ee == ort).sum()),
                   "float32_full_forest_vs_onnxruntime_flips": int((full != ort).sum()),
                   "mean_trees_used": round(float(used.mean()), 2),
                   "rows_decided_at_first_check": float((used == first).mean()),
                   "seconds_early_exit": round(t_ee, 2), "seconds_full_forest": round(t_full, 2)}
            report["datasets"][name]["models"][key] = rep
            print(name, key, rep, flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
