"""Flattened random-forest inference in NumPy: per-tree, lockstep, exact early exit."""

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Forest:
    offset: np.ndarray
    scale: np.ndarray
    tree_offsets: np.ndarray
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    is_leaf: np.ndarray
    leaf_scores: np.ndarray
    dtype: type
    meta: dict

    @property
    def n_trees(self):
        return len(self.tree_offsets) - 1

    @property
    def n_classes(self):
        return int(self.leaf_scores.shape[1])


def load_forest(path, dtype=None):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["metadata_json"]))
    dt = np.dtype(dtype or meta.get("inference_dtype", "float64")).type
    return Forest(
        offset=data["offset"].astype(dt),
        scale=data["scale"].astype(dt),
        tree_offsets=data["tree_offsets"].astype(np.int64),
        left=data["left"].astype(np.int64),
        right=data["right"].astype(np.int64),
        feature=data["feature"].astype(np.int64),
        threshold=data["threshold"].astype(dt),
        is_leaf=data["is_leaf"].astype(bool),
        leaf_scores=data["leaf_scores"].astype(dt),
        dtype=dt,
        meta=meta,
    )


def scale_input(forest, matrix):
    x = np.ascontiguousarray(matrix, dtype=forest.dtype)
    return (x - forest.offset) * forest.scale


def predict_per_tree(forest, matrix):
    """Baseline: one Python-level while-loop per tree."""
    scaled = scale_input(forest, matrix)
    n = scaled.shape[0]
    scores = np.zeros((n, forest.n_classes), dtype=forest.dtype)
    for t in range(forest.n_trees):
        node = np.full(n, forest.tree_offsets[t], dtype=np.int64)
        active = np.ones(n, dtype=bool)
        while active.any():
            rows = np.flatnonzero(active)
            cur = node[rows]
            leaf = forest.is_leaf[cur]
            if leaf.any():
                lr = rows[leaf]
                scores[lr] += forest.leaf_scores[node[lr]]
                active[lr] = False
            br = rows[~leaf]
            if br.size:
                cb = node[br]
                go_left = scaled[br, forest.feature[cb]] <= forest.threshold[cb]
                node[br] = np.where(go_left, forest.left[cb], forest.right[cb])
    return np.argmax(scores, axis=1).astype(np.int64), scores


def _leaves_lockstep(forest, scaled, tree_ids):
    """Leaf node id reached by every row in every listed tree: (len(tree_ids), n_rows)."""
    n = scaled.shape[0]
    node = np.repeat(forest.tree_offsets[tree_ids].reshape(-1, 1), n, axis=1)
    active = ~forest.is_leaf[node]
    while active.any():
        t_idx, r_idx = np.nonzero(active)
        cur = node[t_idx, r_idx]
        go_left = scaled[r_idx, forest.feature[cur]] <= forest.threshold[cur]
        nxt = np.where(go_left, forest.left[cur], forest.right[cur])
        node[t_idx, r_idx] = nxt
        active[t_idx, r_idx] = ~forest.is_leaf[nxt]
    return node


# (tree, row) pairs advanced per lockstep block. Whole-forest lockstep builds a
# trees x rows matrix that falls out of cache at large batches; blocks of about
# this many pairs measured fastest across batch sizes 1..2000 on real flows.
PAIRS_PER_BLOCK = 8192


def _sum_over_trees(forest, scaled, tree_ids, values):
    """sum_t values[leaf_t(row)] for every row, advancing cache-sized blocks of trees in lockstep."""
    n = scaled.shape[0]
    block = max(1, PAIRS_PER_BLOCK // max(n, 1))
    total = np.zeros((n,) + values.shape[1:], dtype=values.dtype)
    for a in range(0, len(tree_ids), block):
        leaves = _leaves_lockstep(forest, scaled, tree_ids[a:a + block])
        total += values[leaves].sum(axis=0)
    return total


def predict_lockstep(forest, matrix):
    """Trees advance together: O(depth) Python iterations per block instead of O(depth) per tree."""
    scaled = scale_input(forest, matrix)
    scores = _sum_over_trees(forest, scaled, np.arange(forest.n_trees), forest.leaf_scores)
    return np.argmax(scores, axis=1).astype(np.int64), scores


def margin_bounds(forest):
    """Per-tree [min, max] of (score_attack - score_benign) over that tree's leaves."""
    lo = np.empty(forest.n_trees)
    hi = np.empty(forest.n_trees)
    diff = (forest.leaf_scores[:, 1] - forest.leaf_scores[:, 0]).astype(np.float64)
    for t in range(forest.n_trees):
        a, b = forest.tree_offsets[t], forest.tree_offsets[t + 1]
        d = diff[a:b][forest.is_leaf[a:b]]
        lo[t], hi[t] = d.min(), d.max()
    return lo, hi


def predict_early_exit(forest, matrix, schedule, order=None, bounds=None):
    """Exact early exit for a binary forest.

    Trees are evaluated in chunks (lockstep within a chunk). After each chunk a
    row stops as soon as the trees not yet evaluated can no longer flip its
    label, so labels are identical to the full forest by construction.
    Returns (labels, trees_used_per_row).
    """
    lo, hi = bounds if bounds is not None else margin_bounds(forest)
    order = np.arange(forest.n_trees) if order is None else np.asarray(order)
    scaled = scale_input(forest, matrix)
    n = scaled.shape[0]
    diff = (forest.leaf_scores[:, 1] - forest.leaf_scores[:, 0]).astype(np.float64)

    margin = np.zeros(n, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=np.int64)
    alive = np.arange(n)
    rem_lo, rem_hi = lo[order].sum(), hi[order].sum()

    pos = 0
    for size in schedule:
        if alive.size == 0 or pos >= len(order):
            break
        chunk = order[pos:pos + size]
        pos += len(chunk)
        margin[alive] += _sum_over_trees(forest, scaled[alive], chunk, diff)
        used[alive] += len(chunk)
        rem_lo -= lo[chunk].sum()
        rem_hi -= hi[chunk].sum()

        m = margin[alive]
        sure_attack = m + rem_lo > 0
        sure_benign = m + rem_hi <= 0
        labels[alive[sure_attack]] = 1
        alive = alive[~(sure_attack | sure_benign)]

    if alive.size:
        labels[alive] = (margin[alive] > 0).astype(np.int64)
    return labels, used
