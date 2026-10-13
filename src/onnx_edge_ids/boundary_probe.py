#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Boundary-directed probes for the ONNX-vs-NumPy parity gate.

Two tree runtimes fed the same row can only disagree at a split whose scaled
feature value sits within rounding distance of the threshold. Replayed traffic
almost never lands there, so a replay-based parity check passes even when the
fallback runtime scales or compares in a different precision. This module
builds one probe per internal node of the forest, without any dataset:

1. walk from the node to the root and intersect the interval each ancestor
   split imposes on each feature, then write a value from inside every
   interval into the row, so the row is routed to that node;
2. overwrite the node's own split feature with the raw float32 value that the
   scaler maps onto the threshold, and with its float32 predecessor and
   successor.

Features the path does not constrain take the scaler offset (scaled value 0).
The construction runs in float64 so the row lands where intended; whether a
runtime then agrees with ONNX Runtime at that boundary is what the gate
measures.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VARIANTS = ("on_threshold", "one_ulp_below", "one_ulp_above")


@dataclass(frozen=True)
class ForestArrays:
    offset: np.ndarray
    scale: np.ndarray
    tree_offsets: np.ndarray
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    is_leaf: np.ndarray
    n_features: int


@dataclass(frozen=True)
class ProbeSet:
    rows: np.ndarray        # (n_probes, n_features) raw float32, engine input
    node: np.ndarray        # (n_probes,) global node id each probe targets
    variant: np.ndarray     # (n_probes,) index into VARIANTS
    internal_nodes: int
    distinct_pairs: int     # distinct (feature, threshold) pairs in the forest

    def __len__(self):
        return int(self.rows.shape[0])


def load_forest_arrays(npz_path, dtype=np.float64):
    data = np.load(Path(npz_path), allow_pickle=False)
    meta = json.loads(str(data["metadata_json"]))
    return ForestArrays(
        offset=data["offset"].astype(dtype),
        scale=data["scale"].astype(dtype),
        tree_offsets=data["tree_offsets"].astype(np.int64),
        left=data["left"].astype(np.int64),
        right=data["right"].astype(np.int64),
        feature=data["feature"].astype(np.int64),
        threshold=data["threshold"].astype(dtype),
        is_leaf=data["is_leaf"].astype(bool),
        n_features=int(meta.get("n_features", len(data["offset"]))),
    )


def _parents(forest):
    parent = np.full(len(forest.left), -1, dtype=np.int64)
    went_left = np.zeros(len(forest.left), dtype=bool)
    internal = np.flatnonzero(~forest.is_leaf)
    parent[forest.left[internal]] = internal
    went_left[forest.left[internal]] = True
    parent[forest.right[internal]] = internal
    return parent, went_left


def _path_rows(forest, targets):
    """One raw row per target node that satisfies every split on its root path."""
    parent, went_left = _parents(forest)
    scaled = np.zeros((len(targets), forest.n_features), dtype=np.float64)
    for i, node in enumerate(targets):
        lo, hi = {}, {}
        cur = int(node)
        while parent[cur] >= 0:
            p = int(parent[cur])
            g, t = int(forest.feature[p]), float(forest.threshold[p])
            if went_left[cur]:
                hi[g] = min(hi.get(g, np.inf), t)   # went left: value <= t
            else:
                lo[g] = max(lo.get(g, -np.inf), t)  # went right: value > t
            cur = p
        for g in set(lo) | set(hi):
            a, b = lo.get(g, -np.inf), hi.get(g, np.inf)
            if np.isfinite(a) and np.isfinite(b):
                scaled[i, g] = (a + b) / 2.0
            elif np.isfinite(b):
                scaled[i, g] = b - 1.0
            else:
                scaled[i, g] = a + 1.0
    # invert scaled = (raw - offset) * scale
    raw = scaled / forest.scale + forest.offset
    return raw.astype(np.float32)


def build_probes(npz_path, variants=VARIANTS):
    """Build the boundary probe set for the NumPy artifact at ``npz_path``."""
    forest = load_forest_arrays(npz_path, np.float64)
    internal = np.flatnonzero(~forest.is_leaf)
    feat = forest.feature[internal]
    thr = forest.threshold[internal]
    pairs = np.unique(np.stack([feat, thr.astype(np.float32).view(np.int32)]), axis=1).shape[1]

    base = _path_rows(forest, internal)
    on = (thr / forest.scale[feat] + forest.offset[feat]).astype(np.float32)
    values = {
        "on_threshold": on,
        "one_ulp_below": np.nextafter(on, np.float32(-np.inf)),
        "one_ulp_above": np.nextafter(on, np.float32(np.inf)),
    }
    rows, node, variant = [], [], []
    for k, name in enumerate(VARIANTS):
        if name not in variants:
            continue
        block = base.copy()
        block[np.arange(len(internal)), feat] = values[name]
        rows.append(block)
        node.append(internal)
        variant.append(np.full(len(internal), k, dtype=np.int64))
    return ProbeSet(
        rows=np.concatenate(rows),
        node=np.concatenate(node),
        variant=np.concatenate(variant),
        internal_nodes=int(len(internal)),
        distinct_pairs=int(pairs),
    )


def probes_reaching_node(npz_path, probes, dtype=np.float32):
    """Fraction of probes that pass through their target node under ``dtype`` arithmetic."""
    forest = load_forest_arrays(npz_path, dtype)
    x = probes.rows.astype(dtype)
    scaled = (x - forest.offset) * forest.scale
    targets = probes.node
    tree = np.searchsorted(forest.tree_offsets, targets, side="right") - 1
    cur = forest.tree_offsets[tree].copy()
    idx = np.arange(len(targets))
    hit = cur == targets
    active = ~forest.is_leaf[cur] & ~hit
    while active.any():
        a = np.flatnonzero(active)
        c = cur[a]
        go_left = scaled[idx[a], forest.feature[c]] <= forest.threshold[c]
        cur[a] = np.where(go_left, forest.left[c], forest.right[c])
        hit[a] = cur[a] == targets[a]
        active[a] = ~forest.is_leaf[cur[a]] & ~hit[a]
    return float(hit.mean()) if len(hit) else 0.0
