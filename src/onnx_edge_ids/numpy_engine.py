#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np

from .inference_engine import InferenceEngine


class NumpyInferenceEngine(InferenceEngine):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = Path(model_path)
        self.meta = {}
        self.offset = None
        self.scale = None
        self.tree_offsets = None
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.is_leaf = None
        self.leaf_scores = None
        self.n_classes = 0
        self.inference_dtype = np.float64
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"NumPy model not found: {self.model_path}")

        data = np.load(self.model_path, allow_pickle=False)
        self.meta = json.loads(str(data["metadata_json"]))
        dtype_name = self.meta.get("inference_dtype", "float64")
        if dtype_name not in {"float32", "float64"}:
            raise ValueError(f"Unsupported NumPy inference dtype: {dtype_name}")
        self.inference_dtype = np.dtype(dtype_name).type
        self.offset = data["offset"].astype(self.inference_dtype, copy=False)
        self.scale = data["scale"].astype(self.inference_dtype, copy=False)
        self.tree_offsets = data["tree_offsets"].astype(np.int64, copy=False)
        self.left = data["left"].astype(np.int64, copy=False)
        self.right = data["right"].astype(np.int64, copy=False)
        self.feature = data["feature"].astype(np.int64, copy=False)
        self.threshold = data["threshold"].astype(self.inference_dtype, copy=False)
        self.is_leaf = data["is_leaf"].astype(bool, copy=False)
        self.leaf_scores = data["leaf_scores"].astype(self.inference_dtype, copy=False)
        # Artifacts exported before BRANCH_LT / base-value support carry neither:
        # all splits are BRANCH_LEQ and the base values are zero.
        files = set(data.files)
        self.strict = (data["strict"].astype(bool, copy=False) if "strict" in files
                       else np.zeros(len(self.left), dtype=bool))
        self.any_strict = bool(self.strict.any())
        # Where nodes_missing_value_tracks_true is set, ONNX Runtime sends a NaN
        # input to the true (left) branch; elsewhere the comparison is false and
        # NaN goes right. Older artifacts carry no flags: NaN always goes right.
        self.missing_true = (data["missing_true"].astype(bool, copy=False) if "missing_true" in files
                             else np.zeros(len(self.left), dtype=bool))
        self.any_missing_true = bool(self.missing_true.any())
        n_targets = int(self.leaf_scores.shape[1])
        self.base_values = (data["base_values"].astype(self.inference_dtype, copy=False) if "base_values" in files
                            else np.zeros(n_targets, dtype=self.inference_dtype))
        self.post_transform = self.meta.get("post_transform", "NONE")
        if self.post_transform not in {"NONE", "LOGISTIC", "SOFTMAX"}:
            raise ValueError(f"Unsupported post_transform: {self.post_transform}")
        self.n_targets = n_targets
        self.n_classes = max(n_targets, 2)

    # (tree, row) pairs advanced per lockstep block. A whole-forest trees x rows
    # matrix falls out of cache at large batches and ends up slower than the
    # per-tree loop; blocks of about this many pairs measured fastest across
    # batch sizes 1..2000 on real CICIDS2017 flows.
    PAIRS_PER_BLOCK = 8192

    def _infer(self, matrix):
        """Advance cache-sized blocks of trees in lockstep instead of one tree at a time.

        Node ids are globally unique across the flattened forest (tree_offsets
        partitions the flat arrays), so a (block_trees, n_rows) matrix of global
        node ids can be gathered directly without per-tree index arithmetic.
        The per-tree version pays NumPy's per-call dispatch overhead once per
        tree per level, which dominates at the small batches a streaming
        classifier serves.
        """
        x = np.ascontiguousarray(matrix, dtype=self.inference_dtype)
        if x.ndim != 2 or x.shape[1] != len(self.offset):
            raise ValueError(f"Expected matrix shape (n, {len(self.offset)}), got {x.shape}")

        scaled = (x - self.offset) * self.scale
        n_rows = scaled.shape[0]
        n_trees = len(self.tree_offsets) - 1
        block = max(1, self.PAIRS_PER_BLOCK // max(n_rows, 1))
        scores = np.zeros((n_rows, self.n_targets), dtype=self.inference_dtype)

        for start in range(0, n_trees, block):
            roots = self.tree_offsets[start:min(start + block, n_trees)]
            node = np.repeat(roots.reshape(-1, 1), n_rows, axis=1)
            active = ~self.is_leaf[node]
            while np.any(active):
                t_idx, r_idx = np.nonzero(active)
                cur = node[t_idx, r_idx]
                value, thr = scaled[r_idx, self.feature[cur]], self.threshold[cur]
                if self.any_strict:
                    go_left = np.where(self.strict[cur], value < thr, value <= thr)
                else:
                    go_left = value <= thr
                if self.any_missing_true:
                    go_left |= self.missing_true[cur] & np.isnan(value)
                nxt = np.where(go_left, self.left[cur], self.right[cur])
                node[t_idx, r_idx] = nxt
                active[t_idx, r_idx] = ~self.is_leaf[nxt]
            scores += self.leaf_scores[node].sum(axis=0)

        return self._outputs(scores + self.base_values)

    def _outputs(self, scores):
        """Labels and per-class probabilities the way ONNX Runtime derives them from the scores."""
        with np.errstate(over="ignore"):  # exp(-s) overflows to inf for very negative logits: p -> 0, as in ONNX Runtime
            return self._post_transform(scores)

    def _post_transform(self, scores):
        dt = self.inference_dtype
        one, half = dt(1), dt(0.5)
        if self.n_targets == 1:
            s = scores[:, 0]
            p1 = one / (one + np.exp(-s)) if self.post_transform == "LOGISTIC" else s
            probs = np.stack([one - p1, p1], axis=1)
            return (p1 > half).astype(np.int64), probs
        if self.post_transform == "SOFTMAX":
            z = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs = z / z.sum(axis=1, keepdims=True)
        elif self.post_transform == "LOGISTIC":
            probs = one / (one + np.exp(-scores))
        else:
            probs = scores
        return np.argmax(probs, axis=1).astype(np.int64), probs
