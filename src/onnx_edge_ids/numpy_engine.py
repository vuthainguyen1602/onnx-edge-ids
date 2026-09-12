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
        self.n_classes = int(self.leaf_scores.shape[1])

    def _infer(self, matrix):
        x = np.ascontiguousarray(matrix, dtype=self.inference_dtype)
        if x.ndim != 2 or x.shape[1] != len(self.offset):
            raise ValueError(f"Expected matrix shape (n, {len(self.offset)}), got {x.shape}")

        scaled = (x - self.offset) * self.scale
        scores = np.zeros((scaled.shape[0], self.n_classes), dtype=self.inference_dtype)

        for t in range(len(self.tree_offsets) - 1):
            base = int(self.tree_offsets[t])
            end = int(self.tree_offsets[t + 1])
            node = np.full(scaled.shape[0], base, dtype=np.int64)
            active = np.ones(scaled.shape[0], dtype=bool)

            while np.any(active):
                rows = np.flatnonzero(active)
                cur = node[rows]
                leaf = self.is_leaf[cur]
                if np.any(leaf):
                    leaf_rows = rows[leaf]
                    scores[leaf_rows] += self.leaf_scores[node[leaf_rows]]
                    active[leaf_rows] = False

                branch_rows = rows[~leaf]
                if branch_rows.size:
                    cur_branch = node[branch_rows]
                    feat = self.feature[cur_branch]
                    go_left = scaled[branch_rows, feat] <= self.threshold[cur_branch]
                    node[branch_rows] = np.where(
                        go_left, self.left[cur_branch], self.right[cur_branch]
                    )

                if np.any((node < base) | (node >= end)):
                    raise RuntimeError("Tree traversal left the exported node range")

        preds = np.argmax(scores, axis=1).astype(np.int64)
        return preds, scores
