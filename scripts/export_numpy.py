#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export an ONNX tree-ensemble artifact to the compact NumPy runtime format.

The graph is read by ``onnx_edge_ids.onnx_artifact`` (the same parser the
release gate builds its probes from): an optional Scaler in front of one
TreeEnsembleClassifier with BRANCH_LEQ or BRANCH_LT splits, one or two score
columns, base values and the NONE / LOGISTIC / SOFTMAX post-transforms. Other
split modes raise.

The exported ``.npz`` contains:
  * fitted scaler offset/scale,
  * flattened tree nodes with their split mode (``strict``: BRANCH_LT),
  * per-leaf scores and base values,
  * feature-column and post-transform metadata.

Usage:
    python scripts/export_numpy.py \
        --model artifacts/ids_rf.onnx \
        --out   artifacts/ids_rf_numpy.npz \
        --features artifacts/feature_columns.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from onnx_edge_ids.onnx_artifact import read_onnx_forest  # noqa: E402


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="ONNX tree ensemble -> NumPy artifact")
    ap.add_argument("--model", required=True, help="Input .onnx model")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--features", default=None, help="feature_columns.json (order check)")
    args = ap.parse_args()

    g = read_onnx_forest(args.model)
    n_features = g["n_features"]
    doc_columns = g["doc_feature_columns"]
    feature_columns = _read_json(args.features) if args.features else None
    if feature_columns is None:
        feature_columns = doc_columns or [f"feature_{i}" for i in range(n_features)]
    elif doc_columns is not None and list(feature_columns) != list(doc_columns):
        raise ValueError(
            "[ERR] ONNX feature order differs from feature_columns.json; "
            "the served matrix would be permuted.\n"
            f"  onnx[:5]={list(doc_columns)[:5]}\n"
            f"  file[:5]={list(feature_columns)[:5]}"
        )
    if len(feature_columns) != n_features:
        raise ValueError(f"[ERR] feature columns={len(feature_columns)} != model features={n_features}")

    metadata = {
        "producer": "ONNX-EdgeIDS/scripts/export_numpy.py",
        "source_model": os.path.abspath(args.model),
        "feature_columns": list(feature_columns),
        "n_features": int(n_features),
        "n_classes": int(len(g["class_labels"])),
        "class_labels": [int(c) for c in g["class_labels"]],
        "n_targets": int(g["leaf_scores"].shape[1]),
        "post_transform": g["post_transform"],
        "n_trees": int(len(g["tree_offsets"]) - 1),
        "n_nodes": int(len(g["left"])),
        "inference_dtype": "float32",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        metadata_json=json.dumps(metadata),
        **{k: g[k] for k in ("offset", "scale", "tree_offsets", "left", "right", "feature",
                             "threshold", "is_leaf", "strict", "leaf_scores", "base_values")},
    )
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[OK] Feature order matches {args.features}" if args.features else "[OK] Loaded ONNX")
    print(f"[OK] Wrote {args.out} ({size_mb:.2f} MB)")
    modes = "BRANCH_LT" if g["strict"][~g["is_leaf"]].all() else (
        "BRANCH_LEQ" if not g["strict"].any() else "BRANCH_LEQ+BRANCH_LT")
    print(f"  Trees: {metadata['n_trees']} | nodes: {metadata['n_nodes']} | "
          f"split mode: {modes} | post_transform: {g['post_transform']}")


if __name__ == "__main__":
    main()
