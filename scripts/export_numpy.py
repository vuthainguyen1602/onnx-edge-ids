#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export an ONNX RandomForest artifact to the compact NumPy runtime format.

The exported ``.npz`` contains:
  * fitted scaler offset/scale,
  * flattened RandomForest nodes,
  * per-leaf class scores,
  * feature-column metadata for order checks.

Usage:
    python scripts/export_numpy.py \
        --model artifacts/ids_rf.onnx \
        --out   artifacts/ids_rf_numpy.npz \
        --features artifacts/feature_columns.json
"""

import argparse
import json
import os

import numpy as np


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _attr_map(node):
    return {attr.name: attr for attr in node.attribute}


def _floats(attr):
    return np.asarray(attr.floats, dtype=np.float64)


def _ints(attr):
    return np.asarray(attr.ints, dtype=np.int64)


def _strings(attr):
    return [value.decode("utf-8") for value in attr.strings]


def _find_node(graph, op_type):
    matches = [node for node in graph.node if node.op_type == op_type]
    if len(matches) != 1:
        raise ValueError(f"[ERR] Expected exactly one {op_type} node, found {len(matches)}")
    return matches[0]


def _input_dim(model):
    shape = model.graph.input[0].type.tensor_type.shape.dim
    if len(shape) < 2 or not shape[1].dim_value:
        raise ValueError("[ERR] ONNX input must have a fixed feature dimension")
    return int(shape[1].dim_value)


def _feature_columns_from_doc(model):
    if not model.doc_string:
        return None
    try:
        metadata = json.loads(model.doc_string)
    except json.JSONDecodeError:
        return None
    columns = metadata.get("feature_columns")
    if isinstance(columns, list) and all(isinstance(col, str) for col in columns):
        return columns
    return None


def load_onnx_arrays(model_path, feature_columns):
    import onnx

    model = onnx.load(model_path)
    n_features = _input_dim(model)

    scaler_attrs = _attr_map(_find_node(model.graph, "Scaler"))
    offset = _floats(scaler_attrs["offset"])
    scale = _floats(scaler_attrs["scale"])
    if len(offset) != n_features or len(scale) != n_features:
        raise ValueError(
            f"[ERR] Scaler dim mismatch: input={n_features}, "
            f"offset={len(offset)}, scale={len(scale)}"
        )

    forest_attrs = _attr_map(_find_node(model.graph, "TreeEnsembleClassifier"))
    treeids = _ints(forest_attrs["nodes_treeids"])
    nodeids = _ints(forest_attrs["nodes_nodeids"])
    featureids = _ints(forest_attrs["nodes_featureids"])
    values = _floats(forest_attrs["nodes_values"])
    modes = _strings(forest_attrs["nodes_modes"])
    truenodeids = _ints(forest_attrs["nodes_truenodeids"])
    falsenodeids = _ints(forest_attrs["nodes_falsenodeids"])

    class_labels = _ints(forest_attrs["classlabels_int64s"])
    class_index = {int(label): i for i, label in enumerate(class_labels)}
    n_classes = len(class_labels)

    pairs = [(int(tid), int(nid)) for tid, nid in zip(treeids, nodeids)]
    global_index = {pair: i for i, pair in enumerate(sorted(pairs))}
    nodes_by_tree = {}
    for tid, nid in pairs:
        nodes_by_tree.setdefault(tid, []).append(nid)

    left = []
    right = []
    feature = []
    threshold = []
    is_leaf = []
    leaf_scores = np.zeros((len(pairs), n_classes), dtype=np.float64)
    tree_offsets = [0]

    row_by_pair = {pair: i for i, pair in enumerate(pairs)}
    for tid in sorted(nodes_by_tree):
        for nid in sorted(nodes_by_tree[tid]):
            src = row_by_pair[(tid, nid)]
            mode = modes[src]
            is_leaf_node = mode == "LEAF"

            is_leaf.append(is_leaf_node)
            if is_leaf_node:
                left.append(-1)
                right.append(-1)
                feature.append(-1)
                threshold.append(0.0)
            elif mode == "BRANCH_LEQ":
                left.append(global_index[(tid, int(truenodeids[src]))])
                right.append(global_index[(tid, int(falsenodeids[src]))])
                feature.append(int(featureids[src]))
                threshold.append(float(values[src]))
            else:
                raise NotImplementedError(f"[ERR] Unsupported tree node mode: {mode}")

        tree_offsets.append(len(left))

    class_treeids = _ints(forest_attrs["class_treeids"])
    class_nodeids = _ints(forest_attrs["class_nodeids"])
    class_ids = _ints(forest_attrs["class_ids"])
    class_weights = _floats(forest_attrs["class_weights"])
    for tid, nid, class_id, weight in zip(
        class_treeids, class_nodeids, class_ids, class_weights
    ):
        row = global_index[(int(tid), int(nid))]
        leaf_scores[row, class_index[int(class_id)]] = float(weight)

    doc_columns = _feature_columns_from_doc(model)
    if feature_columns is None:
        feature_columns = doc_columns or [f"feature_{i}" for i in range(n_features)]
    elif doc_columns is not None and list(feature_columns) != list(doc_columns):
        raise ValueError(
            "[ERR] ONNX feature order differs from feature_columns.json; "
            "the served matrix would be permuted.\n"
            f"  onnx[:5]={list(doc_columns)[:5]}\n"
            f"  file[:5]={list(feature_columns)[:5]}"
        )

    arrays = {
        "tree_offsets": np.asarray(tree_offsets, dtype=np.int64),
        "left": np.asarray(left, dtype=np.int64),
        "right": np.asarray(right, dtype=np.int64),
        "feature": np.asarray(feature, dtype=np.int64),
        "threshold": np.asarray(threshold, dtype=np.float64),
        "is_leaf": np.asarray(is_leaf, dtype=np.bool_),
        "leaf_scores": leaf_scores,
    }
    return offset, scale, arrays, feature_columns, n_features, n_classes


def numpy_predict(matrix, offset, scale, arrays):
    x = np.ascontiguousarray(matrix, dtype=np.float64)
    scaled = (x - offset) * scale
    n_classes = arrays["leaf_scores"].shape[1]
    scores = np.zeros((scaled.shape[0], n_classes), dtype=np.float64)

    tree_offsets = arrays["tree_offsets"]
    for t in range(len(tree_offsets) - 1):
        base = int(tree_offsets[t])
        node = np.full(scaled.shape[0], base, dtype=np.int64)
        active = np.ones(scaled.shape[0], dtype=bool)

        while np.any(active):
            rows = np.flatnonzero(active)
            cur = node[rows]
            leaf = arrays["is_leaf"][cur]
            if np.any(leaf):
                leaf_rows = rows[leaf]
                scores[leaf_rows] += arrays["leaf_scores"][node[leaf_rows]]
                active[leaf_rows] = False

            branch_rows = rows[~leaf]
            if branch_rows.size:
                cur_branch = node[branch_rows]
                feat = arrays["feature"][cur_branch]
                go_left = scaled[branch_rows, feat] <= arrays["threshold"][cur_branch]
                node[branch_rows] = np.where(
                    go_left, arrays["left"][cur_branch], arrays["right"][cur_branch]
                )

    return np.argmax(scores, axis=1).astype(np.int64), scores


def main():
    ap = argparse.ArgumentParser(description="ONNX RandomForest -> NumPy artifact")
    ap.add_argument("--model", required=True, help="Input .onnx model")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--features", default=None, help="feature_columns.json (order check)")
    args = ap.parse_args()

    feature_columns = _read_json(args.features) if args.features else None
    offset, scale, arrays, input_cols, n_features, n_classes = load_onnx_arrays(
        args.model, feature_columns
    )

    if len(input_cols) != n_features:
        raise ValueError(f"[ERR] feature columns={len(input_cols)} != model features={n_features}")

    metadata = {
        "producer": "ONNX-EdgeIDS/scripts/export_numpy.py",
        "source_model": os.path.abspath(args.model),
        "feature_columns": list(input_cols),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "n_trees": int(len(arrays["tree_offsets"]) - 1),
        "n_nodes": int(len(arrays["left"])),
        "inference_dtype": "float32",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        offset=offset.astype(np.float64),
        scale=scale.astype(np.float64),
        metadata_json=json.dumps(metadata),
        **arrays,
    )
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[OK] Feature order matches {args.features}" if args.features else "[OK] Loaded ONNX")
    print(f"[OK] Wrote {args.out} ({size_mb:.2f} MB)")
    print(f"  Trees: {metadata['n_trees']} | nodes: {metadata['n_nodes']}")


if __name__ == "__main__":
    main()
