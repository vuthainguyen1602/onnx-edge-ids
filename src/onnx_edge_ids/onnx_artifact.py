#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read a Scaler + TreeEnsembleClassifier ONNX graph into flat arrays.

This is the single parser shared by the NumPy exporter, the boundary probes and
the release gate's structure check, so the probes are built from the graph
ONNX Runtime serves, not from the exporter's output.

Supported: an optional ``ai.onnx.ml.Scaler`` in front of one
``TreeEnsembleClassifier`` with ``BRANCH_LEQ`` or ``BRANCH_LT`` splits (the
modes scikit-learn, LightGBM and XGBoost exports use), one or two score
columns, ``base_values``, and the NONE / LOGISTIC / SOFTMAX post-transforms.
Any other split mode raises, so an unsupported graph fails loudly.
"""

import json

import numpy as np

SPLIT_MODES = {"BRANCH_LEQ": False, "BRANCH_LT": True}  # mode -> strict comparison
POST_TRANSFORMS = ("NONE", "LOGISTIC", "SOFTMAX")


def _attrs(node):
    out = {}
    for a in node.attribute:
        if a.floats:
            out[a.name] = np.asarray(a.floats, dtype=np.float32)
        elif a.ints:
            out[a.name] = np.asarray(a.ints, dtype=np.int64)
        elif a.strings:
            out[a.name] = [s.decode("utf-8") for s in a.strings]
        elif a.s:
            out[a.name] = a.s.decode("utf-8")
    return out


def _one(graph, op_type, required=True):
    nodes = [n for n in graph.node if n.op_type == op_type]
    if len(nodes) > 1 or (required and not nodes):
        raise ValueError(f"[ERR] Expected {'exactly' if required else 'at most'} one {op_type} node, found {len(nodes)}")
    return nodes[0] if nodes else None


def _input_dim(model):
    shape = model.graph.input[0].type.tensor_type.shape.dim
    if len(shape) < 2 or not shape[1].dim_value:
        raise ValueError("[ERR] ONNX input must have a fixed feature dimension")
    return int(shape[1].dim_value)


def feature_columns_from_doc(model):
    if not model.doc_string:
        return None
    try:
        metadata = json.loads(model.doc_string)
    except json.JSONDecodeError:
        return None
    columns = metadata.get("feature_columns")
    if isinstance(columns, list) and all(isinstance(c, str) for c in columns):
        return columns
    return None


def read_onnx_forest(model_path):
    """Flat float64 arrays of the graph, nodes in sorted (tree, node) order.

    Thresholds, scaler parameters and leaf weights are the graph's float32
    values widened exactly to float64.
    """
    import onnx

    model = onnx.load(str(model_path))
    n_features = _input_dim(model)
    ens = _attrs(_one(model.graph, "TreeEnsembleClassifier"))
    scaler = _one(model.graph, "Scaler", required=False)

    modes = ens["nodes_modes"]
    bad = sorted({m for m in modes if m != "LEAF" and m not in SPLIT_MODES})
    if bad:
        raise NotImplementedError(f"[ERR] Unsupported tree node mode(s): {bad}")
    post = ens.get("post_transform", "NONE")
    if post not in POST_TRANSFORMS:
        raise NotImplementedError(f"[ERR] Unsupported post_transform: {post}")

    tree_ids, node_ids = ens["nodes_treeids"], ens["nodes_nodeids"]
    order = np.lexsort((node_ids, tree_ids))
    tree_s, node_s = tree_ids[order], node_ids[order]
    lookup = {(int(t), int(n)): i for i, (t, n) in enumerate(zip(tree_s, node_s))}
    n = len(order)
    mode_s = [modes[i] for i in order]
    is_leaf = np.array([m == "LEAF" for m in mode_s], dtype=bool)
    strict = np.array([SPLIT_MODES.get(m, False) for m in mode_s], dtype=bool)
    left = np.full(n, -1, dtype=np.int64)
    right = np.full(n, -1, dtype=np.int64)
    tn, fn = ens["nodes_truenodeids"][order], ens["nodes_falsenodeids"][order]
    for i in np.flatnonzero(~is_leaf):
        t = int(tree_s[i])
        left[i] = lookup[(t, int(tn[i]))]
        right[i] = lookup[(t, int(fn[i]))]
    feature = np.where(is_leaf, -1, ens["nodes_featureids"][order]).astype(np.int64)
    threshold = np.where(is_leaf, 0.0, ens["nodes_values"][order].astype(np.float64))
    trees = np.unique(tree_s)
    tree_offsets = np.append(np.searchsorted(tree_s, trees, side="left"), n).astype(np.int64)

    class_ids = ens["class_ids"]
    n_targets = int(class_ids.max()) + 1
    leaf_scores = np.zeros((n, n_targets), dtype=np.float64)
    rows = np.array([lookup[(int(t), int(m))] for t, m in zip(ens["class_treeids"], ens["class_nodeids"])],
                    dtype=np.int64)
    np.add.at(leaf_scores, (rows, class_ids), ens["class_weights"].astype(np.float64))
    base = np.asarray(ens.get("base_values", np.zeros(n_targets)), dtype=np.float64)
    if base.size == 0:
        base = np.zeros(n_targets)

    if "classlabels_int64s" in ens:
        labels = np.asarray(ens["classlabels_int64s"], dtype=np.int64)
    else:
        labels = np.arange(len(ens["classlabels_strings"]), dtype=np.int64)

    if scaler is not None:
        s = _attrs(scaler)
        offset = np.broadcast_to(s["offset"].astype(np.float64), (n_features,)).copy()
        scale = np.broadcast_to(s["scale"].astype(np.float64), (n_features,)).copy()
    else:
        offset, scale = np.zeros(n_features), np.ones(n_features)

    return {
        "offset": offset,
        "scale": scale,
        "tree_offsets": tree_offsets,
        "left": left,
        "right": right,
        "feature": feature,
        "threshold": threshold,
        "is_leaf": is_leaf,
        "strict": strict,
        "leaf_scores": leaf_scores,
        "base_values": base,
        "post_transform": post,
        "class_labels": labels,
        "n_features": n_features,
        "doc_feature_columns": feature_columns_from_doc(model),
    }


ARRAY_KEYS = ("offset", "scale", "tree_offsets", "left", "right", "feature", "threshold",
              "is_leaf", "strict", "leaf_scores", "base_values")


def check_npz_matches_onnx(npz_path, onnx_path):
    """Differences between an exported NumPy artifact and the ONNX graph it claims to mirror.

    Every float is compared at float32, the precision both artifacts are served
    in. An empty list means the artifact is structurally the graph: same tree
    layout, split features, thresholds, split modes, leaf weights, scaler and
    base values. It does not check arithmetic; that is the probes' job.
    """
    ref = read_onnx_forest(onnx_path)
    data = np.load(str(npz_path), allow_pickle=False)
    meta = json.loads(str(data["metadata_json"]))
    got = {k: data[k] for k in data.files}
    got.setdefault("strict", np.zeros(len(got["left"]), dtype=bool))
    got.setdefault("base_values", np.zeros(got["leaf_scores"].shape[1]))
    problems = []
    for key in ARRAY_KEYS:
        a, b = np.asarray(got[key]), np.asarray(ref[key])
        if a.shape != b.shape:
            problems.append(f"{key}: shape {a.shape} != graph {b.shape}")
            continue
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            diff = int(np.count_nonzero(a.astype(np.float32) != b.astype(np.float32)))
        else:
            diff = int(np.count_nonzero(a != b))
        if diff:
            problems.append(f"{key}: {diff} value(s) differ from the graph")
    post = meta.get("post_transform", "NONE")
    if post != ref["post_transform"]:
        problems.append(f"post_transform: {post} != graph {ref['post_transform']}")
    doc_cols = ref["doc_feature_columns"]
    if doc_cols is not None and meta.get("feature_columns") not in (None, doc_cols):
        problems.append("feature_columns: order differs from the graph's embedded list")
    return problems
