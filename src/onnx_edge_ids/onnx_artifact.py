#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read a Scaler + TreeEnsembleClassifier ONNX graph into flat arrays.

This is the single parser shared by the NumPy exporter, the boundary probes and
the release gate's structure check, so the probes are built from the graph
ONNX Runtime serves, not from the exporter's output.

Supported: an optional ``ai.onnx.ml.Scaler`` in front of one
``TreeEnsembleClassifier`` with ``BRANCH_LEQ`` or ``BRANCH_LT`` splits (the
modes scikit-learn, LightGBM and XGBoost exports use), one or two score
columns, ``base_values``, ``nodes_missing_value_tracks_true`` (a NaN input goes
to the true branch where the flag is set, as ONNX Runtime routes it), and the
NONE / LOGISTIC / SOFTMAX post-transforms. Any other split mode raises, and so
does any other operator in the graph, except value-preserving output wrappers
(Identity, a widening Cast, Mul by 1, an ArgMax label), so an unsupported graph
fails loudly instead of being half-read.
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


# Output-side wrappers converters add around the tree node; none changes a value
# the engines compare. Cast targets: FLOAT, INT32, INT64, DOUBLE (exact widenings
# of a float32 probability, or integer labels); a narrowing cast such as FLOAT16 is refused.
_PASS_THROUGH_CASTS = {1, 6, 7, 11}


def _check_graph_form(model, ens_node, scaler_node):
    """Refuse any graph that is not [Scaler ->] TreeEnsembleClassifier plus value-preserving outputs.

    The exporter reads only the Scaler and the tree node, so an operator anywhere
    else that changes a value (an Imputer or encoder in front, a calibration step
    behind) would be silently left out of the NumPy artifact and the probes.
    """
    from onnx import numpy_helper

    graph = model.graph
    consts = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for n in graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
    inputs = [i.name for i in graph.input if i.name not in consts]
    if len(inputs) != 1:
        raise NotImplementedError(f"[ERR] Expected one graph input, found {len(inputs)}: {inputs}")
    if scaler_node is not None:
        if list(scaler_node.input) != inputs:
            raise NotImplementedError("[ERR] Scaler must read the graph input directly")
        front = scaler_node.output[0]
    else:
        front = inputs[0]
    if list(ens_node.input) != [front]:
        raise NotImplementedError(
            f"[ERR] TreeEnsembleClassifier must read {'the Scaler output' if scaler_node is not None else 'the graph input'}"
            f", reads {list(ens_node.input)}")

    producer = {o: n for n in graph.node for o in n.output}
    label_src, prob_src = ens_node.output[0], ens_node.output[1] if len(ens_node.output) > 1 else None

    def source(name, allow_argmax):
        """Follow value-preserving wrappers back to the tree node's output (or ArgMax of its probabilities)."""
        while name not in (label_src, prob_src):
            node = producer.get(name)
            if node is None:
                raise NotImplementedError(f"[ERR] Graph output depends on '{name}', not on the tree ensemble")
            op = node.op_type
            attrs = {a.name: a for a in node.attribute}
            if op == "Identity":
                name = node.input[0]
            elif op == "Cast" and "to" in attrs and attrs["to"].i in _PASS_THROUGH_CASTS:
                name = node.input[0]
            elif op == "Mul" and len(node.input) == 2:
                data = [i for i in node.input if i not in consts]
                ones = [i for i in node.input if i in consts]
                if len(data) != 1 or len(ones) != 1 or not np.all(consts[ones[0]] == 1):
                    raise NotImplementedError(f"[ERR] Unsupported Mul node '{node.name or node.output[0]}' (only by 1)")
                name = data[0]
            elif op == "ArgMax" and allow_argmax:
                axis = attrs["axis"].i if "axis" in attrs else 0
                if axis not in (1, -1) or source(node.input[0], False) != prob_src:
                    raise NotImplementedError("[ERR] ArgMax label must take axis 1 of the probabilities")
                return "argmax"
            else:
                raise NotImplementedError(
                    f"[ERR] Unsupported operator {node.domain or 'ai.onnx'}:{op} ('{node.name or node.output[0]}'); "
                    "only [Scaler ->] TreeEnsembleClassifier with Identity, widening Cast, Mul by 1 "
                    "or an ArgMax label is supported")
        return name

    outputs = [o.name for o in graph.output]
    if len(outputs) < 2 or prob_src is None:
        raise NotImplementedError("[ERR] Graph must output a label and a probability tensor")
    if source(outputs[0], True) not in (label_src, "argmax"):
        raise NotImplementedError("[ERR] First graph output must be the tree ensemble's label")
    if source(outputs[1], False) != prob_src:
        raise NotImplementedError("[ERR] Second graph output must be the tree ensemble's probabilities")
    allowed = {id(ens_node)} | ({id(scaler_node)} if scaler_node is not None else set())
    for n in graph.node:
        if id(n) in allowed or n.op_type in ("Identity", "Cast", "Mul", "ArgMax", "Constant"):
            continue
        raise NotImplementedError(f"[ERR] Unsupported operator {n.domain or 'ai.onnx'}:{n.op_type} "
                                  f"('{n.name or n.output[0]}')")


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
    ens_node = _one(model.graph, "TreeEnsembleClassifier")
    scaler = _one(model.graph, "Scaler", required=False)
    _check_graph_form(model, ens_node, scaler)
    ens = _attrs(ens_node)

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
    flags = ens.get("nodes_missing_value_tracks_true", np.zeros(n, dtype=np.int64))
    if len(flags) != n:
        flags = np.zeros(n, dtype=np.int64)  # attribute absent (protobuf drops an empty list)
    missing_true = (flags[order] != 0) & ~is_leaf
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
        "missing_true": missing_true,
        "leaf_scores": leaf_scores,
        "base_values": base,
        "post_transform": post,
        "class_labels": labels,
        "n_features": n_features,
        "doc_feature_columns": feature_columns_from_doc(model),
    }


ARRAY_KEYS = ("offset", "scale", "tree_offsets", "left", "right", "feature", "threshold",
              "is_leaf", "strict", "missing_true", "leaf_scores", "base_values")


def check_npz_matches_onnx(npz_path, onnx_path):
    """Differences between an exported NumPy artifact and the ONNX graph it claims to mirror.

    Floats are compared for exact equality after casting both sides to float32,
    the precision ONNX Runtime and the NumPy engine serve in. The graph stores
    float32, so a float64 copy that round-trips to the graph's float32 counts as
    equal, which is right because the engine casts it the same way before use;
    any value that rounds to a different float32 is a mismatch. For the same
    reason the artifact must declare float32 inference: an engine serving the
    same values in float64 is the precision fault this gate exists to catch.
    An empty list means the artifact is structurally the graph: same tree
    layout, split features, thresholds, split modes, missing-value routing, leaf
    weights, scaler, base values and serving precision. It does not check arithmetic; that is the
    probes' job.
    """
    ref = read_onnx_forest(onnx_path)
    data = np.load(str(npz_path), allow_pickle=False)
    meta = json.loads(str(data["metadata_json"]))
    got = {k: data[k] for k in data.files}
    got.setdefault("strict", np.zeros(len(got["left"]), dtype=bool))
    got.setdefault("base_values", np.zeros(got["leaf_scores"].shape[1]))
    got.setdefault("missing_true", np.zeros(len(got["left"]), dtype=bool))
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
    dtype = meta.get("inference_dtype", "float64 (engine default)")
    if dtype != "float32":
        problems.append(f"inference_dtype: {dtype}, the graph is served in float32")
    post = meta.get("post_transform", "NONE")
    if post != ref["post_transform"]:
        problems.append(f"post_transform: {post} != graph {ref['post_transform']}")
    doc_cols = ref["doc_feature_columns"]
    if doc_cols is not None and meta.get("feature_columns") not in (None, doc_cols):
        problems.append("feature_columns: order differs from the graph's embedded list")
    return problems
