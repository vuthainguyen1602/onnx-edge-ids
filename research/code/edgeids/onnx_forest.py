"""Generic NumPy traversal of an ONNX (Scaler +) TreeEnsembleClassifier graph.

Unlike ``forest.py``, which reads the project's flattened ``.npz`` artifact,
this module reads any ONNX tree ensemble directly: every split mode the
operator defines, an optional ``ai.onnx.ml.Scaler`` in front of it, base
values, and the NONE / LOGISTIC / SOFTMAX post-transforms. It exists so the
boundary-directed parity test can be run on forests from any training
framework, not only on the deployed forest's export.
"""

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper

_MODES = {b"BRANCH_LEQ": 0, b"BRANCH_LT": 1, b"BRANCH_GTE": 2, b"BRANCH_GT": 3, b"BRANCH_EQ": 4, b"BRANCH_NEQ": 5, b"LEAF": 6}
LEAF = 6


@dataclass
class OnnxForest:
    offset: np.ndarray        # Scaler offset per feature (0 when no scaler)
    scale: np.ndarray         # Scaler scale per feature (1 when no scaler)
    tree_offsets: np.ndarray  # first node id of each tree, plus total
    left: np.ndarray          # true-branch child (global node id)
    right: np.ndarray         # false-branch child
    feature: np.ndarray
    threshold: np.ndarray     # float32 as stored in the graph
    mode: np.ndarray          # _MODES code per node
    leaf_scores: np.ndarray   # (n_nodes, n_targets), zero for internal nodes
    base_values: np.ndarray   # (n_targets,)
    post_transform: str
    class_labels: np.ndarray
    n_features: int

    @property
    def n_trees(self):
        return len(self.tree_offsets) - 1

    @property
    def is_leaf(self):
        return self.mode == LEAF

    @property
    def n_targets(self):
        return int(self.leaf_scores.shape[1])


def _attrs(node):
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.FLOATS:
            out[a.name] = np.array(a.floats, dtype=np.float32)
        elif a.type == onnx.AttributeProto.INTS:
            out[a.name] = np.array(a.ints, dtype=np.int64)
        elif a.type == onnx.AttributeProto.STRINGS:
            out[a.name] = list(a.strings)
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode()
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = float(a.f)
        elif a.type == onnx.AttributeProto.INT:
            out[a.name] = int(a.i)
        elif a.type == onnx.AttributeProto.TENSOR:
            out[a.name] = numpy_helper.to_array(a.t)
    return out


def load_onnx_forest(path):
    model = onnx.load(path)
    graph = model.graph
    n_features = None
    for inp in graph.input:
        dims = inp.type.tensor_type.shape.dim
        if len(dims) == 2 and dims[1].dim_value > 0:
            n_features = dims[1].dim_value
    scaler = [n for n in graph.node if n.op_type == "Scaler"]
    ens = [n for n in graph.node if n.op_type == "TreeEnsembleClassifier"]
    if len(ens) != 1:
        raise ValueError(f"expected one TreeEnsembleClassifier, found {len(ens)}")
    if len(scaler) > 1:
        raise ValueError("more than one Scaler node")
    a = _attrs(ens[0])
    if any(m not in _MODES for m in a["nodes_modes"]):
        raise ValueError("unknown split mode in graph")

    tree_ids = a["nodes_treeids"]
    node_ids = a["nodes_nodeids"]
    trees = np.unique(tree_ids)
    # global node id = position in the sorted (tree, node) order
    order = np.lexsort((node_ids, tree_ids))
    tree_s, node_s = tree_ids[order], node_ids[order]
    lookup = {(int(t), int(n)): i for i, (t, n) in enumerate(zip(tree_s, node_s))}
    n = len(order)
    mode = np.array([_MODES[m] for m in a["nodes_modes"]], dtype=np.int64)[order]
    left = np.full(n, -1, dtype=np.int64)
    right = np.full(n, -1, dtype=np.int64)
    tn, fn = a["nodes_truenodeids"][order], a["nodes_falsenodeids"][order]
    for i in np.flatnonzero(mode != LEAF):
        t = int(tree_s[i])
        left[i] = lookup[(t, int(tn[i]))]
        right[i] = lookup[(t, int(fn[i]))]
    feature = a["nodes_featureids"][order]
    threshold = a["nodes_values"][order].astype(np.float32)
    tree_offsets = np.searchsorted(tree_s, trees, side="left")
    tree_offsets = np.append(tree_offsets, n).astype(np.int64)

    if "classlabels_int64s" in a and len(a["classlabels_int64s"]):
        labels = np.asarray(a["classlabels_int64s"], dtype=np.int64)
    else:
        labels = np.arange(len(a["classlabels_strings"]))
    n_targets = int(a["class_ids"].max()) + 1
    leaf_scores = np.zeros((n, n_targets), dtype=np.float32)
    ct, cn, cid, cw = a["class_treeids"], a["class_nodeids"], a["class_ids"], a["class_weights"]
    gi = np.array([lookup[(int(t), int(m))] for t, m in zip(ct, cn)], dtype=np.int64)
    np.add.at(leaf_scores, (gi, cid), cw)
    base = np.asarray(a.get("base_values", np.zeros(n_targets, dtype=np.float32)), dtype=np.float32)
    if base.size == 0:
        base = np.zeros(n_targets, dtype=np.float32)

    if n_features is None:
        n_features = int(feature.max()) + 1
    if scaler:
        s = _attrs(scaler[0])
        offset = np.asarray(s["offset"], dtype=np.float32)
        scale = np.asarray(s["scale"], dtype=np.float32)
        if offset.size == 1:
            offset = np.repeat(offset, n_features)
        if scale.size == 1:
            scale = np.repeat(scale, n_features)
    else:
        offset = np.zeros(n_features, dtype=np.float32)
        scale = np.ones(n_features, dtype=np.float32)

    return OnnxForest(
        offset=offset, scale=scale, tree_offsets=tree_offsets, left=left, right=right,
        feature=feature, threshold=threshold, mode=mode, leaf_scores=leaf_scores,
        base_values=base, post_transform=a.get("post_transform", "NONE"),
        class_labels=labels, n_features=int(n_features),
    )


def scale_input(forest, matrix, dtype):
    """(x - offset) * scale in the requested dtype, exactly as ai.onnx.ml.Scaler defines it."""
    x = np.ascontiguousarray(matrix, dtype=dtype)
    return (x - forest.offset.astype(dtype)) * forest.scale.astype(dtype)


def _go_true(mode, x, t):
    out = np.empty(len(x), dtype=bool)
    for code, op in ((0, np.less_equal), (1, np.less), (2, np.greater_equal), (3, np.greater), (4, np.equal), (5, np.not_equal)):
        m = mode == code
        if m.any():
            out[m] = op(x[m], t[m])
    return out


def raw_scores(forest, matrix, dtype):
    """Sum of leaf weights (plus base values) per row, traversing in ``dtype``."""
    scaled = scale_input(forest, matrix, dtype)
    thr = forest.threshold.astype(dtype)
    n = scaled.shape[0]
    # accumulate in the traversal dtype too, so a float32 traversal sums like ONNX Runtime does
    scores = np.zeros((n, forest.n_targets), dtype=dtype)
    leaf_scores = forest.leaf_scores.astype(dtype)
    is_leaf = forest.is_leaf
    for t in range(forest.n_trees):
        node = np.full(n, forest.tree_offsets[t], dtype=np.int64)
        active = ~is_leaf[node]
        while active.any():
            rows = np.flatnonzero(active)
            cur = node[rows]
            go = _go_true(forest.mode[cur], scaled[rows, forest.feature[cur]], thr[cur])
            nxt = np.where(go, forest.left[cur], forest.right[cur])
            node[rows] = nxt
            active[rows] = ~is_leaf[nxt]
        scores += leaf_scores[node]
    return (scores + forest.base_values.astype(dtype)).astype(np.float64)


def labels_from_scores(forest, scores):
    """Label the way ONNX Runtime does: post-transform, expand a single binary score, argmax."""
    s = scores
    if forest.n_targets == 1 and len(forest.class_labels) == 2:
        if forest.post_transform == "LOGISTIC":
            p1 = 1.0 / (1.0 + np.exp(-s[:, 0]))
        else:
            p1 = s[:, 0]
        # ORT's binary case: [1 - p, p] when weights are probabilities/logits, else sign of the score.
        idx = (p1 > 0.5).astype(np.int64) if forest.post_transform in ("LOGISTIC", "PROBIT") else (p1 > 0.5).astype(np.int64)
        return forest.class_labels[idx]
    if forest.post_transform == "SOFTMAX":
        z = s - s.max(axis=1, keepdims=True)
        s = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    elif forest.post_transform == "LOGISTIC":
        s = 1.0 / (1.0 + np.exp(-s))
    return forest.class_labels[np.argmax(s, axis=1)]


def predict(forest, matrix, dtype, batch=50000):
    out = []
    for i in range(0, len(matrix), batch):
        out.append(labels_from_scores(forest, raw_scores(forest, matrix[i:i + batch], dtype)))
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def parents(forest):
    parent = np.full(len(forest.left), -1, dtype=np.int64)
    went_true = np.zeros(len(forest.left), dtype=bool)
    internal = np.flatnonzero(~forest.is_leaf)
    parent[forest.left[internal]] = internal
    went_true[forest.left[internal]] = True
    parent[forest.right[internal]] = internal
    return parent, went_true


def path_rows(forest, targets, base):
    """One row per target node whose (scaled) features satisfy every split on the way to it.

    Intervals are intersected in float64 scaled space and mapped back through
    the scaler; a value strictly inside an interval is chosen so that the
    inclusive/exclusive distinction between LEQ and LT does not matter.
    """
    parent, went_true = parents(forest)
    scale = forest.scale.astype(np.float64)
    offset = forest.offset.astype(np.float64)
    rows = base.astype(np.float64).copy()
    for i, node in enumerate(targets):
        lo, hi = {}, {}
        cur = node
        while parent[cur] >= 0:
            p = parent[cur]
            g, t, m = int(forest.feature[p]), float(forest.threshold[p]), int(forest.mode[p])
            true_is_low = m in (0, 1)   # LEQ / LT: the true branch is x below t
            below = went_true[cur] == true_is_low
            if below:
                hi[g] = min(hi.get(g, np.inf), t)
            else:
                lo[g] = max(lo.get(g, -np.inf), t)
            cur = p
        for g in set(lo) | set(hi):
            a, b = lo.get(g, -np.inf), hi.get(g, np.inf)
            if np.isfinite(a) and np.isfinite(b):
                s = (a + b) / 2.0
            elif np.isfinite(b):
                s = b - 1.0
            else:
                s = a + 1.0
            rows[i, g] = s / scale[g] + offset[g] if scale[g] != 0 else offset[g]
    return rows.astype(np.float32)


def reaches(forest, rows, targets, dtype=np.float32):
    """Whether row i passes through node targets[i] under this forest's arithmetic in ``dtype``."""
    scaled = scale_input(forest, rows, dtype)
    thr = forest.threshold.astype(dtype)
    tree = np.searchsorted(forest.tree_offsets, targets, side="right") - 1
    cur = forest.tree_offsets[tree].copy()
    idx = np.arange(len(targets))
    hit = cur == targets
    active = ~forest.is_leaf[cur] & ~hit
    while active.any():
        a = np.flatnonzero(active)
        c = cur[a]
        go = _go_true(forest.mode[c], scaled[idx[a], forest.feature[c]], thr[c])
        cur[a] = np.where(go, forest.left[c], forest.right[c])
        hit[a] = cur[a] == targets[a]
        active[a] = ~forest.is_leaf[cur[a]] & ~hit[a]
    return hit


def threshold_sharing(forest):
    """How many distinct (feature, threshold) pairs the splits use and how many trees share each."""
    internal = np.flatnonzero(~forest.is_leaf)
    key = np.stack([forest.feature[internal], forest.threshold[internal].view(np.int32)])
    pairs, inverse, counts = np.unique(key, axis=1, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    tree_of = np.searchsorted(forest.tree_offsets, internal, side="right") - 1
    tp = np.unique(np.stack([inverse, tree_of]), axis=1)
    trees_per_pair = np.bincount(tp[0], minlength=pairs.shape[1])
    return {
        "internal_nodes": int(len(internal)),
        "distinct_feature_threshold_pairs": int(pairs.shape[1]),
        "nodes_per_pair_mean": float(counts.mean()),
        "trees_per_pair_mean": float(trees_per_pair.mean()),
        "trees_per_pair_max": int(trees_per_pair.max()),
        "share_of_nodes_whose_pair_is_in_2plus_trees": float(np.mean(trees_per_pair[inverse] >= 2)),
        "share_of_nodes_whose_pair_is_in_half_the_trees": float(np.mean(trees_per_pair[inverse] >= forest.n_trees / 2)),
    }


def _leaf_of_tree(forest, scaled, thr, t):
    n = scaled.shape[0]
    node = np.full(n, forest.tree_offsets[t], dtype=np.int64)
    is_leaf = forest.is_leaf
    active = ~is_leaf[node]
    while active.any():
        rows = np.flatnonzero(active)
        cur = node[rows]
        go = _go_true(forest.mode[cur], scaled[rows, forest.feature[cur]], thr[cur])
        nxt = np.where(go, forest.left[cur], forest.right[cur])
        node[rows] = nxt
        active[rows] = ~is_leaf[nxt]
    return node


def trees_moved(forest, matrix, dtype_a=np.float32, dtype_b=np.float64):
    """Per row, how many trees reach a different leaf under the two traversal dtypes."""
    sa, sb = scale_input(forest, matrix, dtype_a), scale_input(forest, matrix, dtype_b)
    ta, tb = forest.threshold.astype(dtype_a), forest.threshold.astype(dtype_b)
    moved = np.zeros(matrix.shape[0], dtype=np.int64)
    for t in range(forest.n_trees):
        moved += _leaf_of_tree(forest, sa, ta, t) != _leaf_of_tree(forest, sb, tb, t)
    return moved


def rows_on_a_threshold(forest, matrix, dtype=np.float32):
    """Share of rows with at least one feature value exactly equal, after scaling in ``dtype``,
    to some split threshold of that feature anywhere in the forest."""
    scaled = scale_input(forest, matrix, dtype)
    internal = np.flatnonzero(~forest.is_leaf)
    hit = np.zeros(matrix.shape[0], dtype=bool)
    for f in np.unique(forest.feature[internal]):
        thr = np.unique(forest.threshold[internal][forest.feature[internal] == f]).astype(dtype)
        hit |= np.isin(scaled[:, f], thr)
    return float(hit.mean())
