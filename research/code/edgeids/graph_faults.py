"""Faults injected into an exported ONNX tree-ensemble graph itself (converter faults, exp43).

``faults.py`` mutates the NumPy artifact exported from a correct graph, i.e. a second
runtime. Here the graph is what is wrong, as a faulty converter would write it, so a
checker that compares the training model with its export (Leafparity) and the
data-free probes run against the training model can be compared on the same faults.

Node ids are the global ids of ``onnx_forest.load_onnx_forest`` (sorted by tree, then
node); ``Graph.order[i]`` is the attribute index of global node ``i``.
"""

import copy

import numpy as np
import onnx

from edgeids.onnx_forest import load_onnx_forest


class Graph:
    """A TreeEnsembleClassifier graph, its arrays and a way to write mutated copies."""

    def __init__(self, path):
        self.path = path
        self.model = onnx.load(path)
        self.forest = load_onnx_forest(path)
        ens = [n for n in self.model.graph.node if n.op_type == "TreeEnsembleClassifier"][0]
        self._ens_index = list(self.model.graph.node).index(ens)
        a = {x.name: x for x in ens.attribute}
        self.tree_ids = np.array(a["nodes_treeids"].ints)
        self.node_ids = np.array(a["nodes_nodeids"].ints)
        self.order = np.lexsort((self.node_ids, self.tree_ids))
        lookup = {(int(t), int(n)): i for i, (t, n) in enumerate(zip(self.tree_ids[self.order], self.node_ids[self.order]))}
        self.class_global = np.array([lookup[(int(t), int(n))] for t, n in zip(a["class_treeids"].ints, a["class_nodeids"].ints)])
        self.class_ids = np.array(a["class_ids"].ints)
        self.internal = np.flatnonzero(~self.forest.is_leaf)
        self.leaves = np.flatnonzero(self.forest.is_leaf)

    def write(self, out, values=None, features=None, modes=None, weights=None, base=None):
        """Copy of the graph with any of the per-node arrays (global order) or class weights replaced."""
        m = copy.deepcopy(self.model)
        ens = m.graph.node[self._ens_index]
        a = {x.name: x for x in ens.attribute}
        if values is not None:
            v = np.array(a["nodes_values"].floats, dtype=np.float32)
            v[self.order] = np.asarray(values, dtype=np.float32)
            del a["nodes_values"].floats[:]; a["nodes_values"].floats.extend(v.tolist())
        if features is not None:
            f = np.array(a["nodes_featureids"].ints)
            f[self.order] = np.asarray(features)
            del a["nodes_featureids"].ints[:]; a["nodes_featureids"].ints.extend(int(x) for x in f)
        if modes is not None:
            md = np.array(list(a["nodes_modes"].strings), dtype=object)
            md[self.order] = np.asarray(modes, dtype=object)
            del a["nodes_modes"].strings[:]; a["nodes_modes"].strings.extend(md.tolist())
        if weights is not None:
            del a["class_weights"].floats[:]; a["class_weights"].floats.extend(np.asarray(weights, dtype=np.float32).tolist())
        if base is not None:
            del a["base_values"].floats[:]; a["base_values"].floats.extend(np.asarray(base, dtype=np.float32).tolist())
        onnx.save_model(m, out)
        return out

    # per-node views in global order
    def values(self):
        return self.forest.threshold.copy()

    def features(self):
        return self.forest.feature.copy()

    def modes(self):
        ens = [n for n in self.model.graph.node if n.op_type == "TreeEnsembleClassifier"][0]
        md = [x for x in ens.attribute if x.name == "nodes_modes"][0].strings
        return np.array(list(md), dtype=object)[self.order]

    def weights(self):
        ens = [n for n in self.model.graph.node if n.op_type == "TreeEnsembleClassifier"][0]
        return np.array([x for x in ens.attribute if x.name == "class_weights"][0].floats, dtype=np.float32)

    def base_values(self):
        ens = [n for n in self.model.graph.node if n.op_type == "TreeEnsembleClassifier"][0]
        b = [x for x in ens.attribute if x.name == "base_values"]
        return np.array(b[0].floats, dtype=np.float32) if b else np.zeros(0, dtype=np.float32)


def _up(v):
    return np.nextafter(np.float32(v), np.float32(np.inf))


def past_crossing(g, nodes):
    """For each node, the threshold that sends the first raw float32 past ONNX Runtime's crossing to the
    true branch (one node's decision moved by one float32 input). Nodes without a crossing are dropped."""
    from onnx_edge_ids.boundary_probe import crossing_values, load_forest_arrays
    f = load_forest_arrays(g.path, np.float64)
    nodes = np.asarray(nodes)
    feat = f.feature[nodes]
    start = (f.threshold[nodes] / f.scale[feat] + f.offset[feat]).astype(np.float32)
    ok, last_true = crossing_values(f, nodes, start)
    first_false = np.nextafter(last_true, np.float32(np.inf))
    z = ((first_false - f.offset[feat].astype(np.float32)) * f.scale[feat].astype(np.float32)).astype(np.float32)
    out = {}
    for v, zv, good in zip(nodes, z, ok):
        if good:
            out[int(v)] = float(_up(zv)) if f.strict[v] else float(zv)
    return out


def _leaf_rows(g, leaf):
    return np.flatnonzero(g.class_global == leaf)


def swap_leaf(g, weights, leaf):
    """One leaf's class weights swapped (two columns) or negated (one column); None if that changes nothing."""
    rows = _leaf_rows(g, leaf)
    w = weights.copy()
    if len(rows) == 2:
        w[rows] = weights[rows[::-1]]
    else:
        w[rows] = -weights[rows]
    return None if np.array_equal(w, weights) else w


def element_fault(g, kind, node, rng, state):
    """Apply one single-element fault of ``kind`` at ``node`` to ``state`` (dict of arrays); False if inapplicable."""
    if kind == "threshold":
        t = past_crossing(g, [node])
        if node not in t:
            return False
        state["values"][node] = t[node]
    elif kind == "feature":
        other = int(rng.integers(0, g.forest.n_features - 1))
        state["features"][node] = other if other < state["features"][node] else other + 1
    elif kind == "leaf":
        w = swap_leaf(g, state["weights"], node)
        if w is None:
            return False
        state["weights"] = w
    return True


def graph_mutants(g, seed=0, single=20, coupled=10, multi_sizes=(2, 5, 10), multi_count=10):
    """name -> (kwargs for Graph.write, faulty global nodes, fault class) for the converter-fault benchmark."""
    rng = np.random.default_rng(seed)
    internal, leaves = g.internal, g.leaves
    vals, feats, mds, w = g.values(), g.features(), g.modes(), g.weights()
    out = {}

    # whole-graph and one-element exporter-style faults (as exp25, minus the Scaler: these graphs have none)
    v = vals.copy(); v[internal] = np.nextafter(vals[internal], np.float32(np.inf)); out["thr_all_up"] = ({"values": v}, list(internal), "single-switch, every node")
    v = vals.copy(); v[internal[0]] = _up(vals[internal[0]]); out["thr_one_up"] = ({"values": v}, [int(internal[0])], "single-switch, one node")
    used = np.bincount(feats[internal]).argsort()[::-1]
    fa, fb = int(used[0]), int(used[1])
    f = feats.copy(); f[feats == fa], f[feats == fb] = fb, fa
    out["feature_swap"] = ({"features": f}, [int(i) for i in internal if feats[i] in (fa, fb)], "feature mapping, many nodes")
    t0 = [int(i) for i in leaves if g.forest.tree_offsets[0] <= i < g.forest.tree_offsets[1]]
    ww = w.copy()
    for leaf in t0:
        s = swap_leaf(g, ww, leaf)
        ww = s if s is not None else ww
    out["leaf_tree0"] = ({"weights": ww}, t0, "leaf weights, one tree")
    m = mds.copy(); m[internal] = np.where(mds[internal] == b"BRANCH_LT", b"BRANCH_LEQ", b"BRANCH_LT")
    out["mode_flipped"] = ({"modes": m}, list(internal), "single-switch, every node")
    b = g.base_values()
    if b.size and np.any(b != 0):
        out["base_dropped"] = ({"base": np.zeros_like(b)}, [], "base value")

    # single-node threshold faults: each moves one node's decision by one float32 input
    nodes = np.sort(rng.choice(internal, single * 2, replace=False))
    moved = past_crossing(g, nodes)
    for node in list(moved)[:single]:
        v = vals.copy(); v[node] = moved[node]
        out[f"single_node/node_{node}"] = ({"values": v}, [node], "single-switch, one node")

    # coupled one-element faults: one leaf's weights, or one node's split feature
    for leaf in np.sort(rng.choice(leaves, coupled * 3, replace=False)):
        s = swap_leaf(g, w, int(leaf))
        if s is not None and sum(k.startswith("coupled/leaf_") for k in out) < coupled:
            out[f"coupled/leaf_{int(leaf)}"] = ({"weights": s}, [int(leaf)], "leaf weight, one leaf")
    for node in np.sort(rng.choice(internal, coupled, replace=False)):
        st = {"features": feats.copy()}
        element_fault(g, "feature", int(node), rng, st)
        out[f"coupled/feature_{int(node)}"] = ({"features": st["features"]}, [int(node)], "feature mapping, one node")

    # multi-node faults: k elements at once, kinds drawn at random (threshold, feature, leaf)
    for k in multi_sizes:
        made = 0
        while made < multi_count:
            st = {"values": vals.copy(), "features": feats.copy(), "weights": w.copy()}
            faulty, kinds = [], []
            while len(faulty) < k:
                kind = rng.choice(["threshold", "feature", "leaf"])
                node = int(rng.choice(leaves if kind == "leaf" else internal))
                if node in faulty:
                    continue
                if element_fault(g, kind, node, rng, st):
                    faulty.append(node); kinds.append(str(kind))
            kw = {key: st[key] for key in ("values", "features", "weights")}
            out[f"multi_{k}/draw_{made}"] = (kw, faulty, "multi-node: " + ",".join(sorted(set(kinds))))
            made += 1
    return out
