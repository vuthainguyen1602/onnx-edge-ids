"""Graph-level fault injection (exp43) on the shipped ONNX graph: python -m pytest research/code/tests"""

import os
import tempfile

import numpy as np

from edgeids.data import ARTIFACTS
from edgeids.graph_faults import Graph, graph_mutants, past_crossing
from edgeids.onnx_forest import load_onnx_forest
from edgeids.runtime import ort_probs

GRAPH = Graph(os.path.join(ARTIFACTS, "ids_rf.onnx"))
ROWS = np.random.default_rng(0).lognormal(3.0, 3.0, (300, GRAPH.forest.n_features)).astype(np.float32)


def test_unchanged_write_is_identical():
    out = GRAPH.write(os.path.join(tempfile.mkdtemp(), "same.onnx"), values=GRAPH.values(), features=GRAPH.features(),
                      modes=GRAPH.modes(), weights=GRAPH.weights())
    assert np.array_equal(ort_probs(GRAPH.path, ROWS), ort_probs(out, ROWS))


def test_write_changes_exactly_the_named_node():
    node = int(GRAPH.internal[5])
    v = GRAPH.values(); v[node] = np.nextafter(v[node], np.float32(np.inf))
    out = load_onnx_forest(GRAPH.write(os.path.join(tempfile.mkdtemp(), "one.onnx"), values=v))
    changed = np.flatnonzero(out.threshold != GRAPH.forest.threshold)
    assert changed.tolist() == [node]


def test_past_crossing_moves_one_float32_input():
    from onnx_edge_ids.boundary_probe import crossing_values, load_forest_arrays
    node = int(GRAPH.internal[7])
    moved = past_crossing(GRAPH, [node])[node]
    f = load_forest_arrays(GRAPH.path, np.float64)
    g = f.feature[node]
    start = np.array([np.float32(f.threshold[node] / f.scale[g] + f.offset[g])])
    _, last_true = crossing_values(f, np.array([node]), start)
    first_false = np.nextafter(last_true[0], np.float32(np.inf))
    z = np.float32((first_false - np.float32(f.offset[g])) * np.float32(f.scale[g]))
    assert z > np.float32(f.threshold[node])           # took the false branch before the fault
    assert z <= np.float32(moved)                      # takes the true branch after it (BRANCH_LEQ)


def test_mutant_set_is_labelled():
    muts = graph_mutants(GRAPH, single=3, coupled=2, multi_sizes=(2,), multi_count=2)
    assert {"thr_all_up", "thr_one_up", "feature_swap", "leaf_tree0", "mode_flipped"} <= set(muts)
    assert sum(k.startswith("single_node/") for k in muts) == 3
    assert sum(k.startswith("multi_2/") for k in muts) == 2
    assert all(len(faulty) == 2 for k, (_, faulty, _) in muts.items() if k.startswith("multi_2/"))
