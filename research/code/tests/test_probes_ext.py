"""Extended probes (every split mode, missing values) on the shipped graph: python -m pytest research/code/tests"""

import os
import tempfile

import numpy as np

from edgeids.data import ARTIFACTS
from edgeids.graph_faults import Graph
from edgeids.onnx_forest import load_onnx_forest, parents, reaches
from edgeids.probes_ext import KINDS, build
from edgeids.runtime import numpy_probs, ort_probs

GRAPH = os.path.join(ARTIFACTS, "ids_rf.onnx")
FOREST = load_onnx_forest(GRAPH)
NODES = np.flatnonzero(~FOREST.is_leaf)[:300]


def _agree(path, forest, rows):
    a, b = ort_probs(path, rows), numpy_probs(forest, rows, np.float32)
    return not np.any(((a[:, 1] > 0.5) != (b[:, 1] > 0.5)) | np.any(np.abs(a - b) > 1e-5, axis=1))


def test_monotone_and_nan_probes_reach_and_agree():
    pr = build(FOREST, NODES)
    assert set(np.unique(pr.kind)) == {KINDS.index("monotone_crossing"), KINDS.index("missing")}
    finite = ~np.isnan(pr.rows[np.arange(len(pr.node)), FOREST.feature[pr.node]])
    assert reaches(FOREST, pr.rows[finite], pr.node[finite]).mean() > 0.99
    assert _agree(GRAPH, FOREST, pr.rows)                # NaN included: no missing flags, false branch in both


def test_equality_split_probes():
    g = Graph(GRAPH)
    modes = g.modes()
    eq = NODES[:20]
    modes[eq] = b"BRANCH_EQ"
    path = g.write(os.path.join(tempfile.mkdtemp(), "eq.onnx"), modes=modes)
    f = load_onnx_forest(path)
    pr = build(f, eq)
    kinds = {KINDS[k] for k in np.unique(pr.kind)}
    assert "missing" in kinds and ({"equality_on", "equality_neighbour"} & kinds)
    assert _agree(path, f, pr.rows)                       # the NumPy traversal reads BRANCH_EQ as ORT does


def test_parents_are_consistent_with_children():
    parent, went_true = parents(FOREST)
    internal = np.flatnonzero(~FOREST.is_leaf)
    kids = np.flatnonzero(parent >= 0)
    assert np.all(np.where(went_true[kids], FOREST.left[parent[kids]], FOREST.right[parent[kids]]) == kids)
    assert len(kids) == 2 * len(internal)
