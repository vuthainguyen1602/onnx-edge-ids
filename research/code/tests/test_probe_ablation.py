"""exp2b helpers, on the shipped artifact alone (no dataset needed)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import ARTIFACTS
from edgeids.forest import load_forest
from experiments.exp2b_probe_ablation import pair_table, walk_marking_pair

F32 = load_forest(os.path.join(ARTIFACTS, "ids_rf_numpy.npz"), "float32")
ROWS = np.random.default_rng(0).lognormal(3.0, 3.0, (50, len(F32.offset))).astype(np.float32)


def test_pair_table_matches_the_paper_and_round_trips():
    pair_feat, pair_thr, node_pair, n_internal = pair_table(F32)
    assert n_internal == 95741
    assert len(pair_feat) == 815
    internal = np.flatnonzero(~F32.is_leaf)
    assert (node_pair[internal] >= 0).all()
    assert (node_pair[F32.is_leaf] == -1).all()
    # every internal node's (feature, threshold) reconstructs bit-exactly from its pair id
    assert (pair_feat[node_pair[internal]] == F32.feature[internal]).all()
    assert (pair_thr[node_pair[internal]].view(np.int32) == F32.threshold[internal].view(np.int32)).all()


def test_walk_marks_a_pair_every_row_must_visit():
    _, _, node_pair, _ = pair_table(F32)
    root = int(F32.tree_offsets[0])  # tree 0's root: on every row's path
    exercised = np.zeros(len(F32.left), dtype=bool)
    hit = walk_marking_pair(F32, ROWS, int(node_pair[root]), node_pair, exercised)
    assert hit.all()
    assert exercised[root]
    # only nodes carrying the target pair may be marked
    assert (node_pair[np.flatnonzero(exercised)] == node_pair[root]).all()
