"""Runs on the shipped artifact alone (no dataset needed): python -m pytest lncs/code/tests"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import ARTIFACTS
from edgeids.forest import (load_forest, margin_bounds, predict_early_exit,
                            predict_lockstep, predict_per_tree)

FOREST = load_forest(os.path.join(ARTIFACTS, "ids_rf_numpy.npz"))
ROWS = np.random.default_rng(0).lognormal(3.0, 3.0, (300, len(FOREST.offset))).astype(np.float32)


def test_lockstep_matches_per_tree():
    a, sa = predict_per_tree(FOREST, ROWS)
    b, sb = predict_lockstep(FOREST, ROWS)
    assert np.array_equal(a, b)
    assert np.abs(sa - sb).max() < 1e-5


def test_lockstep_is_batch_size_invariant():
    whole, _ = predict_lockstep(FOREST, ROWS)
    single = np.concatenate([predict_lockstep(FOREST, ROWS[i:i + 1])[0] for i in range(len(ROWS))])
    assert np.array_equal(whole, single)


def test_early_exit_is_exact():
    first = FOREST.n_trees // 2 + 4
    schedule = [first] + [8] * ((FOREST.n_trees - first + 7) // 8)
    labels, used = predict_early_exit(FOREST, ROWS, schedule, bounds=margin_bounds(FOREST))
    assert np.array_equal(labels, predict_lockstep(FOREST, ROWS)[0])
    assert used.min() >= first and used.max() <= FOREST.n_trees


def test_no_exit_before_half_the_trees_can_be_exact():
    labels, used = predict_early_exit(FOREST, ROWS, [10] * 20)
    assert np.array_equal(labels, predict_lockstep(FOREST, ROWS)[0])
    assert used.min() > FOREST.n_trees // 2


def test_matches_the_packaged_numpy_engine():
    """The paper's traversal and the one the package ships must not drift apart."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "src")
    sys.path.insert(0, src)
    from onnx_edge_ids import create_inference_engine

    engine = create_inference_engine("numpy", os.path.join(ARTIFACTS, "ids_rf_numpy.npz"))
    assert np.array_equal(engine.predict_batch(ROWS).predictions, predict_lockstep(FOREST, ROWS)[0])
