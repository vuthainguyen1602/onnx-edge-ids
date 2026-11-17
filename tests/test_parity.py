"""Label differences at an exact score tie are indeterminate only when --tie-band asks for it."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_edge_ids.parity import p_class1, split_label_disagreements  # noqa: E402

# row 0: agree; row 1: tie, labels differ; row 2: far from a tie, labels differ;
# row 3: one side at a tie, the other not
PRED_A = np.array([1, 1, 1, 1])
CONF_A = np.array([0.9, 0.5000002, 0.8, 0.5000002])
PRED_B = np.array([1, 0, 0, 0])
CONF_B = np.array([0.9, 0.5000001, 0.7, 0.6])


def test_p_class1_from_label_and_confidence():
    assert np.allclose(p_class1(PRED_B, CONF_B), [0.9, 0.4999999, 0.3, 0.4])


def test_default_band_fails_every_label_difference():
    fail, tie = split_label_disagreements(PRED_A, CONF_A, PRED_B, CONF_B)
    assert fail.tolist() == [False, True, True, True]
    assert not tie.any()


def test_band_sets_apart_only_rows_where_both_runtimes_tie():
    fail, tie = split_label_disagreements(PRED_A, CONF_A, PRED_B, CONF_B, band=1e-6)
    assert tie.tolist() == [False, True, False, False]
    assert fail.tolist() == [False, False, True, True]
