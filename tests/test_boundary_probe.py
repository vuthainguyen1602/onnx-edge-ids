"""Boundary probes are built from the shipped artifact alone; no dataset needed."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_edge_ids.boundary_probe import VARIANTS, build_probes, probes_reaching_node  # noqa: E402
from onnx_edge_ids import create_inference_engine  # noqa: E402

NPZ = ROOT / "artifacts" / "ids_rf_numpy.npz"
ONNX = ROOT / "artifacts" / "ids_rf.onnx"


@pytest.fixture(scope="module")
def probes():
    return build_probes(NPZ)


def test_one_probe_per_internal_node_and_variant(probes):
    assert len(probes) == probes.internal_nodes * len(VARIANTS)
    assert probes.rows.dtype == np.float32
    assert np.isfinite(probes.rows).all()
    assert probes.distinct_pairs < probes.internal_nodes


def test_probes_reach_their_node_under_float32(probes):
    # a few paths are narrower than float32 can represent after inverse scaling
    assert probes_reaching_node(NPZ, probes, np.float32) > 0.99
    assert probes_reaching_node(NPZ, probes, np.float64) == 1.0


def test_numpy_engine_agrees_with_onnx_on_every_probe(probes):
    pytest.importorskip("onnxruntime")
    onnx = create_inference_engine("onnx", str(ONNX))
    numpy_engine = create_inference_engine("numpy", str(NPZ))
    po, pn, co, cn = [], [], [], []
    for i in range(0, len(probes), 20000):
        chunk = probes.rows[i:i + 20000]
        a, b = onnx.predict_batch(chunk), numpy_engine.predict_batch(chunk)
        po.append(a.predictions); pn.append(b.predictions)
        co.append(a.confidences); cn.append(b.confidences)
    po, pn = np.concatenate(po), np.concatenate(pn)
    assert int((po != pn).sum()) == 0
    assert float(np.abs(np.concatenate(co) - np.concatenate(cn)).max()) <= 1e-5
