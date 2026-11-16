"""BRANCH_LT graphs, single-column logistic output and the NumPy-vs-ONNX structure check.

The graph is built by hand, so the test needs neither a dataset nor a training
framework: two trees on two features, every split BRANCH_LT, a base value and a
LOGISTIC post-transform, as an XGBoost export has them.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_edge_ids import create_inference_engine  # noqa: E402
from onnx_edge_ids.boundary_probe import build_probes, probes_reaching_node  # noqa: E402
from onnx_edge_ids.onnx_artifact import check_npz_matches_onnx, read_onnx_forest  # noqa: E402

onnx = pytest.importorskip("onnx")


def _graph(path, mode="BRANCH_LT"):
    from onnx import TensorProto, helper

    # tree 0: x0 < 0.5 ? (x1 < -1.0 ? -2.0 : 0.5) : 1.5 ; tree 1: x1 < 0.25 ? -0.75 : 0.75
    nodes = dict(
        nodes_treeids=[0, 0, 0, 0, 0, 1, 1, 1],
        nodes_nodeids=[0, 1, 2, 3, 4, 0, 1, 2],
        nodes_featureids=[0, 1, 0, 0, 0, 1, 0, 0],
        nodes_values=[0.5, -1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
        nodes_modes=[mode, mode, "LEAF", "LEAF", "LEAF", mode, "LEAF", "LEAF"],
        nodes_truenodeids=[1, 3, 0, 0, 0, 1, 0, 0],
        nodes_falsenodeids=[2, 4, 0, 0, 0, 2, 0, 0],
        nodes_missing_value_tracks_true=[0] * 8,
        class_treeids=[0, 0, 0, 1, 1],
        class_nodeids=[2, 3, 4, 1, 2],
        class_ids=[0, 0, 0, 0, 0],
        class_weights=[1.5, -2.0, 0.5, -0.75, 0.75],
    )
    scaler = helper.make_node("Scaler", ["X"], ["Z"], domain="ai.onnx.ml",
                              offset=[0.1, -0.3], scale=[3.0, 0.7])
    ens = helper.make_node("TreeEnsembleClassifier", ["Z"], ["label", "probabilities"], domain="ai.onnx.ml",
                           classlabels_int64s=[0, 1], post_transform="LOGISTIC", base_values=[0.125], **nodes)
    g = helper.make_graph(
        [scaler, ens], "lt",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
         helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 3)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return path


def _export(onnx_path, npz_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_numpy.py"),
                    "--model", str(onnx_path), "--out", str(npz_path)], check=True, capture_output=True)
    return npz_path


def test_reader_keeps_split_mode_and_base_value(tmp_path):
    g = read_onnx_forest(_graph(tmp_path / "lt.onnx"))
    assert g["strict"][~g["is_leaf"]].all()
    assert g["post_transform"] == "LOGISTIC"
    assert g["leaf_scores"].shape[1] == 1 and g["base_values"].tolist() == [0.125]


def test_unsupported_split_mode_raises(tmp_path):
    with pytest.raises(NotImplementedError):
        read_onnx_forest(_graph(tmp_path / "gte.onnx", mode="BRANCH_GTE"))


def test_numpy_engine_matches_onnx_runtime_on_branch_lt_probes(tmp_path):
    pytest.importorskip("onnxruntime")
    graph = _graph(tmp_path / "lt.onnx")
    npz = _export(graph, tmp_path / "lt.npz")
    assert check_npz_matches_onnx(npz, graph) == []
    probes = build_probes(graph)
    assert probes_reaching_node(graph, probes, np.float32) == 1.0
    a = create_inference_engine("onnx", str(graph)).predict_batch(probes.rows)
    b = create_inference_engine("numpy", str(npz)).predict_batch(probes.rows)
    assert (a.predictions == b.predictions).all()
    assert float(np.abs(a.confidences - b.confidences).max()) <= 1e-6


def test_structure_check_and_probes_catch_a_dropped_split_mode(tmp_path):
    """An exporter that writes BRANCH_LT splits as BRANCH_LEQ: the structure check names it,
    and the on-threshold probes, built from the graph, see the routing change."""
    pytest.importorskip("onnxruntime")
    graph = _graph(tmp_path / "lt.onnx")
    npz = _export(graph, tmp_path / "lt.npz")
    data = dict(np.load(npz, allow_pickle=False))
    data["strict"] = np.zeros_like(data["strict"])
    bad = tmp_path / "lt_as_leq.npz"
    np.savez(bad, **data)
    assert any(p.startswith("strict") for p in check_npz_matches_onnx(bad, graph))
    probes = build_probes(graph)
    a = create_inference_engine("onnx", str(graph)).predict_batch(probes.rows)
    b = create_inference_engine("numpy", str(bad)).predict_batch(probes.rows)
    assert float(np.abs(a.confidences - b.confidences).max()) > 1e-5


def test_shipped_artifact_matches_its_graph():
    assert check_npz_matches_onnx(ROOT / "artifacts" / "ids_rf_numpy.npz", ROOT / "artifacts" / "ids_rf.onnx") == []


def test_crossing_probes_straddle_and_catch_a_one_step_threshold_shift(tmp_path):
    """Moving one threshold by a single float32 step: the exact-crossing probes straddle every
    split in float32 and see it; the engine on the correct artifact still agrees with ONNX Runtime."""
    pytest.importorskip("onnxruntime")
    from onnx_edge_ids.boundary_probe import ALL_VARIANTS, CROSSING_VARIANTS

    graph = _graph(tmp_path / "lt.onnx")
    npz = _export(graph, tmp_path / "lt.npz")
    probes = build_probes(graph, ALL_VARIANTS)
    assert len(probes) == 5 * probes.internal_nodes
    ort = create_inference_engine("onnx", str(graph)).predict_batch(probes.rows)
    ok = create_inference_engine("numpy", str(npz)).predict_batch(probes.rows)
    assert (ort.predictions == ok.predictions).all()
    assert float(np.abs(ort.confidences - ok.confidences).max()) <= 1e-6

    data = dict(np.load(npz, allow_pickle=False))
    node = int(np.flatnonzero((~data["is_leaf"]) & (data["feature"] == 1))[0])
    data["threshold"][node] = float(np.nextafter(np.float32(data["threshold"][node]), np.float32(np.inf)))
    bad = tmp_path / "shifted.npz"
    np.savez(bad, **data)
    cross = build_probes(graph, CROSSING_VARIANTS)
    a = create_inference_engine("onnx", str(graph)).predict_batch(cross.rows)
    b = create_inference_engine("numpy", str(bad)).predict_batch(cross.rows)
    assert float(np.abs(a.confidences - b.confidences).max()) > 1e-5
