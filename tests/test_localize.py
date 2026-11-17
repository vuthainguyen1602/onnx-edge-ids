"""A failing gate points at the fault: probe targets (L1) and first divergences artifact vs. graph (L2)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx_edge_ids import create_inference_engine  # noqa: E402
from onnx_edge_ids.boundary_probe import build_probes  # noqa: E402
from onnx_edge_ids.localize import first_divergences, summarize_failures  # noqa: E402
from test_split_modes import _export, _graph  # noqa: E402


def _failing(graph, npz, probes):
    a = create_inference_engine("onnx", str(graph)).predict_batch(probes.rows)
    b = create_inference_engine("numpy", str(npz)).predict_batch(probes.rows)
    return (a.predictions != b.predictions) | (np.abs(a.confidences - b.confidences) > 1e-5)


def _mutated(tmp_path, npz, name, change):
    data = dict(np.load(npz, allow_pickle=False))
    change(data)
    out = tmp_path / f"{name}.npz"
    np.savez(out, **data)
    return out


def test_correct_artifact_has_no_divergence(tmp_path):
    graph = _graph(tmp_path / "g.onnx")
    npz = _export(graph, tmp_path / "g.npz")
    probes = build_probes(graph)
    assert not _failing(graph, npz, probes).any()
    l2 = first_divergences(graph, npz, probes.rows)
    assert l2["divergent_nodes"] == 0 and not l2["base_value_differs"]


def test_a_shifted_scaler_offset_is_localized_to_its_feature(tmp_path):
    graph = _graph(tmp_path / "g.onnx")
    npz = _export(graph, tmp_path / "g.npz")
    bad = _mutated(tmp_path, npz, "offset", lambda d: d["offset"].__setitem__(1, d["offset"][1] + 0.05))
    probes = build_probes(graph)
    failing = _failing(graph, bad, probes)
    assert failing.any()
    l1 = summarize_failures(graph, probes, failing)
    assert l1["scope"] in ("feature", "threshold") and l1["top_features"][0]["feature"] == 1
    l2 = first_divergences(graph, bad, probes.rows[failing])
    assert set(l2["by_kind"]) == {"scaler"} and {d["feature"] for d in l2["top_nodes"]} == {1}


def test_wrong_leaf_weights_are_localized_to_their_tree(tmp_path):
    graph = _graph(tmp_path / "g.onnx")
    npz = _export(graph, tmp_path / "g.npz")

    def negate_tree1(d):
        start, end = int(d["tree_offsets"][1]), int(d["tree_offsets"][2])
        d["leaf_scores"][start:end] *= -1

    bad = _mutated(tmp_path, npz, "leaf", negate_tree1)
    probes = build_probes(graph)
    failing = _failing(graph, bad, probes)
    l2 = first_divergences(graph, bad, probes.rows[failing])
    assert set(l2["by_kind"]) == {"leaf weight"} and {d["tree"] for d in l2["top_nodes"]} == {1}
