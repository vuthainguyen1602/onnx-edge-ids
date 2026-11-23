"""The reader refuses graphs it would only half-read.

The exporter and the probes read the Scaler and the TreeEnsembleClassifier and
nothing else, so any other operator that changes a value must make the reader
raise instead of being silently dropped from the NumPy artifact.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_edge_ids.onnx_artifact import read_onnx_forest  # noqa: E402

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

TREE = dict(
    nodes_treeids=[0, 0, 0], nodes_nodeids=[0, 1, 2], nodes_featureids=[0, 0, 0],
    nodes_values=[0.5, 0.0, 0.0], nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
    nodes_truenodeids=[1, 0, 0], nodes_falsenodeids=[2, 0, 0],
    class_treeids=[0, 0, 0, 0], class_nodeids=[1, 1, 2, 2], class_ids=[0, 1, 0, 1],
    class_weights=[1.0, 0.0, 0.0, 1.0], classlabels_int64s=[0, 1],
)


def _save(path, nodes, outputs=("label", "probabilities"), initializers=(), inputs=("X",)):
    g = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info(i, TensorProto.FLOAT, [None, 2]) for i in inputs],
        [helper.make_tensor_value_info(outputs[0], TensorProto.INT64, [None]),
         helper.make_tensor_value_info(outputs[1], TensorProto.FLOAT, [None, 2])],
        initializer=list(initializers))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 3)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return path


def _scaler(inp="X", out="Z"):
    return helper.make_node("Scaler", [inp], [out], domain="ai.onnx.ml", offset=[0.0, 0.0], scale=[1.0, 1.0])


def _ens(inp="Z", outs=("label", "probabilities")):
    return helper.make_node("TreeEnsembleClassifier", [inp], list(outs), domain="ai.onnx.ml", **TREE)


def test_converter_wrappers_are_accepted(tmp_path):
    # LightGBM via onnxmltools: Identity, Cast of the label, Mul of the probabilities by 1.
    one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one")
    nodes = [_scaler(), _ens(outs=("lt", "pt")),
             helper.make_node("Identity", ["lt"], ["li"]), helper.make_node("Cast", ["li"], ["label"], to=7),
             helper.make_node("Identity", ["pt"], ["pi"]), helper.make_node("Mul", ["pi", "one"], ["probabilities"])]
    assert read_onnx_forest(_save(tmp_path / "w.onnx", nodes, initializers=[one]))["n_features"] == 2


def test_argmax_label_is_accepted(tmp_path):
    nodes = [_scaler(), _ens(outs=("raw", "probabilities")),
             helper.make_node("ArgMax", ["probabilities"], ["label"], axis=1, keepdims=0)]
    read_onnx_forest(_save(tmp_path / "a.onnx", nodes))


def test_no_scaler_is_accepted(tmp_path):
    read_onnx_forest(_save(tmp_path / "n.onnx", [_ens(inp="X")]))


def test_preprocessing_operator_is_refused(tmp_path):
    imputer = helper.make_node("Imputer", ["X"], ["I"], domain="ai.onnx.ml",
                               imputed_value_floats=[0.0, 0.0], replaced_value_float=float("nan"))
    with pytest.raises(NotImplementedError, match="Scaler must read the graph input"):
        read_onnx_forest(_save(tmp_path / "i.onnx", [imputer, _scaler(inp="I"), _ens()]))


def test_preprocessing_without_scaler_is_refused(tmp_path):
    imputer = helper.make_node("Imputer", ["X"], ["I"], domain="ai.onnx.ml",
                               imputed_value_floats=[0.0, 0.0], replaced_value_float=float("nan"))
    with pytest.raises(NotImplementedError, match="must read the graph input"):
        read_onnx_forest(_save(tmp_path / "i2.onnx", [imputer, _ens(inp="I")]))


def test_mul_by_other_constant_is_refused(tmp_path):
    two = numpy_helper.from_array(np.array([2.0], dtype=np.float32), "two")
    nodes = [_scaler(), _ens(outs=("label", "pt")), helper.make_node("Mul", ["pt", "two"], ["probabilities"])]
    with pytest.raises(NotImplementedError, match="Mul"):
        read_onnx_forest(_save(tmp_path / "m.onnx", nodes, initializers=[two]))


def test_postprocessing_operator_is_refused(tmp_path):
    nodes = [_scaler(), _ens(outs=("label", "pt")), helper.make_node("Sqrt", ["pt"], ["probabilities"])]
    with pytest.raises(NotImplementedError, match="Sqrt"):
        read_onnx_forest(_save(tmp_path / "s.onnx", nodes))


def test_narrowing_cast_is_refused(tmp_path):
    nodes = [_scaler(), _ens(outs=("label", "pt")),
             helper.make_node("Cast", ["pt"], ["p16"], to=10), helper.make_node("Cast", ["p16"], ["probabilities"], to=1)]
    with pytest.raises(NotImplementedError, match="Cast"):
        read_onnx_forest(_save(tmp_path / "c.onnx", nodes))


def test_dangling_operator_is_refused(tmp_path):
    # An extra node off the output path still means the graph is not the supported form.
    nodes = [_scaler(), _ens(), helper.make_node("Relu", ["Z"], ["unused"])]
    with pytest.raises(NotImplementedError, match="Relu"):
        read_onnx_forest(_save(tmp_path / "d.onnx", nodes))


def test_swapped_outputs_are_refused(tmp_path):
    nodes = [_scaler(), _ens(outs=("lt", "pt")),
             helper.make_node("Identity", ["pt"], ["label"]), helper.make_node("Identity", ["lt"], ["probabilities"])]
    with pytest.raises(NotImplementedError, match="First graph output"):
        read_onnx_forest(_save(tmp_path / "x.onnx", nodes))
