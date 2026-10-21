"""The generic ONNX tree-ensemble traversal must agree with ONNX Runtime and, on the
deployed artifact, reproduce the threshold-sharing numbers quoted in the paper."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("onnx")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import ARTIFACTS
from edgeids.onnx_forest import load_onnx_forest, path_rows, predict, reaches, threshold_sharing

DEPLOYED = os.path.join(ARTIFACTS, "ids_rf.onnx")
ort = pytest.importorskip("onnxruntime")


def _ort_labels(path, x):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return np.asarray(s.run(None, {s.get_inputs()[0].name: x})[0]).reshape(-1)


@pytest.mark.skipif(not os.path.exists(DEPLOYED), reason="deployed artifact not present")
def test_deployed_forest_parses_and_shares_thresholds_as_in_the_paper():
    f = load_onnx_forest(DEPLOYED)
    assert f.n_trees == 200 and f.n_targets == 2 and f.post_transform == "NONE"
    sharing = threshold_sharing(f)
    assert sharing["internal_nodes"] == 95741
    assert sharing["distinct_feature_threshold_pairs"] == 815
    assert sharing["trees_per_pair_max"] == 200


@pytest.mark.skipif(not os.path.exists(DEPLOYED), reason="deployed artifact not present")
def test_float32_traversal_matches_onnx_runtime_on_random_rows_and_probes():
    f = load_onnx_forest(DEPLOYED)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, f.n_features)).astype(np.float32) * 1000
    assert np.array_equal(predict(f, x, np.float32), _ort_labels(DEPLOYED, x))

    internal = np.flatnonzero(~f.is_leaf)
    targets = np.sort(rng.choice(internal, size=500, replace=False))
    rows = path_rows(f, targets, x[rng.choice(len(x), size=len(targets))])
    assert reaches(f, rows, targets).mean() > 0.98
    feat = f.feature[targets]
    rows[np.arange(len(targets)), feat] = (f.threshold[targets].astype(np.float64) / f.scale[feat] + f.offset[feat]).astype(np.float32)
    assert np.array_equal(predict(f, rows, np.float32), _ort_labels(DEPLOYED, rows))


def test_skl2onnx_random_forest_export_round_trips_through_the_generic_parser(tmp_path):
    skl2onnx = pytest.importorskip("skl2onnx")
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(1)
    x = rng.normal(size=(3000, 6))
    y = ((x[:, 0] + x[:, 1] ** 2) > 0.8).astype(int)
    pipe = Pipeline([("s", StandardScaler()), ("m", RandomForestClassifier(n_estimators=20, max_depth=6, random_state=0))]).fit(x, y)
    model = skl2onnx.convert_sklearn(pipe, initial_types=[("input", FloatTensorType([None, 6]))],
                                     options={id(pipe.steps[-1][1]): {"zipmap": False}},
                                     target_opset={"": 17, "ai.onnx.ml": 3})
    path = tmp_path / "rf.onnx"
    path.write_bytes(model.SerializeToString())

    f = load_onnx_forest(str(path))
    assert f.n_trees == 20
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    xt = rng.normal(size=(5000, 6)).astype(np.float32)
    out = s.run(None, {"input": xt})
    prob_label = (np.asarray(out[1])[:, 1] > 0.5).astype(np.int64)
    # the probability-derived label is what exp5 compares against; the raw label output
    # changed meaning in ONNX Runtime 1.26/1.27 for this single-weight binary export
    assert np.array_equal(predict(f, xt, np.float32), prob_label)
    assert np.mean(predict(f, xt, np.float64) == pipe.predict(xt.astype(np.float64))) > 0.999
