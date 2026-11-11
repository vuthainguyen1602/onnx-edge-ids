"""A converter defect the parity gate catches: HistGradientBoosting thresholds in skl2onnx 1.20.0.

scikit-learn compares a float32 input, widened to float64, with a float64
threshold. skl2onnx 1.20.0 (the latest release on PyPI when this was run)
writes that threshold rounded to the nearest float32 for HistGradientBoosting,
although it adjusts it for decision trees and forests. When the float32 lands
above the float64 threshold, the float32 input equal to it goes right in
scikit-learn and left in ONNX Runtime. Upstream fixed this on main
(onnx/sklearn-onnx pull request 1227, merged 2026-02-25).

No dataset is needed. A classifier is trained on scikit-learn's breast-cancer
data in its native float64, exported with default settings, and both sides are
given the same float32 inputs: the training rows, and for every split the
largest float32 not above its threshold and that value's two neighbours,
written into the mean row. Run it once per skl2onnx version:

    python exp12_hgb_threshold_rounding.py                       # installed release
    pip install git+https://github.com/onnx/sklearn-onnx.git@main && python exp12_hgb_threshold_rounding.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import RESULTS


def convert(model, n_features):
    from skl2onnx import convert_sklearn
    from skl2onnx.common import _container
    from skl2onnx.common.data_types import FloatTensorType

    kwargs = dict(initial_types=[("x", FloatTensorType([None, n_features]))], options={id(model): {"zipmap": False}})
    try:
        return convert_sklearn(model, **kwargs), False
    except ValueError:
        # skl2onnx 1.20.0 hands numpy booleans to an INTS attribute and protobuf refuses them.
        orig = _container.ModelComponentContainer.add_node

        def add_node(self, *a, **k):
            for key, v in list(k.items()):
                if isinstance(v, (list, tuple)) and len(v) and all(isinstance(e, (bool, np.bool_, int, np.integer)) for e in v):
                    k[key] = [int(e) for e in v]
            return orig(self, *a, **k)

        _container.ModelComponentContainer.add_node = add_node
        return convert_sklearn(model, **kwargs), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import onnxruntime as ort
    import skl2onnx
    import sklearn
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import HistGradientBoostingClassifier

    x64, y = load_breast_cancer(return_X_y=True)
    x32 = x64.astype(np.float32)
    model = HistGradientBoostingClassifier(max_iter=50, random_state=0).fit(x64, y)
    onx, needed_workaround = convert(model, x64.shape[1])
    session = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])

    def compare(rows):
        lab, prob = session.run(None, {"x": rows})
        delta = np.abs(prob - model.predict_proba(rows)).max(axis=1)
        return {"rows": int(len(rows)), "label_flips": int((lab != model.predict(rows)).sum()),
                "rows_with_probability_delta_over_1e-4": int((delta > 1e-4).sum()),
                "max_probability_delta": float(delta.max())}, lab

    base = x32.mean(axis=0).astype(np.float32)
    probes, meta = [], []
    for iteration in model._predictors:
        for predictor in iteration:
            nodes = predictor.nodes
            for node in nodes[~nodes["is_leaf"].astype(bool)]:
                t, f = float(node["num_threshold"]), int(node["feature_idx"])
                lo = np.float32(t)
                if float(lo) > t:
                    lo = np.nextafter(lo, np.float32(-np.inf))
                for v in (lo, np.nextafter(lo, np.float32(np.inf)), np.nextafter(lo, np.float32(-np.inf))):
                    row = base.copy()
                    row[f] = v
                    probes.append(row)
                    meta.append((f, t, float(v)))
    probes = np.array(probes, dtype=np.float32)

    on_train, _ = compare(x32)
    on_probes, lab = compare(probes)
    flipped = np.flatnonzero(lab != model.predict(probes))
    report = {
        "versions": {"sklearn": sklearn.__version__, "skl2onnx": skl2onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
        "default_conversion_needed_bool_workaround": needed_workaround,
        "split_nodes": int(len(meta) // 3),
        "thresholds_not_float32_representable": int(sum(float(np.float32(t)) != t for _, t, _ in meta) // 3),
        "training_rows_as_float32": on_train,
        "boundary_probes": on_probes,
        "example_flips": [{"feature": meta[i][0], "float64_threshold": meta[i][1],
                           "float32_of_threshold": float(np.float32(meta[i][1])), "input": meta[i][2]} for i in flipped[:3]],
    }
    print(json.dumps(report, indent=2))
    out = args.out or os.path.join(RESULTS, f"exp12_hgb_threshold_rounding_skl2onnx_{skl2onnx.__version__}.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
